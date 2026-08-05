import math
import os
import random
import time
from collections import namedtuple, Counter
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

Position = namedtuple("Position", ["r", "c"])
Action = namedtuple("Action", ["label", "direction", "target"])

RED = "红"
BLUE = "蓝"

START_POSITIONS = {
    RED: [Position(0, 0), Position(0, 1), Position(0, 2), Position(1, 0), Position(1, 1), Position(2, 0)],
    BLUE: [Position(4, 4), Position(4, 3), Position(4, 2), Position(3, 4), Position(3, 3), Position(2, 4)],
}

GOAL_CORNERS = {
    RED: Position(4, 4),
    BLUE: Position(0, 0),
}

MOVE_OFFSETS = {
    RED: [Position(0, 1), Position(1, 0), Position(1, 1)],
    BLUE: [Position(0, -1), Position(-1, 0), Position(-1, -1)],
}

BOARD_SIZE = 5
ACTION_SPACE_SIZE = 6 * 3
BOARD_CHANNELS = 13
LEGACY_FEATURE_SIZE = BOARD_SIZE * BOARD_SIZE * BOARD_CHANNELS + 2
FEATURE_SIZE = BOARD_SIZE * BOARD_SIZE * BOARD_CHANNELS + 2 + 6 + 2 + 4
MODEL_ARCHIVE_VERSION = "einstein-ai-model-v2"
DEFAULT_MODEL_PATHS = {
    "mlp": "einstein_value_model_mlp.pt",
    "gnn": "einstein_value_model_gnn.pt",
}
LEGACY_MODEL_PATHS = {
    "mlp": "einstein_value_model.pt",
}

PLAYER_NAMES = {
    RED: "RED",
    BLUE: "BLUE",
    None: None,
}

ADJACENCY = {}
for r in range(BOARD_SIZE):
    for c in range(BOARD_SIZE):
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    neighbors.append((nr, nc))
        ADJACENCY[(r, c)] = neighbors


def _extract_state_dict(saved_data, prefer_best=True):
    if isinstance(saved_data, dict) and prefer_best and "best_state_dict" in saved_data:
        return saved_data["best_state_dict"]
    if isinstance(saved_data, dict) and "state_dict" in saved_data:
        return saved_data["state_dict"]
    if isinstance(saved_data, dict) and "history" in saved_data:
        history = saved_data["history"]
        if not history:
            raise ValueError("Saved model history is empty.")
        return history[-1]["state_dict"]
    return saved_data


def _state_dict_shape(state_dict, key):
    value = state_dict.get(key) if isinstance(state_dict, dict) else None
    return tuple(value.shape) if hasattr(value, "shape") else None


def _infer_mlp_shape(state_dict):
    layer_shape = _state_dict_shape(state_dict, "layer1.weight")
    hidden_shape = _state_dict_shape(state_dict, "layer2.weight")
    input_size = layer_shape[1] if layer_shape else FEATURE_SIZE
    hidden_size = hidden_shape[0] if hidden_shape else 128
    return input_size, hidden_size


def _infer_gnn_hidden_size(state_dict):
    node_shape = _state_dict_shape(state_dict, "node_emb.weight")
    return node_shape[0] if node_shape else 64


def _clone_state_dict(state_dict):
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _load_state_dict(module, state_dict):
    device = getattr(module, "device", "cpu")
    try:
        module.load_state_dict({key: value.to(device) for key, value in state_dict.items()})
    except RuntimeError as ex:
        expected = getattr(module, "input_size", FEATURE_SIZE)
        raise RuntimeError(
            f"模型结构与当前网络不匹配，当前编码维度为 {expected}。"
            f"请重新训练模型，或加载与该版本代码匹配的模型文件。原始错误：{ex}"
        ) from ex


def default_model_path(value_kind):
    return DEFAULT_MODEL_PATHS.get(value_kind)


def resolve_model_path(value_kind, path=None):
    if path:
        return path
    preferred = DEFAULT_MODEL_PATHS.get(value_kind)
    if preferred:
        return preferred
    return LEGACY_MODEL_PATHS.get(value_kind)


def resolve_ensemble_paths(model_path=None, mlp_path=None, gnn_path=None):
    if mlp_path or gnn_path:
        return mlp_path, gnn_path
    if not model_path:
        return DEFAULT_MODEL_PATHS["mlp"], DEFAULT_MODEL_PATHS["gnn"]

    root, ext = os.path.splitext(model_path)
    ext = ext or ".pt"
    lower_root = root.lower()
    if lower_root.endswith("_gnn"):
        return f"{root[:-4]}_mlp{ext}", model_path
    if lower_root.endswith("_mlp"):
        return model_path, f"{root[:-4]}_gnn{ext}"
    if os.path.basename(model_path) == LEGACY_MODEL_PATHS["mlp"]:
        return model_path, DEFAULT_MODEL_PATHS["gnn"]
    return model_path, f"{root}_gnn{ext}"


def _history_score(entry):
    metadata = entry.get("metadata") or {}
    if metadata.get("accepted") is False:
        return -float("inf")
    value = metadata.get("best_win_rate")
    if value is not None:
        return value
    value = metadata.get("validation_loss")
    if value is not None:
        return -value
    return -float("inf")


def pick_best_history_entry(history):
    if not history:
        raise ValueError("Saved model history is empty.")
    scored = [(_history_score(entry), index, entry) for index, entry in enumerate(history)]
    best_score, _, best_entry = max(scored, key=lambda item: (item[0], item[1]))
    if best_score == -float("inf"):
        return history[-1]
    return best_entry


def save_model_archive(model, path, value_kind, metadata=None, replay_buffer=None, history_limit=None):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required to save model archives.")
    metadata = dict(metadata or {})
    metadata.setdefault("value_kind", value_kind)
    metadata.setdefault("feature_size", getattr(model, "input_size", FEATURE_SIZE))
    metadata.setdefault("archive_version", MODEL_ARCHIVE_VERSION)
    metadata.setdefault("timestamp", time.time())

    history = []
    if os.path.exists(path):
        try:
            data = torch.load(path, map_location=getattr(model, "device", "cpu"))
            if isinstance(data, dict) and "history" in data:
                history = list(data["history"])
            else:
                history = [{
                    "state_dict": _extract_state_dict(data),
                    "timestamp": time.time(),
                    "metadata": {"imported": True},
                }]
        except Exception as ex:
            metadata["previous_archive_warning"] = str(ex)

    state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history.append({"state_dict": state_dict, "timestamp": time.time(), "metadata": metadata})
    if history_limit is not None and history_limit > 0:
        history = history[-history_limit:]
    best_entry = pick_best_history_entry(history)

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save({
        "format": MODEL_ARCHIVE_VERSION,
        "value_kind": value_kind,
        "feature_size": metadata.get("feature_size", FEATURE_SIZE),
        "state_dict": state_dict,
        "best_state_dict": best_entry["state_dict"],
        "history": history,
        "replay_buffer": replay_buffer or [],
        "metadata": metadata,
    }, path)


def encode_for_network(value_network, state, die=None):
    input_size = getattr(value_network, "input_size", FEATURE_SIZE)
    if input_size == LEGACY_FEATURE_SIZE:
        return state.encode_legacy()
    return state.encode(die=die)


def clamp_value(value):
    return max(-1.0, min(1.0, float(value)))


def predict_for_player(value_network, state, player):
    """Value networks predict from state.turn; convert to any player's view."""
    value = value_network.predict(state)
    return value if state.turn == player else -value


def evaluate_action_value(value_network, state, label, target):
    root_player = state.turn
    captured = state.board.get((target.r, target.c))
    origin = state.label_position(root_player, label)
    next_state = state.copy()
    next_state.move(label, target)
    if next_state.winner == root_player:
        return 1.0

    value = predict_for_player(value_network, next_state, root_player)
    if captured is not None and captured[0] != root_player:
        value += 0.08 + 0.01 * captured[1]
    if origin is not None:
        goal = GOAL_CORNERS[root_player]
        old_dist = abs(goal.r - origin.r) + abs(goal.c - origin.c)
        new_dist = abs(goal.r - target.r) + abs(goal.c - target.c)
        value += 0.02 * (old_dist - new_dist)
    return clamp_value(value)


def parse_order_string(text):
    if not text:
        return None
    try:
        labels = [int(token) for token in text.replace("，", ",").split(",") if token.strip()]
    except ValueError:
        return None
    if len(labels) != 6 or len(set(labels)) != 6 or any(label < 1 or label > 6 for label in labels):
        return None
    return labels


def layout_from_order(order, player):
    return {label: START_POSITIONS[player][index] for index, label in enumerate(order)}


def other_player(player):
    return BLUE if player == RED else RED


class GameState:
    def __init__(self, board=None, turn=RED, winner=None, red_layout=None, blue_layout=None,
                 winner_reason=None):
        if board is None:
            board = {}
            if red_layout is None:
                red_layout = {i + 1: pos for i, pos in enumerate(START_POSITIONS[RED])}
            if blue_layout is None:
                blue_layout = {i + 1: pos for i, pos in enumerate(START_POSITIONS[BLUE])}
            for label, pos in red_layout.items():
                board[(pos.r, pos.c)] = (RED, label)
            for label, pos in blue_layout.items():
                board[(pos.r, pos.c)] = (BLUE, label)
        self.board = board
        self.turn = turn
        self.winner = winner
        self.winner_reason = winner_reason

    @classmethod
    def from_einstein(cls, game):
        state = cls(board=dict(game.board), turn=game.turn, winner=game.winner)
        return state

    def copy(self):
        return GameState(
            board=dict(self.board),
            turn=self.turn,
            winner=self.winner,
            winner_reason=self.winner_reason,
        )

    def clone(self):
        return self.copy()

    def active_labels(self, player):
        return sorted(label for (owner, label) in self.board.values() if owner == player)

    def label_position(self, player, label):
        for (r, c), (owner, num) in self.board.items():
            if owner == player and num == label:
                return Position(r, c)
        return None

    def possible_moves(self, player, label):
        origin = self.label_position(player, label)
        if origin is None:
            return []
        moves = []
        for offset in MOVE_OFFSETS[player]:
            target = Position(origin.r + offset.r, origin.c + offset.c)
            if 0 <= target.r < BOARD_SIZE and 0 <= target.c < BOARD_SIZE:
                moves.append(target)
        return moves

    def legal_move_targets(self, player, label):
        targets = []
        for target in self.possible_moves(player, label):
            current = self.board.get((target.r, target.c))
            if current is None or current[0] != player:
                targets.append(target)
        return targets

    def choose_labels(self, player, die):
        active = self.active_labels(player)
        if not active:
            return []
        movable = [label for label in active if self.legal_move_targets(player, label)]
        if not movable:
            return []
        if die in movable:
            return [die]
        if die in active:
            closest = min(abs(label - die) for label in movable)
            return [label for label in movable if abs(label - die) == closest]
        best_diff = min(abs(label - die) for label in movable)
        return [label for label in movable if abs(label - die) == best_diff]

    def legal_actions(self, die):
        actions = []
        for label in self.choose_labels(self.turn, die):
            origin = self.label_position(self.turn, label)
            if origin is None:
                continue
            for direction, offset in enumerate(MOVE_OFFSETS[self.turn]):
                target = Position(origin.r + offset.r, origin.c + offset.c)
                if target in self.legal_move_targets(self.turn, label):
                    actions.append(Action(label, direction, target))
        return actions

    def action_to_index(self, action):
        if action is None or action[0] is None:
            return None
        if isinstance(action, Action):
            label, direction = action.label, action.direction
        else:
            label, target = action
            origin = self.label_position(self.turn, label)
            if origin is None:
                return None
            direction = None
            for index, offset in enumerate(MOVE_OFFSETS[self.turn]):
                candidate = Position(origin.r + offset.r, origin.c + offset.c)
                if candidate == target:
                    direction = index
                    break
            if direction is None:
                return None
        if label < 1 or label > 6 or direction < 0 or direction > 2:
            return None
        return (label - 1) * 3 + direction

    def index_to_action(self, index, die=None):
        if index is None or index < 0 or index >= ACTION_SPACE_SIZE:
            return None
        label = index // 3 + 1
        direction = index % 3
        if die is not None and label not in self.choose_labels(self.turn, die):
            return None
        origin = self.label_position(self.turn, label)
        if origin is None:
            return None
        offset = MOVE_OFFSETS[self.turn][direction]
        target = Position(origin.r + offset.r, origin.c + offset.c)
        action = Action(label, direction, target)
        if target not in self.legal_move_targets(self.turn, label):
            return None
        return action

    def legal_action_mask(self, die):
        mask = [0] * ACTION_SPACE_SIZE
        for action in self.legal_actions(die):
            index = self.action_to_index(action)
            if index is not None:
                mask[index] = 1
        return mask

    def get_available_actions(self, die):
        return [(action.label, action.target) for action in self.legal_actions(die)]

    def apply_action(self, action):
        next_state = self.copy()
        if action is None or action[0] is None:
            next_state.skip_turn()
            return next_state
        if isinstance(action, Action):
            label, target = action.label, action.target
        else:
            label, target = action
        next_state.move(label, target)
        return next_state

    def move(self, label, target):
        origin = self.label_position(self.turn, label)
        if origin is None:
            raise ValueError("棋子不存在")
        if target not in self.legal_move_targets(self.turn, label):
            raise ValueError("目标位置非法")
        player = self.turn
        origin_key = (origin.r, origin.c)
        target_key = (target.r, target.c)
        self.board.pop(origin_key)
        self.board.pop(target_key, None)
        self.board[target_key] = (player, label)
        if target == GOAL_CORNERS[player]:
            self.winner = player
            self.winner_reason = "goal"
        else:
            opponent = other_player(player)
            if not self.active_labels(opponent):
                self.winner = player
                self.winner_reason = "capture_all"
        self.turn = other_player(player)

    def skip_turn(self):
        self.turn = other_player(self.turn)
        if not self.active_labels(other_player(self.turn)):
            self.winner = self.turn
            self.winner_reason = "capture_all"

    def is_terminal(self):
        return self.winner is not None

    def winner_value(self, player):
        if self.winner is None:
            return 0.0
        return 1.0 if self.winner == player else -1.0

    def encode_legacy(self):
        features = [0.0] * LEGACY_FEATURE_SIZE
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                base = (r * BOARD_SIZE + c) * BOARD_CHANNELS
                piece = self.board.get((r, c))
                if piece is None:
                    continue
                owner, label = piece
                if owner == RED:
                    features[base + (label - 1)] = 1.0
                else:
                    features[base + 6 + (label - 1)] = 1.0
        features[-2] = 1.0 if self.turn == RED else 0.0
        features[-1] = 1.0 if self.turn == BLUE else 0.0
        return features

    def encode(self, die=None, perspective=True):
        features = [0.0] * FEATURE_SIZE
        current = self.turn

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board.get((r, c))
                if piece is None:
                    continue
                owner, label = piece
                rr, cc = r, c
                channel_owner = owner
                if perspective and current == BLUE:
                    rr = BOARD_SIZE - 1 - r
                    cc = BOARD_SIZE - 1 - c
                    channel_owner = RED if owner == BLUE else BLUE
                elif perspective:
                    channel_owner = RED if owner == current else BLUE

                base = (rr * BOARD_SIZE + cc) * BOARD_CHANNELS
                if channel_owner == RED:
                    features[base + (label - 1)] = 1.0
                else:
                    features[base + 6 + (label - 1)] = 1.0
                features[base + 12] = 1.0

        offset = BOARD_SIZE * BOARD_SIZE * BOARD_CHANNELS
        features[offset] = 1.0 if self.turn == RED else 0.0
        features[offset + 1] = 1.0 if self.turn == BLUE else 0.0

        if die is not None and 1 <= die <= 6:
            features[offset + 2 + (die - 1)] = 1.0

        own = self.turn if perspective else RED
        opp = other_player(own)
        own_labels = self.active_labels(own)
        opp_labels = self.active_labels(opp)
        count_offset = offset + 8
        features[count_offset] = len(own_labels) / 6.0
        features[count_offset + 1] = len(opp_labels) / 6.0

        def normalized_distances(player):
            values = []
            for (r, c), (owner, _) in self.board.items():
                if owner != player:
                    continue
                goal = GOAL_CORNERS[player]
                values.append((abs(goal.r - r) + abs(goal.c - c)) / (2 * (BOARD_SIZE - 1)))
            if not values:
                return 1.0, 1.0
            return sum(values) / len(values), min(values)

        own_avg, own_min = normalized_distances(own)
        opp_avg, opp_min = normalized_distances(opp)
        dist_offset = count_offset + 2
        features[dist_offset] = own_avg
        features[dist_offset + 1] = own_min
        features[dist_offset + 2] = opp_avg
        features[dist_offset + 3] = opp_min
        return features

    def square_attacked_by(self, player, target):
        for label in self.active_labels(player):
            if target in self.legal_move_targets(player, label):
                return True
        return False

    def heuristic_value(self, player):
        score = 0.0
        my_count = len(self.active_labels(player))
        opp_count = len(self.active_labels(other_player(player)))
        opponent = other_player(player)
        score += (my_count - opp_count) * 18
        for (r, c), (owner, label) in self.board.items():
            goal = GOAL_CORNERS[owner]
            dist = abs(goal.r - r) + abs(goal.c - c)
            factor = 1 if owner == player else -1
            progress = (2 * (BOARD_SIZE - 1) - dist)
            score += factor * progress * 2.0
            score += factor * (7 - label) * 0.5
            if owner == player and self.square_attacked_by(opponent, Position(r, c)):
                score -= 4.0 + 0.3 * label
            if owner == opponent:
                opp_goal = GOAL_CORNERS[owner]
                opp_dist = abs(opp_goal.r - r) + abs(opp_goal.c - c)
                if opp_dist <= 2:
                    score -= (3 - opp_dist) * 6.0
            if owner == player and Position(r, c) == GOAL_CORNERS[player]:
                score += 100.0
        return math.tanh(score / 40.0)


class ValueNetwork:
    def predict(self, state: GameState) -> float:
        raise NotImplementedError


class HeuristicValueNetwork(ValueNetwork):
    def predict(self, state: GameState) -> float:
        return state.heuristic_value(state.turn)


if TORCH_AVAILABLE:
    class MLPValueNetwork(ValueNetwork, nn.Module):
        def __init__(self, hidden_size=128, input_size=FEATURE_SIZE, device="cpu"):
            nn.Module.__init__(self)
            self.device = device
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.layer1 = nn.Linear(input_size, hidden_size)
            self.layer2 = nn.Linear(hidden_size, hidden_size)
            self.layer3 = nn.Linear(hidden_size, 1)
            self.to(device)

        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = torch.tanh(self.layer3(x))
            return x

        def predict(self, state: GameState) -> float:
            self.eval()
            with torch.no_grad():
                array = torch.tensor(encode_for_network(self, state), dtype=torch.float32, device=self.device).unsqueeze(0)
                return self.forward(array).item()

        def save(self, path: str):
            save_model_archive(self, path, "mlp", metadata={
                "saved_by": "MLPValueNetwork.save",
                "hidden_size": self.hidden_size,
            })

        @classmethod
        def load(cls, path: str, device="cpu"):
            data = torch.load(path, map_location=device)
            state_dict = _extract_state_dict(data)
            input_size, hidden_size = _infer_mlp_shape(state_dict)
            model = cls(hidden_size=hidden_size, input_size=input_size, device=device)
            _load_state_dict(model, state_dict)
            return model

    class GNNValueNetwork(ValueNetwork, nn.Module):
        def __init__(self, hidden_size=64, device="cpu"):
            nn.Module.__init__(self)
            self.device = device
            self.input_size = FEATURE_SIZE
            self.hidden_size = hidden_size
            self.node_emb = nn.Linear(13, hidden_size)
            self.conv = nn.Linear(hidden_size * 2, hidden_size)
            self.output = nn.Linear(hidden_size, 1)
            self.to(device)

        def forward(self, state_tensor):
            # Only use board features, ignore turn (last 2 features)
            board_features = state_tensor[:, :BOARD_SIZE * BOARD_SIZE * 13]
            nodes = board_features.view(-1, BOARD_SIZE * BOARD_SIZE, 13)
            x = F.relu(self.node_emb(nodes))
            for _ in range(2):
                neighbor_embeds = []
                for idx in range(BOARD_SIZE * BOARD_SIZE):
                    row = idx // BOARD_SIZE
                    col = idx % BOARD_SIZE
                    neighbors = [n[0] * BOARD_SIZE + n[1] for n in ADJACENCY[(row, col)]]
                    if neighbors:
                        neigh = x[:, neighbors, :].mean(dim=1)
                    else:
                        neigh = torch.zeros_like(x[:, idx, :])
                    neighbor_embeds.append(neigh)
                neighbor_embeds = torch.stack(neighbor_embeds, dim=1)
                x = F.relu(self.conv(torch.cat([x, neighbor_embeds], dim=-1)))
            graph = x.mean(dim=1)
            return torch.tanh(self.output(graph))

        def predict(self, state: GameState) -> float:
            self.eval()
            with torch.no_grad():
                array = torch.tensor(encode_for_network(self, state), dtype=torch.float32, device=self.device).unsqueeze(0)
                return self.forward(array).item()

        def save(self, path: str):
            save_model_archive(self, path, "gnn", metadata={
                "saved_by": "GNNValueNetwork.save",
                "hidden_size": self.hidden_size,
            })

        @classmethod
        def load(cls, path: str, device="cpu"):
            data = torch.load(path, map_location=device)
            state_dict = _extract_state_dict(data)
            hidden_size = _infer_gnn_hidden_size(state_dict)
            model = cls(hidden_size=hidden_size, device=device)
            _load_state_dict(model, state_dict)
            return model


class EnsembleValueNetwork(ValueNetwork):
    def __init__(self, mlp_path=None, gnn_path=None, device="cpu",
                 mlp_weight=0.45, gnn_weight=0.45, heuristic_weight=0.10):
        self.device = device
        self.load_errors = []
        self.mlp = None
        self.gnn = None
        if mlp_path and os.path.exists(mlp_path):
            try:
                self.mlp = MLPValueNetwork.load(mlp_path, device=device)
            except Exception as ex:
                self.load_errors.append(f"MLP 加载失败: {ex}")
        if gnn_path and os.path.exists(gnn_path):
            try:
                self.gnn = GNNValueNetwork.load(gnn_path, device=device)
            except Exception as ex:
                self.load_errors.append(f"GNN 加载失败: {ex}")
        self.heuristic = HeuristicValueNetwork()
        self.mlp_weight = mlp_weight
        self.gnn_weight = gnn_weight
        self.heuristic_weight = heuristic_weight

    def component_values(self, state: GameState):
        values = {}
        if self.mlp is not None:
            values["mlp"] = self.mlp.predict(state)
        if self.gnn is not None:
            values["gnn"] = self.gnn.predict(state)
        values["heuristic"] = self.heuristic.predict(state)
        return values

    def predict(self, state: GameState) -> float:
        values = self.component_values(state)
        learned = []
        if "mlp" in values:
            learned.append(("mlp", values["mlp"], self.mlp_weight))
        if "gnn" in values:
            learned.append(("gnn", values["gnn"], self.gnn_weight))
        if not learned:
            return values["heuristic"]

        if len(learned) >= 2:
            disagreement = abs(values["mlp"] - values["gnn"])
            heuristic_weight = min(0.45, self.heuristic_weight + 0.30 * disagreement)
        else:
            heuristic_weight = max(0.25, self.heuristic_weight)
        learned_weight = 1.0 - heuristic_weight
        learned_total = sum(weight for _, _, weight in learned)
        blended = heuristic_weight * values["heuristic"]
        for _, value, weight in learned:
            blended += learned_weight * (weight / learned_total) * value
        return clamp_value(blended)


def make_value_network(kind="heuristic", device="cpu", model_path=None, mlp_path=None, gnn_path=None) -> ValueNetwork:
    if kind == "ensemble" and TORCH_AVAILABLE:
        mlp_path, gnn_path = resolve_ensemble_paths(model_path=model_path, mlp_path=mlp_path, gnn_path=gnn_path)
        return EnsembleValueNetwork(mlp_path=mlp_path, gnn_path=gnn_path, device=device)
    if kind == "gnn" and TORCH_AVAILABLE:
        path = gnn_path or model_path
        if path and os.path.exists(path):
            return GNNValueNetwork.load(path, device=device)
        return GNNValueNetwork(device=device)
    if kind == "mlp" and TORCH_AVAILABLE:
        path = mlp_path or model_path
        if path and os.path.exists(path):
            return MLPValueNetwork.load(path, device=device)
        return MLPValueNetwork(device=device)
    return HeuristicValueNetwork()


def find_best_start_layout(value_network, player=RED, opponent_layout=None):
    from itertools import permutations

    if opponent_layout is None:
        opponent = BLUE if player == RED else RED
        opponent_layout = {i + 1: START_POSITIONS[opponent][i] for i in range(6)}

    best_score = -math.inf
    best_order = None
    for order in permutations(range(1, 7)):
        layout = layout_from_order(list(order), player)
        if player == RED:
            state = GameState(red_layout=layout, blue_layout=opponent_layout, turn=player)
        else:
            state = GameState(red_layout=opponent_layout, blue_layout=layout, turn=player)
        score = value_network.predict(state)
        if score > best_score:
            best_score = score
            best_order = list(order)
    return best_order, best_score


class MCTSNode:
    def __init__(self, state: GameState, die_value=None, parent=None, move=None, prior=1.0):
        self.state = state
        self.die_value = die_value
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.N = 0
        self.W = 0.0

    def q_value(self):
        return self.W / self.N if self.N else 0.0

    def u_value(self, c_puct=1.2):
        return c_puct * self.prior * math.sqrt(self.parent.N) / (1 + self.N) if self.parent else 0.0

    def score(self):
        return self.q_value() + self.u_value()

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self):
        die = self.die_value if self.die_value is not None else random.randint(1, 6)
        self.die_value = die
        actions = self.state.get_available_actions(die)
        if not actions:
            child_state = self.state.copy()
            child_state.skip_turn()
            self.children.append(MCTSNode(child_state, die_value=None, parent=self, move=None))
            return
        prior = 1.0 / len(actions)
        for label, target in actions:
            child_state = self.state.copy()
            child_state.move(label, target)
            self.children.append(MCTSNode(child_state, die_value=None, parent=self, move=(label, target), prior=prior))

    def best_child(self):
        def score_for_parent(child):
            value = child.q_value()
            if child.state.turn != self.state.turn:
                value = -value
            return value + child.u_value()
        return max(self.children, key=score_for_parent)

    def best_policy_child(self):
        return max(self.children, key=lambda child: child.N)


class ParallelMCTS:
    def __init__(self, value_network: ValueNetwork, simulations=120, workers=4, max_depth=40):
        self.value_network = value_network
        self.simulations = simulations
        self.workers = workers
        self.max_depth = max_depth

    def _simulate_tree(self, root_state, die, sims):
        root = MCTSNode(root_state.copy(), die_value=die)
        root.expand()
        for _ in range(sims):
            node = root
            path = [node]
            while not node.is_leaf() and not node.state.is_terminal():
                node = node.best_child()
                path.append(node)
            if not node.state.is_terminal():
                node.expand()
                if node.children:
                    node = random.choice(node.children)
                    path.append(node)
            value = self._evaluate(node, path[0].state.turn)
            self._backup(path, value)
        return root

    def _evaluate(self, node, root_player):
        player = node.state.turn
        if node.state.is_terminal():
            return node.state.winner_value(player)
        network_value = self.value_network.predict(node.state)
        rollout_value = self._rollout(node.state, player)
        return self._second_estimate(network_value, rollout_value)

    def _rollout(self, state, root_player):
        current = state.copy()
        for _ in range(self.max_depth):
            if current.is_terminal():
                return current.winner_value(root_player)
            die = random.randint(1, 6)
            actions = current.get_available_actions(die)
            if not actions:
                current.skip_turn()
                continue
            label, target = random.choice(actions)
            current.move(label, target)
        return current.heuristic_value(root_player)

    def _second_estimate(self, net_value, rollout_value):
        delta = rollout_value - net_value
        return 0.6 * net_value + 0.4 * rollout_value + 0.1 * delta * abs(delta)

    def _backup(self, path, value):
        for node in reversed(path):
            node.N += 1
            node.W += value
            value = -value

    def search(self, state, die):
        if state.is_terminal():
            return None, None
        if self.simulations < 1:
            return self._expected_search(state, die)
        workers = min(self.workers, self.simulations)
        per_worker = max(1, self.simulations // workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._simulate_tree, state, die, per_worker) for _ in range(workers)]
            roots = [future.result() for future in futures]

        aggregate = Counter()
        for root in roots:
            for child in root.children:
                key = child.move
                aggregate[key] += child.N

        if not aggregate:
            return self._expected_search(state, die)
        best_move, _ = max(aggregate.items(), key=lambda item: item[1])
        return best_move

    def _expected_search(self, state, die):
        best_move = None
        best_score = -math.inf
        actions = state.get_available_actions(die)
        for label, target in actions:
            value = evaluate_action_value(self.value_network, state, label, target)
            if value > best_score:
                best_score = value
                best_move = (label, target)
        return best_move


class HybridAI:
    def __init__(self, value_kind="heuristic", simulations=120, workers=4, time_limit=1.0,
                 device="cpu", model_path=None, mlp_path=None, gnn_path=None):
        self.value_kind = value_kind
        self.device = device
        self.value_network = make_value_network(
            value_kind,
            device=device,
            model_path=model_path,
            mlp_path=mlp_path,
            gnn_path=gnn_path,
        )
        self.mcts = ParallelMCTS(value_network=self.value_network, simulations=simulations, workers=workers)
        self.time_limit = time_limit
        self.simulations = simulations
        self.workers = workers

    def choose_action(self, state: GameState, die: int):
        move = self.choose_move(state, die)
        if move[0] is None:
            return None
        index = state.action_to_index(move)
        return state.index_to_action(index, die)

    def choose_move(self, state: GameState, die: int):
        actions = state.get_available_actions(die)
        if not actions:
            return None, None
        move = self.mcts.search(state, die)
        if move is None:
            return self._expected_search(state, die)
        return move

    def _expected_search(self, state: GameState, die: int):
        best_move = None
        best_score = -math.inf
        for label, target in state.get_available_actions(die):
            value = evaluate_action_value(self.value_network, state, label, target)
            if value > best_score:
                best_score = value
                best_move = (label, target)
        return best_move

    def load_model(self, path: str = None, mlp_path=None, gnn_path=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch required to load model.")
        if self.value_kind == "ensemble":
            mlp_path, gnn_path = resolve_ensemble_paths(model_path=path, mlp_path=mlp_path, gnn_path=gnn_path)
            self.value_network = EnsembleValueNetwork(mlp_path=mlp_path, gnn_path=gnn_path, device=self.device)
        elif self.value_kind == "mlp":
            path = resolve_model_path("mlp", path)
            self.value_network = MLPValueNetwork.load(path, device=self.device)
        elif self.value_kind == "gnn":
            path = resolve_model_path("gnn", path)
            self.value_network = GNNValueNetwork.load(path, device=self.device)
        else:
            raise ValueError("Heuristic 模型无需加载文件")
        self.mcts = ParallelMCTS(value_network=self.value_network, simulations=self.simulations, workers=self.workers)


class SelfPlayTrainer:
    def __init__(self, ai: HybridAI, episodes=20, max_moves=100, random_move_rate=0.1,
                 evaluate_games=20, evaluation_interval=1, history_capacity=50000,
                 history_sample_ratio=0.5, discount=0.98):
        self.ai = ai
        self.episodes = episodes
        self.max_moves = max_moves
        self.random_move_rate = random_move_rate
        self.evaluate_games = evaluate_games
        self.evaluation_interval = evaluation_interval
        self.history_capacity = history_capacity
        self.history_sample_ratio = history_sample_ratio
        self.discount = discount
        self.training_history = []
        self.last_episode_winner = None
        self.last_training_summary = {}

    def load_training_history(self, samples):
        if not samples or self.history_capacity <= 0:
            self.training_history = []
            return
        expected = getattr(self.ai.value_network, "input_size", FEATURE_SIZE)
        filtered = []
        for item in samples:
            try:
                features, value = item
            except (TypeError, ValueError):
                continue
            if len(features) == expected:
                filtered.append((features, value))
        self.training_history = filtered[-self.history_capacity:]

    def export_training_history(self):
        return list(self.training_history)

    def _remember_training_data(self, samples):
        if not samples or self.history_capacity <= 0:
            return
        self.training_history.extend(samples)
        if len(self.training_history) > self.history_capacity:
            self.training_history = self.training_history[-self.history_capacity:]

    def generate_episode(self):
        state = GameState()
        history = []
        while not state.is_terminal() and len(history) < self.max_moves:
            player = state.turn
            die = random.randint(1, 6)
            actions = state.get_available_actions(die)
            if not actions:
                play = (None, None)
            elif random.random() < self.random_move_rate:
                play = random.choice(actions)
            else:
                play = self.ai.choose_move(state, die)
                if play[0] is None:
                    play = random.choice(actions)
            history.append((state.copy(), die, play, player))
            if play[0] is None:
                state.skip_turn()
                continue
            label, target = play
            state.move(label, target)
        self.last_episode_winner = state.winner
        return history

    def generate_training_data(self):
        new_data = []
        previous_history = list(self.training_history)
        for _ in range(self.episodes):
            episode = self.generate_episode()
            final_winner = self.last_episode_winner
            for index, (state, die, _, player) in enumerate(episode):
                if final_winner is None:
                    value = 0.0
                else:
                    value = 1.0 if final_winner == player else -1.0
                    remaining = len(episode) - index - 1
                    value *= self.discount ** remaining
                new_data.append((encode_for_network(self.ai.value_network, state, die), value))

        replay_count = min(len(previous_history), int(len(new_data) * self.history_sample_ratio))
        replay_data = random.sample(previous_history, replay_count) if replay_count > 0 else []
        self._remember_training_data(new_data)
        return new_data + replay_data

    def evaluate_against_heuristic(self, games=20):
        baseline = HybridAI(value_kind="heuristic", simulations=0, workers=1)
        model_wins = 0
        for i in range(games):
            state = GameState()
            model_player = RED if i % 2 == 0 else BLUE
            moves = 0
            while not state.is_terminal() and moves < self.max_moves:
                die = random.randint(1, 6)
                ai = self.ai if state.turn == model_player else baseline
                action = ai.choose_move(state, die)
                if action[0] is None:
                    state.skip_turn()
                    moves += 1
                    continue
                label, target = action
                state.move(label, target)
                moves += 1
            if state.winner == model_player:
                model_wins += 1
        return model_wins / games if games else 0.0

    def evaluate_against_old_model(self, old_model_path, games=20):
        if not TORCH_AVAILABLE or not os.path.exists(old_model_path):
            return self.evaluate_against_heuristic(games)
        try:
            old_ai = HybridAI(value_kind=self.ai.value_kind, simulations=self.ai.simulations, workers=self.ai.workers)
            old_ai.load_model(old_model_path)
            baseline = old_ai
        except Exception:
            baseline = HybridAI(value_kind="heuristic", simulations=0, workers=1)
        model_wins = 0
        for i in range(games):
            state = GameState()
            model_player = RED if i % 2 == 0 else BLUE
            moves = 0
            while not state.is_terminal() and moves < self.max_moves:
                die = random.randint(1, 6)
                ai = self.ai if state.turn == model_player else baseline
                action = ai.choose_move(state, die)
                if action[0] is None:
                    state.skip_turn()
                    moves += 1
                    continue
                label, target = action
                state.move(label, target)
                moves += 1
            if state.winner == model_player:
                model_wins += 1
        return model_wins / games if games else 0.0

    def train_network(self, epochs=5, learning_rate=1e-3, batch_size=16, eval_against_old=False,
                      model_path=None, min_accept_win_rate=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("Torch is required for network training.")
        if not hasattr(self.ai.value_network, "train"):
            raise RuntimeError("Current value network does not support training.")

        original_state = _clone_state_dict(self.ai.value_network.state_dict())
        best_state = _clone_state_dict(self.ai.value_network.state_dict())
        best_win_rate = None
        final_win_rate = None
        accepted = True
        evaluation_target = "启发式 AI"
        if self.evaluate_games > 0:
            if eval_against_old and model_path and os.path.exists(model_path):
                initial_win_rate = self.evaluate_against_old_model(model_path, self.evaluate_games)
                evaluation_target = "旧模型"
                print(f"初始评估: 对旧模型胜率 {initial_win_rate:.2%}")
            else:
                initial_win_rate = self.evaluate_against_heuristic(self.evaluate_games)
                print(f"初始评估: 对启发式 AI 胜率 {initial_win_rate:.2%}")
            best_win_rate = initial_win_rate

        dataset = self.generate_training_data()
        optimizer = torch.optim.Adam(self.ai.value_network.parameters(), lr=learning_rate)
        device = getattr(self.ai.value_network, "device", "cpu")
        eval_interval = max(1, self.evaluation_interval)
        batch_size = max(1, batch_size)
        for epoch in range(epochs):
            random.shuffle(dataset)
            total_loss = 0.0
            for start in range(0, len(dataset), batch_size):
                batch = dataset[start:start + batch_size]
                inputs = torch.tensor([item[0] for item in batch], dtype=torch.float32, device=device)
                targets = torch.tensor([item[1] for item in batch], dtype=torch.float32, device=device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = self.ai.value_network(inputs)
                loss = F.mse_loss(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            avg_loss = total_loss / len(dataset) if dataset else 0.0
            print(f"Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}, dataset={len(dataset)}")
            if self.evaluate_games > 0 and (epoch + 1) % eval_interval == 0:
                if eval_against_old and model_path and os.path.exists(model_path):
                    win_rate = self.evaluate_against_old_model(model_path, self.evaluate_games)
                    print(f"  评估: 对旧模型胜率 {win_rate:.2%}")
                else:
                    win_rate = self.evaluate_against_heuristic(self.evaluate_games)
                    print(f"  评估: 对启发式 AI 胜率 {win_rate:.2%}")
                if best_win_rate is None or win_rate >= best_win_rate:
                    best_win_rate = win_rate
                    best_state = _clone_state_dict(self.ai.value_network.state_dict())
        if self.evaluate_games > 0:
            if eval_against_old and model_path and os.path.exists(model_path):
                final_win_rate = self.evaluate_against_old_model(model_path, self.evaluate_games)
                print(f"最终评估: 对旧模型胜率 {final_win_rate:.2%}")
            else:
                final_win_rate = self.evaluate_against_heuristic(self.evaluate_games)
                print(f"最终评估: 对启发式 AI 胜率 {final_win_rate:.2%}")
            if best_win_rate is None or final_win_rate >= best_win_rate:
                best_win_rate = final_win_rate
                best_state = _clone_state_dict(self.ai.value_network.state_dict())
            _load_state_dict(self.ai.value_network, best_state)
            print(f"已回放训练评估历史，保留对{evaluation_target}胜率最高的版本：{best_win_rate:.2%}")

        if min_accept_win_rate is not None and best_win_rate is not None and best_win_rate < min_accept_win_rate:
            _load_state_dict(self.ai.value_network, original_state)
            accepted = False
            print(f"本轮训练未达到最低接受胜率 {min_accept_win_rate:.2%}，已回退到训练前模型。")

        self.last_training_summary = {
            "accepted": accepted,
            "best_win_rate": best_win_rate,
            "final_win_rate": final_win_rate,
            "dataset_size": len(dataset),
            "history_size": len(self.training_history),
            "evaluation_target": evaluation_target,
        }
        return self.ai.value_network
