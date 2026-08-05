import os
import random
import tkinter as tk
from tkinter import messagebox
from collections import namedtuple
from einstein_ai import (
    HybridAI,
    GameState,
    TORCH_AVAILABLE,
    ParallelMCTS,
    parse_order_string,
    layout_from_order,
    default_model_path,
    resolve_ensemble_paths,
)

if TORCH_AVAILABLE:
    from einstein_ai import MLPValueNetwork, GNNValueNetwork
else:
    MLPValueNetwork = None
    GNNValueNetwork = None

Position = namedtuple("Position", ["r", "c"])

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


class EinsteinGame:
    def __init__(self, red_layout=None, blue_layout=None):
        self.board = {}
        self.turn = RED
        self.winner = None
        self.setup(red_layout, blue_layout)

    def setup(self, red_layout, blue_layout):
        if red_layout is None:
            red_layout = self.random_layout(RED)
        if blue_layout is None:
            blue_layout = self.random_layout(BLUE)
        self.place_starting_pieces(RED, red_layout)
        self.place_starting_pieces(BLUE, blue_layout)

    def random_layout(self, player):
        positions = START_POSITIONS[player].copy()
        random.shuffle(positions)
        return {i + 1: positions[i] for i in range(6)}

    def place_starting_pieces(self, player, layout):
        for label, pos in layout.items():
            self.board[(pos.r, pos.c)] = (player, label)

    def piece_positions(self, player):
        return {(r, c): label for (r, c), (owner, label) in self.board.items() if owner == player}

    def active_labels(self, player):
        return sorted(label for (owner, label) in self.board.values() if owner == player)

    def label_position(self, player, label):
        for (r, c), (owner, num) in self.board.items():
            if owner == player and num == label:
                return Position(r, c)
        return None

    def occupied(self, pos):
        return (pos.r, pos.c) in self.board

    def in_bounds(self, pos):
        return 0 <= pos.r < BOARD_SIZE and 0 <= pos.c < BOARD_SIZE

    def possible_moves(self, player, label):
        origin = self.label_position(player, label)
        if origin is None:
            return []
        moves = []
        for offset in MOVE_OFFSETS[player]:
            target = Position(origin.r + offset.r, origin.c + offset.c)
            if self.in_bounds(target):
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
            # 对应点数棋子在盘上但被己方封堵时，改走与点数最近的可动棋子
            closest = min(abs(label - die) for label in movable)
            return [label for label in movable if abs(label - die) == closest]

        best_diff = min(abs(label - die) for label in movable)
        return [label for label in movable if abs(label - die) == best_diff]

    def move_piece(self, player, label, target):
        origin = self.label_position(player, label)
        if origin is None:
            raise ValueError("棋子不存在")
        if target not in self.legal_move_targets(player, label):
            raise ValueError("目标位置非法")
        origin_key = (origin.r, origin.c)
        target_key = (target.r, target.c)
        captured = self.board.pop(target_key, None)
        self.board.pop(origin_key)
        self.board[target_key] = (player, label)
        self.check_victory(player, target, captured)

    def check_victory(self, player, target, captured):
        if target == GOAL_CORNERS[player]:
            self.winner = player
            return
        opponent = BLUE if player == RED else RED
        if not self.active_labels(opponent):
            self.winner = player

    def roll_die(self):
        return random.randint(1, 6)

    def next_player(self):
        self.turn = BLUE if self.turn == RED else RED

    def board_str(self):
        grid = [[" . " for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for (r, c), (owner, label) in self.board.items():
            grid[r][c] = f"{owner[0]}{label}"
        rows = [" ".join(cell.rjust(3) for cell in row) for row in grid]
        return "\n".join(rows)

    def print_board(self):
        print("当前棋盘：")
        print(self.board_str())

    def is_game_over(self):
        return self.winner is not None

    def get_valid_targets(self, player, label):
        return self.legal_move_targets(player, label)


class EinsteinGUI:
    COLORS = {
        RED: "#ffcccc",
        BLUE: "#cce5ff",
        None: "#f0f0f0",
        "valid": "#c8ffa0",
        "selected": "#fff78a",
    }

    def __init__(self, root):
        self.root = root
        root.title("爱因斯坦棋")
        root.resizable(False, False)

        self.game = EinsteinGame()
        self.die_value = None
        self.candidates = []
        self.selected_label = None
        self.buttons = {}
        self.undo_stack = []
        self.pending_ai_job = None
        self.move_history = []
        self.move_number = 0
        self.finished_recorded = False
        self.free_walk_var = tk.BooleanVar(value=False)
        self.start_first_var = tk.StringVar(value=RED)
        self.mode_var = tk.StringVar(value="pvp")
        self.human_side_var = tk.StringVar(value=RED)
        self.model_kind_var = tk.StringVar(value="heuristic")
        self.model_path_var = tk.StringVar(value=default_model_path("mlp"))
        self.red_order_var = tk.StringVar(value="1,2,3,4,5,6")
        self.blue_order_var = tk.StringVar(value="1,2,3,4,5,6")
        self.model_loaded = False
        self.ai = HybridAI(value_kind="heuristic", simulations=120, workers=4)

        top_frame = tk.Frame(root)
        top_frame.pack(padx=10, pady=10, fill="x")

        self.status_label = tk.Label(top_frame, text="", font=("Arial", 12), anchor="w", justify="left")
        self.status_label.pack(fill="x")

        config_frame = tk.Frame(root)
        config_frame.pack(padx=10, pady=(0, 8), fill="x")

        tk.Label(config_frame, text="游戏模式:", font=("Arial", 10)).pack(side="left")
        tk.Radiobutton(config_frame, text="玩家 vs 玩家", variable=self.mode_var, value="pvp",
                       command=self.on_mode_change).pack(side="left", padx=(6, 2))
        tk.Radiobutton(config_frame, text="玩家 vs AI", variable=self.mode_var, value="pvai",
                       command=self.on_mode_change).pack(side="left", padx=2)

        tk.Checkbutton(config_frame, text="自由行走", variable=self.free_walk_var,
                   command=self.on_free_walk_change).pack(side="left", padx=(12, 2))

        side_frame = tk.Frame(root)
        side_frame.pack(padx=10, pady=(0, 10), fill="x")
        tk.Label(side_frame, text="玩家执:", font=("Arial", 10)).pack(side="left")
        tk.Radiobutton(side_frame, text="红方", variable=self.human_side_var, value=RED).pack(side="left", padx=(6, 2))
        tk.Radiobutton(side_frame, text="蓝方", variable=self.human_side_var, value=BLUE).pack(side="left", padx=2)
        self.side_frame = side_frame
        tk.Label(side_frame, text="   先手:", font=("Arial", 10)).pack(side="left")
        tk.Radiobutton(side_frame, text="红方", variable=self.start_first_var, value=RED).pack(side="left", padx=(6, 2))
        tk.Radiobutton(side_frame, text="蓝方", variable=self.start_first_var, value=BLUE).pack(side="left", padx=2)
        tk.Radiobutton(side_frame, text="随机", variable=self.start_first_var, value="random").pack(side="left", padx=2)

        model_frame = tk.Frame(root)
        model_frame.pack(padx=10, pady=(0, 10), fill="x")
        tk.Label(model_frame, text="AI 模型:", font=("Arial", 10)).pack(side="left")
        tk.Radiobutton(model_frame, text="启发式", variable=self.model_kind_var, value="heuristic",
                       command=self.on_model_kind_change).pack(side="left", padx=(6, 2))
        tk.Radiobutton(model_frame, text="MLP", variable=self.model_kind_var, value="mlp",
                       command=self.on_model_kind_change).pack(side="left", padx=2)
        tk.Radiobutton(model_frame, text="GNN", variable=self.model_kind_var, value="gnn",
                       command=self.on_model_kind_change).pack(side="left", padx=2)
        tk.Radiobutton(model_frame, text="Ensemble", variable=self.model_kind_var, value="ensemble",
                       command=self.on_model_kind_change).pack(side="left", padx=2)
        tk.Entry(model_frame, textvariable=self.model_path_var, width=28).pack(side="left", padx=6)
        tk.Button(model_frame, text="加载模型", command=self.load_model, width=10).pack(side="left", padx=2)
        self.model_status_label = tk.Label(model_frame, text="使用启发式 AI", font=("Arial", 10))
        self.model_status_label.pack(side="left", padx=6)

        layout_frame = tk.Frame(root)
        layout_frame.pack(padx=10, pady=(0, 10), fill="x")
        tk.Label(layout_frame, text="红方摆放顺序:", font=("Arial", 10)).pack(side="left")
        tk.Entry(layout_frame, textvariable=self.red_order_var, width=18).pack(side="left", padx=(6, 2))
        tk.Label(layout_frame, text="蓝方摆放顺序:", font=("Arial", 10)).pack(side="left", padx=(12, 0))
        tk.Entry(layout_frame, textvariable=self.blue_order_var, width=18).pack(side="left", padx=6)
        tk.Label(layout_frame, text="格式: 1,2,3,4,5,6", font=("Arial", 8)).pack(side="left", padx=4)

        control_frame = tk.Frame(root)
        control_frame.pack(padx=10, pady=(0, 10), fill="x")

        self.die_label = tk.Label(control_frame, text="骰子: -", font=("Arial", 12), width=12)
        self.die_label.pack(side="left")

        self.roll_button = tk.Button(control_frame, text="掷骰子", command=self.roll_die, width=10)
        self.roll_button.pack(side="left", padx=5)

        self.undo_button = tk.Button(control_frame, text="悔棋", command=self.undo_move, width=10, state="disabled")
        self.undo_button.pack(side="left", padx=5)

        reset_button = tk.Button(control_frame, text="重新开始", command=self.reset_game, width=10)
        reset_button.pack(side="left", padx=5)

        board_frame = tk.Frame(root)
        board_frame.pack(padx=10, pady=(0, 10))

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                button = tk.Button(board_frame, text="", width=6, height=3,
                                   command=lambda r=r, c=c: self.cell_clicked(r, c))
                button.grid(row=r, column=c, padx=1, pady=1)
                self.buttons[(r, c)] = button

        record_frame = tk.LabelFrame(root, text="棋谱", font=("Arial", 10))
        record_frame.pack(padx=10, pady=(0, 10), fill="x")

        record_body = tk.Frame(record_frame)
        record_body.pack(fill="x", padx=6, pady=(4, 4))
        record_scroll = tk.Scrollbar(record_body, orient="vertical")
        self.record_text = tk.Text(
            record_body,
            width=72,
            height=8,
            state="disabled",
            wrap="word",
            font=("Consolas", 10),
            yscrollcommand=record_scroll.set,
        )
        record_scroll.config(command=self.record_text.yview)
        self.record_text.pack(side="left", fill="x", expand=True)
        record_scroll.pack(side="right", fill="y")

        record_actions = tk.Frame(record_frame)
        record_actions.pack(fill="x", padx=6, pady=(0, 6))
        tk.Button(record_actions, text="复制棋谱", command=self.copy_move_record, width=10).pack(side="right")

        self.start_game()

    def reset_game(self):
        self.start_game()

    def set_status(self, message):
        self.status_label.config(text=message)

    def refresh_move_record(self):
        if not hasattr(self, "record_text"):
            return
        text = "\n".join(self.move_history) if self.move_history else "暂无棋谱记录。"
        self.record_text.config(state="normal")
        self.record_text.delete("1.0", tk.END)
        self.record_text.insert(tk.END, text)
        self.record_text.config(state="disabled")
        self.record_text.see(tk.END)

    def record_entry(self, text):
        self.move_history.append(text)
        self.refresh_move_record()

    def copy_move_record(self):
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(self.move_history))
        self.set_status("棋谱已复制到剪贴板。")

    def format_position(self, pos):
        return f"({pos.r + 1},{pos.c + 1})" if pos is not None else "(-,-)"

    def format_die(self, die):
        return "自由" if die is None else str(die)

    def starting_order_text(self, player):
        labels_by_pos = self.game.piece_positions(player)
        labels = [labels_by_pos.get((pos.r, pos.c), "-") for pos in START_POSITIONS[player]]
        return ",".join(str(label) for label in labels)

    def record_game_start(self):
        self.record_entry(
            f"开局：红方摆放 {self.starting_order_text(RED)}；"
            f"蓝方摆放 {self.starting_order_text(BLUE)}；先手 {self.game.turn}方。"
        )

    def record_move(self, player, actor, die, label, origin, target, captured):
        self.move_number += 1
        captured_text = ""
        if captured is not None and captured[0] != player:
            captured_text = f"，吃 {captured[0]}{captured[1]}"
        winner_text = f"，{player}方胜" if self.game.winner == player else ""
        self.record_entry(
            f"{self.move_number:02d}. {player}方({actor}) 骰子 {self.format_die(die)}："
            f"{player}{label} {self.format_position(origin)} -> {self.format_position(target)}"
            f"{captured_text}{winner_text}"
        )

    def record_skip(self, player, die, reason):
        self.move_number += 1
        self.record_entry(
            f"{self.move_number:02d}. {player}方 骰子 {self.format_die(die)}：跳过（{reason}）"
        )

    def record_game_result(self):
        if self.finished_recorded or self.game.winner is None:
            return
        self.finished_recorded = True
        self.record_entry(f"终局：{self.game.winner}方获胜。")

    def schedule_ai(self, delay, callback):
        self.cancel_pending_ai()
        self.pending_ai_job = self.root.after(delay, lambda: self.run_scheduled_ai(callback))

    def run_scheduled_ai(self, callback):
        self.pending_ai_job = None
        callback()

    def cancel_pending_ai(self):
        if self.pending_ai_job is None:
            return
        try:
            self.root.after_cancel(self.pending_ai_job)
        except tk.TclError:
            pass
        self.pending_ai_job = None

    def make_snapshot(self, status_message="已悔棋，回到上一步。"):
        return {
            "board": dict(self.game.board),
            "turn": self.game.turn,
            "winner": self.game.winner,
            "die_value": self.die_value,
            "candidates": list(self.candidates),
            "selected_label": None,
            "die_label": self.die_label.cget("text"),
            "status": status_message,
            "move_history": list(self.move_history),
            "move_number": self.move_number,
            "finished_recorded": self.finished_recorded,
        }

    def push_undo_snapshot(self, status_message="已悔棋，回到上一步。"):
        self.undo_stack.append(self.make_snapshot(status_message))
        self.update_undo_button()

    def restore_snapshot(self, snapshot):
        self.game.board = dict(snapshot["board"])
        self.game.turn = snapshot["turn"]
        self.game.winner = snapshot["winner"]
        self.die_value = snapshot["die_value"]
        self.candidates = list(snapshot["candidates"])
        self.selected_label = snapshot["selected_label"]
        self.move_history = list(snapshot.get("move_history", []))
        self.move_number = snapshot.get("move_number", len(self.move_history))
        self.finished_recorded = snapshot.get("finished_recorded", False)
        self.die_label.config(text=snapshot["die_label"])
        self.update_board()
        self.refresh_move_record()
        self.update_controls()
        self.set_status(snapshot["status"])

    def update_undo_button(self):
        if hasattr(self, "undo_button"):
            state = "normal" if self.undo_stack else "disabled"
            self.undo_button.config(state=state)

    def undo_move(self):
        if not self.undo_stack:
            self.set_status("当前没有可悔棋的步骤。")
            return
        self.cancel_pending_ai()
        snapshot = self.undo_stack.pop()
        if self.mode_var.get() == "pvai":
            human_side = self.human_side_var.get()
            while snapshot["turn"] != human_side and self.undo_stack:
                snapshot = self.undo_stack.pop()
        self.restore_snapshot(snapshot)

    def update_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                button = self.buttons[(r, c)]
                cell_text = " . "
                bg = self.COLORS[None]
                piece = self.game.board.get((r, c))
                if piece is not None:
                    owner, label = piece
                    cell_text = f"{owner[0]}{label}"
                    bg = self.COLORS[owner]
                button.config(text=cell_text, bg=bg, relief="raised")

        if self.selected_label is not None:
            origin = self.game.label_position(self.game.turn, self.selected_label)
            if origin is not None:
                self.buttons[(origin.r, origin.c)].config(bg=self.COLORS["selected"])
            for target in self.game.legal_move_targets(self.game.turn, self.selected_label):
                self.buttons[(target.r, target.c)].config(bg=self.COLORS["valid"])

    def on_mode_change(self):
        self.update_controls()

    def on_free_walk_change(self):
        if self.free_walk_var.get():
            self.set_status("已启用自由行走模式：无需掷骰子，任意可动棋子均可行走。")
        else:
            self.set_status("已禁用自由行走模式：回合需掷骰子决定棋子。")
        self.update_controls()

    def on_model_kind_change(self):
        self.update_model_path_hint()
        self.update_ai_from_controls()
        self.update_controls()

    def update_model_path_hint(self):
        kind = self.model_kind_var.get()
        current = self.model_path_var.get().strip()
        known_defaults = {
            "",
            "einstein_value_model.pt",
            default_model_path("mlp"),
            default_model_path("gnn"),
        }
        if current not in known_defaults:
            return
        if kind == "gnn":
            self.model_path_var.set(default_model_path("gnn"))
        elif kind in ("mlp", "ensemble"):
            self.model_path_var.set(default_model_path("mlp"))

    def update_ai_from_controls(self):
        kind = self.model_kind_var.get()
        if kind == "heuristic":
            self.ai = HybridAI(value_kind="heuristic", simulations=0, workers=1)
            self.model_loaded = False
            self.model_status_label.config(text="使用启发式 AI")
            return
        if not TORCH_AVAILABLE:
            messagebox.showwarning("Torch 未安装", "Torch 未安装，无法使用 MLP/GNN 模型。请安装 torch 后重试。")
            self.model_kind_var.set("heuristic")
            self.ai = HybridAI(value_kind="heuristic", simulations=0, workers=1)
            self.model_status_label.config(text="使用启发式 AI")
            return
        if kind == "ensemble":
            self.ai = HybridAI(value_kind=kind, simulations=120, workers=4)
            self.model_loaded = False
            self.model_status_label.config(text="Ensemble AI（未加载模型）")
        else:
            self.ai = HybridAI(value_kind=kind, simulations=120, workers=4)
            self.model_loaded = False
            self.model_status_label.config(text=f"{kind.upper()} AI（未加载模型）")

    def load_model(self):
        path = self.model_path_var.get().strip()
        kind = self.model_kind_var.get()
        if kind == "heuristic":
            self.set_status("启发式 AI 无需加载模型。")
            return
        if not TORCH_AVAILABLE:
            messagebox.showerror("错误", "Torch 未安装，无法加载模型。")
            return
        if not path:
            path = default_model_path("mlp" if kind == "ensemble" else kind)
            self.model_path_var.set(path)
        if kind != "ensemble" and not os.path.exists(path):
            messagebox.showerror("错误", f"模型文件不存在：{path}")
            return
        try:
            if kind == "ensemble":
                mlp_path, gnn_path = resolve_ensemble_paths(model_path=path)
                self.ai = HybridAI(value_kind=kind, simulations=120, workers=4, mlp_path=mlp_path, gnn_path=gnn_path)
                loaded = []
                if getattr(self.ai.value_network, "mlp", None) is not None:
                    loaded.append("MLP")
                if getattr(self.ai.value_network, "gnn", None) is not None:
                    loaded.append("GNN")
                if not loaded:
                    messagebox.showerror("错误", f"未能加载 MLP/GNN 模型：{mlp_path}，{gnn_path}")
                    return
                errors = getattr(self.ai.value_network, "load_errors", [])
                if errors:
                    messagebox.showwarning("部分模型未加载", "\n".join(errors))
                self.model_status_label.config(text=f"已加载 Ensemble({'+'.join(loaded)})")
            else:
                self.ai = HybridAI(value_kind=kind, simulations=120, workers=4, model_path=path)
                self.ai.mcts = ParallelMCTS(value_network=self.ai.value_network, simulations=120, workers=4)
                self.model_status_label.config(text=f"已加载 {kind.upper()} 模型")
            self.model_loaded = True
            self.set_status(f"已加载模型：{path}。AI 将使用训练好的价值网络。")
        except Exception as ex:
            messagebox.showerror("加载失败", f"模型加载失败：{ex}")

    def is_ai_turn(self):
        return self.mode_var.get() == "pvai" and self.game.turn != self.human_side_var.get()

    def update_controls(self):
        self.update_undo_button()
        if self.game.is_game_over():
            self.roll_button.config(state="disabled")
            return
        # If free-walk is enabled for the human-controlled side and it's that side's turn,
        # disable the roll button because the human doesn't need to roll.
        if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
            self.roll_button.config(state="disabled")
            return
        if self.is_ai_turn():
            self.roll_button.config(state="disabled")
        else:
            self.roll_button.config(state="normal")

    def get_custom_layouts(self):
        red_order = parse_order_string(self.red_order_var.get())
        blue_order = parse_order_string(self.blue_order_var.get())
        red_layout = None
        blue_layout = None
        if red_order is not None:
            red_layout = layout_from_order(red_order, RED)
        if blue_order is not None:
            blue_layout = layout_from_order(blue_order, BLUE)
        if self.red_order_var.get().strip() and red_order is None:
            self.set_status("红方摆放顺序格式错误，已使用随机摆放。")
        if self.blue_order_var.get().strip() and blue_order is None:
            self.set_status("蓝方摆放顺序格式错误，已使用随机摆放。")
        return red_layout, blue_layout

    def start_game(self):
        self.cancel_pending_ai()
        self.undo_stack = []
        red_layout, blue_layout = self.get_custom_layouts()
        self.game = EinsteinGame(red_layout=red_layout, blue_layout=blue_layout)
        # 设置先手
        sf = self.start_first_var.get()
        if sf == "random":
            self.game.turn = random.choice([RED, BLUE])
        else:
            self.game.turn = sf
        self.move_history = []
        self.move_number = 0
        self.finished_recorded = False
        self.record_game_start()
        self.die_value = None
        self.candidates = []
        self.selected_label = None
        # 显示掷骰子或自由行走提示，仅当当前回合为人类且启用自由行走时显示“自由行走”
        if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
            self.die_label.config(text="自由行走")
        else:
            self.die_label.config(text="骰子: -")
        self.update_board()
        self.update_controls()
        self.update_ai_from_controls()
        if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
            self.set_status(f"新局开始，轮到{self.game.turn}方（自由行走）。")
        else:
            self.set_status(f"新局开始，轮到{self.game.turn}方，请先掷骰子。")
        if self.is_ai_turn():
            self.schedule_ai(400, self.ai_take_turn)

    def ai_take_turn(self):
        if self.game.is_game_over() or not self.is_ai_turn():
            return
        # AI always rolls in its turn (free-walk only affects human-controlled side)
        if self.roll_die() and self.is_ai_turn():
            self.schedule_ai(400, self.ai_make_move)

    def ai_make_move(self):
        if self.game.is_game_over():
            return
        # Normal AI move: use GameState and AI policy (dice already rolled)
        state = GameState.from_einstein(self.game)
        choice, target = self.ai.choose_move(state, self.die_value)
        if choice is None:
            self.push_undo_snapshot("已悔棋，回到 AI 跳过回合前。")
            self.record_skip(self.game.turn, self.die_value, "AI 无可选走法")
            self.set_status(f"{self.game.turn}方没有可走棋子，回合跳过。")
            self.game.next_player()
            self.die_value = None
            self.candidates = []
            self.selected_label = None
            # 更新骰子/自由行走提示：仅当下一方是人类且启用自由行走时显示
            if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
                self.die_label.config(text="自由行走")
            else:
                self.die_label.config(text="骰子: -")
            self.update_board()
            self.update_controls()
            if self.is_ai_turn():
                self.schedule_ai(400, self.ai_take_turn)
            else:
                if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
                    self.set_status(f"轮到{self.game.turn}方（自由行走）。")
                else:
                    self.set_status(f"轮到{self.game.turn}方，请先掷骰子。")
            return

        player = self.game.turn
        origin = self.game.label_position(player, choice)
        captured = self.game.board.get((target.r, target.c))
        self.push_undo_snapshot("已悔棋，回到 AI 走子前。")
        self.game.move_piece(player, choice, target)
        self.record_move(player, "AI", self.die_value, choice, origin, target, captured)
        self.set_status(f"{player}方(AI) 走子 {choice} -> ({target.r + 1},{target.c + 1})。")
        self.update_board()
        if self.game.is_game_over():
            self.finish_game()
            return

        self.game.next_player()
        self.die_value = None
        self.candidates = []
        self.selected_label = None
        # 若下一方为人类并且启用自由行走，显示“自由行走”提示；否则显示骰子为默认
        if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
            self.die_label.config(text="自由行走")
        else:
            self.die_label.config(text="骰子: -")
        self.update_board()
        self.update_controls()
        if self.is_ai_turn():
            self.schedule_ai(400, self.ai_take_turn)
        else:
            if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
                self.set_status(f"轮到{self.game.turn}方（自由行走）。")
            else:
                self.set_status(f"轮到{self.game.turn}方，请先掷骰子。")

    def ai_choose_move(self):
        best_move = None
        best_score = -999
        for label in self.candidates:
            targets = self.game.legal_move_targets(self.game.turn, label)
            for target in targets:
                score = self.ai_eval_move(label, target)
                if score > best_score:
                    best_score = score
                    best_move = (label, target)
        return best_move if best_move is not None else (None, None)

    def ai_eval_move(self, label, target):
        score = 0
        piece = self.game.board.get((target.r, target.c))
        if piece is not None and piece[0] != self.game.turn:
            score += 120
        if target == GOAL_CORNERS[self.game.turn]:
            score += 200
        goal = GOAL_CORNERS[self.game.turn]
        dist = abs(goal.r - target.r) + abs(goal.c - target.c)
        score += 50 - dist
        return score

    def roll_die(self):
        if self.game.is_game_over():
            return False
        # If free-walk is enabled for the human-controlled side and it's that side's turn,
        # the human does not roll dice; otherwise roll as normal (AI still rolls).
        if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
            return False
        pre_roll_snapshot = self.make_snapshot("已悔棋，回到掷骰前。")
        self.die_value = self.game.roll_die()
        self.candidates = self.game.choose_labels(self.game.turn, self.die_value)
        self.selected_label = None
        self.die_label.config(text=f"骰子: {self.die_value}")
        self.roll_button.config(state="disabled")

        if not self.candidates:
            self.undo_stack.append(pre_roll_snapshot)
            self.record_skip(self.game.turn, self.die_value, "无可走棋子")
            self.set_status(f"{self.game.turn}方没有可走棋子，回合自动跳过。")
            self.game.next_player()
            self.die_value = None
            self.candidates = []
            if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
                self.die_label.config(text="自由行走")
            else:
                self.die_label.config(text="骰子: -")
            self.update_board()
            self.update_controls()
            if self.is_ai_turn():
                self.schedule_ai(300, self.ai_take_turn)
            else:
                if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
                    self.set_status(f"轮到{self.game.turn}方（自由行走）。")
                else:
                    self.set_status(f"轮到{self.game.turn}方，请先掷骰子。")
            return False

        self.set_status(
            f"{self.game.turn}方掷出 {self.die_value}，可走棋子：{', '.join(str(x) for x in self.candidates)}。请选择棋子。"
        )
        self.update_board()
        return True

    def cell_clicked(self, r, c):
        if self.game.is_game_over() or self.is_ai_turn():
            return
        # If free-walk is enabled for human side, that human can move without rolling.
        if not (self.free_walk_var.get() and self.game.turn == self.human_side_var.get()) and self.die_value is None:
            self.set_status(f"请先掷骰子开始{self.game.turn}方回合。")
            return

        # Ensure candidates are prepared in free-walk mode for human side
        if self.free_walk_var.get() and self.game.turn == self.human_side_var.get() and not self.candidates:
            active = self.game.active_labels(self.game.turn)
            self.candidates = [label for label in active if self.game.get_valid_targets(self.game.turn, label)]

        if self.selected_label is None:
            piece = self.game.board.get((r, c))
            if piece is not None:
                owner, label = piece
                if owner == self.game.turn and label in self.candidates:
                    self.selected_label = label
                    self.set_status(f"已选择棋子 {label}，请选择目标格子。")
                    self.update_board()
                    return
            self.set_status("请选择一个可走的本方棋子。")
            return

        target = Position(r, c)
        if target not in self.game.legal_move_targets(self.game.turn, self.selected_label):
            self.set_status("请选择合法的目标格子。")
            return

        player = self.game.turn
        label = self.selected_label
        origin = self.game.label_position(player, label)
        captured = self.game.board.get((target.r, target.c))
        self.push_undo_snapshot("已悔棋，回到走子前。")
        self.game.move_piece(player, label, target)
        self.record_move(player, "玩家", self.die_value, label, origin, target, captured)
        if self.game.is_game_over():
            self.finish_game()
            return

        self.game.next_player()
        self.die_value = None
        self.candidates = []
        self.selected_label = None
        # If next player is human and has free-walk enabled, show free-walk label
        if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
            self.die_label.config(text="自由行走")
        else:
            self.die_label.config(text="骰子: -")
        self.update_board()
        self.update_controls()
        if self.is_ai_turn():
            self.schedule_ai(400, self.ai_take_turn)
        else:
            if self.free_walk_var.get() and self.game.turn == self.human_side_var.get():
                self.set_status(f"轮到{self.game.turn}方（自由行走）。")
            else:
                self.set_status(f"轮到{self.game.turn}方，请先掷骰子。")

    def finish_game(self):
        self.update_board()
        self.record_game_result()
        self.set_status(f"游戏结束，胜者：{self.game.winner}方。")
        self.die_label.config(text=f"胜者: {self.game.winner}方")
        messagebox.showinfo("游戏结束", f"游戏结束，胜者：{self.game.winner}方。")


def main():
    root = tk.Tk()
    EinsteinGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
