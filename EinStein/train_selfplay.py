import argparse
import json
import os
import random

from einstein_ai import (
    BLUE,
    RED,
    START_POSITIONS,
    GameState,
    HybridAI,
    PLAYER_NAMES,
    default_model_path,
    resolve_ensemble_paths,
    resolve_model_path,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Generate Einstein self-play data as JSON Lines.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--output", default=os.path.join("data", "selfplay.jsonl"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--ai-kind", choices=["heuristic", "mlp", "gnn", "ensemble"], default="heuristic")
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--append", action="store_true")
    return parser.parse_args(args)


def random_layout(player):
    positions = START_POSITIONS[player].copy()
    random.shuffle(positions)
    return {label: positions[label - 1] for label in range(1, 7)}


def new_random_state():
    return GameState(
        red_layout=random_layout(RED),
        blue_layout=random_layout(BLUE),
        turn=random.choice([RED, BLUE]),
    )


def build_ai(kind, model_path=None, simulations=0, workers=1):
    if kind == "heuristic":
        return HybridAI(value_kind="heuristic", simulations=simulations, workers=workers)
    if kind == "ensemble":
        mlp_path, gnn_path = resolve_ensemble_paths(model_path=model_path)
        return HybridAI(
            value_kind="ensemble",
            simulations=simulations,
            workers=workers,
            mlp_path=mlp_path,
            gnn_path=gnn_path,
        )

    path = resolve_model_path(kind, model_path)
    if path and os.path.exists(path):
        return HybridAI(value_kind=kind, simulations=simulations, workers=workers, model_path=path)
    print(f"Warning: no {kind} model found at {path}; using an untrained network.")
    return HybridAI(value_kind=kind, simulations=simulations, workers=workers)


def play_game(ai, max_moves):
    state = new_random_state()
    records = []
    for _ in range(max_moves):
        if state.is_terminal():
            break
        player = state.turn
        die = random.randint(1, 6)
        legal_actions = state.legal_actions(die)
        legal_mask = state.legal_action_mask(die)
        action = ai.choose_action(state, die) if legal_actions else None
        if action is None and legal_actions:
            action = random.choice(legal_actions)
        action_index = state.action_to_index(action) if action is not None else None
        records.append({
            "state": state.encode(die=die),
            "die": die,
            "legal_mask": legal_mask,
            "action_index": action_index,
            "player": PLAYER_NAMES[player],
            "winner": None,
            "value_target": 0.0,
        })
        state = state.apply_action(action)

    winner = state.winner
    winner_name = PLAYER_NAMES[winner]
    for record in records:
        record["winner"] = winner_name
        if winner is None:
            record["value_target"] = 0.0
        elif record["player"] == winner_name:
            record["value_target"] = 1.0
        else:
            record["value_target"] = -1.0
    return records, winner


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    ai = build_ai(args.ai_kind, args.model_path, simulations=args.simulations, workers=args.workers)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    mode = "a" if args.append else "w"
    wins = {RED: 0, BLUE: 0, None: 0}
    total_positions = 0
    with open(args.output, mode, encoding="utf-8") as handle:
        for game_index in range(1, args.games + 1):
            records, winner = play_game(ai, args.max_moves)
            wins[winner] += 1
            total_positions += len(records)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if game_index % max(1, args.games // 10) == 0:
                print(f"generated {game_index}/{args.games} games, positions={total_positions}")

    print(f"saved {total_positions} positions to {args.output}")
    print(f"wins: RED={wins[RED]}, BLUE={wins[BLUE]}, draws={wins[None]}")


if __name__ == "__main__":
    main()
