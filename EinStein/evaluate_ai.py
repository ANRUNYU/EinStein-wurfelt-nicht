import argparse
import os
import random

from einstein_ai import (
    BLUE,
    RED,
    START_POSITIONS,
    GameState,
    HybridAI,
    default_model_path,
    resolve_ensemble_paths,
    resolve_model_path,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Evaluate two Einstein AI players.")
    parser.add_argument("--ai-a", choices=["heuristic", "mlp", "gnn", "ensemble"], default="mlp")
    parser.add_argument("--model-a", default=None)
    parser.add_argument("--ai-b", choices=["heuristic", "mlp", "gnn", "ensemble"], default="heuristic")
    parser.add_argument("--model-b", default=None)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(args)


def random_layout(player):
    positions = START_POSITIONS[player].copy()
    random.shuffle(positions)
    return {label: positions[label - 1] for label in range(1, 7)}


def new_random_state():
    return GameState(red_layout=random_layout(RED), blue_layout=random_layout(BLUE), turn=RED)


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


def play_game(red_ai, blue_ai, max_moves):
    state = new_random_state()
    moves = 0
    while not state.is_terminal() and moves < max_moves:
        die = random.randint(1, 6)
        ai = red_ai if state.turn == RED else blue_ai
        action = ai.choose_action(state, die)
        state = state.apply_action(action)
        moves += 1
    return state.winner, state.winner_reason, moves


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    ai_a = build_ai(args.ai_a, args.model_a, args.simulations, args.workers)
    ai_b = build_ai(args.ai_b, args.model_b, args.simulations, args.workers)
    stats = {
        "a_wins": 0,
        "b_wins": 0,
        "draws": 0,
        "red_wins": 0,
        "blue_wins": 0,
        "goal_wins": 0,
        "capture_all_wins": 0,
        "total_moves": 0,
    }

    for game_index in range(args.games):
        a_is_red = game_index % 2 == 0
        red_ai = ai_a if a_is_red else ai_b
        blue_ai = ai_b if a_is_red else ai_a
        winner, reason, moves = play_game(red_ai, blue_ai, args.max_moves)
        stats["total_moves"] += moves
        if winner is None:
            stats["draws"] += 1
            continue
        if winner == RED:
            stats["red_wins"] += 1
        else:
            stats["blue_wins"] += 1
        if reason == "goal":
            stats["goal_wins"] += 1
        elif reason == "capture_all":
            stats["capture_all_wins"] += 1

        a_won = (winner == RED and a_is_red) or (winner == BLUE and not a_is_red)
        if a_won:
            stats["a_wins"] += 1
        else:
            stats["b_wins"] += 1

    games = max(1, args.games)
    print(f"AI A ({args.ai_a}) win rate: {stats['a_wins'] / games:.2%}")
    print(f"AI B ({args.ai_b}) win rate: {stats['b_wins'] / games:.2%}")
    print(f"Draw rate: {stats['draws'] / games:.2%}")
    print(f"Average moves: {stats['total_moves'] / games:.2f}")
    print(f"Red win rate: {stats['red_wins'] / games:.2%}")
    print(f"Blue win rate: {stats['blue_wins'] / games:.2%}")
    print(f"Goal wins: {stats['goal_wins']}")
    print(f"Capture-all wins: {stats['capture_all_wins']}")


if __name__ == "__main__":
    main()
