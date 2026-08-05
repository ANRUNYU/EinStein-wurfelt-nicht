import argparse
import os
import time

from einstein_ai import (
    HybridAI,
    SelfPlayTrainer,
    TORCH_AVAILABLE,
    default_model_path,
    save_model_archive,
)

try:
    import torch
except ImportError:
    torch = None


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train Einstein chess value network with self-play.")
    parser.add_argument("--value-kind", choices=["mlp", "gnn", "ensemble"], default="mlp",
                        help="Value network type to train.")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of self-play episodes per training run.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for Adam optimizer.")
    parser.add_argument("--simulations", type=int, default=40,
                        help="MCTS simulations per move during self-play.")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker threads for MCTS.")
    parser.add_argument("--max-moves", type=int, default=100,
                        help="Maximum moves per self-play episode.")
    parser.add_argument("--random-move-rate", type=float, default=0.1,
                        help="Probability of taking a random move during self-play for exploration.")
    parser.add_argument("--eval-games", type=int, default=20,
                        help="Number of evaluation games against heuristic AI.")
    parser.add_argument("--eval-interval", type=int, default=1,
                        help="Number of epochs between evaluation checks.")
    parser.add_argument("--eval-against-old", action="store_true",
                        help="Evaluate against the previous model version instead of heuristic.")
    parser.add_argument("--min-accept-win-rate", type=float, default=None,
                        help="When evaluating, keep the new strategy only if its best evaluated win rate reaches this value.")
    parser.add_argument("--history-capacity", type=int, default=50000,
                        help="Maximum number of historical self-play samples to keep for replay.")
    parser.add_argument("--history-sample-ratio", type=float, default=0.5,
                        help="Historical replay samples added per new sample during training.")
    parser.add_argument("--discount", type=float, default=0.98,
                        help="Discount applied to earlier moves in one self-play game.")
    parser.add_argument("--no-continue-from-model", action="store_true",
                        help="Start from a fresh network even if --model-out already exists.")
    parser.add_argument("--model-out", default=None,
                        help="Path to save the trained model.")
    return parser.parse_args(args)


def load_training_artifacts(path):
    if not os.path.exists(path):
        return None, []
    data = torch.load(path, map_location="cpu")
    replay_buffer = []
    if isinstance(data, dict):
        replay_buffer = data.get("replay_buffer", [])
    return data, replay_buffer


def pick_best_history_entry(history):
    def score(entry):
        metadata = entry.get("metadata") or {}
        if metadata.get("accepted") is False:
            return -float("inf")
        value = metadata.get("best_win_rate")
        return -float("inf") if value is None else value

    scored = [(score(entry), index, entry) for index, entry in enumerate(history)]
    best_score, _, best_entry = max(scored, key=lambda item: (item[0], item[1]))
    if best_score == -float("inf"):
        return history[-1]
    return best_entry


def save_model_history(model, path, metadata=None, replay_buffer=None):
    save_model_archive(
        model,
        path,
        value_kind=(metadata or {}).get("value_kind", "mlp"),
        metadata=metadata,
        replay_buffer=replay_buffer,
    )


def main():
    args = parse_args()
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for training. Install torch and rerun this script.")

    if torch is not None:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    if args.value_kind == "ensemble":
        raise ValueError("Ensemble 网络不支持直接训练。请分别训练 MLP 和 GNN 模型，然后在推理时使用 ensemble。")
    if args.model_out is None:
        args.model_out = default_model_path(args.value_kind)

    ai = HybridAI(value_kind=args.value_kind, simulations=args.simulations, workers=args.workers)
    try:
        _, replay_buffer = load_training_artifacts(args.model_out)
    except Exception as ex:
        replay_buffer = []
        print(f"Previous training history could not be loaded, replay starts empty: {ex}")
    if os.path.exists(args.model_out) and not args.no_continue_from_model:
        try:
            ai.load_model(args.model_out)
            print(f"Loaded previous best model from {args.model_out}; training will continue from history.")
        except Exception as ex:
            print(f"Previous model could not be loaded, starting from a fresh network: {ex}")
    trainer = SelfPlayTrainer(
        ai,
        episodes=args.episodes,
        max_moves=args.max_moves,
        random_move_rate=args.random_move_rate,
        evaluate_games=args.eval_games,
        evaluation_interval=args.eval_interval,
        history_capacity=args.history_capacity,
        history_sample_ratio=args.history_sample_ratio,
        discount=args.discount,
    )
    trainer.load_training_history(replay_buffer)
    if replay_buffer:
        print(f"Loaded {len(replay_buffer)} historical self-play samples for replay.")
    print(f"Starting training: {args.value_kind} network, {args.episodes} episodes, {args.epochs} epochs")
    trainer.train_network(epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
                          eval_against_old=args.eval_against_old, model_path=args.model_out,
                          min_accept_win_rate=args.min_accept_win_rate if args.eval_games > 0 else None)

    output_path = args.model_out
    if os.path.exists(output_path):
        print(f"模型文件已存在，当前训练结果将追加到同一文件：{output_path}")
    else:
        print(f"Saving model to {output_path}")
    training_summary = getattr(trainer, "last_training_summary", {})
    save_model_history(ai.value_network, output_path, metadata={
        "value_kind": args.value_kind,
        "episodes": args.episodes,
        "epochs": args.epochs,
        "simulations": args.simulations,
        "workers": args.workers,
        "timestamp": time.time(),
        **training_summary,
    }, replay_buffer=trainer.export_training_history())


if __name__ == "__main__":
    main()
