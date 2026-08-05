import argparse
import json
import os
import random

from einstein_ai import FEATURE_SIZE, TORCH_AVAILABLE, default_model_path, save_model_archive

if TORCH_AVAILABLE:
    from einstein_ai import MLPValueNetwork
else:
    MLPValueNetwork = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Train the standard MLP value model from self-play JSONL.")
    parser.add_argument("--data", default=os.path.join("data", "selfplay.jsonl"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", default=default_model_path("mlp"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fresh", action="store_true", help="Ignore an existing compatible model.")
    return parser.parse_args(args)


def load_jsonl(path):
    samples = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            state = record.get("state")
            target = record.get("value_target")
            if state is None or target is None or len(state) != FEATURE_SIZE:
                skipped += 1
                continue
            samples.append((state, float(target)))
    if skipped:
        print(f"Skipped {skipped} records with incompatible feature size.")
    return samples


def split_samples(samples, validation_ratio):
    random.shuffle(samples)
    val_count = int(len(samples) * validation_ratio)
    if val_count <= 0 and len(samples) > 1:
        val_count = 1
    validation = samples[:val_count]
    train = samples[val_count:] or samples
    return train, validation


def batch_iter(samples, batch_size, device):
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        inputs = torch.tensor([item[0] for item in batch], dtype=torch.float32, device=device)
        targets = torch.tensor([item[1] for item in batch], dtype=torch.float32, device=device).unsqueeze(1)
        yield inputs, targets


def evaluate(model, samples, batch_size):
    if not samples:
        return 0.0
    device = model.device
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in batch_iter(samples, batch_size, device):
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            total_loss += loss.item() * len(inputs)
    return total_loss / len(samples)


def main():
    args = parse_args()
    if not TORCH_AVAILABLE or torch is None:
        raise SystemExit("Torch is not installed. Install dependencies with: python -m pip install -r requirements.txt")
    if not os.path.exists(args.data):
        raise SystemExit(f"Training data not found: {args.data}")

    torch.set_num_threads(1)
    random.seed(args.seed)
    samples = load_jsonl(args.data)
    if not samples:
        raise SystemExit("No usable training samples found.")
    train_samples, validation_samples = split_samples(samples, args.validation_ratio)

    model = None
    if os.path.exists(args.output) and not args.fresh:
        try:
            model = MLPValueNetwork.load(args.output)
            if model.input_size != FEATURE_SIZE:
                print(
                    f"Existing model input size is {model.input_size}, current feature size is {FEATURE_SIZE}; "
                    "starting a fresh model."
                )
                model = None
        except Exception as ex:
            print(f"Existing model could not be loaded, starting fresh: {ex}")
    if model is None:
        model = MLPValueNetwork(input_size=FEATURE_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    batch_size = max(1, args.batch_size)
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_samples)
        model.train()
        total_loss = 0.0
        for inputs, targets in batch_iter(train_samples, batch_size, model.device):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(inputs)
        train_loss = total_loss / len(train_samples)
        val_loss = evaluate(model, validation_samples, batch_size)
        print(f"epoch={epoch}/{args.epochs} train_loss={train_loss:.5f} validation_loss={val_loss:.5f}")

    metadata = {
        "script": "train_value_model.py",
        "data": args.data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "validation_loss": evaluate(model, validation_samples, batch_size),
    }
    save_model_archive(model, args.output, value_kind="mlp", metadata=metadata)
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()
