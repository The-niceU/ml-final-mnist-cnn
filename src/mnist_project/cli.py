import argparse
from pathlib import Path

from .engine import ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST CNN training and evaluation")
    parser.add_argument("--data-root", type=Path, default=Path("data/MNIST/raw"), help="Raw MNIST IDX folder")
    parser.add_argument("--runs", type=int, default=5, help="How many repeated runs")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per run")
    parser.add_argument("--batch-size", type=int, default=64, help="Train batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="Test batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu", help="Execution device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick-train-samples", type=int, default=0, help="Use only first N train samples")
    parser.add_argument("--quick-test-samples", type=int, default=0, help="Use only first N test samples")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = ExperimentConfig(
        data_root=args.data_root,
        runs=args.runs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        learning_rate=args.lr,
        device=args.device,
        seed=args.seed,
        quick_train_samples=args.quick_train_samples,
        quick_test_samples=args.quick_test_samples,
    )

    result = run_experiment(config)
    print("\n--- Final Results ---")
    print(f"Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
    print(f"F1-score: {result['f1_mean']:.4f} ± {result['f1_std']:.4f}")
    print(f"Used Time: {result['elapsed_seconds']:.2f} seconds")
    print(f"Hardware: {result['device']}")
    print(
        "Tuned Parameters: "
        f"optimizer=Adam, lr={result['learning_rate']}, batch={result['batch_size']}, epoch={result['epochs']}"
    )


if __name__ == "__main__":
    main()
