from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import random
import time

import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from .data import load_mnist_from_raw
from .model import CNN


class MNISTDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.tensor(images, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


@dataclass
class ExperimentConfig:
    data_root: Path
    runs: int = 5
    epochs: int = 5
    batch_size: int = 64
    test_batch_size: int = 1000
    learning_rate: float = 1e-3
    device: str = "cpu"
    seed: int = 42
    quick_train_samples: int = 0
    quick_test_samples: int = 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable on this machine")
    return torch.device(device_name)


def _train_one_epoch(model, loader, criterion, optimizer, device: torch.device) -> None:
    model.train()
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()


def _evaluate(model, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(target.cpu().tolist())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="macro")
    return {"accuracy": accuracy, "f1": f1}


def run_experiment(config: ExperimentConfig) -> Dict[str, object]:
    _set_seed(config.seed)
    device = _resolve_device(config.device)

    train_images, train_labels, test_images, test_labels = load_mnist_from_raw(config.data_root)

    train_dataset = MNISTDataset(train_images, train_labels)
    test_dataset = MNISTDataset(test_images, test_labels)

    if config.quick_train_samples > 0:
        train_dataset = Subset(train_dataset, range(min(config.quick_train_samples, len(train_dataset))))
    if config.quick_test_samples > 0:
        test_dataset = Subset(test_dataset, range(min(config.quick_test_samples, len(test_dataset))))

    generator = torch.Generator().manual_seed(config.seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

    accuracies: List[float] = []
    f1_scores: List[float] = []
    start_time = time.time()

    for run_index in range(config.runs):
        model = CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for _ in range(config.epochs):
            _train_one_epoch(model, train_loader, criterion, optimizer, device)

        metrics = _evaluate(model, test_loader, device)
        accuracies.append(metrics["accuracy"])
        f1_scores.append(metrics["f1"])

        print(
            f"Run {run_index + 1}/{config.runs}: "
            f"accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}"
        )

    elapsed = time.time() - start_time
    result = {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "elapsed_seconds": elapsed,
        "device": str(device),
        "runs": config.runs,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
    }
    return result
