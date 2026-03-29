from pathlib import Path
import struct
from typing import Tuple

import numpy as np


TRAIN_IMAGES = "train-images-idx3-ubyte"
TRAIN_LABELS = "train-labels-idx1-ubyte"
TEST_IMAGES = "t10k-images-idx3-ubyte"
TEST_LABELS = "t10k-labels-idx1-ubyte"


def _read_idx_images(file_path: Path) -> np.ndarray:
    with file_path.open("rb") as file:
        magic, num, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid image IDX magic number in {file_path}: {magic}")
        images = np.frombuffer(file.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
    return images.astype(np.float32) / 255.0


def _read_idx_labels(file_path: Path) -> np.ndarray:
    with file_path.open("rb") as file:
        magic, num = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid label IDX magic number in {file_path}: {magic}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
        if labels.shape[0] != num:
            raise ValueError(f"Label count mismatch in {file_path}: header={num}, actual={labels.shape[0]}")
    return labels


def load_mnist_from_raw(data_root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = Path(data_root)
    train_images = _read_idx_images(root / TRAIN_IMAGES)
    train_labels = _read_idx_labels(root / TRAIN_LABELS)
    test_images = _read_idx_images(root / TEST_IMAGES)
    test_labels = _read_idx_labels(root / TEST_LABELS)
    return train_images, train_labels, test_images, test_labels
