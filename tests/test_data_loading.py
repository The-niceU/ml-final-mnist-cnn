import struct
from pathlib import Path

from mnist_project.data import load_mnist_from_raw


def _write_idx_images(path: Path, count: int = 2, rows: int = 28, cols: int = 28) -> None:
    with path.open("wb") as file:
        file.write(struct.pack(">IIII", 2051, count, rows, cols))
        file.write(bytes([0] * (count * rows * cols)))


def _write_idx_labels(path: Path, labels: list[int]) -> None:
    with path.open("wb") as file:
        file.write(struct.pack(">II", 2049, len(labels)))
        file.write(bytes(labels))


def test_load_mnist_from_raw_shapes(tmp_path: Path) -> None:
    _write_idx_images(tmp_path / "train-images-idx3-ubyte", count=2)
    _write_idx_labels(tmp_path / "train-labels-idx1-ubyte", labels=[1, 2])
    _write_idx_images(tmp_path / "t10k-images-idx3-ubyte", count=1)
    _write_idx_labels(tmp_path / "t10k-labels-idx1-ubyte", labels=[3])

    train_images, train_labels, test_images, test_labels = load_mnist_from_raw(tmp_path)

    assert train_images.shape == (2, 1, 28, 28)
    assert train_labels.tolist() == [1, 2]
    assert test_images.shape == (1, 1, 28, 28)
    assert test_labels.tolist() == [3]
