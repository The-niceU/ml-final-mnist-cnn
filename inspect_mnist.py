# 保存为 inspect_mnist.py，放在项目根目录执行
import struct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

root = Path("data/MNIST/raw")

img_file = root / "t10k-images-idx3-ubyte"
lbl_file = root / "t10k-labels-idx1-ubyte"

with img_file.open("rb") as f:
    magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
    images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)

with lbl_file.open("rb") as f:
    magic_lbl, num_lbl = struct.unpack(">II", f.read(8))
    labels = np.frombuffer(f.read(), dtype=np.uint8)

print(f"train images: {images.shape}, labels: {labels.shape}")
print(f"first label: {labels[0]}")

plt.imshow(images[0], cmap="gray")
plt.title(f"label={labels[0]}")
plt.axis("off")
plt.show()