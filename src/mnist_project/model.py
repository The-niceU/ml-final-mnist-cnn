import torch
import torch.nn as nn
import torch.nn.functional as functional


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.conv1(x))
        x = functional.max_pool2d(x, 2)
        x = functional.relu(self.conv2(x))
        x = functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = functional.relu(self.fc1(x))
        return self.fc2(x)
