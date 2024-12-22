import torch
import torch.nn as nn
import torch.nn.functional as F

from .phisher_model import PhisherModel


class PhisherOneHotModel(PhisherModel, nn.Module):
    def __init__(self: "PhisherOneHotModel", out_features: int = 2) -> None:
        super().__init__()
        
        # Input: (batch_size, 1, 200, 84)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))  # (200-5+1)x(84-5+1) = 196x80
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5))  # (196-5+1)x(80-5+1) = 192x76
        
        # After pool1: 96x40
        # After pool2: 92x38
        self.fc1 = nn.Linear(in_features=12 * 92 * 38, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=out_features)


    def forward(self: "PhisherOneHotModel", x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension: (batch_size, 1, 200, 84)
        x = x.unsqueeze(1)

        x = self.conv1(x)  # (batch_size, 6, 196, 80)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (batch_size, 6, 96, 40)

        x = self.conv2(x)  # (batch_size, 12, 92, 38)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # (batch_size, 12, 46, 19)

        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 12 * 46 * 19)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x
