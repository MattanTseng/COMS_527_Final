import torch
import torch.nn as nn
import torch.nn.functional as F




# Taken from https://github.com/aome510/chrome-dino-game-rl.git
class DQN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(3072, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    
#https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class DQN_pytorch(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_pytorch, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_observations, 32, kernel_size=8, stride=4),
            nn.ReLU(),
        )

        self.layer1 = nn.Linear(14880, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(64, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)