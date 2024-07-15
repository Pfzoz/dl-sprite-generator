from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear

class Discriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Conv2d(32, 16, 3, stride=2),
            nn.Flatten(),
            nn.Linear(576, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)
        return y
