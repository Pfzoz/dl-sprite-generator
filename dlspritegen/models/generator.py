from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear

class Generator(nn.Module):

    def __init__(self, latent_space_size: int, output_size: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_space_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.ReLU(),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.model(x).view(x.size(0), 3, 16, 16)
        return y
