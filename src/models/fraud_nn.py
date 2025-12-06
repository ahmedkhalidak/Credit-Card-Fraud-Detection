import torch.nn as nn

class FraudNN(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.5),

            nn.Linear(128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.5),

            nn.Linear(128, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.Tanh(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)
