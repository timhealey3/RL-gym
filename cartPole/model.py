import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(4, 150),
            nn.LeakyReLU(),
            nn.Linear(150, 2),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.model(x)