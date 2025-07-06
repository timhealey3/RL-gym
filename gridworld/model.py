import torch.nn as nn

'''
Model Info:
Input: 4x4 tensor, each tensor encodes state of one of the objects, flattened for 64 elements
Output: 4 options: Right, Left, Down, Up
'''
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )
        
    def forward(self, x):
        return self.model(x)