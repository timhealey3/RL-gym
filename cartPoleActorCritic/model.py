import torch
import torch.nn as nn
from torch.nn import functional as F

'''
This model is both the Actor and the Critic model.
Args:
State: 4 values representing the state
Returns:
Actor -> 1D Tensor of probabilites of actions
Critic -> 1D scalar Tensor of expected average reward from state
'''
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4, 25)
        self.l2 = nn.Linear(25, 50)
        self.actor_lin1 = nn.Linear(50, 2)
        self.l3 = nn.Linear(50, 25)
        self.critic_lin1 = nn.Linear(25, 1)

    def forward(self, x):
        # normalize inputs
        x = F.normalize(x, dim=0)
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        # probabilities of the 2 actions
        actor = F.log_softmax(self.actor_lin1(y), dim=0)
        # detach y node from graph so critics loss won't back prop first two layers, only actor will cause weights to be modified
        c = F.relu(self.l3(y.detach()))
        # critic returns scalar between -1 and +1 of average reward from state
        critic = torch.tanh(self.critic_lin1(c))
        return actor, critic
