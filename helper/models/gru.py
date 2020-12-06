import torch
import torch.nn as nn
from torch.distributions import Normal
from helper.model_init import weight_init
from .distributions import DiagGaussianDistribution


class GRUActorCritic(nn.Module):
  def __init__(self, output_size, input_size, hidden_size=256):
    super(GRUActorCritic, self).__init__()
    self.is_recurrent = True
    self.hidden_size = hidden_size

    self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
    self.relu1 = nn.ReLU()
    self.policy = nn.Linear(hidden_size, output_size) 
    self.value = nn.Linear(hidden_size, 1)
    self.apply(weight_init)
    self.distribution = DiagGaussianDistribution(action_dim=7)
    self.deterministic = False

  def forward(self, x, h):
    x, h = self.gru(x, h)
    x = self.relu1(x)
    val = self.value(x)
    dist = self.policy(x).squeeze(0)
    
    # dist = self.distribution(dist) ??

    return Normal(dist, scale=1), val, h

  def init_hidden_state(self, batchsize=1):
    return torch.zeros([1, batchsize, self.hidden_size])