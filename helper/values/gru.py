import torch
import torch.nn as nn
from helper.values.value import Value
import torch.nn.init as I


class GRUValue(Value):
    def __init__(self, output_size, init_state, input_size=1, hidden_size=256):
        super(GRUValue, self).__init__(output_size)
        self.is_recurrent = True
        self.hidden_size = hidden_size
        self.init_state = init_state

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.value = nn.Linear(hidden_size, 1)
        I.xavier_normal_(self.value.weight)
        self.prev_state = self.init_state

    def forward(self, x):
        x, h = self.gru(x, self.prev_state)
        self.prev_state = h
        return self.value(x)

    def reset_hidden_state(self):
        self.prev_state = self.init_state
