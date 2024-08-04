import torch
from torch.nn import Linear, ReLU


class MaskedMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, *, bias=True):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim, bias=bias)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, output_dim, bias=bias)

    def forward(self, X, input_mask, output_mask):
        X = input_mask * X
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc2(X)
        X = output_mask * X
        return X
