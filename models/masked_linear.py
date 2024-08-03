import torch


class MaskedLinear(torch.nn.Linear):
    def __init__(self, indim, outdim, bias=True):
        super().__init__(indim, outdim, bias)

    def forward(self, x, mask):
        return torch.nn.functional.linear(x, self.weight * mask.T, self.bias)
