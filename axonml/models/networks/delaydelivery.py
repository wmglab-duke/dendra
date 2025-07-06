import torch


class VariableDelayDelivery(torch.nn.Module):
    def __init__(self, pre, pre_idx, post, post_idx, delay):
        super(VariableDelayDelivery, self).__init__()
        self.delay = delay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x