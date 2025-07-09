import torch

from ..mechanisms._mechanism import Mechanism as M
from ..mechanisms._state import State as S
from ..mechanisms.ops import *


class t(S):
    S.STATE('t')
    S.DERIVATIVE("t' = 1")

    def inf(self, v):
        return {'t': torch.zeros(1, dtype=v.dtype, device=v.device)}
    

class alphasynapse(M):
    M.STATE(t)
    M.RANGE(onset=0.0, tau=0.1, gmax=0.0, e=0.0)    
    M.NONSPECIFIC_CURRENT("i")

    @property
    def g(self):
        return self.gmax * alpha((self.t - self.onset) / self.tau)

    def i(self, v):
        return self.g * (v - self.e)


def alpha(x: torch.Tensor) -> torch.Tensor:
    """
    Efficient PyTorch implementation of the NEURON alpha function.

    alpha(x) = 0,                if x < 0 or x > 10
             = x * exp(1 - x),    if 0 <= x <= 10

    Args:
        x (torch.Tensor): A tensor of input values.

    Returns:
        torch.Tensor: The result of the alpha function applied element-wise.
    """
    # Calculate the value for the 'else' case: x * exp(1 - x)
    # This is done for all elements in a single vectorized operation.
    y_calc = x * torch.exp(1.0 - x)
    
    # Define the condition for the 'if' case.
    # The '|' operator is the element-wise OR for boolean tensors.
    condition = (x < 0.0) | (x > 10.0)

    # Use torch.where to select elements:
    # torch.where(condition, value_if_true, value_if_false)
    # If condition is true, take 0.0. Otherwise, take the calculated value.
    return torch.where(condition, 0.0, y_calc)