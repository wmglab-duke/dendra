import torch

class Distribution(torch.nn.Module):

    __constants__ = ['seed']

    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def _sample(self, buffer):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        n = buffer.shape[0]
        dist = self.sample(n).to(buffer.device).to(buffer.dtype).unsqueeze(1).unsqueeze(1)
        return dist.expand(buffer.shape)


    def sample(self, n: int):
        raise NotImplementedError
