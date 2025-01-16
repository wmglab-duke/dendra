import torch


class Distribution(torch.nn.Module):
    __constants__ = ["once", "seeded"]

    def __init__(self, seed=None, once=False):
        super().__init__()
        self.rng : torch.Generator = torch.Generator()
        self.register_buffer("seed_holder", torch.zeros(1, dtype=torch.int64))
        if seed is not None:
            self.rng.manual_seed(seed)
            self.seed_holder[0] = seed
        self.once : bool = once
        self.seeded : bool = seed is not None
        self.initiated : bool = False
        self.dist = torch.zeros(1)

    def device(self):
        return self.seed_holder.device

    def _sample(self, buffer):
        n = buffer.shape[0]
        if self.rng.device != self.seed_holder.device:
            self.rng = torch.Generator(device=self.seed_holder.device)
            if self.seeded:
                self.rng.manual_seed(self.seed_holder[0].item())
        if (self.once and not self.initiated) or (not self.once):
            self.dist = (
                self.sample(n).to(buffer.device).to(buffer.dtype).unsqueeze(1).unsqueeze(1)
            )
        self.initiated = True
        return self.dist.expand(buffer.shape)

    def sample(self, n: int):
        raise NotImplementedError
