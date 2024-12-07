import torch


@torch.jit.interface
class AxonInterface:
    def get_state(self, s: str) -> torch.Tensor:
        pass

    def n(self) -> int:
        pass

    def device(self) -> str:
        pass
