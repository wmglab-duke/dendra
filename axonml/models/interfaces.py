from typing import List
import torch


@torch.jit.interface
class AxonInterface:
    pass

@torch.jit.interface
class HandlerInterface:
    def initialize(self, v, v_init, temp) -> None:
        pass

    def generic(self, model) -> None:
        pass

    def advance(self, v, dt) -> None:
        pass

    def detach(self) -> None:
        pass

    def i_intra(self, v, intra) -> torch.Tensor:
        pass

    def i(self, v) -> torch.Tensor:
        pass

    def update(self, temp) -> None:
        pass

    def get(self, mech: str, state: str) -> torch.Tensor:
        pass

    def set(self, name: str, value: float) -> None:
        pass

    def all_states(self) -> List[str]:
        pass