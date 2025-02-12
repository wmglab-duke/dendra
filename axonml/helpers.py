import functools
import os, contextlib
from typing import ClassVar

import torch


@functools.lru_cache(maxsize=None)
def getenv(key: str, default=0):
    return type(default)(os.getenv(key, default))


class Context(contextlib.ContextDecorator):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.old_context: dict[str, int] = {
            k: v.value for k, v in ContextVar._cache.items()
        }
        for k, v in self.kwargs.items():
            ContextVar._cache[k].value = v

    def __exit__(self, *args):
        for k, v in self.old_context.items():
            ContextVar._cache[k].value = v


class ContextVar:
    _cache: ClassVar[dict[str, "ContextVar"]] = {}
    value: int
    key: str

    def __init__(self, key, default_value):
        if key in ContextVar._cache:
            raise RuntimeError(f"attempt to recreate ContextVar {key}")
        ContextVar._cache[key] = self
        self.value, self.key = getenv(key, default_value), key

    def __bool__(self):
        return bool(self.value)

    def __ge__(self, x):
        return self.value >= x

    def __gt__(self, x):
        return self.value > x

    def __lt__(self, x):
        return self.value < x

    def __le__(self, x):
        return self.value <= x


DEBUG = ContextVar("DEBUG", 0)
TF32 = ContextVar("TF32", 0)


# --- pytorch functions --


def ve_from_s_t(space, time, n, device, multicontact=False):
    ve_s = torch.as_tensor(space, device=device)
    ve_t = torch.as_tensor(time, device=device)

    if multicontact:
        ve_s = ve_s.expand(-1, n, -1)
        ve_t = ve_t.expand(-1, n, -1)
        einsum = op_mc
    else:
        ve_s = ve_s.expand(n, -1)
        ve_t = ve_t.expand(n, -1)
        einsum = op_sc

    return einsum(ve_s, ve_t)


@torch.jit.script
def op_mc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("can,cat->tan", s, t).unsqueeze(2)


@torch.jit.script
def op_sc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("an,at->tan", s, t).unsqueeze(2)
