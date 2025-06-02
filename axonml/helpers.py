import functools
import os, contextlib
from typing import ClassVar

import torch
import numpy as np


@functools.lru_cache(maxsize=None)
def getenv(key: str, default=0):
    return type(default)(os.getenv(key, default))


class ctx(contextlib.ContextDecorator):
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
DFITOT = ContextVar("DFITOT", 1)
IMEM = ContextVar("IMEM", 0)
CUDA = ContextVar("CUDA", int(torch.cuda.is_available()))
DTWARN = ContextVar("DTWARN", 1)
PADE = ContextVar("PADE", -1)
DETECT_ANOMALIES = ContextVar("DETECT_ANOMALIES", 0)
NETWORK = ContextVar("NETWORK", 0)


def numpify(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


# --- pytorch functions --


def allow_tf32(allow=True):
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow


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


@torch.compile
def op_mc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("can,cat->tan", s, t)


@torch.compile
def op_sc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("an,at->tan", s, t)

import contextlib


def interp1d(x, y, xnew, out=None):
    """
    Linear 1D interpolation on the GPU for Pytorch.
    This function returns interpolated values of a set of 1-D functions at
    the desired query points `xnew`.
    This function is working similarly to Matlab™ or scipy functions with
    the `linear` interpolation mode on, except that it parallelises over
    any number of desired interpolation problems.
    Values outside the bounds of x are set to 0.
    The code will run on GPU if all the tensors provided are on a cuda
    device.

    Parameters
    ----------
    x : (N, ) or (D, N) Pytorch Tensor
        A 1-D or 2-D tensor of real values.
    y : (N,) or (D, N) Pytorch Tensor
        A 1-D or 2-D tensor of real values. The length of `y` along its
        last dimension must be the same as that of `x`
    xnew : (P,) or (D, P) Pytorch Tensor
        A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
        _both_ `x` and `y` are 1-D. Otherwise, its length along the first
        dimension must be the same as that of whichever `x` and `y` is 2-D.
    out : Pytorch Tensor, same shape as `xnew`
        Tensor for the output. If None: allocated automatically.

    Returns
    -------
    ynew : Pytorch Tensor
        The interpolated values, same shape as xnew.
    """
    # making the vectors at least 2D
    is_flat = {}
    require_grad = {}
    v = {}
    device = []
    eps = torch.finfo(y.dtype).eps
    for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
        assert len(vec.shape) <= 2, "interp1d: all inputs must be at most 2-D."
        if len(vec.shape) == 1:
            v[name] = vec[None, :]
        else:
            v[name] = vec
        is_flat[name] = v[name].shape[0] == 1
        require_grad[name] = vec.requires_grad
        device = list(set(device + [str(vec.device)]))
    assert len(device) == 1, "All parameters must be on the same device."
    device = device[0]

    # Checking for the dimensions
    assert v["x"].shape[1] == v["y"].shape[1] and (
        v["x"].shape[0] == v["y"].shape[0]
        or v["x"].shape[0] == 1
        or v["y"].shape[0] == 1
    ), (
        "x and y must have the same number of columns, and either "
        "the same number of row or one of them having only one "
        "row."
    )

    reshaped_xnew = False
    if (v["x"].shape[0] == 1) and (v["y"].shape[0] == 1) and (v["xnew"].shape[0] > 1):
        # if there is only one row for both x and y, there is no need to
        # loop over the rows of xnew because they will all have to face the
        # same interpolation problem. We should just stack them together to
        # call interp1d and put them back in place afterwards.
        original_xnew_shape = v["xnew"].shape
        v["xnew"] = v["xnew"].contiguous().view(1, -1)
        reshaped_xnew = True

    # identify the dimensions of output and check if the one provided is ok
    D = max(v["x"].shape[0], v["xnew"].shape[0])
    shape_ynew = (D, v["xnew"].shape[-1])
    if out is not None:
        if out.numel() != shape_ynew[0] * shape_ynew[1]:
            # The output provided is of incorrect shape.
            # Going for a new one
            out = None
        else:
            ynew = out.reshape(shape_ynew)
    if out is None:
        ynew = torch.zeros(*shape_ynew, device=device)

    # moving everything to the desired device in case it was not there
    # already (not handling the case things do not fit entirely, user will
    # do it if required.)
    for name in v:
        v[name] = v[name].to(device)

    # calling searchsorted on the x values.
    ind = ynew.long()

    # expanding xnew to match the number of rows of x in case only one xnew is
    # provided
    if v["xnew"].shape[0] == 1:
        v["xnew"] = v["xnew"].expand(v["x"].shape[0], -1)

    # the squeeze is because torch.searchsorted does accept either a nd with
    # matching shapes for x and xnew or a 1d vector for x. Here we would
    # have (1,len) for x sometimes
    torch.searchsorted(v["x"].contiguous().squeeze(), v["xnew"].contiguous(), out=ind)

    # the `-1` is because searchsorted looks for the index where the values
    # must be inserted to preserve order. And we want the index of the
    # preceeding value.
    ind -= 1
    # we clamp the index, because the number of intervals is x.shape-1,
    # and the left neighbour should hence be at most number of intervals
    # -1, i.e. number of columns in x -2
    ind = torch.clamp(ind, 0, v["x"].shape[1] - 1 - 1)

    # helper function to select stuff according to the found indices.
    def sel(name):
        if is_flat[name]:
            return v[name].contiguous().view(-1)[ind]
        return torch.gather(v[name], 1, ind)

    # activating gradient storing for everything now
    enable_grad = False
    saved_inputs = []
    for name in ["x", "y", "xnew"]:
        if require_grad[name]:
            enable_grad = True
            saved_inputs += [v[name]]
        else:
            saved_inputs += [
                None,
            ]
    # assuming x are sorted in the dimension 1, computing the slopes for
    # the segments
    is_flat["slopes"] = is_flat["x"]
    # now we have found the indices of the neighbors, we start building the
    # output. Hence, we start also activating gradient tracking
    with torch.enable_grad() if enable_grad else contextlib.suppress():
        v["slopes"] = (v["y"][:, 1:] - v["y"][:, :-1]) / (
            eps + (v["x"][:, 1:] - v["x"][:, :-1])
        )

        # now build the linear interpolation
        ynew = sel("y") + sel("slopes") * (v["xnew"] - sel("x"))

        # Create masks for values outside the bounds of x
        x_min = v["x"].min(dim=1, keepdim=True)[0]
        x_max = v["x"].max(dim=1, keepdim=True)[0]

        # Set values outside bounds to zero
        outside_bounds = (v["xnew"] < x_min) | (v["xnew"] > x_max)
        ynew = torch.where(outside_bounds, torch.zeros_like(ynew), ynew)

        if reshaped_xnew:
            ynew = ynew.view(original_xnew_shape)

    return ynew


import time
import logging

TIME_STACK = []

# -- define base logger and formatting options --
logger = logging.getLogger("axonml")
logFormatter = logging.Formatter(
    "%(asctime)s %(name)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)


def tic(message=None, log=True):
    TIME_STACK.append(time.time())
    if message and log:
        logger.info(str(message))


def toc(message=None, log=True):
    try:
        t = time.time() - TIME_STACK.pop()
        output = f"Elapsed: {t:.3f}s"
        if message:
            output = f"{message}:: {output}"
        if log:
            logger.info(output)
        return t
    except IndexError:
        logger.error("You have to tic() before you toc()")
