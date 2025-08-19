import os
import pickle
import time
import warnings
from importlib.resources import files
from pathlib import Path

import torch

import axonml.models.callbacks as callbacks
import axonml.models.mod as mod

from .const import (
    FARADAY,
    PI,
    TAU,
    E,
    R,
)
from .helpers import TF32, allow_tf32, ctx
from .models import (
    Axon,
    ExtCellAxon,
    ExtCellTree,
    Myelinated,
    NetStim,
    Network,
    Population,
    Tree,
    Unmyelinated,
)
from .models.fields import (
    PreComputedInterpolate1D,
    anisotropic_point,
    isotropic_point,
    parametric_efield,
    precomputed_interpolate_1d,
)
from .models.integrators import (
    bwd_euler_bt,
    bwd_euler_sc,
    bwd_euler_ub,
    df,
    dfh,
    dhs,
    dufort_frankel,
    dufort_frankel_homogeneous,
    euler,
    eulerv1,
    rk1,
    rk2,
    rk4,
)
from .models.mod import load_mechanisms
from .models.stim.waveform import *

__all__ = [
    "PI",
    "E",
    "TAU",
    "R",
    "FARADAY",
    "ctx",
    "callbacks",
    "mod",
    "load_mechanisms",
    "Population",
    "Axon",
    "Unmyelinated",
    "Myelinated",
    "Tree",
    "ExtCellAxon",
    "ExtCellTree",
    "NetStim",
    "Network",
    "anisotropic_point",
    "isotropic_point",
    "parametric_efield",
    "PreComputedInterpolate1D",
    "precomputed_interpolate_1d",
    "euler",
    "eulerv1",
    "rk1",
    "rk2",
    "rk4",
    "dufort_frankel",
    "dufort_frankel_homogeneous",
    "bwd_euler_sc",
    "bwd_euler_ub",
    "bwd_euler_bt",
    "dhs",
    "df",
    "dfh",
]


def _cache_cpu_isa_list():
    """
    Checks for a cached CPU ISA string. If not found, runs the slow
    detection and caches the result for future runs.
    """
    # Use a standard cache location.
    cache_dir = Path.home() / ".cache" / "axonml"
    cache_file = cache_dir / f"cpu_isa_list_{torch.__version__}"

    def get_valid_vec_isa_list():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # --- Cache Hit ---
    if cache_file.exists():
        # Check if the PyTorch version matches the one used to create the cache.
        torch._inductor.cpu_vec_isa.valid_vec_isa_list = get_valid_vec_isa_list
        return

    # --- Cache Miss or Stale Cache ---
    print("AxonML: Performing one-time CPU capability check. This may take a minute...")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        start = time.time()
        valid_vec_isa_list = torch._inductor.cpu_vec_isa.valid_vec_isa_list()
        print(f"AxonML: CPU capability check took {(time.time() - start):.3f}s.")
        with open(cache_file, "wb") as f:
            pickle.dump(valid_vec_isa_list, f)
    except Exception as e:
        warnings.warn(
            f"AxonML: CPU capability check failed: {e}. Falling back to default behavior."
        )
        return


_cache_cpu_isa_list()

# setup environment
allow_tf32(bool(TF32))
