"""Dendra: A PyTorch-based framework for simulating and modeling neural dynamics."""

import os

__version__ = "0.15.0"

os.environ.setdefault("OMP_DYNAMIC", "FALSE")
os.environ.setdefault("MKL_DYNAMIC", "FALSE")
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"

from ._bootstrap import configure_torchinductor_cache_for_dendra

configure_torchinductor_cache_for_dendra()

import pickle
import time
import warnings
from pathlib import Path

import torch

import dendra.models.callbacks as callbacks
import dendra.models.mod as mod

from .const import (
    FARADAY,
    PI,
    TAU,
    E,
    R,
)
from .helpers import (
    DEVICE,
    DTYPE,
    TF32,
    allow_tf32,
    ctx,
    current_device,
    current_dtype,
    set_jit_enabled,
    set_jit_in_network_enabled,
    set_jit_network_ops_enabled,
    set_jit_network_solves_enabled,
)
from .models import (
    Axon,
    ExtCellAxon,
    ExtCellTree,
    Myelinated,
    NetStim,
    Network,
    Population,
    SingleCompartment,
    Tree,
    Unmyelinated,
    concat_models,
    concat_slices,
)
from .models.fields import *
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
    scnv,
)
from .models.mod import load_mechanisms
from .models.stim.waveform import *
from .utils.inductor import refresh_torchinductor_precompiled_headers

__all__ = [
    "PI",
    "E",
    "TAU",
    "R",
    "FARADAY",
    "ctx",
    "DEVICE",
    "DTYPE",
    "current_device",
    "current_dtype",
    "set_jit_enabled",
    "set_jit_network_solves_enabled",
    "set_jit_network_ops_enabled",
    "set_jit_in_network_enabled",
    "callbacks",
    "mod",
    "load_mechanisms",
    "Population",
    "SingleCompartment",
    "Axon",
    "Unmyelinated",
    "Myelinated",
    "Tree",
    "ExtCellAxon",
    "ExtCellTree",
    "NetStim",
    "Network",
    "concat_models",
    "concat_slices",
    "anisotropic_point",
    "isotropic_point",
    "parametric_efield",
    "PreComputedInterpolate1D",
    "precomputed_interpolate_1d",
    "EfieldInterpolate3DRect",
    "efield_interpolate_3d_rect",
    "PreComputedInterpolate3DRect",
    "precomputed_interpolate_3d_rect",
    "PreComputedInterpolate3DScattered",
    "precomputed_interpolate_3d_scattered",
    "EfieldInterpolate3DScattered",
    "efield_interpolate_3d_scattered",
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
    "scnv",
]


def _cache_cpu_isa_list():
    """
    Checks for a cached CPU ISA string. If not found, runs the slow
    detection and caches the result for future runs.
    """
    # Use a standard cache location.
    cache_dir = Path.home() / ".cache" / "dendra"
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
    print("Dendra: Performing one-time CPU capability check. This may take a minute...")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        start = time.time()
        valid_vec_isa_list = torch._inductor.cpu_vec_isa.valid_vec_isa_list()
        print(f"Dendra: CPU capability check took {(time.time() - start):.3f}s.")
        with open(cache_file, "wb") as f:
            pickle.dump(valid_vec_isa_list, f)
    except Exception as e:
        warnings.warn(
            f"Dendra: CPU capability check failed: {e}. Falling back to default behavior."
        )
        return


_cache_cpu_isa_list()

# setup environment
allow_tf32(bool(TF32))

import torch._inductor.config as inductor_config

inductor_config.cpp_wrapper = True
