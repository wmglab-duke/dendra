"""Dendra: A PyTorch-based framework for simulating and modeling neural dynamics."""

import os

__version__ = "0.23.1"

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

from .const import FARADAY, PI, TAU, E, R
from .diagnostics import doctor
from .helpers import (
    COMPILE_OPTIONS,
    DEVICE,
    DTYPE,
    NATIVE_EXTENSION_POLICY,
    RUNTIME_CONTRACT_VALIDATION,
    TF32,
    allow_tf32,
    ctx,
    current_compile_options,
    current_device,
    current_dtype,
    current_native_extension_policy,
    current_runtime_contract_validation,
    normalize_native_extension_policy,
    normalize_runtime_contract_validation,
    set_compile_options,
    set_jit_enabled,
    set_jit_in_network_enabled,
    set_jit_network_ops_enabled,
    set_jit_network_solves_enabled,
)
from .models import (
    Axon,
    Cable,
    CompartmentGeometry,
    CompartmentGraph,
    CompartmentMetadata,
    CompartmentTopology,
    DistributionSpec,
    ExtCellAxon,
    ExtCellTree,
    Morphology,
    Myelinated,
    NetStim,
    Network,
    Population,
    RandomParameterSpec,
    RuntimeNoiseSpec,
    Section,
    SectionLocation,
    SingleCompartment,
    SynapseSlots,
    Tree,
    Unmyelinated,
    available_random_distributions,
    concat_models,
    concat_slices,
    connect_morphologies,
    get_random_distribution,
    make_random_parameter_spec,
    make_runtime_noise_spec,
    register_random_distribution,
    sample_random_parameter,
    sample_runtime_noise,
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
from .stepping import step, step_network, step_population

__all__ = [
    "PI",
    "E",
    "TAU",
    "R",
    "FARADAY",
    "ctx",
    "doctor",
    "step",
    "step_population",
    "step_network",
    "DEVICE",
    "DTYPE",
    "NATIVE_EXTENSION_POLICY",
    "RUNTIME_CONTRACT_VALIDATION",
    "COMPILE_OPTIONS",
    "current_compile_options",
    "current_device",
    "current_dtype",
    "current_native_extension_policy",
    "current_runtime_contract_validation",
    "normalize_runtime_contract_validation",
    "normalize_native_extension_policy",
    "set_compile_options",
    "set_jit_enabled",
    "set_jit_network_solves_enabled",
    "set_jit_network_ops_enabled",
    "set_jit_in_network_enabled",
    "callbacks",
    "mod",
    "load_mechanisms",
    "DistributionSpec",
    "RandomParameterSpec",
    "RuntimeNoiseSpec",
    "available_random_distributions",
    "get_random_distribution",
    "register_random_distribution",
    "make_random_parameter_spec",
    "make_runtime_noise_spec",
    "sample_random_parameter",
    "sample_runtime_noise",
    "Population",
    "Morphology",
    "Section",
    "SectionLocation",
    "CompartmentTopology",
    "CompartmentGeometry",
    "CompartmentMetadata",
    "CompartmentGraph",
    "SingleCompartment",
    "Cable",
    "Axon",
    "Unmyelinated",
    "Myelinated",
    "Tree",
    "ExtCellAxon",
    "ExtCellTree",
    "NetStim",
    "Network",
    "SynapseSlots",
    "concat_models",
    "concat_slices",
    "connect_morphologies",
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
