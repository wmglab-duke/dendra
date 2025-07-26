from importlib.resources import files

import glob
import os
import torch

from .helpers import *

from .models import *
from .models.mod import load_mechanisms
from .models.stim.waveform import *
from .models.fields import *
from .models.integrators import *
from .const import *

import axonml.models.callbacks as callbacks
import axonml.models.mod as mod

import time
from pathlib import Path
import warnings
import pickle

# --- This is the core logic ---


def _cache_cpu_isa_list():
    """
    Checks for a cached CPU ISA string. If not found, runs the slow
    detection and caches the result for future runs.
    """
    # Use a standard cache location.
    cache_dir = Path.home() / ".cache" / "axonml"
    cache_file = cache_dir / "cpu_isa_list"
    platform_file = cache_dir / "platform.txt"

    def get_valid_vec_isa_list():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # --- Cache Hit ---
    if cache_file.exists():
        # Check if the PyTorch version matches the one used to create the cache.
        # This helps invalidate the cache if the user updates PyTorch.
        content = platform_file.read_text()
        if content == torch.__version__:
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
        platform_file.write_text(f"{torch.__version__}")
    except Exception as e:
        warnings.warn(
            f"AxonML: CPU capability check failed: {e}. Falling back to default behavior."
        )
        return


_cache_cpu_isa_list()

# register trained models
__this = files(__package__) / "trained"
all_trained = glob.glob(str(__this) + "/*.pt")
all_trained = {os.path.split(t)[1][:-3]: t for t in all_trained}

# setup environment
allow_tf32(bool(TF32))
