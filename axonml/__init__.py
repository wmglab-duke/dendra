from importlib import import_module
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

# register trained models
__this = files(__package__) / "trained"
all_trained = glob.glob(str(__this) + "/*.pt")
all_trained = {os.path.split(t)[1][:-3]: t for t in all_trained}

# setup environment
allow_tf32(bool(TF32))
