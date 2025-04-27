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
__this = os.path.dirname(__file__)
all_trained = glob.glob(__this + "/trained/*")
all_trained = {os.path.split(t)[1]: t for t in all_trained}

# setup environment
allow_tf32(bool(TF32))
torch.set_default_device("cuda" if CUDA else "cpu")
