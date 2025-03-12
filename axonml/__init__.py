import glob
import os
import torch

from .models.mod import load_mechanisms
from .helpers import TF32, CUDA


this = os.path.dirname(__file__)
all_trained = glob.glob(this + "/trained/*")

all_trained = {os.path.split(t)[1]: t for t in all_trained}


def allow_tf32(allow=True):
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow


allow_tf32(bool(TF32))
torch.set_default_device("cuda" if CUDA else "cpu")
