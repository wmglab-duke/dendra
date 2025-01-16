import glob
import os
import torch

from .models.mod import load_mechanisms


this = os.path.dirname(__file__)
all_trained = glob.glob(this + "/trained/*")

trained = {os.path.split(t)[1]: t for t in all_trained}


def allow_tf32(allow=True):
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow


allow_tf32(False)
