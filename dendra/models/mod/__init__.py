import importlib
import importlib.util
import sys
from pathlib import Path

from .ALPHASYNAPSE import alphasynapse
from .ALPHASYNAPSE_D import alpha_d, alphasynapse_d
from .APCOUNT import apcount
from .APCOUNT_D import apcount_d
from .EXP2SYN import exp2syn
from .EXPSYN import expsyn
from .FIRE import fire
from .FIRE_D import fire_d
from .FIRE_R import fire_r
from .FIRE_R_D import fire_r_d
from .GRADED_SYNAPSE import graded_release_gate, graded_syn, sigmoid_release
from .HH import hh
from .PAS import pas
from .SPIKEDETECT import spikedetect


class MechanismContainer:
    """Mechanism definitions returned by :func:`load_mechanisms`.

    Access definitions as ``container.name`` or ``container["name"]``. Use
    :meth:`available` to list the stored names.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def available(self):
        """Return the loaded definition names in insertion order."""
        return list(self.__dict__.keys())


def load_mechanisms(*paths):
    """Load mechanism definitions from Python files in the supplied directories.

    Parameters
    ----------
    *paths : str or path-like
        Directories to scan for ``*.py`` files, excluding ``__init__.py``.

    Returns
    -------
    MechanismContainer
        Definitions accessible by attribute or string key. Each file must export
        a mechanism whose name is the lower-case file stem; ``MyChannel.py``
        must therefore export ``mychannel``. Later directories replace earlier
        definitions with the same lower-case name.

    Notes
    -----
    Loading executes the Python modules. Import or definition errors propagate
    to the caller. Only load trusted mechanism code.
    """

    mechanisms = {}

    for root in paths:
        root = Path(root).resolve()
        for file in root.glob("*.py"):
            if file.name == "__init__.py":
                continue

            module_name = file.stem.lower()  # e.g. "nats"
            unique_name = f"{module_name}_{file.stat().st_ino}"  # avoids collisions

            spec = importlib.util.spec_from_file_location(unique_name, file)
            module = importlib.util.module_from_spec(spec)

            # ★ Make the module discoverable
            sys.modules[unique_name] = module

            spec.loader.exec_module(module)

            try:
                mechanisms[module_name] = getattr(module, module_name)
            except AttributeError as e:
                raise AttributeError(
                    f"{file.name} does not define a class called {module_name}"
                ) from e

    return MechanismContainer(**mechanisms)


__all__ = [
    "MechanismContainer",
    "load_mechanisms",
    "alphasynapse",
    "alphasynapse_d",
    "alpha_d",
    "apcount",
    "apcount_d",
    "exp2syn",
    "expsyn",
    "fire",
    "fire_d",
    "fire_r",
    "fire_r_d",
    "graded_release_gate",
    "graded_syn",
    "sigmoid_release",
    "hh",
    "pas",
    "spikedetect",
]
