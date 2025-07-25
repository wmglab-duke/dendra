import os
import importlib

import sys
import importlib.util
from pathlib import Path


from .hh import hh
from .exp2syn import exp2syn
from .expsyn import expsyn
from .alphasynapse import alphasynapse
from .pas import pas


class MechanismContainer:
    """Simple attribute / dict‑style container."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def available(self):
        return list(self.__dict__.keys())


def load_mechanisms(*paths):
    """
    Load mechanism modules from the specified file paths.
    This function dynamically imports Python files from the given directories,
    extracting modules that are expected to contain mechanism implementations.
    Each mechanism is assumed to have a class with the same name as the file.

    Parameters
    ----------
    *paths : str
        Variable number of directory paths where mechanism files are located.

    Returns
    -------
    MechanismContainer
        A container object that holds all successfully loaded mechanisms.
        The mechanisms are accessible as attributes of the container,
        with attribute names corresponding to the module names.

    Notes
    -----
    - Files must have a .py extension and not be named "__init__.py"
    - Each file should define a class with the same name as the file itself
    - The function assumes the module structure follows the convention where
      the class name matches the file name

    Examples
    --------
    >>> mechanisms = load_mechanisms('/path/to/mechanisms', '/another/path')
    >>> my_mechanism = mechanisms.mechanism_name
    """

    mechanisms = {}

    for root in paths:
        root = Path(root).resolve()
        for file in root.glob("*.py"):
            if file.name == "__init__.py":
                continue

            module_name = file.stem  # e.g. "nats"
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
