"""axonml.const

Mathematical and physical constants used throughout AxonML.

All values are plain Python ``float`` constants (TorchScript-friendly).
"""

from __future__ import annotations

import math

import torch

__all__ = ["PI", "E", "TAU", "R", "FARADAY"]

PI: torch.jit.Final[float] = math.pi
""":math:`\\pi`, ratio of a circle's circumference to its diameter."""

E: torch.jit.Final[float] = math.e
"""Euler's number."""

TAU: torch.jit.Final[float] = math.tau
""":math:`\\tau = 2\\pi`."""

R: torch.jit.Final[float] = 8314.46261815324
"""Molar gas constant in SI units of J/(kmol·K)."""

FARADAY: torch.jit.Final[float] = 96485.33212331001
"""Faraday constant in SI units of C/mol."""
