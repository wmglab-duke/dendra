from typing import Dict
import sys

import torch

from ..declarations import add_to_namespace, add_to_namespace_dict


R = 1e3 * 8.31446261815324
FARADAY = 96485.33212331001


# default reversal potentials from NEURON
REVERSALS = {"ena": 50.0, "ek": -77.0, "eca": 132.0}


# default initial concentrations from NEURON
CINIT = {
    "nao0": 140.0,
    "nai0": 10.0,
    "ko0": 2.5,
    "ki0": 54.4,
    "cao0": 2.0,
    "cai0": 5e-5,
}


template = """
class Ion(torch.nn.Module):
  def __init__(self, {mechanisms}):
    super().__init__()
    self.register_buffer("R", torch.tensor({R}), persistent=False)
    self.register_buffer("F", torch.tensor({FARADAY}), persistent=False)
    self.register_buffer("i{ion}", torch.tensor({iion_init}), persistent=False)
    self.register_buffer("e{ion}", torch.tensor({eion_init}), persistent=False)
    self.register_buffer("{ion}i", torch.tensor({ion_i_init}), persistent=False)
    self.register_buffer("{ion}o", torch.tensor({ion_o_init}), persistent=False)
    self.z = {valence}
    {assignments}

  def initialize(self) -> None:
    {initialize}

  def einit(self, temp) -> None:
    {einit}

  def eadvance(self) -> None:
    {eadvance}
  
  def update(self) -> None:
    {update}
"""


def parse_mechanisms(mechanisms):
    return ", ".join(mechanism.__class__.__name__ for mechanism in mechanisms)


def parse_assignments(mechanisms):
    return "\n".join(
        f"self.{m.__class__.__name__} = {m.__class__.__name__}" for m in mechanisms
    )


def parse_einit(ion, einit):
    if einit == 0:
        return "pass"
    return f"self.e{ion} = torch.log(self.{ion}o / self.{ion}i) * self.R * (273.15 + temp) / (self.F * self.z)"


def parse_eadvance(ion, eadvance):
    if eadvance == 0:
        return "pass"
    return f"self.e{ion} = torch.log(self.{ion}o / self.{ion}i) * self.R * (273.15 + temp) / (self.F * self.z)"


def build_ion(ion, mechanisms, c_style, e_style, einit, eadvance):
    pass


def USEION(ion, read=[], write=[], state=[]):
    if not read and not write and not state:
        return

    frame = sys._getframe(1)
    namespace = frame.f_locals
