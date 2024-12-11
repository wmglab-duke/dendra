import linecache
import sys

import torch

from .handler import parse_args, parse_assignments
from ..declarations import add_to_namespace_dict


def ion_register(ion, valence, e, i0, o0):
    global VALENCES
    global REVERSAL
    global CINIT
    VALENCES[ion] = valence
    REVERSAL[ion] = e
    CINIT[f"{ion}o0"] = o0
    CINIT[f"{ion}i0"] = i0


R = 1e3 * 8.31446261815324
FARADAY = 96485.33212331001


# default reversal potentials from NEURON
REVERSAL = {"na": 50.0, "k": -77.0, "ca": 132.0}

VALENCES = {"na": 1.0, "k": 1.0, "ca": 2.0}

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
  def __init__(self, {arguments}):
    super().__init__()
    self.rzf = {R} / ({valence} * {FARADAY})
    self.register_buffer("i{ion}", torch.tensor(0.0), persistent=False)
    self.register_buffer("e{ion}", torch.tensor({e_ion}), persistent=False)
    self.register_buffer("{ion}i", torch.tensor({ion_i_0}), persistent=False)
    self.register_buffer("{ion}o", torch.tensor({ion_o_0}), persistent=False)
    {assignments}

  @torch.jit.export
  def initialize(self, temp) -> None:
    self.i{ion} = torch.tensor(0.0, device=self.i{ion}.device)
    self.e{ion} = torch.tensor({e_ion}, device=self.e{ion}.device)
    self.{ion}i = torch.tensor({ion_i_0}, device=self.{ion}i.device)
    self.{ion}o = torch.tensor({ion_o_0}, device=self.{ion}o.device)
    self.einit(temp)
    self.write_after_init()

  @torch.jit.export
  def advance(self, temp) -> None:
    {read_c_self}
    self.eadvance(temp)
    self.write()

  def einit(self, temp) -> None:
    {einit}

  def eadvance(self, temp) -> None:
    {eadvance}

  def write_after_init(self) -> None:
    {write_c_after_init}
    {write_i_after_init}
    {write_e_after_init}
    pass

  def write(self) -> None:
    # write things which get read
    {write_c}
    {write_i}
    {write_e}
    pass
"""


def parse_einit(ion, einit):
    if einit == 0:
        return "pass"
    return f"self.e{ion} = torch.log(self.{ion}o / self.{ion}i) * self.rzf * (273.15 + temp)"


def parse_eadvance(ion, eadvance):
    return parse_einit(ion, eadvance)


def parse_read_c_self(ion_write_c):
    if not ion_write_c:
        return ""
    res = []
    for k, v in ion_write_c.items():
        res.append(f"self.{v} = self.{k}.{v}")
    return "\n    ".join(res)


def parse_write_c_after_init(c_style):
    pass


def parse_write_c(c_style):
    if c_style < 3:
        return ""
    pass


def parse_write_i(ion_read_i):
    res = []
    for k, v in ion_read_i.items():
        res.append(f"self.{k}.{v} = self.{v}")
    return "\n    ".join(res)


def parse_write_e(ion_read_e, eadvance):
    if not eadvance:
        return ""
    res = []
    for k, v in ion_read_e.items():
        res.append(f"self.{k}.{v} = self.{v}")
    return "\n    ".join(res)


def parse_write_c(ion_read_c, c_style):
    if c_style < 3:
        return ""
    res = []
    for k, v in ion_read_c.items():
        res.append(f"self.{k}.{v} = self.{v}")
    return "\n    ".join(res)


def parse_write_c_after_init(ion_read_c, cinit):
    if not cinit:
        return ""
    res = []
    for k, v in ion_read_c.items():
        res.append(f"self.{k}.{v} = self.{v}")
    return "\n    ".join(res)


def parse_write_i_after_init(ion_read_i):
    if not ion_read_i:
        return ""
    res = []
    for k, v in ion_read_i.items():
        res.append(f"self.{k}.{v} = self.{v}")
    return "\n    ".join(res)


def parse_write_e_after_init(ion_read_e, einit):
    if not einit:
        return ""
    res = []
    for k, v in ion_read_e.items():
        res.append(f"self.{k}.{v} = self.{v}")
    return "\n    ".join(res)


def build_ion(
    ion, mechanisms, mechanism_names, ion_read, ion_write_c, c_style, e_style, einit, eadvance, cinit
):
    ion_read_i = {}
    ion_read_e = {}
    ion_read_c = {}

    for mech, v in ion_read.items():
        for v_ in v:
            if v_ == f"e{ion}":
                ion_read_e.setdefault(mech, []).append(v_)
            elif v_ == f"i{ion}":
                ion_read_i.setdefault(mech, []).append(v_)
            elif v_ in [f"{ion}i", f"{ion}o"]:
                ion_read_c.setdefault(mech, []).append(v_)
    
    # args
    arguments = parse_args(mechanism_names)

    # assignments
    assignments = parse_assignments(mechanism_names)

    # initialize (cinit here)

    # einit
    einit_str = parse_einit(ion, einit)

    # eadvance
    eadvance_str = parse_eadvance(ion, eadvance)

    # update -> copy to mechanisms
    update = "pass"

    forward_str = template.format(
        arguments=arguments,
        assignments=assignments,
        ion=ion,
        R=R,
        FARADAY=FARADAY,
        e_ion=REVERSAL[ion],
        ion_i_0=CINIT[f"{ion}i0"],
        ion_o_0=CINIT[f"{ion}o0"],
        valence=VALENCES[ion],
        read_c_self=parse_read_c_self(ion_write_c),
        einit=einit_str,
        eadvance=eadvance_str,
        write_c=parse_write_c(ion_read_c, c_style),
        write_i=parse_write_i(ion_read_i),
        write_e=parse_write_e(ion_read_e, eadvance),
        write_c_after_init=parse_write_c_after_init(ion_read_c, cinit),
        write_i_after_init=parse_write_i_after_init(ion_read_i),
        write_e_after_init=parse_write_e_after_init(ion_read_e, einit),
    )

    filename = "<forward_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = torch.jit.script(locals()["Ion"](*mechanisms))

    return m


def USEION(ion, read=[], write=[]):
    if not read and not write:
        return

    assert ion in VALENCES, f"ion {ion} is not registered"

    if (common := set(read).intersection(write)):
        raise ValueError(f"{common} is/are both read and written")

    valid = {f"{ion}i", f"{ion}o", f"e{ion}", f"i{ion}"}

    # check that all elements in read and write are valid
    for r in read:
        assert r in valid, f"read {r} is not valid"
    for w in write:
        assert w in valid, f"write {w} is not valid"

    frame = sys._getframe(1)
    namespace = frame.f_locals

    if read:
        add_to_namespace_dict(namespace, "_read_ion", {ion: read})

    if write:
        c_write = []
        other = []
        for w in write:
            if w in {f"{ion}i", f"{ion}o"}:
                c_write.append(w)
            else:
                other.append(w)

        if c_write:
            add_to_namespace_dict(namespace, "_write_ion_c", {ion: c_write})
        if other:
            add_to_namespace_dict(namespace, "_write_ion", {ion: other})
