import linecache
import sys

import torch

from .handler import parse_args, parse_assignments
from ...declarations import add_to_namespace_dict

from .defaults import reversals, VALENCES, cinits
from axonml.helpers import DETECT_ANOMALIES, DEBUG

R = 1e3 * 8.31446261815324
FARADAY = 96485.33212331001


template = """
class Ion(torch.nn.Module):
  def __init__(self, shape):
    super().__init__()
    self.name = "{ion}"
    self.rzf = {R} / ({valence} * {FARADAY})
    self.register_buffer("i{ion}", torch.zeros(shape))
    self.register_{e_buffer_or_param}("e{ion}", {einit_tensor})
    self.register_{c_buffer_or_param}("{ion}i", {cinit_i_tensor})
    self.register_{c_buffer_or_param}("{ion}o", {cinit_o_tensor})

  def register_mech_writes_c(self, mech):
    setattr(self, mech._name, mech)

  @torch.jit.export
  def initialize(self, temp) -> None:
    self.i{ion}[:] = 0.0
    self.i{ion}.detach_()
    {initialize_e}
    {initialize_i}
    {initialize_o}
    self.detach()
    self.einit(temp)

  def detach(self):
    self.i{ion}.detach_()
    self.{ion}i.detach_()
    self.{ion}o.detach_()

  @torch.jit.export
  def advance(self, temp) -> None:
    self.eadvance(temp)

  def einit(self, temp) -> None:
    {einit}
    return

  def eadvance(self, temp) -> None:
    {clamp}
    {eadvance}
    {detect_anomalies}
    return

  def set(self, key: str, value):
    p = getattr(self, key)
    if isinstance(p, torch.Tensor) or isinstance(p, torch.nn.Parameter):
      p.data.copy_(torch.as_tensor(value, dtype=p.data.dtype, device=p.device))
"""


def init_tensor(val, buffer_or_param):
    if buffer_or_param == "buffer":
        return f"torch.full(shape, {val})"
    return f"torch.nn.Parameter(torch.tensor({val}), requires_grad=False)"


def parse_einit(ion, einit):
    if einit == 0:
        return ""
    return f"self.e{ion}[:] = (torch.log(self.{ion}o / self.{ion}i) * self.rzf * (273.15 + temp)); self.e{ion}.detach_()"


def parse_eadvance(ion, eadvance):
    if not eadvance:
        return ""
    return f"self.e{ion} = (torch.log(self.{ion}o / self.{ion}i) * self.rzf * (273.15 + temp))"


def clamp(ion):
    return f"self.{ion}i.clamp_(min=1e-9); self.{ion}o.clamp_(min=1e-9)"


def parse_read_c_self(ion_write_c):
    if not ion_write_c:
        return ""
    res = []
    for k, v in ion_write_c.items():
        for v_ in v:
            res.append(f"self.{v_}.data.copy_(self.{k}.{v_})")
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
        for v_ in v:
            res.append(f"self.{k}.{v_} = self.{v_}")
    return "\n    ".join(res)


def parse_write_e(ion_read_e, eadvance):
    if not eadvance:
        return ""
    res = []
    for k, v in ion_read_e.items():
        for v_ in v:
            res.append(f"self.{k}.{v_} = self.{v_}")
    return "\n    ".join(res)


def parse_write_c(ion_read_c, c_style):
    if c_style < 3:
        return ""
    res = []
    for k, v in ion_read_c.items():
        for v_ in v:
            res.append(f"self.{k}.{v_} = self.{v_}")
    return "\n    ".join(res)


def parse_write_c_after_init(ion_read_c, cinit):
    if not cinit:
        return ""
    res = []
    for k, v in ion_read_c.items():
        for v_ in v:
            res.append(f"self.{k}.{v_} = self.{v_}")
    return "\n    ".join(res)


def parse_write_i_after_init(ion_read_i):
    if not ion_read_i:
        return ""
    res = []
    for k, v in ion_read_i.items():
        for v_ in v:
            res.append(f"self.{k}.{v_} = self.{v_}")
    return "\n    ".join(res)


def parse_write_e_after_init(ion_read_e, einit):
    if not einit:
        return ""
    res = []
    for k, v in ion_read_e.items():
        for v_ in v:
            res.append(f"self.{k}.{v_} = self.{v_}")
    return "\n    ".join(res)


def parse_e_buffer_or_param(e_style):
    if e_style < 2:
        return "parameter"
    return "buffer"


def parse_c_buffer_or_param(c_style):
    if c_style < 2:
        return "parameter"
    return "buffer"


def build_ion(
    ion,
    shape,
    n_ax,
    n_comp,
    mechanisms,
    mechanism_names,
    ion_read,
    ion_write_c,
    c_style,
    e_style,
    einit,
    eadvance,
    cinit,
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

    fi = f"{ion}i0"
    fo = f"{ion}o0"
    fe = f"e{ion}"

    e_init_str = f"self.e{ion}.data.copy_({reversals()[fe]})"
    i_init_str = f"self.{ion}i.data.copy_({cinits()[fi]})"
    o_init_str = f"self.{ion}o.data.copy_({cinits()[fo]})"

    # eadvance
    eadvance_str = parse_eadvance(ion, eadvance)

    e_is_buffer = parse_e_buffer_or_param(e_style)
    c_is_buffer = parse_c_buffer_or_param(c_style)

    if DETECT_ANOMALIES:
        detect_anomalies = f"assert torch.all(torch.isfinite(self.e{ion})), 'Anomaly detected in e{ion}'\n    assert torch.all(torch.isfinite(self.{ion}i)), 'Anomaly detected in {ion}i'\n    assert torch.all(torch.isfinite(self.{ion}o)), 'Anomaly detected in {ion}o'"
    else:
        detect_anomalies = ""

    forward_str = template.format(
        arguments=arguments,
        assignments=assignments,
        e_buffer_or_param=e_is_buffer,
        c_buffer_or_param=c_is_buffer,
        einit_tensor=init_tensor(reversals()[f"e{ion}"], e_is_buffer),
        cinit_i_tensor=init_tensor(cinits()[f"{ion}i0"], c_is_buffer),
        cinit_o_tensor=init_tensor(cinits()[f"{ion}o0"], c_is_buffer),
        initialize_e=e_init_str,
        initialize_i=i_init_str,
        initialize_o=o_init_str,
        ion=ion,
        R=R,
        FARADAY=FARADAY,
        e_ion=reversals()[f"e{ion}"],
        ion_i_0=cinits()[f"{ion}i0"],
        ion_o_0=cinits()[f"{ion}o0"],
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
        write_e_immediate=parse_write_e(ion_read_e, 1),
        write_c_immediate=parse_write_c(ion_read_c, 3),
        clamp="",
        detect_anomalies=detect_anomalies,
    )

    if DEBUG >= 3:
        print(forward_str)

    filename = f"<{ion}_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = locals()["Ion"](shape)

    return m


def USEION(ion, read=[], write=[]):
    if not read and not write:
        return

    assert ion in VALENCES, f"ion {ion} is not registered"

    if f"e{ion}" in write:
        raise ValueError(f"e{ion} cannot be written")

    if common := set(read).intersection(write):
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
        add_to_namespace_dict(namespace, "_read_ion", **{ion: read})

    if write:
        c_write = []
        other = []
        for w in write:
            if w in {f"{ion}i", f"{ion}o"}:
                c_write.append(w)
            else:
                other.append(w)

        if c_write:
            add_to_namespace_dict(namespace, "_write_ion_c", **{ion: c_write})
        if other:
            add_to_namespace_dict(namespace, "_write_ion", **{ion: other})
