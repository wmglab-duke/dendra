from typing import List
import re
import string
import random

import torch
import linecache
import textwrap


def indent(text, level=0):
    return textwrap.indent(text, " " * (4 * level))


template = """
class MechanismHandler(torch.nn.Module):
  def __init__(self, temp, {arguments}):
    super().__init__()
    self.temp = temp
    {assignments}

  def initialize(self, v, v_init, area, temp) -> None:
    self.ion_init(temp)
    self.init_buffers(v_init)
    self.itot(v)

  @torch.jit.export
  def set_buffers(self):
{set_buffers}
    return

  def advance(self, v, dt, temp) -> None:
    {mech_advance}
    self.update(temp)
    return

  def gtot(self) -> torch.Tensor:
    return {gtot}

  def i(self, v) -> torch.Tensor:
    {breakpoint}
    {currents}
    total = {total}
    return total

  def itot(self, v):
    {currents_tot}
    return

  def update(self, temp) -> None:
    {ion_advance}
    return

  def ion_init(self, temp) -> None:
    {ion_init}
    return

  def init_buffers(self, v_init) -> None:
    {init_buffers}
    return

  @torch.jit.ignore
  def get(self, mech: str, state: str) -> torch.Tensor:
    return getattr(self, mech).get(state)

  @torch.jit.export
  def all_states(self) -> List[str]:
    return [{all_states}]
    
"""


set_buffer_template = """
self.{mech}.{v}.set_(self.{ion}_ion.{v})
for _, s in self.{mech}.DE.items():
  s.{v}.set_(self.{ion}_ion.{v})
"""


set_ion_write_c_buffer_template = """
self.{mech}.{v}.set_(self.{ion}_ion.{v})
"""


def parse_set_buffers(mechanisms, ions):
    out = []
    for m in mechanisms:
        for ion in ions:
            if ion in m.read_ion:
                for v in m.read_ion[ion]:
                    out.append(set_buffer_template.format(mech=m._name, v=v, ion=ion))
            if ion in m.write_ion_c:
                for v in m.write_ion_c[ion]:
                    out.append(
                        set_ion_write_c_buffer_template.format(
                            mech=m._name, v=v, ion=ion
                        )
                    )
    out = "\n".join(out)
    return indent(out, 1)


def parse_assignments(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key} = {key}")
    return "\n    ".join(result)


def parse_args(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"{key}: torch.nn.Module")
    return ", ".join(result)


def parse_ions(ion_names) -> str:
    result = []
    for key in ion_names:
        result.append(f"{key}_ion: torch.nn.Module")
    return ", ".join(result)


def parse_current_string(s: str, total=False, write=True) -> str:
    if not write:
        return s
    pattern = r"^i([A-Za-z]+)$"
    match = re.match(pattern, s)
    if match:
        # Extract the letters following 'i'
        letters = match.group(1)
        # Construct the transformed string
        if total:
            return f"self.{letters}_ion.i{letters}"
        return f"self.{letters}_ion.i{letters}[:]"
    else:
        # If no match, leave the string unchanged
        return s


def parse_dictionary_to_sum(data: dict, current: str, df=False, write=True) -> str:
    if not data:
        return ""
    result = []
    for key, value_set in data.items():
        for value in value_set:
            if df:
                result.append(f"self.{key}.{value}_tot(v)")
            else:
                result.append(f"self.{key}.{value}(v)")
    s = " + ".join(result)
    return f"{parse_current_string(current, write=write)} = {s}"


def parse_currents(currents, write=True) -> str:
    result = []
    for key, value in currents.items():
        result.append(parse_dictionary_to_sum(value, key, write=write))
    s = "\n    ".join(result)
    return s


def parse_scale(currents) -> str:
    result = []
    for key in currents.keys():
        result.append(f"{key} = {key} * area")
    s = "\n    ".join(result)
    return s


def parse_total(currents, write=True) -> str:
    s = " + ".join([parse_current_string(key, True, write) for key in currents.keys()])
    return s


def parse_advance(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key}._advance(v, dt)")
    s = "\n    ".join(result)
    return s


def parse_inflate(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key}._inflate_s(v)")
    s = "\n    ".join(result)
    return s


def parse_init_buffers(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key}._init_buffers_s(v_init)")
    s = "\n    ".join(result)
    return s


def parse_write_ions(ions_to_write) -> str:
    if ions_to_write is None:
        return ""
    result = []
    for ion in ions_to_write:
        result.append(f"self.{ion}_ion.i{ion} = i{ion}")
    s = "\n    ".join(result)
    return s


def parse_ion_init(ions) -> str:
    if ions is None:
        return ""
    result = []
    for ion in ions:
        result.append(f"self.{ion}_ion.initialize(temp)")
    s = "\n    ".join(result)
    return s


def parse_ion_advance(ions) -> str:
    if ions is None:
        return ""
    result = []
    for ion in ions:
        result.append(f"self.{ion}_ion.advance(temp)")
    s = "\n    ".join(result)
    return s


def parse_setattr(ions) -> str:
    if ions is None:
        return ""
    result = []
    for ion in ions:
        for attr, p in [(f"{ion}o", "c"), (f"{ion}i", "c"), (f"e{ion}", "e")]:
            result.append(
                f"if name == '{attr}': self.{ion}_ion.{attr} = torch.as_tensor(value, device=self.{ion}_ion.{attr}.device); self.{ion}_ion.immediate_update_{p}(); return"
            )
    s = "\n    ".join(result)
    return s


def parse_all_states(mechanisms) -> str:
    result = []
    for mech in mechanisms:
        for state in mech.DE:
            result.append(f"'{mech._name}.{state}'")
    s = ", ".join(result)
    return s


def parse_defaults(ions) -> str:
    res = []
    for ion in ions:
        res.append(f"self.{ion}_ion.immediate_update_e()")
    return "\n    ".join(res)


def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def gtot(mechanisms):
    return " + ".join([f"self.{m._name}.gtot()" for m in mechanisms])


def breakpoint(mechanisms):
    ret = []
    for m in mechanisms:
        ret.append(f"self.{m._name}.breakpoint(v)")
    return "\n    ".join(ret)


def tot_currents(currents, df=False):
    result = []
    for key, value in currents.items():
        result.append(parse_dictionary_to_sum(value, key, df))
    s = "\n    ".join(result)
    return s


def build_handler(mechanisms, names, currents, temp, ions=None, df=False):
    arguments = parse_args(names)

    all_names = names

    if ions is not None:
        ion_names = list(ions.keys())
        all_names = names + [f"{i}_ion" for i in ion_names]
        arguments = arguments + ", " + parse_ions(ion_names)

    forward_str = template.format(
        arguments=arguments,
        assignments=parse_assignments(all_names),
        defaults=parse_defaults(ions),
        currents=parse_currents(currents, write=not df),
        total=parse_total(currents, write=not df),
        inflate=parse_inflate(names),
        mech_advance=parse_advance(names),
        init_buffers=parse_init_buffers(names),
        ion_init=parse_ion_init(ions),
        ion_advance=parse_ion_advance(ions),
        define_setattr=parse_setattr(ions),
        all_states=parse_all_states(mechanisms),
        gtot=gtot(mechanisms),
        breakpoint=breakpoint(mechanisms),
        set_buffers=parse_set_buffers(mechanisms, list(ions.keys())),
        currents_tot=tot_currents(currents, df),
    )

    filename = f"<{randomword(10)}_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    if ions is not None:
        inputs = mechanisms + list(ions.values())
    else:
        inputs = mechanisms

    m = locals()["MechanismHandler"](temp, *inputs)
    return m
