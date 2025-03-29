from typing import List
import re
import string
import random

import torch
import linecache
import textwrap

from axonml.helpers import DEBUG, DFITOT, IMEM


def indent(text, level=0):
    return textwrap.indent(text, " " * (4 * level))


template = """
class MechanismHandler(torch.nn.Module):
  def __init__(self, temp, {arguments}):
    super().__init__()
    self.temp = temp
    {assignments}
    {imem_assignment}

  def initialize(self, v, v_init, temp) -> None:
    self.ion_init(temp)
    self.init_buffers(v_init)
    self.itot(v)

  @torch.jit.export
  def set_buffers(self, diameters):
{set_buffers}
    return

  def advance(self, v, dt, temp) -> None:
    {mech_advance}
    {assign_post_advance}
    self.update(temp)
    return

  @torch.jit.export
  def detach(self) -> None:
    {mech_detach}
    {ion_detach}
    return

  def gtot(self, v) -> torch.Tensor:
    return {gtot}

  def i(self, v_prev, v) -> torch.Tensor:
    {breakpoint}
    {currents}
    {assign_currents_non_df}
    total = {total}
    {imem_write}
    return total

  def itot(self, v):
    {currents_tot}
    {assign_currents_df}
    {imem_write_df}
    return

  def update(self, temp) -> None:
    {ion_advance}
    {assign_equilibrium}
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


def v_args(df=False):
    if df:
        return "v_prev, v"
    return "v"


set_buffer_template = """
self.{mech}.{v}.set_(self.{ion}_ion.{v})
for _, s in self.{mech}.DE.items():
  s.{v}.set_(self.{ion}_ion.{v})
"""


set_ion_write_c_buffer_template = """
self.{mech}.{v}.set_(self.{ion}_ion.{v})
"""


set_diam_buffer_template = """
self.{mech}.diam.set_(diameters.view(-1, 1, 1))
"""


def is_prefix_letters(s: str, prefix_char: str) -> bool:
    # Construct a regex pattern to match, for example, '^i[a-zA-Z]+$'
    pattern = re.compile(r'^' + re.escape(prefix_char) + r'[a-zA-Z]+$')
    return bool(pattern.match(s))


def assign(mechanisms, ions, kind='i') -> str:
    out = []
    for m in mechanisms:
        for ion in ions:
            if ion in m.read_ion:
                for v in m.read_ion[ion]:
                    if is_prefix_letters(v, kind):
                        out.append(f"self.{m._name}.{v} = self.{ion}_ion.{v}")
                        for k, _ in m.DE.items():
                            out.append(f"self.{m._name}.DE['{k}'].{v} = self.{ion}_ion.{v}")
    out = "\n    ".join(out)
    return out


def assign_post_advance(mechanisms, ions) -> str:
    out = []
    for m in mechanisms:
        for ion in ions:
            if ion in m.write_ion_c:
                for v in m.write_ion_c[ion]:
                    out.append(f"self.{ion}_ion.{v} = self.{m._name}.{v}")
    for m in mechanisms:
        for ion in ions:
            if ion in m.read_ion:
                for v in m.read_ion[ion]:
                    if v == f"{ion}i" or v == f"{ion}o":
                        out.append(f"self.{m._name}.{v} = self.{ion}_ion.{v}")
                        for k, _ in m.DE.items():
                            out.append(f"self.{m._name}.DE['{k}'].{v} = self.{ion}_ion.{v}")
    out = "\n    ".join(out)
    return out


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
        out.append(set_diam_buffer_template.format(mech=m._name))
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
        return f"self.{letters}_ion.i{letters}"
    else:
        # If no match, leave the string unchanged
        return s


def parse_dictionary_to_sum(
    data: dict,
    current: str,
    df=False,
    dufort=False,
    unfactorable=None,
    write=True,
    mech_has_conductance=None,
    divide_by_two=None,
) -> str:
    if not data:
        return ""
    result = []
    for key, value_set in data.items():
        for value in value_set:
            if df:
                result.append(f"self.{key}.{value}_tot(v)")
            else:
                if not dufort:
                    result.append(f"self.{key}.{value}(v)")
                else:
                    if unfactorable is None:
                        raise ValueError("unfactorable must be provided")
                    if value in unfactorable[key]:
                        if mech_has_conductance[key]:
                            if divide_by_two[key][value]:
                                result.append(f"self.{key}.{value}(0.5 * v_prev)")
                            else:
                                result.append(f"self.{key}.{value}(v_prev)")
                        else:
                            result.append(f"self.{key}.{value}(v)")
                    else:
                        result.append(f"self.{key}.{value}(v_prev)")
    s = " + ".join(result)
    return f"{parse_current_string(current, write=write)} = {s}"


def parse_currents(
    currents,
    write=True,
    df=False,
    unfactorable=None,
    mech_has_conductance=None,
    divide_by_two=None,
) -> str:
    result = []
    for key, value in currents.items():
        result.append(
            parse_dictionary_to_sum(
                value,
                key,
                dufort=df,
                unfactorable=unfactorable,
                write=write,
                mech_has_conductance=mech_has_conductance,
                divide_by_two=divide_by_two,
            )
        )
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


def gtot(mechanisms, has_gtot):
    assignments = [f"self.{m._name}.gtot(v)" for m in mechanisms if has_gtot[m._name]]
    if len(assignments) == 0:
        return "torch.tensor(0.0)"
    return " + ".join(assignments)


def breakpoint(mechanisms):
    ret = []
    for m in mechanisms:
        ret.append(f"self.{m._name}.breakpoint(v)")
    return "\n    ".join(ret)


def tot_currents(currents, df=False):
    if not DFITOT:
        return ""
    result = []
    for key, value in currents.items():
        result.append(parse_dictionary_to_sum(value, key, df))
    s = "\n    ".join(result)
    return s


def mech_detach(mechanisms):
    result = []
    for m in mechanisms:
        result.append(f"self.{m._name}.detach()")
    return "\n    ".join(result)


def ion_detach(ions):
    result = []
    for ion in ions:
        result.append(f"self.{ion}_ion.detach()")
    return "\n    ".join(result)


def build_handler(
    mechanisms,
    names,
    currents,
    unfactorable,
    has_gtot,
    divide_by_two,
    temp,
    ions=None,
    df=False,
):
    arguments = parse_args(names)

    all_names = names

    if ions is not None:
        ion_names = list(ions.keys())
        all_names = names + [f"{i}_ion" for i in ion_names]
        arguments = arguments + ", " + parse_ions(ion_names)

    if IMEM:
        imem_assignment = "self.register_buffer('imem', torch.zeros(1))"
        if df:
            imem_write = ""
            imem_write_df = f"self.imem = {' + '.join([parse_current_string(key, total=True, write=True) for key in currents.keys()])}"
        else:
            imem_write = "self.imem = total"
            imem_write_df = ""
    else:
        imem_assignment = ""
        imem_write = ""
        imem_write_df = ""

    mech_has_conductance = {}
    for mech in mechanisms:
        mech_has_conductance[mech._name] = hasattr(mech, "conductance")

    if df:
        assign_currents_df = assign(mechanisms, ions)
        assign_currents_non_df = ""
    else:
        assign_currents_df = ""
        assign_currents_non_df = assign(mechanisms, ions)

    forward_str = template.format(
        arguments=arguments,
        assignments=parse_assignments(all_names),
        defaults=parse_defaults(ions),
        currents=parse_currents(
            currents,
            df=df,
            unfactorable=unfactorable,
            write=not df,
            mech_has_conductance=mech_has_conductance,
            divide_by_two=divide_by_two,
        ),
        total=parse_total(currents, write=not df),
        inflate=parse_inflate(names),
        mech_advance=parse_advance(names),
        init_buffers=parse_init_buffers(names),
        ion_init=parse_ion_init(ions),
        ion_advance=parse_ion_advance(ions),
        define_setattr=parse_setattr(ions),
        all_states=parse_all_states(mechanisms),
        gtot=gtot(mechanisms, has_gtot),
        breakpoint=breakpoint(mechanisms),
        set_buffers=parse_set_buffers(mechanisms, list(ions.keys())),
        currents_tot=tot_currents(currents, df),
        mech_detach=mech_detach(mechanisms),
        ion_detach=ion_detach(ions),
        imem_assignment=imem_assignment,
        imem_write=imem_write,
        imem_write_df=imem_write_df,
        assign_currents_df=assign_currents_df,
        assign_currents_non_df=assign_currents_non_df,
        assign_equilibrium=assign(mechanisms, ions, kind="e"),
        assign_post_advance=assign_post_advance(mechanisms, ions),
    )

    if DEBUG > 0:
        print(forward_str)

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
