from typing import List
import re
import string
import random

import torch
import linecache

from axonml.helpers import DEBUG, DFITOT, IMEM, NETWORK

from axonml.models.interfaces import AxonInterface
from .functions import (
    parse_args, 
    parse_ions,
    parse_assignments,
    parse_set_buffers,
    parse_advance,
    parse_breakpoint,
    parse_ion_advance,
    parse_ion_init,
    parse_init_buffers,
    parse_all_states,
    assign_post_advance,
    mech_detach,
    ion_detach
)

from .base import HandlerBuilder

def randomword(length):
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))

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
    self.i(v_init)

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

  def irev(self) -> torch.Tensor:
    return {irev}

  def i(self, v) -> torch.Tensor:
    {breakpoint}
    {local_assign}
    {currents}
    {assign_currents}
    total = {total}
    {imem_write}
    return {explicit_current}

  def update(self, temp) -> None:
    {ion_advance}
    {assign_equilibrium}
    return

  def generic(self, model: AxonInterface) -> None:
    {mech_update_fs}
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


def local_assign(currents) -> str:
    assignments = []
    for k, v in currents.items():
        for mech, currents in v.items():
            for current in currents:
                assignments.append(f"{mech}_{current} = self.{mech}.{current}(v)")
    return "\n    ".join(assignments)


def parse_dictionary_to_sum(
    data: dict,
    current: str,
    write=True,
) -> str:
    if not data:
        return ""
    result = []
    for key, value_set in data.items():
        for value in value_set:
            result.append(f"{key}_{value}")
    s = " + ".join(result)
    return f"{parse_current_string(current, write=write)} = {s}"


def explicit_current(currents, unfactorable) -> str:
    """
    Parse the currents dictionary to create a total current string.
    """
    if not currents:
        return "0.0"
    result = []
    for k, v in currents.items():
        for mech, currents in v.items():
            for current in currents:
                if current in unfactorable[mech]:
                    result.append(f"{mech}_{current}")
    if not result:
        return 'torch.tensor(0.0)'
    s = " + ".join(result)
    return s


def gtot(mechanisms, has_gtot):
    assignments = [f"self.{m._name}.gtot(v)" for m in mechanisms if has_gtot[m._name]]
    if len(assignments) == 0:
        return "torch.tensor(0.0)"
    return " + ".join(assignments)


def irev(mechanisms, has_gtot):
    result = []
    for m in mechanisms:
        if has_gtot[m._name]:
            result.append(f"self.{m._name}.irev()")
    if len(result) == 0:
        return "torch.tensor(0.0)"
    return " + ".join(result)


def parse_total(currents) -> str:
    s = " + ".join([parse_current_string(key, True, True) for key in currents.keys()])
    return s


def parse_currents(
    currents,
    write=True,
) -> str:
    result = []
    for key, value in currents.items():
        result.append(
            parse_dictionary_to_sum(
                value,
                key,
                write=write,
            )
        )
    s = "\n    ".join(result)
    return s


class ImplicitHandlerBuilder(HandlerBuilder):
    """
    ImplicitHandlerBuilder is a class that builds an implicit handler for a given mechanism.
    """

    def __init__(self, DEBUG=0, IMEM=0):
        super().__init__(DEBUG, IMEM)

    def build(
            self,
            mechanisms,
            names,
            currents,
            unfactorable,
            has_gtot,
            divide_by_two,
            temp,
            ions=None):
        """
        Build the implicit handler for the mechanism.
        """
        # Implement the logic to build the implicit handler
        arguments = parse_args(names)

        all_names = names

        if ions is not None:
            ion_names = list(ions.keys())
            all_names = names + [f"{i}_ion" for i in ion_names]
            arguments = arguments + ", " + parse_ions(ion_names)

        if self.IMEM:
            imem_assignment = "self.register_buffer('imem', torch.zeros(1))"
            imem_write = "self.imem = total"
        else:
            imem_assignment = ""
            imem_write = ""

        assign_currents = assign(mechanisms, ions)

        forward_str = template.format(
            arguments=arguments,
            assignments = parse_assignments(all_names),
            imem_assignment=imem_assignment,
            set_buffers=parse_set_buffers(mechanisms, list(ions.keys())),
            mech_advance=parse_advance(names),
            assign_post_advance=assign_post_advance(mechanisms, ions),
            mech_detach=mech_detach(mechanisms),
            ion_detach=ion_detach(ions),
            gtot=gtot(mechanisms, has_gtot),
            irev=irev(mechanisms, has_gtot),
            breakpoint=parse_breakpoint(mechanisms),
            local_assign=local_assign(currents),
            currents=parse_currents(currents, write=True),
            assign_currents=assign_currents,
            total=parse_total(currents),
            imem_write=imem_write,
            explicit_current=explicit_current(currents, unfactorable),
            ion_advance=parse_ion_advance(ions),
            assign_equilibrium=assign(mechanisms, ions, kind='e'),
            mech_update_fs="",
            ion_init=parse_ion_init(ions),
            init_buffers=parse_init_buffers(names),
            all_states=parse_all_states(mechanisms)
        )

        if self.DEBUG > 0:
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




        
