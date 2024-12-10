import torch
from typing import Optional, Dict, List
import linecache


template = """
class MechanismHandler(torch.nn.Module):
  def __init__(self, {arguments}):
    super().__init__()
    {assignments}

  def i_no_intra(self, v, area) -> torch.Tensor:
    {currents}
    {scale}
    total = {total}
    return total

  def i_intra(self, v, area, intra) -> torch.Tensor:
    {currents}
    {scale}
    total = {total}
    total = total - intra
    return total

  def inflate(self, v) -> None:
    {inflate}
  
  def advance(self, v, dt) -> None:
    {advance}

  def init_buffers(self, v_init) -> None:
    {init_buffers}

  @torch.jit.ignore
  def get(self, mech: str, state: str) -> torch.Tensor:
    return getattr(self, mech).get(state)
"""

def parse_assignments(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key} = {key}")
    return "\n".join(result)


def parse_args(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"{key}: torch.nn.Module")
    return ", ".join(result)


def parse_dictionary_to_sum(data: dict, current: str) -> str:
    if not data:
        return ""
    result = []
    for key, value_set in data.items():
        for value in value_set:
            result.append(f'self.{key}.{value}(v)')
    s = " + ".join(result)
    return f"{current} = {s}"


def parse_currents(currents) -> str:
    result = []
    for key, value in currents.items():
        result.append(parse_dictionary_to_sum(value, key))
    s = "\n".join(result)
    return s


def parse_scale(currents) -> str:
    result = []
    for key in currents.keys():
        result.append(f"{key} = {key} * area[:, None, None]")
    s = "\n".join(result)
    return s


def parse_total(currents) -> str:
    s = " + ".join(currents.keys())
    return s


def parse_advance(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key}._advance(v, dt)")
    s = "\n".join(result)
    return s


def parse_inflate(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key}.inflate(v)")
    s = "\n".join(result)
    return s


def parse_init_buffers(mechanism_names) -> str:
    result = []
    for key in mechanism_names:
        result.append(f"self.{key}.init(v_init)")
    s = "\n".join(result)
    return s


def build_handler(mechanisms, names, currents):

    forward_str = template.format(
        arguments=parse_args(names),
        assignments=parse_assignments(names),
        currents=parse_currents(currents),
        scale=parse_scale(currents),
        total=parse_total(currents),
        inflate=parse_inflate(names),
        advance=parse_advance(names),
        init_buffers=parse_init_buffers(names),
    )

    filename = "<forward_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = torch.jit.script(locals()["MechanismHandler"](*mechanisms))
    return m