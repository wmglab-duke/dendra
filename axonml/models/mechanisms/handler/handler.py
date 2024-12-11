import torch
import linecache


template = """
class MechanismHandler(torch.nn.Module):
  def __init__(self, temp, {arguments}):
    super().__init__()
    self.temp = temp
    {assignments}

  def initialize(self, v, v_init, area, temp) -> None:
    self.init_buffers(v_init)
    self.inflate(v)
    self.ion_init(temp)
    self.i_no_intra(v, area)

  def advance(self, v, dt) -> None:
    {mech_advance}

  def i_no_intra(self, v, area) -> torch.Tensor:
    {currents}
    {scale}
    {write_ion_currents}
    total = {total}
    return total

  def i_intra(self, v, area, intra) -> torch.Tensor:
    {currents}
    {scale}
    {write_ion_currents}
    total = {total}
    total = total - intra
    return total

  def update(self, temp) -> None:
    {ion_advance}

  def ion_init(self, temp) -> None:
    {ion_init}

  def inflate(self, v) -> None:
    {inflate}

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
    return "\n    ".join(result)


def parse_dictionary_to_sum(data: dict, current: str) -> str:
    if not data:
        return ""
    result = []
    for key, value_set in data.items():
        for value in value_set:
            result.append(f"self.{key}.{value}(v)")
    s = " + ".join(result)
    return f"{current} = {s}"


def parse_currents(currents) -> str:
    result = []
    for key, value in currents.items():
        result.append(parse_dictionary_to_sum(value, key))
    s = "\n    ".join(result)
    return s


def parse_scale(currents) -> str:
    result = []
    for key in currents.keys():
        result.append(f"{key} = {key} * area[:, None, None]")
    s = "\n    ".join(result)
    return s


def parse_total(currents) -> str:
    s = " + ".join(currents.keys())
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


def build_handler(mechanisms, names, currents, temp, ions=None, ions_write=None):
    arguments = parse_args(names)

    all_names = names

    if ions is not None:
        ion_names = list(ions.keys())
        all_names = names + [f"{i}_ion" for i in ion_names]
        arguments = arguments + ", " + parse_ions(ion_names)

    forward_str = template.format(
        arguments=arguments,
        assignments=parse_assignments(all_names),
        currents=parse_currents(currents),
        scale=parse_scale(currents),
        write_ion_currents=parse_write_ions(ions_write),
        total=parse_total(currents),
        inflate=parse_inflate(names),
        mech_advance=parse_advance(names),
        init_buffers=parse_init_buffers(names),
        ion_init=parse_ion_init(ions),
        ion_advance=parse_ion_advance(ions),
    )

    filename = "<forward_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    if ions is not None:
        inputs = mechanisms + list(ions.values())
    else:
        inputs = mechanisms

    m = torch.jit.script(locals()["MechanismHandler"](temp, *inputs))
    return m
