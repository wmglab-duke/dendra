import torch
from typing import Optional, Dict
import linecache


forward_template = """
def cal_i(self, v, area, i: int, intra: Optional[torch.Tensor] = None) -> torch.Tensor:
  {nonspecific}
  {nonspecific_scale}
  total = {total}
  print('compiled')
  if intra is not None:
    total = total - intra[i]
  return total
"""


def parse_dictionary_to_sum(data: dict, current: str) -> str:
    if not data:
        return ""
    result = []
    for key, value_set in data.items():
        for value in value_set:
            result.append(f"self.{key}.{value}(v)")
    s = " + ".join(result)
    return f"{current} = {s}"


def parse_dictionary_to_scale(data: dict, current: str) -> str:
    if not data:
        return ""
    return f"{current} = {current} * area[:, None, None]"


def parse_dictionaries_to_total(
    ns_dct=None, k_dct=None, na_dct=None, ca_dct=None
) -> str:
    result = []
    if ns_dct:
        result.append("nonspecfic")
    s = " + ".join(result)
    return s


def make_calc_i(ns_dct):
    print(ns_dct)

    forward_str = forward_template.format(
        nonspecific=parse_dictionary_to_sum(ns_dct, "nonspecific"),
        nonspecific_scale=parse_dictionary_to_scale(ns_dct, "nonspecific"),
        total=parse_dictionaries_to_total(ns_dct),
    )

    print(forward_str)

    filename = "<forward_template>"
    code = compile(forward_str, filename, "exec")
    exec(code)

    lines = [line + "\n" for line in forward_str.splitlines()]
    linecache.cache[filename] = (len(forward_str), None, lines, filename)

    m = torch.jit.script_method(locals()["calc_i"])
    return m
