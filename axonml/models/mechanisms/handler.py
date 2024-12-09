import torch
from typing import Optional, Dict, List
import linecache


forward_str = """
class MechanismHandler(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, v, area, i: int, intra: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.tensor(5.0)
"""

filename = "<forward_template>"
code = compile(forward_str, filename, "exec")
exec(code)

lines = [line + "\n" for line in forward_str.splitlines()]
linecache.cache[filename] = (len(forward_str), None, lines, filename)

m = torch.jit.script(globals()["MechanismHandler"]())