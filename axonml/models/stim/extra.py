import torch


def ve_from_s_t(space, time, n, device, multicontact=False):
    ve_s = torch.as_tensor(space, device=device)
    ve_t = torch.as_tensor(time, device=device)

    if multicontact:
        ve_s = ve_s.expand(-1, n, -1)
        ve_t = ve_t.expand(-1, n, -1)
        einsum = op_mc
    else:
        ve_s = ve_s.expand(n, -1)
        ve_t = ve_t.expand(n, -1)
        einsum = op_sc

    return einsum(ve_s, ve_t)


def op_mc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("can,cat->tan", s, t).contiguous()


def op_sc(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.einsum("an,at->tan", s, t).contiguous()


class Extra(torch.nn.Module):
    pass
