import torch


def to_param(val):
    return torch.nn.Parameter(torch.tensor(val), requires_grad=False)


class Parameterized(torch.nn.Module):
    _params = None

    def __init__(self):
        super(Parameterized, self).__init__()
        self.instantiate_parameters()

    def instantiate_parameters(self):
        if self.__class__._params is not None:
            for name, value in self.__class__._params.items():
                if isinstance(value, dict):
                    setattr(self, name, [])
                    for pname, pval in value.items():
                        setattr(self, pname, to_param(pval))
                        getattr(self, name).append(getattr(self, pname))
                else:
                    setattr(self, name, to_param(value))
