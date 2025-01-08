import torch


def to_param(val):
    return torch.nn.Parameter(torch.tensor(val), requires_grad=False)


class Parameterized(torch.nn.Module):
    _params = None

    def __init__(self, **kwargs):
        super(Parameterized, self).__init__()
        self.instantiate_parameters(**kwargs)

    def instantiate_parameters(self, **kwargs):
        _params = self.__class__._params
        if _params is not None:
            _params = dict((k, kwargs.get(k, v)) for k, v in _params.items())
            for name, value in _params.items():
                if isinstance(value, dict):
                    setattr(self, name, [])
                    for pname, pval in value.items():
                        setattr(self, pname, to_param(pval))
                        getattr(self, name).append(getattr(self, pname))
                else:
                    setattr(self, name, to_param(value))
