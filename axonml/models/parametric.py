import torch


class Functional:
    def fn(self, model):
        raise NotImplementedError


class Lambda(Functional):
    def __init__(self, f):
        self.f = f

    def fn(self, model):
        return self.f(model)


def to_param(val, model=None):
    if isinstance(val, Functional):
        return torch.nn.Parameter(torch.as_tensor(val.fn(model)), requires_grad=False)
    return torch.nn.Parameter(torch.as_tensor(val), requires_grad=False)


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
                    setattr(self, name, torch.nn.ParameterDict())
                    for pname, pval in value.items():
                        if isinstance(pval, Functional):
                            pval = 1.0
                        setattr(self, pname, to_param(pval, self))
                        getattr(self, name)[pname] = getattr(self, pname)
                else:
                    if isinstance(value, Functional):
                        value = 1.0
                    setattr(self, name, to_param(value, self))

    def instantiate_parameters_lambda(self, **kwargs):
        _params = self.__class__._params
        changed = False
        if _params is not None:
            _params = dict((k, kwargs.get(k, v)) for k, v in _params.items())
            for name, value in _params.items():
                if isinstance(value, dict):
                    setattr(self, name, torch.nn.ParameterDict())
                    for pname, pval in value.items():
                        if isinstance(pval, Functional):
                            changed = True
                            setattr(self, pname, to_param(pval, self))
                            getattr(self, name)[pname] = getattr(self, pname)
                else:
                    if isinstance(value, Functional):
                        changed = True
                        setattr(self, name, to_param(value, self))
        return changed

    def check_kwargs(self, kwargs):
        _params = self.__class__._params
        if _params is not None:
            for name in kwargs.keys():
                if name not in _params:
                    raise ValueError(
                        f"Unknown parameter {name}. Valid parameters are {list(_params.keys())}."
                    )
        return True
