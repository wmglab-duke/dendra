import inspect

import torch
from torch.nn import functional as F
from torch.nn.utils import parametrize


class Functional:
    def fn(self, model):
        raise NotImplementedError


class Lambda(Functional):
    def __init__(self, f):
        self.f = f

    def fn(self, model):
        return self.f(model)
    

class positive:
    def __init__(self, val):
        self.val = val


class PositiveSoftplus(torch.nn.Module):
    def forward(self, x):
        return F.softplus(x)                # f(x)

    def right_inverse(self, y):
        return softplus_inv(y)              # f⁻¹(y)  ← same helper as §2


def softplus_inv(y, beta=1., eps=1e-6):
    # y must be >0; eps keeps the log well-behaved numerically
    return (torch.log(torch.exp(beta*(y-eps)) - 1.0) / beta)


def to_param(val, model=None):
    if isinstance(val, torch.nn.Parameter):
        return val
    if isinstance(val, Functional):
        return torch.nn.Parameter(torch.as_tensor(val.fn(model)), requires_grad=False)
    return torch.nn.Parameter(torch.as_tensor(val), requires_grad=False)


def distribute_over(val, over='a'):
    valid = {'p', 'c', 'pc'}
    if over not in valid:
        raise ValueError('over must be one of {}'.format(valid[kind]))
    val = torch.as_tensor(val)
    if over == 'p':
        return val[:, None]
    elif over == 'c':
        return val[None, :]
    else:
        return val
    

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


def made_of_slices(key):
    """
    Check if the key is a slice or a list of slices.
    """
    if isinstance(key, slice):
        return True
    if isinstance(key, list) or isinstance(key, tuple):
        return all(isinstance(k, slice) for k in key)
    return False


class _Parameterized(torch.nn.Module):
    """
    A base class that allows subclasses to declare parameters which are
    automatically inherited and aggregated.
    """
    _params = {}
    _param_declarations = []

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called automatically whenever a class
        inherits from Parameterized.
        """
        # Call the parent's __init_subclass__ WITHOUT our custom kwargs,
        # as the base 'object' class does not accept them.
        super().__init_subclass__()
        
        # Start with a fresh dictionary for the new class's parameters.
        new_params = {}
        
        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for a _params attribute defined directly on the base
            if '_params' in base.__dict__:
                new_params.update(base._params)
        
        # Add parameters declared via the PARAMETER() method
        if _Parameterized._param_declarations:
            for p_dict in _Parameterized._param_declarations:
                new_params.update(p_dict)
            _Parameterized._param_declarations = [] # Clear for next class
        
        # Add parameters from class definition keywords (e.g., a=10)
        # These will override anything set by parents.
        new_params.update(kwargs)
        
        cls._params = new_params

    @staticmethod
    def PARAMETER(**kwargs):
        """
        A static method to declare parameters. This has the side effect of
        appending the parameters to a temporary class-level list.
        """
        _Parameterized._param_declarations.append(kwargs)

    def __init__(self, shape, additional_parameters=None, **kwargs):
        """
        Instance constructor. Initializes the instance with class-level
        parameters, allowing for instance-specific overrides.
        """
        super().__init__()
        self.shape  = shape
        self.params = self.__class__._params.copy()

        if kwargs:
            self.params.update(kwargs)

        self.keys = {}
        self.additional_parameters = {}
        self.instantiate_parameters(**self.params)
        self.instantiate_additional_parameters(additional_parameters)
            
    def instantiate_parameters(self, **kwargs):
        # this is only called once, on __init__
        if kwargs is not None:
            for name, value in kwargs.items():
                if isinstance(value, dict):
                    setattr(self, name, torch.nn.ParameterDict())
                    for pname, pval in value.items():
                        setattr(self, pname, to_param(pval, self))
                        getattr(self, name)[pname] = getattr(self, pname)
                else:
                    p_name = f"{name}_"
                    setattr(self, p_name, to_param(value, self))
                    self.register_buffer(name, torch.empty(self.shape))

    def instantiate_additional_parameters(self, additional_parameters=None):
        if additional_parameters is not None:
            for name, list_of_aliases_values_and_keys in additional_parameters.items():
                if name in self.params:
                    count = 0
                    keys = []
                    for (alias, value, key) in list_of_aliases_values_and_keys:
                        if alias is not None:
                            p_name = f"{name}_{alias}"
                        else:
                            p_name = f"{name}_{count}"
                            count += 1
                        key = torch.as_tensor(key, dtype=torch.long)
                        parameter = to_param(value, self)
                        setattr(self, p_name, to_param(value, self).expand_as(key))
                        self.additional_parameters.setdefault(name, []).append(getattr(self, p_name))
                        keys.append(key)
                    self.keys[name] = torch.cat(keys).to(torch.long)

    def load_additional_parameters(self):
        for name, list_of_parameters in self.additional_parameters.items():
            buffer = getattr(self, name)
            additional_params = torch.cat(list_of_parameters)
            key = self.keys[name].to(buffer.device)
            buffer.view(-1).index_copy_(0, key, additional_params)

    @staticmethod
    def set_slice(buffer, key, parameter):
        buffer[key].copy_(parameter)

    @staticmethod
    def set_fancy(buffer, key, parameter):
        buffer.view(-1).index_put_((key.to(buffer.device),), parameter)

    def populate_parameter_buffers(self):
        for name in self.__class__._params:
            p_name = f"{name}_"
            getattr(self, name).detach_()
            getattr(self, name).copy_(getattr(self, p_name))
        self.load_additional_parameters()

    def detach_(self):
        for n, b in self.named_buffers():
            b.detach_()

    def check_kwargs(self, kwargs):
        _params = self.__class__._params
        if _params is not None:
            for name in kwargs.keys():
                if name not in _params:
                    raise ValueError(
                        f"Unknown parameter {name}. Valid parameters are {list(_params.keys())}."
                    )
        return True