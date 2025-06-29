import inspect
from typing import Callable

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


def create_param_expander(
    param: torch.Tensor,
    key: torch.LongTensor,
    main_shape: tuple[int, int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Creates a specialized, efficient function to expand a parameter for indexed assignment.

    This function analyzes the relationship between a parameter's shape, a flat
    index key, and a target 2D shape. It returns a new function, `expander(p)`,
    optimized for the specific parameterization scheme.

    The intended use is:
    `flat_target[key].copy_(expander(new_param_value))`

    Supported Parameterization Schemes:
    1.  **Scalar (0-dim):** The parameter is a single value applied to all key locations.
    2.  **Pre-Sized (1D):** The parameter is a 1D tensor with the same number of elements
        as `key`, providing a one-to-one mapping. `param.numel() == key.numel()`.
    3.  **Row-Broadcast (shape `(N, 1)`):** The `key` indexes into `N` unique rows. The
        parameter provides a single value for each unique row, which is broadcast
        across all columns for that row. `param.shape == (N, 1)`.
    4.  **Column-Broadcast (shape `(1, M)`):** The `key` indexes into `M` unique columns.
        The parameter provides a single value for each unique column, which is
        broadcast down all rows for that column. `param.shape == (1, M)`.

    Args:
        param (torch.Tensor): The parameter tensor whose shape defines the expansion logic.
        key (torch.LongTensor): A 1D tensor of flat indices.
        main_shape (tuple[int, int]): The HxW shape of the conceptual 2D tensor.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]:
            A new function that takes a tensor `p` and returns the expanded 1D tensor.
    """
    num_keys = key.numel()

    # --- Condition 1: Pre-Sized Parameter ---
    # The parameter is already the correct size, one value per key.
    if param.numel() == num_keys:
        # This is the simplest case. The expander is an identity function (with a reshape for safety).
        def expander(p: torch.Tensor) -> torch.Tensor:
            return p.reshape(num_keys)
        return expander

    # --- Condition 2: Scalar Parameter ---
    if param.dim() == 0:
        # Expand the scalar to all key locations.
        def expander(p: torch.Tensor) -> torch.Tensor:
            return p.expand(num_keys)
        return expander

    # --- For broadcast cases, we need to know the unique rows/cols in the key ---
    # This setup is done only once, making the returned expander fast.
    rows = torch.div(key, main_shape[1], rounding_mode='floor')
    cols = key % main_shape[1]

    unique_rows, row_inverse = torch.unique(rows, return_inverse=True)
    unique_cols, col_inverse = torch.unique(cols, return_inverse=True)

    # --- Condition 3: Row-Broadcast ---
    # The param has shape (num_unique_rows, 1).
    if param.shape == (len(unique_rows), 1):
        # We use `row_inverse` to map from the dense param vector back to the sparse keys.
        def expander(p: torch.Tensor) -> torch.Tensor:
            # p[row_inverse] selects the correct row value for each key
            return p[row_inverse.to(p.device)].squeeze(-1)
        return expander

    # --- Condition 4: Column-Broadcast ---
    # The param has shape (1, num_unique_cols).
    if param.shape == (1, len(unique_cols)):
        # We use `col_inverse` to map from the dense param vector back to the sparse keys.
        def expander(p: torch.Tensor) -> torch.Tensor:
            # p[0, col_inverse] selects the correct column value for each key
            return p[0, col_inverse.to(p.device)]
        return expander

    # If none of the conditions were met, raise a helpful error.
    raise ValueError(
        f"Parameter shape {param.shape} does not match any supported condition.\n"
        f"  - For pre-sized, expected numel: {num_keys}\n"
        f"  - For row-broadcast, expected shape: ({len(unique_rows)}, 1)\n"
        f"  - For column-broadcast, expected shape: (1, {len(unique_cols)})"
    )
    

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
                    p_name = f"{name}_default"
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
                        setattr(self, p_name, parameter)
                        fill = create_param_expander(parameter, key, self.shape)
                        self.additional_parameters.setdefault(name, []).append((fill, getattr(self, p_name)))
                        keys.append(key)
                    self.keys[name] = torch.cat(keys).to(torch.long)

    def load_additional_parameters(self):
        for name, list_of_parameters in self.additional_parameters.items():
            buffer = getattr(self, name)
            additional_params = torch.cat([fill(p) for fill, p in list_of_parameters])
            key = self.keys[name].to(buffer.device)
            buffer.view(-1).index_copy_(0, key, additional_params)

    def populate_parameter_buffers(self):
        for name in self.__class__._params:
            p_name = f"{name}_default"
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