import inspect
from typing import Callable

import torch
from torch.nn import functional as F
from torch.nn.utils import parametrize


def to_param(val, model=None):
    if isinstance(val, torch.nn.Parameter):
        return val
    if isinstance(val, torch.nn.Module):
        return val
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


class Functional(torch.nn.Module):
    """
    A functional that can be used as a parameter in a model.
    This is useful for cases where you want to use a function as a parameter,
    such as in a neural network layer.
    """
    
    def __init__(self, func: torch.nn.Module, fill=None, key=None):
        super(Functional, self).__init__()
        self.func = func
        self.fill = fill
        if key is not None:
            self.register_buffer('key', torch.as_tensor(key, dtype=torch.long))
        else:
            self.key = None

    def forward(self, buffer):
        p = self.func(buffer)
        if self.key is None:
            return p
        b = buffer.clone()
        b.view(-1).index_copy_(0, self.key, self.fill(p))
        return b


def build_parametrization(
    module,
    output,
    key: torch.LongTensor,
    main_shape: tuple[int, int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    if key is None:
        return Functional(module)
    fill = create_param_expander(output, key, main_shape)
    return Functional(module, fill=fill, key=key)


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

    _range = {}
    _range_declarations = []

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
        new_range = {}
        
        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for a _params attribute defined directly on the base
            if '_params' in base.__dict__:
                new_params.update(base._params)
            if '_range' in base.__dict__:
                new_range.update(base._range)
        
        # Add parameters declared via the PARAMETER() method
        if _Parameterized._param_declarations:
            for p_dict in _Parameterized._param_declarations:
                new_params.update(p_dict)
            _Parameterized._param_declarations = [] # Clear for next class
        # Add range declarations
        if _Parameterized._range_declarations:
            for r_dict in _Parameterized._range_declarations:
                new_range.update(r_dict)
            _Parameterized._range_declarations = []
        
        # Add parameters from class definition keywords (e.g., a=10)
        # These will override anything set by parents.
        new_params.update(kwargs)
        new_range.update(kwargs)
        
        cls._params = new_params
        cls._range = new_range

    @staticmethod
    def PARAMETER(**kwargs):
        """
        A static method to declare parameters. This has the side effect of
        appending the parameters to a temporary class-level list.
        """
        _Parameterized._param_declarations.append(kwargs)

    @staticmethod
    def RANGE(**kwargs):
        """
        A static method to declare ranges. This has the side effect of
        appending the ranges to a temporary class-level list.
        """
        _Parameterized._range_declarations.append(kwargs)

    def __init__(self, shape, additional_parameters=None, **kwargs):
        super().__init__()
        self.shape  = shape
        self.params = self.__class__._params.copy()
        self.range  = self.__class__._range.copy()

        self.parametrizations = torch.nn.ModuleDict()

        if kwargs:
            self.params = {key: kwargs.get(key, value) for key, value in self.params.items()}
            self.range  = {key: kwargs.get(key, value) for key, value in self.range.items()}

        self.keys = {}
        self.additional_parameters = {}
        self.instantiate_parameters(**self.params)
        self.instantiate_range(**self.range)
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
                    self.register_buffer(name, torch.empty(()))
                    getattr(self, name).copy_(getattr(self, p_name))

    def instantiate_range(self, **kwargs):
        # this is only called once, on __init__
        if kwargs is not None:
            for name, value in kwargs.items():
                p_name = f"{name}_default"
                setattr(self, p_name, to_param(value, self))
                self.register_buffer(name, torch.empty(self.shape))
                getattr(self, name).copy_(getattr(self, p_name))

    def instantiate_additional_parameters(self, additional_parameters=None):
        if additional_parameters is not None:
            for name, list_of_aliases_values_and_keys in additional_parameters.items():
                if name in self.range:
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
                        if isinstance(parameter, torch.nn.Module):
                            p = parameter(torch.empty(self.shape))
                            parametrization = build_parametrization(p, parameter, key, self.shape)
                            self.parametrizations.setdefault(name, []).append(parametrization)
                            setattr(self, p_name, parameter)
                        else:
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
        self.apply_parametrizations()

    def apply_parametrizations(self):
        """
        Apply all parametrizations to the parameters of this model.
        """
        for name, param_list in self.parametrizations.items():
            b = getattr(self, name)
            for param in param_list:
                b = param(b)
            setattr(self, name, b)

    def detach(self):
        for n, b in self.named_buffers():
            try:
                b.detach_()
            except Exception as e:
                setattr(self, n, b.detach())

    def check_kwargs(self, kwargs):
        _params = self.__class__._params
        if _params is not None:
            for name in kwargs.keys():
                if name not in _params:
                    raise ValueError(
                        f"Unknown parameter {name}. Valid parameters are {list(_params.keys())}."
                    )
        return True

    def parameters_dict(self):
        """
        Returns a dictionary of all parameters in the model.
        """
        return {name: param for name, param in self.named_parameters()}

    def batch(self, batch_size: int):
        """
        Returns a new instance of the model with the parameters
        distributed over the specified batch size.
        """
        for name in self.__class__._params:
            p = getattr(self, name)
            p = p.unsqueeze(0)
            setattr(self, name, p)