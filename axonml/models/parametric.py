"""Parameter handling utilities and mixins for AxonML models."""

import itertools
import math
from types import MethodType
from typing import Callable, Union

import torch
import torch.nn.functional as F

from axonml.helpers import DEBUG, REQUIRE_GRAD, logger
from axonml.utils import PreparedInterp1d

from .modular import AxModule
from .rng import RNGModule

_valid_param_type = Union[float, torch.Tensor, torch.nn.Parameter, torch.nn.Module]


def to_param(val, positive=False, requires_grad=None):
    """
    Convert a value into a parameter-like object.

    Parameters
    ----------
    val : Any
        Input value to coerce into a tensor-backed parameter.
    positive : bool, optional
        If True, clamp the value to non-negative range and wrap it in
        :class:`PositiveParam`.
    requires_grad : bool, optional
        If True, the created parameter will require gradients.

    Returns
    -------
    torch.nn.Parameter or PositiveParam or torch.nn.Module
        Parameterized representation of ``val`` suitable for registration.
    """
    if requires_grad is None:
        requires_grad = bool(REQUIRE_GRAD)
    if isinstance(val, torch.nn.Parameter):
        return val
    if isinstance(val, torch.nn.Module):
        return val
    val = torch.as_tensor(val, dtype=torch.float32)
    if positive:
        val = torch.clamp(val, min=0.0)
        return PositiveParam(val)
    param = torch.nn.Parameter(val, requires_grad=requires_grad)
    return param


def is_parametric(val):
    """
    Check whether a value is treated as a parametric object.

    Parameters
    ----------
    val : Any
        Value to inspect.

    Returns
    -------
    bool
        True if ``val`` is a parameter or :class:`Bounded`.
    """
    _parametric_types = (torch.nn.Parameter, Parametric)
    return isinstance(val, _parametric_types)


def distribute_over(val, over="a"):
    """
    Broadcast values across population or compartment dimensions.

    Parameters
    ----------
    val : array_like
        Values to broadcast.
    over : {'p', 'c', 'pc'}, optional
        Axis selection: ``'p'`` expands over populations, ``'c'`` over
        compartments, ``'pc'`` leaves shape unchanged.

    Returns
    -------
    torch.Tensor
        Broadcast tensor with the selected layout.

    Raises
    ------
    ValueError
        If ``over`` is not one of the supported selectors.
    """
    valid = {"p", "c", "pc"}
    if over not in valid:
        raise ValueError("over must be one of {}".format(valid))
    val = torch.as_tensor(val)
    if over == "p":
        return val[:, None]
    elif over == "c":
        return val[None, :]
    else:
        return val


def softplus_inv(y, beta: float = 1.0, threshold: float = 20.0, eps: float = 1e-12):
    """
    Numerically stable inverse of softplus.

    Parameters
    ----------
    y : Tensor or array_like
        Softplus outputs to invert.
    beta : float, optional
        Softplus sharpness parameter.
    threshold : float, optional
        Transition threshold between small and large branches.
    eps : float, optional
        Minimum clamp to avoid ``-inf`` when ``y`` equals zero.

    Returns
    -------
    torch.Tensor
        Values ``x`` such that ``softplus(x, beta) = y``.
    """
    y = torch.as_tensor(y)
    y = torch.clamp(y, min=eps)  # safer for init; prevents -inf params
    by = beta * y

    small = by <= threshold
    x_small = torch.log(torch.expm1(by)) / beta
    x_large = (by + torch.log1p(-torch.exp(-by))) / beta  # stable for large by

    return torch.where(small, x_small, x_large)


class Parametric(torch.nn.Module):
    """
    Base class for modules that implement parameterized behavior.
    """

    def __init__(self):
        super().__init__()
        assert hasattr(self, "__len__"), "Parametric subclasses must implement __len__."
        assert hasattr(self, "repeat"), "Parametric subclasses must implement repeat()."


class cacheable(Parametric):
    """
    Module mixin that caches the most recent forward computation.

    The cache is cleared automatically when switching between train/eval modes.
    """

    def __init__(self):
        super().__init__()
        self._cache = None

    def train(self, mode: bool = True):
        """
        Toggle training mode and clear any cached outputs.

        Parameters
        ----------
        mode : bool, optional
            If True, set the module to training mode; otherwise evaluation.

        Returns
        -------
        cacheable
            Self for chaining.
        """
        self.clear_cache()
        return super().train(mode)

    def eval(self):
        """
        Switch to evaluation mode and clear cached outputs.

        Returns
        -------
        cacheable
            Self for chaining.
        """
        self.clear_cache()
        return super().eval()

    def clear_cache(self):
        """
        Invalidate the stored forward result.
        """
        self._cache = None

    def forward(self, cache=True, *args, **kwargs):
        """
        Compute the module output, optionally reusing cached results.

        Parameters
        ----------
        cache : bool, optional
            If True, reuse the previous result when inputs are unchanged.
        *args, **kwargs
            Positional and keyword arguments forwarded to ``_compute``.

        Returns
        -------
        Any
            Cached or freshly computed output.
        """
        if not cache:
            return self._compute(*args, **kwargs)
        if self._cache is None:
            self._cache = self._compute(*args, **kwargs)
        return self._cache

    def repeat(self, n: int):
        """
        Repeat the cached value along a new leading dimension.

        Parameters
        ----------
        n : int
            Number of repeats.

        Returns
        -------
        torch.Tensor
            Repeated cached output.
        """
        p = self(cache=(not self.training))
        return p.repeat(n)

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__().")


def ste_clamp(y, *, lo=None, hi=None, alpha_lo: float = 1.0, alpha_hi: float = 1.0):
    """
    Straight-through estimator clamp with configurable slopes.

    Parameters
    ----------
    y : torch.Tensor
        Input tensor to clamp.
    lo : float or Tensor, optional
        Lower bound. When omitted, no lower clamp is applied.
    hi : float or Tensor, optional
        Upper bound. When omitted, no upper clamp is applied.
    alpha_lo : float, optional
        Backward slope used below ``lo``.
    alpha_hi : float, optional
        Backward slope used above ``hi``.

    Returns
    -------
    torch.Tensor
        Tensor that is clamped in the forward pass but keeps surrogate gradients.
    """
    y_sur = y
    if lo is not None:
        y_sur = torch.where(y < lo, lo + alpha_lo * (y - lo), y_sur)
    if hi is not None:
        y_sur = torch.where(y > hi, hi + alpha_hi * (y - hi), y_sur)

    y_fwd = y
    if lo is not None:
        lo_t = torch.as_tensor(lo, device=y.device, dtype=y.dtype)
        y_fwd = torch.maximum(y_fwd, lo_t)
    if hi is not None:
        hi_t = torch.as_tensor(hi, device=y.device, dtype=y.dtype)
        y_fwd = torch.minimum(y_fwd, hi_t)

    return y_sur + (y_fwd - y_sur).detach()


# --- modules ---
class Bounded(cacheable):
    """
    Trainable tensor constrained by optional lower and upper bounds.

    Parameters
    ----------
    init : array_like
        Initial value for the unconstrained parameter ``rho``.
    min_val : float, optional
        Lower bound. When ``None``, no lower constraint is enforced.
    max_val : float, optional
        Upper bound. When ``None``, no upper constraint is enforced.
    beta : float, optional
        Sharpness parameter used by softplus or sigmoid transforms.
    threshold : float, optional
        Softplus threshold used for numerical stability.
    lower_mode : {'softplus', 'hard-ste', 'leaky-ste'}, optional
        Strategy for enforcing the lower bound.
    lower_alpha : float, optional
        Surrogate slope used in ``'leaky-ste'`` mode below the bound.
    cap_mode : {'auto', 'softcap', 'sigmoid', 'hard-ste'}, optional
        Strategy for enforcing the upper bound.
    cap_beta : float, optional
        Softcap temperature. Defaults to ``beta`` when ``None``.

    Notes
    -----
    ``cap_mode='auto'`` selects ``'softcap'`` when only an upper bound exists and
    ``'sigmoid'`` when both bounds are present.
    """

    def __init__(
        self,
        init,
        *,
        min_val: float | None = None,
        max_val: float | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
        lower_mode: str = "softplus",
        lower_alpha: float = 0.1,
        cap_mode: str = "auto",
        cap_beta: float | None = None,
    ):
        super().__init__()
        self.min_val = None if min_val is None else float(min_val)
        self.max_val = None if max_val is None else float(max_val)
        if (
            self.min_val is not None
            and self.max_val is not None
            and not (self.min_val < self.max_val)
        ):
            raise ValueError("Require min_val < max_val when both bounds are set.")

        self.beta = float(beta)
        self.threshold = float(threshold)
        self.lower_mode = lower_mode
        self.lower_alpha = float(lower_alpha)
        self.cap_mode = cap_mode
        self.cap_beta = float(cap_beta) if cap_beta is not None else float(beta)

        init = torch.as_tensor(init, dtype=torch.float32)

        # ---- init rho consistent with forward mapping ----
        if self.min_val is None and self.max_val is None:
            rho0 = init

        elif self.min_val is not None and self.max_val is None:
            if self.lower_mode == "softplus":
                y = torch.clamp(init - self.min_val, min=1e-12)
                rho0 = softplus_inv(y, beta=self.beta, threshold=self.threshold)
            else:
                # STE lower modes use identity param
                rho0 = init

        elif self.min_val is None and self.max_val is not None:
            if self._upper_mode(upper_only=True) == "hard-ste":
                rho0 = init
            else:
                y = torch.clamp(self.max_val - init, min=1e-12)
                rho0 = softplus_inv(y, beta=self.cap_beta, threshold=self.threshold)

        else:
            if self._upper_mode(upper_only=False) == "hard-ste":
                rho0 = init
            else:
                rng = max(self.max_val - self.min_val, 1e-12)
                t = torch.clamp((init - self.min_val) / rng, 1e-6, 1 - 1e-6)
                rho0 = torch.special.logit(t) / self.beta

        self.rho = torch.nn.Parameter(rho0)

    def _upper_mode(self, *, upper_only: bool) -> str:
        if self.max_val is None:
            return "none"
        if self.cap_mode == "auto":
            return "softcap" if upper_only else "sigmoid"
        if self.cap_mode not in {"softcap", "sigmoid", "hard-ste"}:
            raise ValueError(f"Unknown cap_mode: {self.cap_mode}")
        return self.cap_mode

    def _apply_upper(self, y: torch.Tensor) -> torch.Tensor:
        if self.max_val is None:
            return y
        mode = self._upper_mode(upper_only=(self.min_val is None))
        if mode == "hard-ste":
            return ste_clamp(y, hi=self.max_val)
        if mode == "softcap":
            return self.max_val - F.softplus(
                self.max_val - y, beta=self.cap_beta, threshold=self.threshold
            )
        # sigmoid mode handled in both-bounds path
        return y

    def _compute(self):
        # No bounds
        if self.min_val is None and self.max_val is None:
            return self.rho

        # Lower-only
        if self.min_val is not None and self.max_val is None:
            if self.lower_mode == "softplus":
                return self.min_val + F.softplus(
                    self.rho, beta=self.beta, threshold=self.threshold
                )
            elif self.lower_mode in {"hard-ste", "leaky-ste"}:
                alpha = 1.0 if self.lower_mode == "hard-ste" else self.lower_alpha
                return ste_clamp(self.rho, lo=self.min_val, alpha_lo=alpha)
            else:
                raise ValueError(f"Unknown lower_mode: {self.lower_mode}")

        # Upper-only
        if self.min_val is None and self.max_val is not None:
            return self._apply_upper(self.rho)

        # Both bounds
        if self._upper_mode(upper_only=False) == "hard-ste":
            # inclusive [min,max] with STE
            return ste_clamp(self.rho, lo=self.min_val, hi=self.max_val)
        else:
            # default: sigmoid to (min,max)
            s = torch.sigmoid(self.beta * self.rho)
            return self.min_val + (self.max_val - self.min_val) * s

    def __len__(self):
        return self.rho.numel()


class PositiveParam(Bounded):
    """
    Bounded parameter constrained to non-negative values.

    Parameters
    ----------
    init : array_like
        Initial value for the parameter.
    include_zero : bool, optional
        If True, make the zero bound inclusive using a leaky STE transform.
    max_val : float, optional
        Optional upper bound.
    beta : float, optional
        Softplus/sigmoid sharpness parameter.
    threshold : float, optional
        Softplus threshold for numerical stability.
    lower_alpha : float, optional
        Surrogate slope below zero when ``include_zero`` is True.
    cap_mode : {'auto', 'softcap', 'sigmoid', 'hard-ste'}, optional
        Strategy for the optional upper bound.
    cap_beta : float, optional
        Softcap temperature. Defaults to ``beta`` when ``None``.
    """

    def __init__(
        self,
        init,
        *,
        include_zero: bool = False,
        max_val: float | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
        lower_alpha: float = 0.1,  # used only if include_zero=True (leaky-ste)
        cap_mode: str = "auto",
        cap_beta: float | None = None,
    ):
        super().__init__(
            init,
            min_val=0.0,
            max_val=max_val,
            beta=beta,
            threshold=threshold,
            lower_mode=("leaky-ste" if include_zero else "softplus"),
            lower_alpha=lower_alpha,
            cap_mode=cap_mode,
            cap_beta=cap_beta,
        )


class Functional(torch.nn.Module):
    """
    Wrapper turning a module into a parameter-update callable.

    Parameters
    ----------
    func : torch.nn.Module
        Module applied to incoming buffers.
    fill : Callable, optional
        Function used to expand results into the flattened parameter space.
    key : array_like, optional
        Flat indices targeted by the fill function.
    """

    def __init__(self, func: torch.nn.Module, fill=None, key=None):
        super(Functional, self).__init__()
        self.func = func
        self.fill = fill
        if key is not None:
            self.register_buffer("key", torch.as_tensor(key, dtype=torch.long))
        else:
            self.key = None

    def forward(self, buffer):
        """
        Apply the wrapped module and optionally scatter the result.

        Parameters
        ----------
        buffer : torch.Tensor
            Parameter tensor to transform.

        Returns
        -------
        torch.Tensor
            Updated tensor with values written at ``key`` locations when provided.
        """
        p = self.func(buffer)
        if self.key is None:
            return p
        b = buffer.clone()
        b.view(-1).index_copy_(0, self.key, self.fill(p))
        return b


def build_parametrization(
    module, output, key: torch.LongTensor, main_shape: tuple[int, int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Construct a parametrization callable for in-graph updates.

    Parameters
    ----------
    module : torch.nn.Module
        Module producing updated values.
    output : torch.Tensor
        Example output used to infer broadcasting behavior.
    key : torch.LongTensor
        Flat indices where updates should be applied.
    main_shape : tuple of int
        Shape of the target parameter grid.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Functional wrapper applying ``module`` and scattering results.
    """
    if key is None:
        return Functional(module)
    fill = create_param_expander(output, key, main_shape)
    return Functional(module, fill=fill, key=key)


def create_param_expander(
    param: torch.Tensor, key: torch.LongTensor, main_shape: tuple[int, int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a specialized function that expands parameters for indexed assignment.

    Parameters
    ----------
    param : torch.Tensor
        Prototype parameter tensor defining expansion semantics.
    key : torch.LongTensor
        Flat index tensor selecting assignment positions.
    main_shape : tuple of int
        Height and width of the conceptual 2D grid addressed by ``key``.

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Function that maps an input tensor to a flattened vector aligned with ``key``.

    Raises
    ------
    ValueError
        If ``param`` does not match any supported broadcasting scheme.

    Notes
    -----
    Supported patterns include scalar, pre-sized, row-broadcast, and column-broadcast
    parameterizations.
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
    rows = torch.div(key, main_shape[1], rounding_mode="floor")
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


class staticproperty:
    """A property whose value is independent of the instance."""

    def __init__(self, func):
        self.func = func  # zero‑argument callable

    def __get__(self, obj, objtype=None):
        return self.func()  # ignore obj / objtype


def add_instance_property(obj, name, func):
    """
    Attach a computed property to a single instance.

    Parameters
    ----------
    obj : object
        Instance receiving the property.
    name : str
        Property name to install.
    func : Callable[[], Any]
        Zero-argument callable returning the property value.
    """
    sub = type(
        f"_{obj.__class__.__name__}Proxy",
        (obj.__class__,),
        {name: staticproperty(func)},
    )
    obj.__class__ = sub  # replace the instance’s class in‑place


class Referency(AxModule):
    """Mixin that allows modules to expose dynamic property references."""

    def setreference(self, name, func):
        """
        Bind a lazily evaluated property to the instance.

        Parameters
        ----------
        name : str
            Property name to expose.
        func : Callable[[], Any]
            Zero-argument callable invoked when the property is accessed.
        """
        add_instance_property(self, name, func)


class SimpleParameterized(Referency):
    """
    Mixin that manages a flat set of named parameters for subclasses.

    Declare parameters with :meth:`PARAMETER` at class definition time; instances
    receive automatic instantiation of buffers or modules via :func:`to_param`.
    """

    _params = {}
    _params_defined_here = {}
    _params_declarations = []

    _flags = {}
    _flags_defined_here = {}
    _flags_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

        new_params = {}
        new_flags = {}

        for base in reversed(cls.__mro__):
            if "_params" in base.__dict__:
                new_params.update(base._params)
            if "_flags" in base.__dict__:
                new_flags.update(base._flags)
        cls._params_defined_here = {}
        cls._flags_defined_here = {}

        if SimpleParameterized._params_declarations:
            for p_dict in SimpleParameterized._params_declarations:
                cls._params_defined_here.update(p_dict)
            SimpleParameterized._params_declarations = []
        if SimpleParameterized._flags_declarations:
            for f_dict in SimpleParameterized._flags_declarations:
                cls._flags_defined_here.update(f_dict)
            SimpleParameterized._flags_declarations = []

        new_params.update(cls._params_defined_here)
        new_flags.update(cls._flags_defined_here)

        cls._params = new_params
        cls._flags = new_flags

    def __init__(self, **kwargs):
        super(SimpleParameterized, self).__init__()
        self.params = self.__class__._params.copy()
        if kwargs:
            self.params = {
                key: kwargs.get(key, value) for key, value in self.params.items()
            }
        self.instantiate_parameters(**self.params)
        self.flags = self.__class__._flags.copy()
        if kwargs:
            self.flags = {
                key: kwargs.get(key, value) for key, value in self.flags.items()
            }
        for key, value in self.flags.items():
            setattr(self, key, value)

    @classmethod
    def check_kwargs(cls, kwargs):
        """
        Validate keyword arguments against declared mechanism parameters.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments supplied to the initializer.

        Returns
        -------
        bool
            True when all names are valid.

        Raises
        ------
        ValueError
            If an unexpected parameter name is provided.
        """
        all_params = set(cls.all_parameter_names())
        if all_params is not None:
            for name in kwargs.keys():
                if name not in all_params:
                    raise ValueError(
                        f"Unknown parameter {name}. Valid parameters are {all_params}."
                    )
        return True

    @classmethod
    def all_parameter_names(cls):
        """
        Get all declared parameter names for the class.

        Returns
        -------
        dict
            Mapping from parameter names to default values.
        """
        return list(cls._params.keys())

    def instantiate_parameters(self, **kwargs):
        """
        Materialize parameters declared for the subclass.

        Parameters
        ----------
        **kwargs
            Mapping from parameter names to initial values.
        """
        for key, value in kwargs.items():
            setattr(self, key, to_param(value))

    @staticmethod
    def PARAMETER(**kwargs):
        """
        Declare parameters for the next subclass initialization.

        Parameters
        ----------
        **kwargs
            Parameter names with default values. These are per-instance and
            flattened (no shape metadata); use :class:`Parameterized` for
            GLOBAL/RANGE/RNG categories when population-aware shapes are needed.
        """
        SimpleParameterized._params_declarations.append(kwargs)

    @staticmethod
    def FLAG(**kwargs):
        """
        Declare flags for the next subclass initialization.

        Parameters
        ----------
        **kwargs
            Flag names with default boolean values.
        """
        SimpleParameterized._flags_declarations.append(kwargs)

    def device(self):
        """
        Device hosting the module's parameters.

        Returns
        -------
        torch.device
            Device of the first registered parameter.
        """
        return next(iter(self.parameters())).device

    def __repr__(self):
        return f"{self.__class__.__name__}({self.parameters_repr()})"

    def parameters_repr(self):
        """
        String representation of named parameters.

        Returns
        -------
        str
            Comma-separated key/value pairs for parameters.
        """
        return ", ".join(f"{k}={v}" for k, v in self.named_parameters())

    def parameter_set_(self, **kwargs):
        """
        Update parameter values in place.

        Parameters
        ----------
        **kwargs
            Mapping from parameter names to new values.

        Raises
        ------
        ValueError
            If an unknown parameter name is provided.
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter {key}.")
            param = getattr(self, key)
            if not isinstance(param, torch.nn.Parameter):
                raise ValueError(f"Attribute {key} is not a parameter.")
            with torch.no_grad():
                param.data.copy_(torch.as_tensor(value, dtype=param.dtype))


def check_conflicts(
    global_params,
    range_params,
    params_defined_here,
    rng_defined_here=None,
    table_defined_here=None,
):
    """
    Check for conflicts between global parameters, range parameters, and
    parameters defined in the current class.

    Raises ValueError if any parameter is defined in more than one category.
    """
    if rng_defined_here is None:
        rng_defined_here = set()
    if table_defined_here is None:
        table_defined_here = dict()
    all_params = (
        set(global_params.keys())
        .union(range_params.keys())
        .union(params_defined_here.keys())
        .union(rng_defined_here.keys())
        .union(table_defined_here.keys())
    )
    duplicates = set()

    for param in all_params:
        count = (
            (param in global_params)
            + (param in range_params)
            + (param in params_defined_here)
            + (param in rng_defined_here)
            + (param in table_defined_here)
        )
        if count > 1:
            duplicates.add(param)

    if duplicates:
        raise ValueError(
            f"Parameter conflict detected: {duplicates}. "
            "A parameter cannot be defined in multiple categories."
        )


def assign_precendence(cls):
    """
    Resolve parameters that appear in more than one of:
        cls._global, cls._range, cls._params.

    Rule:
      (1) If a duplicated parameter is marked "defined here" in this *class*
          in any of *_defined_here, keep that category and remove it from the others.
      (2) Otherwise walk the MRO (nearest first). The first class whose
          *_defined_here contains the parameter determines the winning category.
      (3) If no class in the MRO marks it as defined_here anywhere, fall back
          to a fixed category order ('_params' > '_range' > '_global') among the
          categories where the parameter currently appears.

    Mutates cls._global / cls._range / cls._params in place.
    Returns a dict {param_name: kept_category_name} for inspection.
    """

    # --- helpers -------------------------------------------------------------
    def _as_names(x):
        """Accept set/dict/iterable; return a set of parameter names."""
        if x is None:
            return set()
        if isinstance(x, set):
            return set(x)
        if isinstance(x, dict):
            return set(x.keys())
        try:
            return set(x)
        except TypeError:
            return set()

    # Containers on the class; treat missing as empty dicts
    containers = {
        "_global": getattr(cls, "_global", {}) or {},
        "_range": getattr(cls, "_range", {}) or {},
        "_params": getattr(cls, "_params", {}) or {},
    }

    # Which params are present where?
    present = {k: set(v.keys()) for k, v in containers.items()}
    all_params = present["_global"] | present["_range"] | present["_params"]
    dupes = {p for p in all_params if sum(p in present[k] for k in present) > 1}
    if not dupes:
        return {}

    # Category preference only for tie-breaking when nobody "defined_here" it.
    FALLBACK_ORDER = ("_params", "_range", "_global")

    kept = {}

    # Precompute "defined here" sets for *this* class
    defined_here_cls = {
        "_global": _as_names(getattr(cls, "_global_defined_here", None)),
        "_range": _as_names(getattr(cls, "_range_defined_here", None)),
        "_params": _as_names(getattr(cls, "_params_defined_here", None)),
    }

    for p in dupes:
        # 1) Check if *this* class defines it here in any category
        here_hits = [cat for cat, s in defined_here_cls.items() if p in s]
        if here_hits:
            # If (pathologically) multiple categories say "defined here", choose a stable order.
            if len(here_hits) > 1:
                # Pick the first that also currently contains p; prefer FALLBACK_ORDER among them.
                candidates = [
                    cat
                    for cat in FALLBACK_ORDER
                    if cat in here_hits and p in present[cat]
                ]
                winner = candidates[0] if candidates else here_hits[0]
            else:
                winner = here_hits[0]
        else:
            # 2) Walk the MRO; the first class that "defined_here" picks the category
            winner = None
            for base in cls.__mro__:  # includes cls itself; fine (we already checked)
                if base is object:
                    continue
                dh = {
                    "_global": _as_names(getattr(base, "_global_defined_here", None)),
                    "_range": _as_names(getattr(base, "_range_defined_here", None)),
                    "_params": _as_names(getattr(base, "_params_defined_here", None)),
                }
                hits = [cat for cat, s in dh.items() if p in s]
                if hits:
                    # Prefer a hit that actually exists in this class' containers;
                    # otherwise use a stable category order.
                    candidates = [cat for cat in hits if p in present[cat]]
                    if candidates:
                        # If multiple, use FALLBACK_ORDER to break ties deterministically
                        for cat in FALLBACK_ORDER:
                            if cat in candidates:
                                winner = cat
                                break
                    else:
                        # None of the hits exist here (rare); keep looking.
                        pass
                    if winner is not None:
                        break

            # 3) If nobody in the MRO "defined_here" it, fall back to category priority
            if winner is None:
                for cat in FALLBACK_ORDER:
                    if p in present[cat]:
                        winner = cat
                        break

        # Remove from non-winners
        for cat, mapping in containers.items():
            if cat != winner and p in mapping:
                mapping.pop(p, None)

        kept[p] = winner

    return kept


class Parameterized(SimpleParameterized):
    """
    A base class that allows subclasses to declare parameters which are
    automatically inherited and aggregated.

    Use uppercase classmethods at definition time:

    - ``GLOBAL``: shared scalar parameters (broadcast across compartments).
    - ``RANGE``: per-compartment parameters (shaped like ``shape_p``).
    - ``PARAMETER``: flat per-instance parameters from :class:`SimpleParameterized`.
    - ``RNG``: declare RNG seeds/generators to be instantiated.

    Subclasses (e.g., :class:`Mechanism`, :class:`State`) build on these
    declarations and expose additional lifecycle hooks.
    """

    _global = {}
    _global_defined_here = {}
    _global_declarations = []

    _range = {}
    _range_defined_here = {}
    _range_declarations = []

    _rng = {}
    _rng_defined_here = {}
    _rng_declarations = []

    _table = {}
    _table_defined_here = {}
    _table_declarations = []

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called automatically whenever a class
        inherits from Parameterized.
        """
        # Call the parent's __init_subclass__ WITHOUT our custom kwargs,
        # as the base 'object' class does not accept them.
        super().__init_subclass__()

        # Start with a fresh dictionary for the new class's parameters.
        new_global = {}
        new_range = {}
        new_rng = {}
        new_table = {}

        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for _global, _range, _rng attributes defined directly on the base
            if "_global" in base.__dict__:
                new_global.update(base._global)
            if "_range" in base.__dict__:
                new_range.update(base._range)
            if "_rng" in base.__dict__:
                new_rng.update(base._rng)
            if "_table" in base.__dict__:
                new_table.update(base._table)

        cls._global_defined_here = {}
        cls._range_defined_here = {}
        cls._rng_defined_here = {}
        cls._table_defined_here = {}

        # Add parameters declared via the GLOBAL() method
        if Parameterized._global_declarations:
            for p_dict in Parameterized._global_declarations:
                cls._global_defined_here.update(p_dict)
            Parameterized._global_declarations = []  # Clear for next class
        # Add range declarations
        if Parameterized._range_declarations:
            for r_dict in Parameterized._range_declarations:
                cls._range_defined_here.update(r_dict)
            Parameterized._range_declarations = []
        # Add rng declarations
        if Parameterized._rng_declarations:
            for rng_dict in Parameterized._rng_declarations:
                cls._rng_defined_here.update(rng_dict)
            Parameterized._rng_declarations = []
        # Add table declarations
        if Parameterized._table_declarations:
            for t_dict in Parameterized._table_declarations:
                cls._table_defined_here.update(t_dict)
            Parameterized._table_declarations = []

        # Update the new global and range dictionaries with the class-specific declarations
        new_global.update(cls._global_defined_here)
        new_range.update(cls._range_defined_here)
        new_rng.update(cls._rng_defined_here)
        new_table.update(cls._table_defined_here)

        check_conflicts(
            cls._global_defined_here,
            cls._range_defined_here,
            cls._params_defined_here,
            cls._rng_defined_here,
            cls._table_defined_here,
        )

        # Add parameters from class definition keywords (e.g., a=10)
        # These will override anything set by parents.
        new_global.update({k: v for k, v in kwargs.items() if k in new_global})
        new_range.update({k: v for k, v in kwargs.items() if k in new_range})

        cls._global = new_global
        cls._range = new_range
        cls._rng = new_rng
        cls._table = new_table

        assign_precendence(cls)

    @staticmethod
    def GLOBAL(**kwargs):
        """
        Declare scalar (compartment-independent) parameters.

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            once per instance and broadcast across compartments.
        """
        Parameterized._global_declarations.append(kwargs)

    @staticmethod
    def RANGE(**kwargs):
        """
        Declare per-compartment parameters (range variables).

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            with shape matching the population ``shape_p``.
        """
        Parameterized._range_declarations.append(kwargs)

    @staticmethod
    def RNG(*args, **kwargs):
        """
        Declare RNG identifiers to instantiate device-local generators.
        May supply initial seed values via keyword arguments.

        Parameters
        ----------
        *args : str
            Names of RNG streams to create. Instances receive generator buffers
            accessible via these names. No initial seed is set.
        **kwargs : str -> int
            Mapping of RNG stream names to initial seed values. Instances receive generator
            buffers initialized with the specified seeds.
        """
        Parameterized._rng_declarations.append(
            {**{name: None for name in args}, **kwargs}
        )

    @staticmethod
    def TABLE(func: str, low: float, high: float, n: int, learnable: bool = False):
        """
        Declare a lookup table to be created for the instance.
        The table maps inputs in [low, high] to outputs of the named function.

        This automatically creates a flag ``usetable_<func>`` that can be used to
        enable or disable table usage at runtime. By default, table usage is enabled,
        and can be disabled by setting the flag to False. Interpolation tables
        can be disabled for an entire State / Mechanism by using
        `.usetables(False)`.

        This does not require any modification of the State / Mechanism
        implementation (beyond the TABLE declaration); internally, AxonML will
        check the flag and use the table when enabled  whenever the State /
        Mechanism calls `func` (within, e.g., `breakpoint`).

        In practice, this can speed up repeated evaluations of expensive
        functions, but for simple functions the overhead of the table lookup
        may outweigh the benefits. May also be useful to optimize functions
        without assuming a specific functional form (besides that used for table
        initialization).

        Parameters
        ----------
        func : str
            Name of the function to tabulate (e.g., 'exp', 'sigmoid').
            This must correspond to a method that belongs to the Parameterized
            class and can be called with a single argument (i.e., can be called
            as ``self.func(x)``).
        low : float
            Lower bound of the tabulation range.
        high : float
            Upper bound of the tabulation range.
        n : int
            Number of points in the table.
        learnable : bool, optional
            If True, the table values are trainable parameters. Defaults to False.
        """
        Parameterized.FLAG(**{f"usetable_{func}": False})
        Parameterized._table_declarations.append(
            {func: {"low": low, "high": high, "n": n, "learnable": learnable}}
        )

    def __init__(self, shape, shape_f, additional_parameters=None, **kwargs):
        super().__init__(**kwargs)
        try:
            shape_p = [int(s) for s in shape]
            shape_f = [int(s) for s in shape_f]
            self.shape_p = tuple(shape_p)
            self.shape_f = tuple(shape_f)
        except Exception as e:
            raise TypeError(f"error assigning shape {shape!r}") from e

        self.globals = self.__class__._global.copy()
        self.range = self.__class__._range.copy()
        self.rng = self.__class__._rng.copy()

        self.in_graph_parametrizations = {}

        if kwargs:
            self.globals = {
                key: kwargs.get(key, value) for key, value in self.globals.items()
            }
            self.range = {
                key: kwargs.get(key, value) for key, value in self.range.items()
            }

        self.keys = {}
        self.additional_parameters = {}
        self.instantiate_global(**self.globals)
        self.instantiate_range(**self.range)
        self.instantiate_rng(**self.rng)
        self.instantiate_additional_parameters(additional_parameters)

    def reshape(self, shape_p, shape_f):
        """
        Update population and full shapes, reinitializing range buffers.

        Parameters
        ----------
        shape_p : tuple of int
            Shape used for per-compartment parameters.
        shape_f : tuple of int
            Full tensor shape including batch axes.
        """
        self.shape_p = shape_p
        self.shape_f = shape_f
        self.instantiate_range(**self.range)

    def instantiate_global(self, **kwargs):
        """
        Instantiate global (scalar) parameters and default buffers.

        Parameters
        ----------
        **kwargs
            Mapping of global parameter names to initial values or dictionaries.
        """
        if kwargs is not None:
            for name, value in kwargs.items():
                if isinstance(value, dict):
                    setattr(self, name, torch.nn.ParameterDict())
                    for pname, pval in value.items():
                        setattr(self, pname, to_param(pval))
                        getattr(self, name)[pname] = getattr(self, pname)
                else:
                    p_name = f"{name}_default"
                    setattr(self, p_name, to_param(value))
                    self.register_buffer(name, torch.empty(()))
                    getattr(self, name).copy_(getattr(self, p_name))

    def instantiate_range(self, **kwargs):
        """
        Instantiate range parameters over the population shape.

        Parameters
        ----------
        **kwargs
            Mapping of parameter names to initial values broadcast over ``shape_p``.
        """
        if kwargs is not None:
            for name, value in kwargs.items():
                p_name = f"{name}_default"
                setattr(self, p_name, to_param(value))
                self.register_buffer(name, torch.empty(self.shape_p))
                getattr(self, name).copy_(getattr(self, p_name))

    def instantiate_rng(self, **kwargs):
        for name, value in kwargs.items():
            rng = RNGModule(value, shape_p=self.shape_p, shape_f=self.shape_f)
            setattr(self, name, rng)

    def init_rng(self):
        """
        Initialize all RNG modules.
        """
        for name in self.__class__._rng.keys():
            rng_module = getattr(self, name)
            if isinstance(rng_module, RNGModule):
                rng_module.init()

    def reset_rng(self):
        """
        Reseed all RNG modules.
        """
        for name in self.__class__._rng.keys():
            rng_module = getattr(self, name)
            if isinstance(rng_module, RNGModule):
                rng_module.reset()

    def register_parametrization_in_graph(self, name: str, param: Callable, args=None):
        """
        Register a parametrization to be applied during buffer population.

        Parameters
        ----------
        name : str
            Buffer name receiving the parametrization.
        param : Callable or torch.nn.Module
            Transform producing updated values given the buffer and optional args.
        args : Sequence[str], optional
            Names of additional buffers passed to the parametrization.
        """
        if name not in self.in_graph_parametrizations:
            self.in_graph_parametrizations[name] = []
        if not isinstance(param, torch.nn.Module):
            # If param is a Module, we register it directly
            param = Functional(param)
        if args is None:
            args = []
        self.in_graph_parametrizations[name].append((param, args))

    def instantiate_additional_parameters(self, additional_parameters=None):
        """
        Materialize alias-specific parameter overrides provided at build time.

        Parameters
        ----------
        additional_parameters : dict, optional
            Mapping from parameter names to lists of ``(alias, value, key)`` tuples
            describing indexed overrides.
        """
        if additional_parameters is not None:
            for name, list_of_aliases_values_and_keys in additional_parameters.items():
                if name in self.range:
                    count = 0
                    keys = []
                    for alias, value, key in list_of_aliases_values_and_keys:
                        if alias is not None:
                            p_name = f"{name}_{alias}"
                        else:
                            p_name = f"{name}_{count}"
                            count += 1
                        key = torch.as_tensor(key, dtype=torch.long)
                        parameter = to_param(value)
                        if isinstance(parameter, torch.nn.Module):
                            p = parameter(torch.empty(self.shape_p))
                            parametrization = build_parametrization(
                                parameter, p, key, self.shape_p[-2:]
                            )
                            self.register_parametrization_in_graph(
                                name, parametrization
                            )
                            setattr(self, p_name, parameter)
                        else:
                            setattr(self, p_name, parameter)
                            fill = create_param_expander(
                                parameter, key, self.shape_p[-2:]
                            )
                            self.additional_parameters.setdefault(name, []).append(
                                (fill, getattr(self, p_name))
                            )
                            keys.append(key)
                    self.keys[name] = torch.cat(keys).to(torch.long)

    def load_additional_parameters(self):
        """
        Scatter alias-specific parameter overrides into their buffers.
        """
        for name, list_of_parameters in self.additional_parameters.items():
            buffer = getattr(self, name)
            additional_params = torch.cat([fill(p) for fill, p in list_of_parameters])
            key = self.keys[name].to(buffer.device)
            buffer.view(-1).index_copy_(0, key, additional_params)

    def parametrize(
        self,
        name: str,
        value: _valid_param_type,
        key: torch.LongTensor = None,
        alias: str = None,
    ):
        """
        Add or update an alias-specific parameter override.

        Parameters
        ----------
        name : str
            Name of the base parameter to override.
        value : Union[float, torch.Tensor, torch.nn.Parameter, torch.nn.Module]
            New parameter value or module.
        key : torch.LongTensor, optional
            Flat indices where the override should be applied. If None, applies to all indices.
        alias : str, optional
            Alias name for the override. If None, a numeric suffix is used.

        Examples
        --------
        >>> print(model.rhoa)  # Original parameter
        tensor([[100., 100., 100.],
                [100., 100., 100.]])
        >>> model.parametrize('rhoa', 150.0)
        >>> model.initialize() # Re-initialize to apply the override
        >>> print(model.rhoa)  # Updated parameter
        tensor([[150., 150., 150.],
                [150., 150., 150.]])
        >>> print(model.rhoa_0)  # Access the override parameter
        tensor(150.)
        """
        if key is None:
            return self.parametrize(
                name, value, key=torch.arange(math.prod(self.shape_p)), alias=alias
            )
        if name in self.range:
            if alias is None:
                count = 0
                while hasattr(self, f"{name}_{count}"):
                    count += 1
                alias = str(count)
            p_name = f"{name}_{alias}"
            if hasattr(self, p_name):
                raise ValueError(
                    f"Parameter override '{p_name}' already exists. Choose a different alias."
                )
            parameter = to_param(value)
            if isinstance(parameter, torch.nn.Module):
                p = parameter(torch.empty(self.shape_p))
                parametrization = build_parametrization(
                    parameter, p, key, self.shape_p[-2:]
                )
                self.register_parametrization_in_graph(name, parametrization)
                setattr(self, p_name, parameter)
            else:
                setattr(self, p_name, parameter)
                fill = create_param_expander(parameter, key, self.shape_p[-2:])
                if name not in self.additional_parameters:
                    self.additional_parameters[name] = []
                self.additional_parameters[name].append((fill, getattr(self, p_name)))
                if name not in self.keys:
                    self.keys[name] = key.to(torch.long)
                else:
                    self.keys[name] = torch.cat([self.keys[name], key.to(torch.long)])

    def populate_parameter_buffers(self):
        """
        Reset parameter buffers to defaults, then apply overrides and parametrizations.
        """
        keys_to_process = itertools.chain(
            self.__class__._global.keys(), self.__class__._range.keys()
        )
        for name in keys_to_process:
            if not torch.is_tensor(getattr(self, name)):
                continue
            if hasattr(self, "parametrizations"):
                if name in self.parametrizations:
                    # If the parameter has parametrizations, we skip it
                    continue
            p_name = f"{name}_default"
            setattr(self, name, getattr(self, name).detach())
            getattr(self, name).copy_(getattr(self, p_name))
        self.load_additional_parameters()
        self.apply_parametrizations()

    def apply_parametrizations(self):
        """
        Apply all parametrizations to the parameter buffers of this model.
        """
        for name, param_list in self.in_graph_parametrizations.items():
            b = getattr(self, name)
            for param, args in param_list:
                b = param(b, *[getattr(self, arg) for arg in args])
            setattr(self, name, b)

    def instantiate_tables(self):
        """
        Instantiate lookup tables declared for this class.
        """
        for name, table_info in self.__class__._table.items():
            func_name = name
            rebind_func_with_table(self, func_name)
            low, high, n, learnable = (
                table_info["low"],
                table_info["high"],
                table_info["n"],
                table_info.get("learnable", False),
            )
            if not hasattr(self, func_name):
                raise ValueError(
                    f"Function '{func_name}' not found in class '{self.__class__.__name__}' for table instantiation."
                )
            func = getattr(self, func_name)
            x = torch.linspace(low, high, n, dtype=torch.float64)
            y = func(x).flatten()
            setattr(
                self,
                f"{func_name}_table",
                PreparedInterp1d(
                    x,
                    y,
                    sort_xy=False,
                    exact_clamp=False,
                    learnable_y=learnable,
                    uniform="always",
                ).to(dtype=self.dtype(), device=self.device()),
            )
            setattr(self, f"usetable_{func_name}", True)

    def usetables(self, usetables=True):
        """
        Switch all function implementations to use lookup tables.
        """
        for name in self.__class__._table.keys():
            setattr(self, f"usetable_{name}", usetables)

    def detach(self):
        """
        Detach registered buffers from the computation graph.
        """
        for n, b in self.named_buffers():
            try:
                b.detach_()
            except Exception:
                setattr(self, n, b.detach())

    def parameters_dict(self):
        """
        Returns a dictionary of all parameters in the model.
        """
        return {name: param for name, param in self.named_parameters()}

    @classmethod
    def all_parameter_names(cls):
        """
        Returns a list of all parameter names in the model.
        """
        return (
            list(cls._params.keys())
            + list(cls._global.keys())
            + list(cls._range.keys())
        )

    def dtype(self):
        """
        Data type of the module's parameters.

        Returns
        -------
        torch.dtype
            Data type of the first registered parameter.
        """
        return next(iter(self.parameters())).dtype

    def device(self):
        """
        Device hosting the module's parameters.

        Returns
        -------
        torch.device
            Device of the first registered parameter.
        """
        return next(iter(self.parameters())).device


table_function_template = """
def {func_name}_with_table(self, x):
    if self.usetable_{func_name}:
        return self.{func_name}_table(x)
    return self.{func_name}_original(x)
"""


def rebind_func_with_table(obj, func_name):
    """
    Rebind a function of an object to use its lookup table if available.

    Parameters
    ----------
    obj : object
        The object containing the function and potential lookup table.
    func_name : str
        The name of the function to rebind.
    """
    func_code = table_function_template.format(func_name=func_name)
    if DEBUG > 0:
        logger.info(f"Generated code for {func_name}:\n{func_code}")
    filename = "<table_function>"
    code = compile(func_code, filename, "exec")
    exec(code)
    meth = locals()[f"{func_name}_with_table"]
    setattr(obj, f"{func_name}_original", getattr(obj, func_name))
    setattr(obj, func_name, MethodType(meth, obj))
