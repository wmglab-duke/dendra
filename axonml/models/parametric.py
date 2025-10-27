import itertools
from typing import Callable

import torch
import torch.nn.functional as F


def to_param(val, positive=False):
    if isinstance(val, torch.nn.Parameter):
        return val
    if isinstance(val, torch.nn.Module):
        return val
    val = torch.as_tensor(val)
    if positive:
        val = torch.clamp(val, min=0.0)
        return PositiveParam(val)
    param = torch.nn.Parameter(val, requires_grad=False)
    return param


def is_parametric(val):
    _parametric_types = (torch.nn.Parameter, PositiveParam)
    return isinstance(val, _parametric_types)


def distribute_over(val, over="a"):
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
    Numerically stable inverse of softplus: x s.t. softplus(x, beta) = y.
    - Uses log(expm1(.)) for small by to avoid cancellation.
    - Uses by + log1p(-exp(-by)) for large by to avoid overflow.
    - Clamps y to avoid -inf at exactly 0 during initialization.
    """
    y = torch.as_tensor(y)
    y = torch.clamp(y, min=eps)  # safer for init; prevents -inf params
    by = beta * y

    small = by <= threshold
    x_small = torch.log(torch.expm1(by)) / beta
    x_large = (by + torch.log1p(-torch.exp(-by))) / beta  # stable for large by

    return torch.where(small, x_small, x_large)


class cacheable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._cache = None

    def clear_cache(self):
        self._cache = None

    def forward(self, cache=True, *args, **kwargs):
        if not cache:
            return self._compute(*args, **kwargs)
        if self._cache is None:
            self._cache = self._compute(*args, **kwargs)
        return self._cache

    def repeat(self, n: int):
        p = self(cache=(not self.training))
        return p.repeat(n)


def ste_clamp(y, *, lo=None, hi=None, alpha_lo: float = 1.0, alpha_hi: float = 1.0):
    """
    Forward: hard clamp to [lo, hi].
    Backward: use surrogate with slope 1 in-range; slope alpha_lo/alpha_hi when clamped.
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
    Trainable tensor with optional lower/upper bounds.

    Bounds & modes:
      - min=None, max=None:            identity
      - min!=None, max=None:           lower bound via:
           lower_mode="softplus"  ->  min + softplus(rho)      (exclusive)
           lower_mode="hard-ste"  ->  ste_clamp(rho, lo=min)   (inclusive)
           lower_mode="leaky-ste" ->  ste_clamp(..., alpha_lo=lower_alpha)
      - min=None,  max!=None:          upper bound via:
           cap_mode="softcap"    ->  max - softplus(max - y)
           cap_mode="hard-ste"   ->  ste_clamp(y, hi=max)
      - min!=None, max!=None:
           cap_mode="sigmoid"    ->  min + (max-min)*sigmoid(beta*rho)
           cap_mode="hard-ste"   ->  ste_clamp(y, lo=min, hi=max)

    Args:
        init, min_val, max_val, beta, threshold as before
        lower_mode: "softplus" | "hard-ste" | "leaky-ste"
        lower_alpha: slope used when clamped below min (for leaky-ste)
        cap_mode: "auto"|"softcap"|"sigmoid"|"hard-ste"
        cap_beta: temperature for softcap
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


class PositiveParam(Bounded):
    """
    Special case of Bounded with min_val=0 by default.
    Set include_zero=True to make the lower bound inclusive (via STE).
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
    A functional that can be used as a parameter in a model.
    This is useful for cases where you want to use a function as a parameter,
    such as in a neural network layer.
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
        p = self.func(buffer)
        if self.key is None:
            return p
        b = buffer.clone()
        b.view(-1).index_copy_(0, self.key, self.fill(p))
        return b


def build_parametrization(
    module, output, key: torch.LongTensor, main_shape: tuple[int, int]
) -> Callable[[torch.Tensor], torch.Tensor]:
    if key is None:
        return Functional(module)
    fill = create_param_expander(output, key, main_shape)
    return Functional(module, fill=fill, key=key)


def create_param_expander(
    param: torch.Tensor, key: torch.LongTensor, main_shape: tuple[int, int]
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
    sub = type(
        f"_{obj.__class__.__name__}Proxy",
        (obj.__class__,),
        {name: staticproperty(func)},
    )
    obj.__class__ = sub  # replace the instance’s class in‑place


class Referency(torch.nn.Module):
    def setreference(self, name, func):
        add_instance_property(self, name, func)


class SimpleParameterized(Referency):
    _params = {}
    _params_defined_here = {}
    _params_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

        new_params = {}

        for base in reversed(cls.__mro__):
            if "_params" in base.__dict__:
                new_params.update(base._params)

        cls._params_defined_here = {}

        if SimpleParameterized._params_declarations:
            for p_dict in SimpleParameterized._params_declarations:
                cls._params_defined_here.update(p_dict)
            SimpleParameterized._params_declarations = []

        new_params.update(cls._params_defined_here)

        cls._params = new_params

    def __init__(self, **kwargs):
        super(SimpleParameterized, self).__init__()
        self.params = self.__class__._params.copy()
        if kwargs:
            self.params = {
                key: kwargs.get(key, value) for key, value in self.params.items()
            }
        self.instantiate_parameters(**self.params)

    def check_kwargs(self, kwargs):
        """
        Check if the provided keyword arguments match the declared parameters.
        Raises ValueError if any unknown parameter is found.
        """
        if not self._params:
            return True
        for key in kwargs:
            if key not in self._params:
                raise ValueError(
                    f"Unknown parameter: {key} for {self.__class__.__name__}. Valid parameters are: {list(self._params.keys())}"
                )
        return True

    def instantiate_parameters(self, **kwargs):
        """
        Instantiate parameters using the provided keyword arguments.
        This method is called during initialization to set up the waveform's parameters.
        """
        for key, value in kwargs.items():
            setattr(self, key, to_param(value))

    @staticmethod
    def PARAMETER(**kwargs):
        SimpleParameterized._params_declarations.append(kwargs)

    def device(self):
        """
        Returns the device of the first parameter.
        """
        return next(iter(self.parameters())).device

    def __repr__(self):
        return f"{self.__class__.__name__}({self.parameters_repr()})"

    def parameters_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.named_parameters())


def check_conflicts(global_params, range_params, params_defined_here):
    """
    Check for conflicts between global parameters, range parameters, and
    parameters defined in the current class.

    Raises ValueError if any parameter is defined in more than one category.
    """
    all_params = (
        set(global_params.keys())
        .union(range_params.keys())
        .union(params_defined_here.keys())
    )
    duplicates = set()

    for param in all_params:
        count = (
            (param in global_params)
            + (param in range_params)
            + (param in params_defined_here)
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
    """

    _global = {}
    _global_defined_here = {}
    _global_declarations = []

    _range = {}
    _range_defined_here = {}
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
        new_global = {}
        new_range = {}

        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for _global and _range attributes defined directly on the base
            if "_global" in base.__dict__:
                new_global.update(base._global)
            if "_range" in base.__dict__:
                new_range.update(base._range)

        cls._global_defined_here = {}
        cls._range_defined_here = {}

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

        # Update the new global and range dictionaries with the class-specific declarations
        new_global.update(cls._global_defined_here)
        new_range.update(cls._range_defined_here)

        check_conflicts(
            cls._global_defined_here, cls._range_defined_here, cls._params_defined_here
        )

        # Add parameters from class definition keywords (e.g., a=10)
        # These will override anything set by parents.
        new_global.update({k: v for k, v in kwargs.items() if k in new_global})
        new_range.update({k: v for k, v in kwargs.items() if k in new_range})

        cls._global = new_global
        cls._range = new_range

        assign_precendence(cls)

    @staticmethod
    def GLOBAL(**kwargs):
        """
        A static method to declare parameters. This has the side effect of
        appending the parameters to a temporary class-level list.
        """
        Parameterized._global_declarations.append(kwargs)

    @staticmethod
    def RANGE(**kwargs):
        """
        A static method to declare ranges. This has the side effect of
        appending the ranges to a temporary class-level list.
        """
        Parameterized._range_declarations.append(kwargs)

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
        self.instantiate_additional_parameters(additional_parameters)

    def reshape(self, shape_p, shape_f):
        self.shape_p = shape_p
        self.shape_f = shape_f
        self.instantiate_range(**self.range)

    def instantiate_global(self, **kwargs):
        # this is only called once, on __init__
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
        # this is only called once, on __init__
        if kwargs is not None:
            for name, value in kwargs.items():
                p_name = f"{name}_default"
                setattr(self, p_name, to_param(value))
                self.register_buffer(name, torch.empty(self.shape_p))
                getattr(self, name).copy_(getattr(self, p_name))

    def register_parametrization_in_graph(self, name: str, param: Callable, args=None):
        if name not in self.in_graph_parametrizations:
            self.in_graph_parametrizations[name] = []
        if not isinstance(param, torch.nn.Module):
            # If param is a Module, we register it directly
            param = Functional(param)
        if args is None:
            args = []
        self.in_graph_parametrizations[name].append((param, args))

    def instantiate_additional_parameters(self, additional_parameters=None):
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
        for name, list_of_parameters in self.additional_parameters.items():
            buffer = getattr(self, name)
            additional_params = torch.cat([fill(p) for fill, p in list_of_parameters])
            key = self.keys[name].to(buffer.device)
            buffer.view(-1).index_copy_(0, key, additional_params)

    def populate_parameter_buffers(self):
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

    def detach(self):
        for n, b in self.named_buffers():
            try:
                b.detach_()
            except Exception:
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
