"""Parameter handling utilities and mixins for Dendra models."""

import itertools
import math
from numbers import Integral
from types import MethodType
from typing import Callable, Union

import torch
import torch.nn.functional as F

from dendra.helpers import (
    DEBUG,
    REQUIRE_GRAD,
    _normalize_dtype_value,
    current_device,
    current_dtype,
    logger,
)
from dendra.utils import PreparedInterp1d
from dendra.utils.dynamic_compilation import compile_generated_function

from ._class_declarations import consume_class_values, declare_class_value
from .modular import DNModule, matches_any_pattern
from .random_parameters import (
    RandomParameterSpec,
    RuntimeNoiseSpec,
    make_random_parameter_spec,
    make_runtime_noise_spec,
    sample_random_parameter,
    sample_runtime_noise,
)
from .rng import RNGModule

_valid_param_type = Union[float, torch.Tensor, torch.nn.Parameter, torch.nn.Module]


def to_param(
    val, positive=False, negative=False, requires_grad=None, *, device=None, dtype=None
):
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
    if negative and positive:
        raise ValueError("Cannot set both positive and negative to True.")
    requested_requires_grad = requires_grad
    if requires_grad is None:
        requires_grad = bool(REQUIRE_GRAD)
    if isinstance(val, torch.nn.Parameter):
        parameter_device = val.device if device is None else torch.device(device)
        parameter_dtype = val.dtype if dtype is None else _normalize_dtype_value(dtype)
        converted = val.to(device=parameter_device, dtype=parameter_dtype)
        constrained_requires_grad = (
            val.requires_grad
            if requested_requires_grad is None
            else bool(requested_requires_grad)
        )
        if positive:
            return PositiveParam(converted, requires_grad=constrained_requires_grad)
        if negative:
            return NegativeParam(converted, requires_grad=constrained_requires_grad)
        if converted is not val or constrained_requires_grad != val.requires_grad:
            return torch.nn.Parameter(
                converted, requires_grad=constrained_requires_grad
            )
        return val
    if isinstance(val, torch.nn.Module):
        return val
    if torch.is_tensor(val):
        # Existing floating tensors are already explicit dtype/device choices.
        # Preserve them unless the caller requests a conversion; routing them
        # through the global construction defaults first irreversibly rounds
        # float64 model inputs (notably Material initial fields) to float32.
        target_device = val.device if device is None else torch.device(device)
        target_dtype = (
            val.dtype
            if dtype is None and val.is_floating_point()
            else (
                current_dtype(torch.float32)
                if dtype is None
                else _normalize_dtype_value(dtype)
            )
        )
    else:
        target_device = current_device(None) if device is None else torch.device(device)
        target_dtype = (
            current_dtype(torch.float32)
            if dtype is None
            else _normalize_dtype_value(dtype)
        )
    val = torch.as_tensor(
        val,
        device=target_device,
        dtype=target_dtype,
    )
    if positive:
        val = torch.clamp(val, min=0.0)
        return PositiveParam(val, requires_grad=requires_grad)
    if negative:
        val = torch.clamp(val, max=0.0)
        return NegativeParam(val, requires_grad=requires_grad)
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
    beta = float(beta)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive.")
    y = torch.as_tensor(y)
    if not y.is_floating_point():
        y = y.to(dtype=current_dtype(torch.float32))
    original_dtype = y.dtype
    work = y.float() if y.dtype in {torch.float16, torch.bfloat16} else y
    effective_eps = max(float(eps), torch.finfo(work.dtype).tiny)
    work = torch.clamp(work, min=effective_eps)
    by = beta * work

    # y + log(1 - exp(-beta*y)) / beta, written with expm1 for
    # stable values and gradients at both small and large arguments.
    nonlinear = work + torch.log(-torch.expm1(-by)) / beta
    result = torch.where(by <= float(threshold), nonlinear, work)
    return result.to(dtype=original_dtype)


def resolve(parameter):
    """Resolve a parameter-like object to a tensor value."""
    if isinstance(parameter, torch.nn.Parameter):
        return parameter
    if isinstance(parameter, torch.nn.Module):
        return parameter()
    return parameter


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

    def _apply(self, fn, recurse=True):
        """Invalidate cached tensors before dtype or device conversion."""
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def requires_grad_(self, requires_grad: bool = True):
        """Invalidate cached graphs when freezing or unfreezing parameters."""
        self.clear_cache()
        return super().requires_grad_(requires_grad)

    def __getstate__(self):
        """Exclude derived tensors from deepcopy and pickle payloads."""
        state = super().__getstate__()
        state["_cache"] = None
        return state

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Invalidate derived values before restoring module state."""
        self.clear_cache()
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, *args, **kwargs):
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
        if self.training:
            return self._compute(*args, **kwargs)
        if self._cache is None:
            self._cache = self._compute(*args, **kwargs)
        return self._cache

    def repeat(self, n: int):
        """
        Repeat the value using the legacy sampling convention.

        Parameters
        ----------
        n : int
            Number of repeats.

        Returns
        -------
        torch.Tensor
            Repeated output. Scalars become length-``n`` vectors; existing
            vectors are concatenated ``n`` times.
        """
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n must be a non-negative integer.")
        n = int(n)
        if n < 0:
            raise ValueError("n must be a non-negative integer.")
        p = self()
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

    _version = 2

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
        requires_grad: bool = None,
    ):
        super().__init__()
        if requires_grad is None:
            requires_grad = bool(REQUIRE_GRAD)
        self.min_val = None if min_val is None else float(min_val)
        self.max_val = None if max_val is None else float(max_val)
        if self.min_val is not None and not math.isfinite(self.min_val):
            raise ValueError(
                "min_val must be finite when provided; use None if unbounded."
            )
        if self.max_val is not None and not math.isfinite(self.max_val):
            raise ValueError(
                "max_val must be finite when provided; use None if unbounded."
            )
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
        if self.lower_mode not in {"softplus", "hard-ste", "leaky-ste"}:
            raise ValueError(f"Unknown lower_mode: {self.lower_mode}")
        if self.cap_mode not in {"auto", "softcap", "sigmoid", "hard-ste"}:
            raise ValueError(f"Unknown cap_mode: {self.cap_mode}")
        if self.max_val is not None and self.min_val is None and cap_mode == "sigmoid":
            raise ValueError(
                "cap_mode='sigmoid' requires both min_val and max_val; use "
                "cap_mode='softcap' for an upper-only bound."
            )
        if (
            self.max_val is not None
            and self.min_val is not None
            and cap_mode == "softcap"
        ):
            raise ValueError(
                "cap_mode='softcap' is only defined for an upper-only bound; use "
                "cap_mode='sigmoid' when both bounds are present."
            )
        if (
            self.max_val is not None
            and self.min_val is not None
            and cap_mode in {"auto", "sigmoid"}
            and self.lower_mode != "softplus"
        ):
            raise ValueError(
                "Non-softplus lower modes with both bounds require cap_mode='hard-ste'."
            )
        if not math.isfinite(self.beta) or self.beta <= 0.0:
            raise ValueError("beta must be finite and positive.")
        if not math.isfinite(self.cap_beta) or self.cap_beta <= 0.0:
            raise ValueError("cap_beta must be finite and positive.")

        init = torch.as_tensor(init)
        if not init.is_floating_point():
            init = init.to(dtype=current_dtype(torch.float32))

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
                rho0 = self.max_val - softplus_inv(
                    y, beta=self.cap_beta, threshold=self.threshold
                )

        else:
            if self._upper_mode(upper_only=False) == "hard-ste":
                if self.lower_mode == "softplus":
                    y = torch.clamp(init - self.min_val, min=1e-12)
                    rho0 = softplus_inv(y, beta=self.beta, threshold=self.threshold)
                else:
                    rho0 = init
            else:
                rng = max(self.max_val - self.min_val, 1e-12)
                work = (
                    init.float()
                    if init.dtype in {torch.float16, torch.bfloat16}
                    else init
                )
                t = torch.clamp((work - self.min_val) / rng, 1e-6, 1 - 1e-6)
                rho0 = (torch.special.logit(t) / self.beta).to(dtype=init.dtype)

        self.rho = torch.nn.Parameter(rho0, requires_grad=requires_grad)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Migrate bounded-parameter coordinates saved before version 2."""
        version = local_metadata.get("version")
        rho_key = f"{prefix}rho"
        if version is not None and version < 2 and rho_key in state_dict:
            old_rho = state_dict[rho_key]
            old_value = None
            if getattr(self, "_auto_promoted_hard_cap", False):
                old_value = self.min_val + (
                    self.max_val - self.min_val
                ) * torch.sigmoid(self.beta * old_rho)
            elif (
                self.min_val is not None
                and self.max_val is not None
                and self.cap_mode == "hard-ste"
                and self.lower_mode == "softplus"
            ):
                old_value = torch.clamp(old_rho, min=self.min_val, max=self.max_val)
            if old_value is not None:
                state_dict[rho_key] = Bounded._inverse_transform(self, old_value)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

    def _compute(self, *args, **kwargs):
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
            # Inclusive [min,max]. Preserve the configured lower transform
            # while applying a straight-through upper cap.
            if self.lower_mode == "softplus":
                lower_bounded = self.min_val + F.softplus(
                    self.rho, beta=self.beta, threshold=self.threshold
                )
                return ste_clamp(lower_bounded, hi=self.max_val)
            alpha_lo = 1.0 if self.lower_mode == "hard-ste" else self.lower_alpha
            return ste_clamp(
                self.rho,
                lo=self.min_val,
                hi=self.max_val,
                alpha_lo=alpha_lo,
            )
        else:
            # default: sigmoid to (min,max)
            s = torch.sigmoid(self.beta * self.rho)
            return self.min_val + (self.max_val - self.min_val) * s

    def __len__(self):
        return self.rho.numel()

    def set(self, value):
        """
        Set the parameter value directly, bypassing the unconstrained ``rho``.

        Parameters
        ----------
        value : array_like
            New value for the parameter, which will be clamped to the valid range.
        """
        with torch.no_grad():
            resolved = torch.as_tensor(
                value, dtype=self.rho.dtype, device=self.rho.device
            )
            self.rho.copy_(self._inverse_transform(resolved))
        self.clear_cache()

    def _inverse_transform(self, value):
        # This method computes the inverse of the forward mapping, used for direct setting.
        if self.min_val is None and self.max_val is None:
            return value

        elif self.min_val is not None and self.max_val is None:
            if self.lower_mode == "softplus":
                y = torch.clamp(value - self.min_val, min=1e-12)
                return softplus_inv(y, beta=self.beta, threshold=self.threshold)
            else:
                return value

        elif self.min_val is None and self.max_val is not None:
            if self._upper_mode(upper_only=True) == "hard-ste":
                return value
            else:
                y = torch.clamp(self.max_val - value, min=1e-12)
                return self.max_val - softplus_inv(
                    y, beta=self.cap_beta, threshold=self.threshold
                )

        else:
            if self._upper_mode(upper_only=False) == "hard-ste":
                if self.lower_mode == "softplus":
                    y = torch.clamp(value - self.min_val, min=1e-12)
                    return softplus_inv(y, beta=self.beta, threshold=self.threshold)
                return value
            else:
                rng = max(self.max_val - self.min_val, 1e-12)
                work = (
                    value.float()
                    if value.dtype in {torch.float16, torch.bfloat16}
                    else value
                )
                t = torch.clamp((work - self.min_val) / rng, 1e-6, 1 - 1e-6)
                return (torch.special.logit(t) / self.beta).to(dtype=value.dtype)

    def repeat_and_reinit(self, n: int):
        """Repeat along a new leading axis before optimizers are constructed."""
        requires_grad = self.rho.requires_grad
        device = self.rho.device
        dtype = self.rho.dtype
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n must be a non-negative integer.")
        n = int(n)
        if n < 0:
            raise ValueError("n must be a non-negative integer.")
        value = self()
        new = value.repeat((n,) + (1,) * value.ndim)
        new_p = self._inverse_transform(new)
        self.rho = to_param(
            new_p,
            requires_grad=requires_grad,
            device=device,
            dtype=dtype,
        )
        self.clear_cache()
        return self

    def batch(self, n: int):
        """Add a leading batch axis before optimizers are constructed."""
        if isinstance(n, bool) or not isinstance(n, Integral):
            raise TypeError("n must be a non-negative integer.")
        n = int(n)
        if n < 0:
            raise ValueError("n must be a non-negative integer.")
        requires_grad = self.rho.requires_grad
        device = self.rho.device
        dtype = self.rho.dtype
        old = torch.atleast_1d(self())
        new = old[None, :].expand(n, *old.shape).clone()
        self.rho = to_param(
            self._inverse_transform(new),
            requires_grad=requires_grad,
            device=device,
            dtype=dtype,
        )
        self.clear_cache()
        return self


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
        requires_grad: bool = None,
    ):
        auto_promoted_hard_cap = (
            include_zero and max_val is not None and cap_mode == "auto"
        )
        if include_zero and max_val is not None:
            if cap_mode == "auto":
                cap_mode = "hard-ste"
            elif cap_mode != "hard-ste":
                raise ValueError(
                    "include_zero with max_val requires cap_mode='hard-ste'."
                )
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
            requires_grad=requires_grad,
        )
        self._auto_promoted_hard_cap = auto_promoted_hard_cap


class NegativeParam(Bounded):
    """
    Bounded parameter constrained to non-positive values.

    Parameters
    ----------
    init : array_like
        Initial value for the parameter.
    include_zero : bool, optional
        If True, make the zero bound inclusive using a leaky STE transform.
    min_val : float, optional
        Optional lower bound (must be negative when provided).
    beta : float, optional
        Softplus/sigmoid sharpness parameter.
    threshold : float, optional
        Softplus threshold for numerical stability.
    upper_alpha : float, optional
        Surrogate slope above zero when ``include_zero`` is True.
    cap_mode : {'auto', 'softcap', 'sigmoid', 'hard-ste'}, optional
        Strategy for the optional lower bound.
    cap_beta : float, optional
        Softcap temperature. Defaults to ``beta`` when ``None``.
    """

    def __init__(
        self,
        init,
        *,
        include_zero: bool = False,
        min_val: float | None = None,
        beta: float = 1.0,
        threshold: float = 20.0,
        upper_alpha: float = 0.1,  # used only if include_zero=True (leaky-ste)
        cap_mode: str = "auto",
        cap_beta: float | None = None,
        requires_grad: bool = None,
    ):
        init = torch.as_tensor(init)
        if min_val is not None and float(min_val) >= 0.0:
            raise ValueError("NegativeParam min_val must be negative when provided.")
        auto_promoted_hard_cap = (
            include_zero and min_val is not None and cap_mode == "auto"
        )
        if include_zero and min_val is not None:
            if cap_mode == "auto":
                cap_mode = "hard-ste"
            elif cap_mode != "hard-ste":
                raise ValueError(
                    "include_zero with min_val requires cap_mode='hard-ste'."
                )
        super().__init__(
            torch.neg(init),
            min_val=0.0,
            max_val=-min_val if min_val is not None else None,
            beta=beta,
            threshold=threshold,
            lower_mode=("leaky-ste" if include_zero else "softplus"),
            lower_alpha=upper_alpha,  # note the flipped role of alpha here
            cap_mode=cap_mode,
            cap_beta=cap_beta,
            requires_grad=requires_grad,
        )
        self._auto_promoted_hard_cap = auto_promoted_hard_cap

    def _compute(self, *args, **kwargs):
        return torch.neg(super()._compute(*args, **kwargs))

    def _inverse_transform(self, value):
        value = torch.as_tensor(value, device=self.rho.device, dtype=self.rho.dtype)
        return super()._inverse_transform(torch.neg(value))


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


class Referency(DNModule):
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

    _params_p = {}
    _params_p_defined_here = {}
    _params_p_declarations = []

    _params_n = {}
    _params_n_defined_here = {}
    _params_n_declarations = []

    _flags = {}
    _flags_defined_here = {}
    _flags_declarations = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__()

        new_params = {}
        new_params_p = {}
        new_params_n = {}
        new_flags = {}

        for base in reversed(cls.__mro__):
            if "_params" in base.__dict__:
                new_params.update(base._params)
            if "_params_p" in base.__dict__:
                new_params_p.update(base._params_p)
            if "_params_n" in base.__dict__:
                new_params_n.update(base._params_n)
            if "_flags" in base.__dict__:
                new_flags.update(base._flags)
        cls._params_defined_here = {}
        cls._params_p_defined_here = {}
        cls._params_n_defined_here = {}
        cls._flags_defined_here = {}

        for p_dict in consume_class_values(
            cls,
            "simple_parameterized.params",
            SimpleParameterized._params_declarations,
        ):
            cls._params_defined_here.update(p_dict)
        for pp_dict in consume_class_values(
            cls,
            "simple_parameterized.params_p",
            SimpleParameterized._params_p_declarations,
        ):
            cls._params_p_defined_here.update(pp_dict)
        for pn_dict in consume_class_values(
            cls,
            "simple_parameterized.params_n",
            SimpleParameterized._params_n_declarations,
        ):
            cls._params_n_defined_here.update(pn_dict)
        for f_dict in consume_class_values(
            cls,
            "simple_parameterized.flags",
            SimpleParameterized._flags_declarations,
        ):
            cls._flags_defined_here.update(f_dict)

        new_params.update(cls._params_defined_here)
        new_params_p.update(cls._params_p_defined_here)
        new_params_n.update(cls._params_n_defined_here)
        new_flags.update(cls._flags_defined_here)

        cls._params = new_params
        cls._params_p = new_params_p
        cls._params_n = new_params_n
        cls._flags = new_flags

    def __init__(self, **kwargs):
        super(SimpleParameterized, self).__init__()
        self.params = self.__class__._params.copy()
        self.params_p = self.__class__._params_p.copy()
        self.params_n = self.__class__._params_n.copy()
        if kwargs:
            self.params = {
                key: kwargs.get(key, value) for key, value in self.params.items()
            }
            self.params_p = {
                key: kwargs.get(key, value) for key, value in self.params_p.items()
            }
            self.params_n = {
                key: kwargs.get(key, value) for key, value in self.params_n.items()
            }
        self.instantiate_parameters(**self.params)
        self.instantiate_parameters(positive=True, **self.params_p)
        self.instantiate_parameters(negative=True, **self.params_n)
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
    def normalize_random_kwargs(cls, kwargs):
        """Normalize insertion kwargs for random distribution parameters.

        Canonical names such as ``rvar_mu`` and ``rvar_sigma`` are left
        unchanged. Bare distribution-parameter names such as ``mu`` or
        ``sigma`` are accepted only when they uniquely identify one declared
        random variable and do not collide with an ordinary declared parameter.
        """
        if not kwargs:
            return dict(kwargs or {})
        specs = []
        specs.extend(getattr(cls, "_random_parameters", {}).values())
        specs.extend(getattr(cls, "_runtime_noises", {}).values())
        if not specs:
            return dict(kwargs or {})
        ordinary_names = set(cls.all_parameter_names())
        bare_to_full = {}
        for spec in specs:
            for param_name in spec.params:
                bare_to_full.setdefault(param_name, []).append(
                    spec.full_parameter_name(param_name)
                )
        out = {}
        for key, value in dict(kwargs).items():
            if key in ordinary_names or key not in bare_to_full:
                out[key] = value
                continue
            full_names = bare_to_full[key]
            if len(full_names) != 1:
                raise ValueError(
                    f"Ambiguous random distribution override {key!r}. Use one of {full_names!r}."
                )
            full_name = full_names[0]
            if full_name in out:
                raise ValueError(
                    f"Both {key!r} and {full_name!r} were provided; use only the canonical random-parameter name."
                )
            out[full_name] = value
        return out

    @classmethod
    def all_parameter_names(cls):
        """
        Get all declared parameter names for the class.

        Returns
        -------
        dict
            Mapping from parameter names to default values.
        """
        ordered_names = itertools.chain(
            cls._params.keys(),
            cls._params_p.keys(),
            cls._params_n.keys(),
            cls._flags.keys(),
        )
        return list(dict.fromkeys(ordered_names))

    def instantiate_parameters(self, positive=False, negative=False, **kwargs):
        """
        Materialize parameters declared for the subclass.

        Parameters
        ----------
        **kwargs
            Mapping from parameter names to initial values.
        """
        for key, value in kwargs.items():
            setattr(self, key, to_param(value, positive=positive, negative=negative))

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
        declare_class_value(
            "simple_parameterized.params",
            kwargs,
            SimpleParameterized._params_declarations,
        )

    @staticmethod
    def PARAMETERP(**kwargs):
        """
        Declare positive parameters for the next subclass initialization.

        Parameters
        ----------
        **kwargs
            Parameter names with default values. These are per-instance and
            flattened (no shape metadata); use :class:`Parameterized` for
            GLOBAL/RANGE/RNG categories when population-aware shapes are needed.
        """
        declare_class_value(
            "simple_parameterized.params_p",
            kwargs,
            SimpleParameterized._params_p_declarations,
        )

    @staticmethod
    def FLAG(**kwargs):
        """
        Declare flags for the next subclass initialization.

        Parameters
        ----------
        **kwargs
            Flag names with default boolean values.
        """
        declare_class_value(
            "simple_parameterized.flags",
            kwargs,
            SimpleParameterized._flags_declarations,
        )

    def device(self):
        """
        Device hosting the module's parameters.

        Returns
        -------
        torch.device
            Device of the first registered parameter.
        """
        return next(iter(self.parameters())).device

    def evaluate(self, parameter_name: str):
        """
        Evaluate and return the value of a parameter by name.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to retrieve.

        Returns
        -------
        torch.Tensor
            The value of the requested parameter.
        """
        p = getattr(self, parameter_name)
        if isinstance(p, torch.nn.Parameter):
            return p
        elif isinstance(p, torch.nn.Module):
            return p()
        else:
            return p

    def resolve(self, parameter):
        if isinstance(parameter, str):
            return self.evaluate(parameter)
        return resolve(parameter)

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
            matched = False
            matched_via_pattern = False
            if hasattr(self, key):
                parameter = getattr(self, key)
                if isinstance(parameter, torch.nn.Parameter):
                    with torch.no_grad():
                        parameter.copy_(
                            torch.as_tensor(
                                value,
                                dtype=parameter.dtype,
                                device=parameter.device,
                            )
                        )
                    matched = True
                elif isinstance(parameter, Bounded):
                    parameter.set(value)
                    matched = True

            if not matched:
                pattern_parameters = [
                    parameter
                    for parameter_name, parameter in self.named_parameters()
                    if matches_any_pattern([key], parameter_name)
                ]
                if pattern_parameters:
                    staged = []
                    for parameter in pattern_parameters:
                        source = torch.as_tensor(
                            value,
                            dtype=parameter.dtype,
                            device=parameter.device,
                        )
                        try:
                            source = torch.broadcast_to(source, parameter.shape).clone()
                        except RuntimeError as exc:
                            raise ValueError(
                                f"Value for pattern {key!r} cannot broadcast to "
                                f"parameter shape {tuple(parameter.shape)}."
                            ) from exc
                        staged.append((parameter, source))

                    with torch.no_grad():
                        backups = [
                            (parameter, parameter.clone())
                            for parameter in pattern_parameters
                        ]
                        try:
                            for parameter, source in staged:
                                parameter.copy_(source)
                        except Exception:
                            for parameter, backup in backups:
                                parameter.copy_(backup)
                            for module in self.modules():
                                if isinstance(module, cacheable):
                                    module.clear_cache()
                            raise
                    matched = True
                    matched_via_pattern = True

            if matched_via_pattern:
                for module in self.modules():
                    if isinstance(module, cacheable):
                        module.clear_cache()

            if not matched:
                raise ValueError(f"Unknown parameter or pattern: {key!r}.")


def _param_key_set(mapping):
    """Return keys from a declaration mapping/set/list, treating None as empty."""
    if mapping is None:
        return set()
    if hasattr(mapping, "keys"):
        return set(mapping.keys())
    return set(mapping)


def check_conflicts(
    global_params=None,
    range_params=None,
    batch_params=None,
    global_p_params=None,
    range_p_params=None,
    batch_p_params=None,
    global_n_params=None,
    range_n_params=None,
    batch_n_params=None,
    params_defined_here=None,
    rng_defined_here=None,
    table_defined_here=None,
):
    """
    Check for parameter declaration conflicts across all declaration categories.

    The first three arguments are kept backward-compatible with the historical
    ``check_conflicts(global, range, batch)`` helper form used by older tests and
    downstream code.  Newer declaration categories default to empty.
    """
    groups = [
        global_params,
        range_params,
        batch_params,
        global_p_params,
        range_p_params,
        batch_p_params,
        global_n_params,
        range_n_params,
        batch_n_params,
        params_defined_here,
        rng_defined_here,
        table_defined_here,
    ]
    key_sets = [_param_key_set(group) for group in groups]
    all_params = set().union(*key_sets) if key_sets else set()
    duplicates = {
        param
        for param in all_params
        if sum(param in key_set for key_set in key_sets) > 1
    }

    if duplicates:
        raise ValueError(
            f"Parameter conflict detected: {duplicates}. "
            "A parameter cannot be defined in multiple categories."
        )


def _random_scope_maps(cls, scope: str, constraint: str):
    if scope == "global":
        if constraint == "positive":
            return cls._global_p_defined_here
        if constraint == "negative":
            return cls._global_n_defined_here
        return cls._global_defined_here
    if scope == "range":
        if constraint == "positive":
            return cls._range_p_defined_here
        if constraint == "negative":
            return cls._range_n_defined_here
        return cls._range_defined_here
    if scope == "batch":
        if constraint == "positive":
            return cls._batch_p_defined_here
        if constraint == "negative":
            return cls._batch_n_defined_here
        return cls._batch_defined_here
    raise ValueError(f"Unknown random parameter scope: {scope!r}")


def _expand_random_distribution_parameters(cls, specs):
    for spec in specs.values():
        for param_name, default in spec.params.items():
            full_name = spec.full_parameter_name(param_name)
            constraint = spec.constraints.get(param_name, "real")
            _random_scope_maps(cls, spec.scope, constraint)[full_name] = default


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
        "_batch": getattr(cls, "_batch", {}) or {},
        "_params": getattr(cls, "_params", {}) or {},
        "_global_p": getattr(cls, "_global_p", {}) or {},
        "_range_p": getattr(cls, "_range_p", {}) or {},
        "_batch_p": getattr(cls, "_batch_p", {}) or {},
        "_global_n": getattr(cls, "_global_n", {}) or {},
        "_range_n": getattr(cls, "_range_n", {}) or {},
        "_batch_n": getattr(cls, "_batch_n", {}) or {},
        "_params_p": getattr(cls, "_params_p", {}) or {},
        "_params_n": getattr(cls, "_params_n", {}) or {},
    }

    # Which params are present where?
    present = {k: set(v.keys()) for k, v in containers.items()}
    all_params = (
        present["_global"]
        | present["_range"]
        | present["_batch"]
        | present["_params"]
        | present["_global_p"]
        | present["_range_p"]
        | present["_batch_p"]
        | present["_global_n"]
        | present["_range_n"]
        | present["_batch_n"]
        | present["_params_p"]
        | present["_params_n"]
    )
    dupes = {p for p in all_params if sum(p in present[k] for k in present) > 1}
    if not dupes:
        return {}

    # Category preference only for tie-breaking when nobody "defined_here" it.
    FALLBACK_ORDER = (
        "_params",
        "_params_p",
        "_params_n",
        "_range",
        "_range_p",
        "_range_n",
        "_batch",
        "_batch_p",
        "_batch_n",
        "_global",
        "_global_p",
        "_global_n",
    )

    kept = {}

    # Precompute "defined here" sets for *this* class
    defined_here_cls = {
        "_global": _as_names(getattr(cls, "_global_defined_here", None)),
        "_range": _as_names(getattr(cls, "_range_defined_here", None)),
        "_batch": _as_names(getattr(cls, "_batch_defined_here", None)),
        "_params": _as_names(getattr(cls, "_params_defined_here", None)),
        "_global_p": _as_names(getattr(cls, "_global_p_defined_here", None)),
        "_range_p": _as_names(getattr(cls, "_range_p_defined_here", None)),
        "_batch_p": _as_names(getattr(cls, "_batch_p_defined_here", None)),
        "_global_n": _as_names(getattr(cls, "_global_n_defined_here", None)),
        "_range_n": _as_names(getattr(cls, "_range_n_defined_here", None)),
        "_batch_n": _as_names(getattr(cls, "_batch_n_defined_here", None)),
        "_params_p": _as_names(getattr(cls, "_params_p_defined_here", None)),
        "_params_n": _as_names(getattr(cls, "_params_n_defined_here", None)),
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
                    "_batch": _as_names(getattr(base, "_batch_defined_here", None)),
                    "_params": _as_names(getattr(base, "_params_defined_here", None)),
                    "_global_p": _as_names(
                        getattr(base, "_global_p_defined_here", None)
                    ),
                    "_range_p": _as_names(getattr(base, "_range_p_defined_here", None)),
                    "_batch_p": _as_names(getattr(base, "_batch_p_defined_here", None)),
                    "_global_n": _as_names(
                        getattr(base, "_global_n_defined_here", None)
                    ),
                    "_range_n": _as_names(getattr(base, "_range_n_defined_here", None)),
                    "_batch_n": _as_names(getattr(base, "_batch_n_defined_here", None)),
                    "_params_p": _as_names(
                        getattr(base, "_params_p_defined_here", None)
                    ),
                    "_params_n": _as_names(
                        getattr(base, "_params_n_defined_here", None)
                    ),
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
    - ``BATCH``: parameters with shape ``shape_p[:-1] + (1,)`` so they
      broadcast across the final compartment dimension.
    - ``RANGE``: per-compartment parameters (shaped like ``shape_p``).
    - ``PARAMETER``: flat per-instance parameters from :class:`SimpleParameterized`.
    - ``RNG``: declare RNG seeds/generators to be instantiated.

    Subclasses (e.g., :class:`Mechanism`, :class:`State`) build on these
    declarations and expose additional lifecycle hooks.
    """

    _global = {}
    _global_defined_here = {}
    _global_declarations = []

    _global_p = {}
    _global_p_defined_here = {}
    _global_p_declarations = []

    _global_n = {}
    _global_n_defined_here = {}
    _global_n_declarations = []

    _range = {}
    _range_defined_here = {}
    _range_declarations = []

    _range_p = {}
    _range_p_defined_here = {}
    _range_p_declarations = []

    _range_n = {}
    _range_n_defined_here = {}
    _range_n_declarations = []

    _batch = {}
    _batch_defined_here = {}
    _batch_declarations = []

    _batch_p = {}
    _batch_p_defined_here = {}
    _batch_p_declarations = []

    _batch_n = {}
    _batch_n_defined_here = {}
    _batch_n_declarations = []

    _rng = {}
    _rng_defined_here = {}
    _rng_declarations = []

    _random_parameters = {}
    _random_parameters_defined_here = {}
    _global_rand_declarations = []
    _range_rand_declarations = []
    _batch_rand_declarations = []

    _runtime_noises = {}
    _runtime_noises_defined_here = {}
    _global_noise_declarations = []
    _range_noise_declarations = []
    _batch_noise_declarations = []

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
        new_global_p = {}
        new_range = {}
        new_range_p = {}
        new_batch = {}
        new_batch_p = {}
        new_rng = {}
        new_random_parameters = {}
        new_runtime_noises = {}
        new_table = {}
        new_global_n = {}
        new_range_n = {}
        new_batch_n = {}

        # Walk MRO in reverse to build up params from parent to child
        for base in reversed(cls.__mro__):
            # We look for _global, _range, _rng attributes defined directly on the base
            if "_global" in base.__dict__:
                new_global.update(base._global)
            if "_global_p" in base.__dict__:
                new_global_p.update(base._global_p)
            if "_range" in base.__dict__:
                new_range.update(base._range)
            if "_range_p" in base.__dict__:
                new_range_p.update(base._range_p)
            if "_batch" in base.__dict__:
                new_batch.update(base._batch)
            if "_batch_p" in base.__dict__:
                new_batch_p.update(base._batch_p)
            if "_rng" in base.__dict__:
                new_rng.update(base._rng)
            if "_random_parameters" in base.__dict__:
                new_random_parameters.update(base._random_parameters)
            if "_runtime_noises" in base.__dict__:
                new_runtime_noises.update(base._runtime_noises)
            if "_table" in base.__dict__:
                new_table.update(base._table)
            if "_global_n" in base.__dict__:
                new_global_n.update(base._global_n)
            if "_range_n" in base.__dict__:
                new_range_n.update(base._range_n)
            if "_batch_n" in base.__dict__:
                new_batch_n.update(base._batch_n)

        cls._global_defined_here = {}
        cls._global_p_defined_here = {}
        cls._range_defined_here = {}
        cls._range_p_defined_here = {}
        cls._batch_defined_here = {}
        cls._batch_p_defined_here = {}
        cls._rng_defined_here = {}
        cls._random_parameters_defined_here = {}
        cls._runtime_noises_defined_here = {}
        cls._table_defined_here = {}
        cls._global_n_defined_here = {}
        cls._range_n_defined_here = {}
        cls._batch_n_defined_here = {}

        # Add parameters declared via the GLOBAL() method
        for p_dict in consume_class_values(
            cls, "parameterized.global", Parameterized._global_declarations
        ):
            cls._global_defined_here.update(p_dict)
        # Add parameters declared via the GLOBALP() method
        for p_dict in consume_class_values(
            cls, "parameterized.global_p", Parameterized._global_p_declarations
        ):
            cls._global_p_defined_here.update(p_dict)
        # Add negative global declarations
        for n_dict in consume_class_values(
            cls, "parameterized.global_n", Parameterized._global_n_declarations
        ):
            cls._global_n_defined_here.update(n_dict)
        # Add range declarations
        for r_dict in consume_class_values(
            cls, "parameterized.range", Parameterized._range_declarations
        ):
            cls._range_defined_here.update(r_dict)
        # Add parameters declared via the RANGEP() method
        for r_dict in consume_class_values(
            cls, "parameterized.range_p", Parameterized._range_p_declarations
        ):
            cls._range_p_defined_here.update(r_dict)
        # Add batch declarations
        for b_dict in consume_class_values(
            cls, "parameterized.batch", Parameterized._batch_declarations
        ):
            cls._batch_defined_here.update(b_dict)
        # Add parameters declared via the BATCHP() method
        for b_dict in consume_class_values(
            cls, "parameterized.batch_p", Parameterized._batch_p_declarations
        ):
            cls._batch_p_defined_here.update(b_dict)
        # Add negative range declarations
        for n_dict in consume_class_values(
            cls, "parameterized.range_n", Parameterized._range_n_declarations
        ):
            cls._range_n_defined_here.update(n_dict)
        # Add negative batch declarations
        for n_dict in consume_class_values(
            cls, "parameterized.batch_n", Parameterized._batch_n_declarations
        ):
            cls._batch_n_defined_here.update(n_dict)
        # Add rng declarations
        for rng_dict in consume_class_values(
            cls, "parameterized.rng", Parameterized._rng_declarations
        ):
            cls._rng_defined_here.update(rng_dict)

        # Add stochastic parameter declarations. Distribution parameters are
        # exposed as ordinary GLOBAL/RANGE/BATCH parameters.
        for declarations in (
            consume_class_values(
                cls,
                "parameterized.global_rand",
                Parameterized._global_rand_declarations,
            ),
            consume_class_values(
                cls,
                "parameterized.range_rand",
                Parameterized._range_rand_declarations,
            ),
            consume_class_values(
                cls,
                "parameterized.batch_rand",
                Parameterized._batch_rand_declarations,
            ),
        ):
            for spec in declarations:
                if spec.name in cls._random_parameters_defined_here:
                    raise ValueError(
                        f"Random parameter {spec.name!r} declared more than once "
                        f"on {cls.__name__}."
                    )
                cls._random_parameters_defined_here[spec.name] = spec
        _expand_random_distribution_parameters(cls, cls._random_parameters_defined_here)

        # Add detached runtime-noise declarations. Distribution parameters are
        # exposed as ordinary GLOBAL/RANGE/BATCH parameters, while the sampled
        # noise buffer itself is updated in-place during simulation.
        for declarations in (
            consume_class_values(
                cls,
                "parameterized.global_noise",
                Parameterized._global_noise_declarations,
            ),
            consume_class_values(
                cls,
                "parameterized.range_noise",
                Parameterized._range_noise_declarations,
            ),
            consume_class_values(
                cls,
                "parameterized.batch_noise",
                Parameterized._batch_noise_declarations,
            ),
        ):
            for spec in declarations:
                if spec.name in cls._runtime_noises_defined_here:
                    raise ValueError(
                        f"Runtime noise {spec.name!r} declared more than once "
                        f"on {cls.__name__}."
                    )
                cls._runtime_noises_defined_here[spec.name] = spec
        _expand_random_distribution_parameters(cls, cls._runtime_noises_defined_here)

        # Add table declarations
        for t_dict in consume_class_values(
            cls, "parameterized.table", Parameterized._table_declarations
        ):
            cls._table_defined_here.update(t_dict)

        # Update the new global and range dictionaries with the class-specific declarations
        new_global.update(cls._global_defined_here)
        new_global_p.update(cls._global_p_defined_here)
        new_range.update(cls._range_defined_here)
        new_range_p.update(cls._range_p_defined_here)
        new_batch.update(cls._batch_defined_here)
        new_batch_p.update(cls._batch_p_defined_here)
        new_rng.update(cls._rng_defined_here)
        new_random_parameters.update(cls._random_parameters_defined_here)
        new_runtime_noises.update(cls._runtime_noises_defined_here)
        new_table.update(cls._table_defined_here)
        new_global_n.update(cls._global_n_defined_here)
        new_range_n.update(cls._range_n_defined_here)
        new_batch_n.update(cls._batch_n_defined_here)

        stochastic_buffer_conflicts = (
            set(new_random_parameters) | set(new_runtime_noises)
        ) & set().union(
            set(new_global),
            set(new_global_p),
            set(new_global_n),
            set(new_range),
            set(new_range_p),
            set(new_range_n),
            set(new_batch),
            set(new_batch_p),
            set(new_batch_n),
            set(getattr(cls, "_params", {}).keys()),
            set(getattr(cls, "_params_p", {}).keys()),
            set(getattr(cls, "_params_n", {}).keys()),
        )
        if stochastic_buffer_conflicts:
            raise ValueError(
                "Stochastic sample/noise buffer names conflict with declared "
                f"parameters: {sorted(stochastic_buffer_conflicts)}."
            )

        check_conflicts(
            cls._global_defined_here,
            cls._range_defined_here,
            cls._batch_defined_here,
            cls._global_p_defined_here,
            cls._range_p_defined_here,
            cls._batch_p_defined_here,
            cls._global_n_defined_here,
            cls._range_n_defined_here,
            cls._batch_n_defined_here,
            cls._params_defined_here,
            cls._rng_defined_here,
            cls._table_defined_here,
        )

        # Add parameters from class definition keywords (e.g., a=10)
        # These will override anything set by parents.
        new_global.update({k: v for k, v in kwargs.items() if k in new_global})
        new_global_p.update({k: v for k, v in kwargs.items() if k in new_global_p})
        new_range.update({k: v for k, v in kwargs.items() if k in new_range})
        new_range_p.update({k: v for k, v in kwargs.items() if k in new_range_p})
        new_batch.update({k: v for k, v in kwargs.items() if k in new_batch})
        new_batch_p.update({k: v for k, v in kwargs.items() if k in new_batch_p})
        new_global_n.update({k: v for k, v in kwargs.items() if k in new_global_n})
        new_range_n.update({k: v for k, v in kwargs.items() if k in new_range_n})
        new_batch_n.update({k: v for k, v in kwargs.items() if k in new_batch_n})

        cls._global = new_global
        cls._global_p = new_global_p
        cls._range = new_range
        cls._range_p = new_range_p
        cls._batch = new_batch
        cls._batch_p = new_batch_p
        cls._global_n = new_global_n
        cls._range_n = new_range_n
        cls._batch_n = new_batch_n
        cls._rng = new_rng
        cls._random_parameters = new_random_parameters
        cls._runtime_noises = new_runtime_noises
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
        declare_class_value(
            "parameterized.global", kwargs, Parameterized._global_declarations
        )

    @staticmethod
    def GLOBALP(**kwargs):
        """
        Declare scalar (compartment-independent) strictly positive parameters.

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            once per instance and broadcast across compartments.
        """
        declare_class_value(
            "parameterized.global_p", kwargs, Parameterized._global_p_declarations
        )

    @staticmethod
    def GLOBALN(**kwargs):
        """
        Declare scalar (compartment-independent) strictly negative parameters.

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            once per instance and broadcast across compartments.
        """
        declare_class_value(
            "parameterized.global_n", kwargs, Parameterized._global_n_declarations
        )

    @staticmethod
    def GLOBAL_SIGNED(**kwargs):
        """
        Declare scalar (compartment-independent) signed parameters.

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            once per instance and broadcast across compartments.
        """
        for k, v in kwargs.items():
            if v > 0:
                declare_class_value(
                    "parameterized.global_p",
                    {k: v},
                    Parameterized._global_p_declarations,
                )
            elif v < 0:
                declare_class_value(
                    "parameterized.global_n",
                    {k: v},
                    Parameterized._global_n_declarations,
                )
            else:
                declare_class_value(
                    "parameterized.global",
                    {k: v},
                    Parameterized._global_declarations,
                )

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
        declare_class_value(
            "parameterized.range", kwargs, Parameterized._range_declarations
        )

    @staticmethod
    def RANGEP(**kwargs):
        """
        Declare per-compartment strictly positive parameters (range variables).

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            with shape matching the population ``shape_p``.
        """
        declare_class_value(
            "parameterized.range_p", kwargs, Parameterized._range_p_declarations
        )

    @staticmethod
    def RANGEN(**kwargs):
        """
        Declare per-compartment strictly negative parameters (range variables).

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            with shape matching the population ``shape_p``.
        """
        declare_class_value(
            "parameterized.range_n", kwargs, Parameterized._range_n_declarations
        )

    @staticmethod
    def BATCH(**kwargs):
        """
        Declare parameters that broadcast over the final compartment dimension.

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            with shape ``shape_p[:-1] + (1,)``.
        """
        declare_class_value(
            "parameterized.batch", kwargs, Parameterized._batch_declarations
        )

    @staticmethod
    def BATCHP(**kwargs):
        """
        Declare strictly positive parameters that broadcast over compartments.

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            with shape ``shape_p[:-1] + (1,)``.
        """
        declare_class_value(
            "parameterized.batch_p", kwargs, Parameterized._batch_p_declarations
        )

    @staticmethod
    def BATCHN(**kwargs):
        """
        Declare strictly negative parameters that broadcast over compartments.

        Parameters
        ----------
        **kwargs
            Mapping of parameter name to default value. Values are instantiated
            with shape ``shape_p[:-1] + (1,)``.
        """
        declare_class_value(
            "parameterized.batch_n", kwargs, Parameterized._batch_n_declarations
        )

    @staticmethod
    def GLOBALRAND(
        name: str,
        *,
        distribution: str = "normal",
        resample_on_initialize: bool = True,
        seed: int | None = None,
        reparameterized: bool = True,
        rng_name: str | None = None,
        **distribution_parameters,
    ):
        """Declare a sampled scalar buffer with parametric distribution parameters."""
        declare_class_value(
            "parameterized.global_rand",
            make_random_parameter_spec(
                name,
                scope="global",
                distribution=distribution,
                resample_on_initialize=resample_on_initialize,
                seed=seed,
                reparameterized=reparameterized,
                rng_name=rng_name,
                **distribution_parameters,
            ),
            Parameterized._global_rand_declarations,
        )

    @staticmethod
    def RANGERAND(
        name: str,
        *,
        distribution: str = "normal",
        resample_on_initialize: bool = True,
        seed: int | None = None,
        reparameterized: bool = True,
        rng_name: str | None = None,
        **distribution_parameters,
    ):
        """Declare a sampled per-compartment buffer.

        Distribution parameters are exposed as ordinary RANGE parameters named
        ``<name>_<distribution_parameter>``. For example,
        ``RANGERAND('rvar', distribution='normal', mu=0, sigma=1)`` declares a
        sampled buffer ``rvar`` plus ``rvar_mu`` and positive ``rvar_sigma``
        range parameters that can be locally overridden at insertion time.
        """
        declare_class_value(
            "parameterized.range_rand",
            make_random_parameter_spec(
                name,
                scope="range",
                distribution=distribution,
                resample_on_initialize=resample_on_initialize,
                seed=seed,
                reparameterized=reparameterized,
                rng_name=rng_name,
                **distribution_parameters,
            ),
            Parameterized._range_rand_declarations,
        )

    @staticmethod
    def BATCHRAND(
        name: str,
        *,
        distribution: str = "normal",
        resample_on_initialize: bool = True,
        seed: int | None = None,
        reparameterized: bool = True,
        rng_name: str | None = None,
        **distribution_parameters,
    ):
        """Declare a sampled buffer with BATCH-shaped distribution parameters."""
        declare_class_value(
            "parameterized.batch_rand",
            make_random_parameter_spec(
                name,
                scope="batch",
                distribution=distribution,
                resample_on_initialize=resample_on_initialize,
                seed=seed,
                reparameterized=reparameterized,
                rng_name=rng_name,
                **distribution_parameters,
            ),
            Parameterized._batch_rand_declarations,
        )

    @staticmethod
    def GLOBALNOISE(
        name: str,
        *,
        distribution: str = "normal",
        seed: int | None = None,
        rng_name: str | None = None,
        cadence: str = "step",
        phase: str = "pre_state",
        scale: str = "standard",
        **distribution_parameters,
    ):
        """Declare a detached scalar runtime-noise buffer.

        Runtime noise is resampled in-place during simulation and does not
        preserve gradients through its distribution parameters.
        """
        declare_class_value(
            "parameterized.global_noise",
            make_runtime_noise_spec(
                name,
                scope="global",
                distribution=distribution,
                seed=seed,
                rng_name=rng_name,
                cadence=cadence,
                phase=phase,
                scale=scale,
                **distribution_parameters,
            ),
            Parameterized._global_noise_declarations,
        )

    @staticmethod
    def RANGENOISE(
        name: str,
        *,
        distribution: str = "normal",
        seed: int | None = None,
        rng_name: str | None = None,
        cadence: str = "step",
        phase: str = "pre_state",
        scale: str = "standard",
        **distribution_parameters,
    ):
        """Declare a detached per-compartment runtime-noise buffer."""
        declare_class_value(
            "parameterized.range_noise",
            make_runtime_noise_spec(
                name,
                scope="range",
                distribution=distribution,
                seed=seed,
                rng_name=rng_name,
                cadence=cadence,
                phase=phase,
                scale=scale,
                **distribution_parameters,
            ),
            Parameterized._range_noise_declarations,
        )

    @staticmethod
    def BATCHNOISE(
        name: str,
        *,
        distribution: str = "normal",
        seed: int | None = None,
        rng_name: str | None = None,
        cadence: str = "step",
        phase: str = "pre_state",
        scale: str = "standard",
        **distribution_parameters,
    ):
        """Declare a detached runtime-noise buffer with BATCH shape."""
        declare_class_value(
            "parameterized.batch_noise",
            make_runtime_noise_spec(
                name,
                scope="batch",
                distribution=distribution,
                seed=seed,
                rng_name=rng_name,
                cadence=cadence,
                phase=phase,
                scale=scale,
                **distribution_parameters,
            ),
            Parameterized._batch_noise_declarations,
        )

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
        declare_class_value(
            "parameterized.rng",
            {**{name: None for name in args}, **kwargs},
            Parameterized._rng_declarations,
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
        implementation (beyond the TABLE declaration); internally, Dendra will
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
        declare_class_value(
            "parameterized.table",
            {func: {"low": low, "high": high, "n": n, "learnable": learnable}},
            Parameterized._table_declarations,
        )

    def __init__(
        self,
        shape,
        shape_f,
        additional_parameters=None,
        *,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._init_device = (
            current_device(torch.device("cpu"))
            if device is None
            else torch.device(device)
        )
        self._init_dtype = (
            current_dtype(torch.float32)
            if dtype is None
            else _normalize_dtype_value(dtype)
        )
        try:
            shape_p = [int(s) for s in shape]
            shape_f = [int(s) for s in shape_f]
            self.shape_p = tuple(shape_p)
            self.shape_f = tuple(shape_f)
        except Exception as e:
            raise TypeError(f"error assigning shape {shape!r}") from e

        self.globals = self.__class__._global.copy()
        self.globals_p = self.__class__._global_p.copy()
        self.range = self.__class__._range.copy()
        self.range_p = self.__class__._range_p.copy()
        self.range_n = self.__class__._range_n.copy()
        self.batch_t = self.__class__._batch.copy()
        self.batch_p = self.__class__._batch_p.copy()
        self.batch_n = self.__class__._batch_n.copy()
        self.global_n = self.__class__._global_n.copy()

        self.rng = self.__class__._rng.copy()
        self.random_parameters = self.__class__._random_parameters.copy()
        self.runtime_noises = self.__class__._runtime_noises.copy()
        self._random_parameter_generation = {}
        self._random_parameter_initialized = {}

        self.in_graph_parametrizations = {}

        if kwargs:
            self.globals = {
                key: kwargs.get(key, value) for key, value in self.globals.items()
            }
            self.range = {
                key: kwargs.get(key, value) for key, value in self.range.items()
            }
            self.batch_t = {
                key: kwargs.get(key, value) for key, value in self.batch_t.items()
            }
            self.globals_p = {
                key: kwargs.get(key, value) for key, value in self.globals_p.items()
            }
            self.range_p = {
                key: kwargs.get(key, value) for key, value in self.range_p.items()
            }
            self.batch_p = {
                key: kwargs.get(key, value) for key, value in self.batch_p.items()
            }
            self.global_n = {
                key: kwargs.get(key, value) for key, value in self.global_n.items()
            }
            self.range_n = {
                key: kwargs.get(key, value) for key, value in self.range_n.items()
            }
            self.batch_n = {
                key: kwargs.get(key, value) for key, value in self.batch_n.items()
            }

        self.keys = {}
        self.additional_parameters = {}
        self.instantiate_global(**self.globals)
        self.instantiate_global(positive=True, **self.globals_p)
        self.instantiate_global(negative=True, **self.global_n)
        self.instantiate_range(**self.range)
        self.instantiate_range(positive=True, **self.range_p)
        self.instantiate_range(negative=True, **self.range_n)
        self.instantiate_batch(**self.batch_t)
        self.instantiate_batch(positive=True, **self.batch_p)
        self.instantiate_batch(negative=True, **self.batch_n)
        self.instantiate_rng(**self.rng)
        self.instantiate_random_parameters(**self.random_parameters)
        self.instantiate_runtime_noises(**self.runtime_noises)
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
        self.instantiate_range(positive=True, **self.range_p)
        self.instantiate_range(negative=True, **self.range_n)
        self.instantiate_batch(**self.batch_t)
        self.instantiate_batch(positive=True, **self.batch_p)
        self.instantiate_batch(negative=True, **self.batch_n)
        self.instantiate_random_parameters(**self.random_parameters)
        self.instantiate_runtime_noises(**self.runtime_noises)

    def _refresh_and_set(self, name, value):
        """
        Remove any existing parameter or buffer with the given name.

        Parameters
        ----------
        name : str
            Name of the parameter or buffer to remove.
        value : Any
            New value to set for the parameter or buffer.
        """
        if hasattr(self, name):
            try:
                self._parameters.pop(name)
            except KeyError:
                pass
            try:
                self._buffers.pop(name)
            except KeyError:
                pass
        setattr(self, name, value)

    def instantiate_global(self, positive=False, negative=False, **kwargs):
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
                        setattr(
                            self,
                            pname,
                            to_param(
                                pval,
                                positive=positive,
                                negative=negative,
                                device=self._init_device,
                                dtype=self._init_dtype,
                            ),
                        )
                        getattr(self, name)[pname] = getattr(self, pname)
                else:
                    p_name = f"{name}_param"
                    self._refresh_and_set(
                        p_name,
                        to_param(
                            value,
                            positive=positive,
                            negative=negative,
                            device=self._init_device,
                            dtype=self._init_dtype,
                        ),
                    )
                    self.register_buffer(
                        name,
                        torch.empty(
                            (), device=self._init_device, dtype=self._init_dtype
                        ),
                    )
                    getattr(self, name).copy_(self.evaluate(p_name))

    def _batch_shape(self):
        if len(self.shape_p) == 0:
            raise ValueError(
                "BATCH parameters require shape_p to have at least one dimension."
            )
        return self.shape_p[:-1] + (1,)

    def _batch_main_shape(self) -> tuple[int, int]:
        return (math.prod(self._batch_shape()[:-1]) or 1, 1)

    def _collapse_batch_key(self, key: torch.LongTensor) -> torch.LongTensor:
        key = torch.as_tensor(key, dtype=torch.long)
        if key.numel() == 0:
            return key
        collapsed = torch.div(key, self.shape_p[-1], rounding_mode="floor")
        # Preserve first-seen ordering so pre-sized tensors map predictably.
        seen = set()
        ordered = []
        for idx in collapsed.detach().cpu().tolist():
            if idx not in seen:
                seen.add(idx)
                ordered.append(idx)
        return torch.as_tensor(ordered, dtype=torch.long, device=collapsed.device)

    def instantiate_range(self, positive=False, negative=False, **kwargs):
        """
        Instantiate range parameters over the population shape.

        Parameters
        ----------
        **kwargs
            Mapping of parameter names to initial values broadcast over ``shape_p``.
        """
        if kwargs is not None:
            for name, value in kwargs.items():
                p_name = f"{name}_param"
                self._refresh_and_set(
                    p_name,
                    to_param(
                        value,
                        positive=positive,
                        negative=negative,
                        device=self._init_device,
                        dtype=self._init_dtype,
                    ),
                )
                self.register_buffer(
                    name,
                    torch.empty(
                        self.shape_p, device=self._init_device, dtype=self._init_dtype
                    ),
                )
                getattr(self, name).copy_(self.evaluate(p_name))

    def instantiate_batch(self, positive=False, negative=False, **kwargs):
        """
        Instantiate parameters that broadcast over the final compartment axis.

        Parameters
        ----------
        **kwargs
            Mapping of parameter names to initial values broadcast over
            ``shape_p[:-1] + (1,)``.
        """
        if kwargs is not None:
            batch_shape = self._batch_shape()
            for name, value in kwargs.items():
                p_name = f"{name}_param"
                self._refresh_and_set(
                    p_name,
                    to_param(
                        value,
                        positive=positive,
                        negative=negative,
                        device=self._init_device,
                        dtype=self._init_dtype,
                    ),
                )
                self.register_buffer(
                    name,
                    torch.empty(
                        batch_shape, device=self._init_device, dtype=self._init_dtype
                    ),
                )
                getattr(self, name).copy_(self.evaluate(p_name))

    def instantiate_rng(self, **kwargs):
        for name, value in kwargs.items():
            rng = RNGModule(value, shape_p=self.shape_p, shape_f=self.shape_f)
            setattr(self, name, rng)

    def instantiate_random_parameters(self, **kwargs):
        """Allocate sampled random-parameter buffers and RNG streams."""
        for name, spec in kwargs.items():
            if not isinstance(spec, RandomParameterSpec):
                raise TypeError(
                    f"Random parameter declarations must be RandomParameterSpec "
                    f"instances; got {type(spec).__name__} for {name!r}."
                )
            shape = self._random_parameter_shape(spec)
            if hasattr(self, name):
                self._buffers.pop(name, None)
            self.register_buffer(
                name,
                torch.empty(shape, device=self._init_device, dtype=self._init_dtype),
            )
            self._random_parameter_generation.pop(name, None)
            self._random_parameter_initialized[name] = False
            if not hasattr(self, spec.effective_rng_name):
                setattr(
                    self,
                    spec.effective_rng_name,
                    RNGModule(spec.seed, shape_p=shape, shape_f=shape),
                )

    def instantiate_runtime_noises(self, **kwargs):
        """Allocate detached runtime-noise buffers and RNG streams."""
        for name, spec in kwargs.items():
            if not isinstance(spec, RuntimeNoiseSpec):
                raise TypeError(
                    f"Runtime noise declarations must be RuntimeNoiseSpec "
                    f"instances; got {type(spec).__name__} for {name!r}."
                )
            shape = self._runtime_noise_shape(spec)
            if hasattr(self, name):
                self._buffers.pop(name, None)
            self.register_buffer(
                name,
                torch.empty(shape, device=self._init_device, dtype=self._init_dtype),
            )
            if not hasattr(self, spec.effective_rng_name):
                setattr(
                    self,
                    spec.effective_rng_name,
                    RNGModule(spec.seed, shape_p=shape, shape_f=shape),
                )

    def _random_parameter_shape(self, spec: RandomParameterSpec):
        return self._stochastic_shape(spec.scope)

    def _runtime_noise_shape(self, spec: RuntimeNoiseSpec):
        return self._stochastic_shape(spec.scope)

    def _stochastic_shape(self, scope: str):
        if scope == "global":
            return ()
        if scope == "range":
            return self.shape_p
        if scope == "batch":
            return self._batch_shape()
        raise ValueError(f"Unknown stochastic scope: {scope!r}")

    def init_rng(self):
        """
        Initialize all RNG modules.
        """
        for name in self.__class__._rng.keys():
            rng_module = getattr(self, name)
            if isinstance(rng_module, RNGModule):
                rng_module.init(self._init_device)
        for spec in self.random_parameters.values():
            rng_module = getattr(self, spec.effective_rng_name)
            if isinstance(rng_module, RNGModule):
                rng_module.init(self._init_device)
        for spec in self.runtime_noises.values():
            rng_module = getattr(self, spec.effective_rng_name)
            if isinstance(rng_module, RNGModule):
                rng_module.init(self._init_device)

    def reset_rng(self):
        """
        Reseed all RNG modules.
        """
        for name in self.__class__._rng.keys():
            rng_module = getattr(self, name)
            if isinstance(rng_module, RNGModule):
                rng_module.reset()
        for spec in self.random_parameters.values():
            rng_module = getattr(self, spec.effective_rng_name)
            if isinstance(rng_module, RNGModule):
                rng_module.reset()
        for spec in self.runtime_noises.values():
            rng_module = getattr(self, spec.effective_rng_name)
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
                positive = (name in self.range_p) or (name in self.batch_p)
                negative = (name in self.range_n) or (name in self.batch_n)
                bounded = positive or negative
                is_range = (
                    name in self.range or name in self.range_p or name in self.range_n
                )
                is_batch = (
                    name in self.batch_t or name in self.batch_p or name in self.batch_n
                )
                if is_range or is_batch:
                    count = 0
                    keys = []
                    param_shape = (
                        self._batch_main_shape() if is_batch else self.shape_p[-2:]
                    )
                    empty_shape = self._batch_shape() if is_batch else self.shape_p
                    for alias, value, key in list_of_aliases_values_and_keys:
                        if alias is not None:
                            p_name = f"{name}_{alias}"
                        else:
                            p_name = f"{name}_param_{count}"
                            count += 1
                        key = torch.as_tensor(key, dtype=torch.long)
                        if is_batch:
                            key = self._collapse_batch_key(key)
                        parameter = to_param(
                            value,
                            positive=positive,
                            negative=negative,
                            device=self._init_device,
                            dtype=self._init_dtype,
                        )
                        if isinstance(parameter, torch.nn.Module) and not bounded:
                            parameter = parameter.to(
                                device=self._init_device, dtype=self._init_dtype
                            )
                            p = parameter(
                                torch.empty(
                                    empty_shape,
                                    device=self._init_device,
                                    dtype=self._init_dtype,
                                )
                            )
                            parametrization = build_parametrization(
                                parameter, p, key, param_shape
                            )
                            self.register_parametrization_in_graph(
                                name, parametrization
                            )
                            setattr(self, p_name, parameter)
                        else:
                            setattr(self, p_name, parameter)
                            if bounded:
                                p = parameter()
                            else:
                                p = parameter
                            fill = create_param_expander(p, key, param_shape)
                            self.additional_parameters.setdefault(name, []).append(
                                (fill, getattr(self, p_name))
                            )
                            keys.append(key)
                    if keys:
                        self.keys[name] = torch.cat(keys).to(torch.long)

    def load_additional_parameters(self):
        """
        Scatter alias-specific parameter overrides into their buffers.
        """
        for name, list_of_parameters in self.additional_parameters.items():
            buffer = getattr(self, name)
            additional_params = torch.cat(
                [fill(self.resolve(p)) for fill, p in list_of_parameters]
            ).to(device=buffer.device, dtype=buffer.dtype)
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
        >>> print(model.rhoa_param_0)  # Access the override parameter
        tensor(150.)
        """
        if key is None:
            return self.parametrize(
                name, value, key=torch.arange(math.prod(self.shape_p)), alias=alias
            )
        is_range = name in self.range or name in self.range_p or name in self.range_n
        is_batch = name in self.batch_t or name in self.batch_p or name in self.batch_n
        if not (is_range or is_batch):
            raise KeyError(
                f"Unknown or non-indexable parameter {name!r}. Expected a declared "
                "RANGE or BATCH parameter."
            )
        if is_range or is_batch:
            positive = (name in self.range_p) or (name in self.batch_p)
            negative = (name in self.range_n) or (name in self.batch_n)
            if alias is None:
                count = 0
                while hasattr(self, f"{name}_param_{count}"):
                    count += 1
                alias = f"param_{count}"
            p_name = f"{name}_{alias}"
            if hasattr(self, p_name):
                raise ValueError(
                    f"Parameter override '{p_name}' already exists. Choose a different alias."
                )
            key = torch.as_tensor(key, dtype=torch.long)
            main_shape = self._batch_main_shape() if is_batch else self.shape_p[-2:]
            empty_shape = self._batch_shape() if is_batch else self.shape_p
            if is_batch:
                key = self._collapse_batch_key(key)
            target = getattr(self, name)
            key = key.to(device=target.device)
            existing_key = self.keys.get(name)
            combined_key = (
                key.to(torch.long)
                if existing_key is None
                else torch.cat((existing_key.to(device=key.device), key.to(torch.long)))
            )
            parameter = to_param(
                value,
                positive=positive,
                negative=negative,
                device=target.device,
                dtype=target.dtype,
            )
            if isinstance(parameter, torch.nn.Module) and not (positive or negative):
                parameter = parameter.to(device=target.device, dtype=target.dtype)
                p = parameter(
                    torch.empty(empty_shape, device=target.device, dtype=target.dtype)
                )
                parametrization = build_parametrization(parameter, p, key, main_shape)
                self.register_parametrization_in_graph(name, parametrization)
                setattr(self, p_name, parameter)
            else:
                p = parameter() if (positive or negative) else parameter
                fill = create_param_expander(p, key, main_shape)
                setattr(self, p_name, parameter)
                if name not in self.additional_parameters:
                    self.additional_parameters[name] = []
                self.additional_parameters[name].append((fill, getattr(self, p_name)))
                self.keys[name] = combined_key

    def populate_parameter_buffers(self, random_generation=None):
        """
        Reset parameter buffers to defaults, then apply overrides and parametrizations.
        """
        keys_to_process = itertools.chain(
            self.__class__._global.keys(),
            self.__class__._range.keys(),
            self.__class__._batch.keys(),
            self.__class__._global_p.keys(),
            self.__class__._range_p.keys(),
            self.__class__._batch_p.keys(),
            self.__class__._global_n.keys(),
            self.__class__._range_n.keys(),
            self.__class__._batch_n.keys(),
        )
        for name in keys_to_process:
            if not torch.is_tensor(getattr(self, name)):
                continue
            if hasattr(self, "parametrizations"):
                if name in self.parametrizations:
                    # If the parameter has parametrizations, we skip it
                    continue
            p_name = f"{name}_param"
            setattr(self, name, getattr(self, name).detach())
            getattr(self, name).copy_(self.evaluate(p_name))
        self.load_additional_parameters()
        self.apply_parametrizations()
        self._sample_random_parameters(random_generation=random_generation)
        self.sample_runtime_noises_(force=True, phase=None, dt=1.0)
        self.make_contiguous()

    def _sample_random_parameters(
        self, names=None, *, force: bool = False, random_generation=None
    ):
        if not self.random_parameters:
            return False
        selected = (
            tuple(self.random_parameters) if not names else tuple(str(n) for n in names)
        )
        any_sampled = False
        for name in selected:
            if name not in self.random_parameters:
                raise KeyError(
                    f"Unknown random parameter {name!r}. Available random parameters: "
                    f"{tuple(self.random_parameters)}."
                )
            spec = self.random_parameters[name]
            initialized = self._random_parameter_initialized.get(name, False)
            if not force:
                if not spec.resample_on_initialize and initialized:
                    continue
                if spec.resample_on_initialize and random_generation is not None:
                    previous = self._random_parameter_generation.get(name, None)
                    if previous is random_generation or previous == random_generation:
                        continue
            buffer = getattr(self, name)
            rng = getattr(self, spec.effective_rng_name)
            if isinstance(rng, RNGModule):
                rng.init(buffer.device)
            distribution_params = {
                p: getattr(self, spec.full_parameter_name(p)) for p in spec.params
            }
            sample = sample_random_parameter(
                spec,
                distribution_params,
                rng,
                tuple(buffer.shape),
                device=buffer.device,
                dtype=buffer.dtype,
            )
            setattr(self, name, sample.to(device=buffer.device, dtype=buffer.dtype))
            self._random_parameter_initialized[name] = True
            if random_generation is not None:
                self._random_parameter_generation[name] = random_generation
            any_sampled = True
        return any_sampled

    def resample_random_parameters(self, *names, force: bool = True):
        """Regenerate sampled random-parameter buffers."""
        self._sample_random_parameters(names or None, force=force)
        return self

    def sample_runtime_noises_(
        self,
        *names,
        dt=None,
        phase: str | None = "pre_state",
        step_index: int | None = None,
        force: bool = False,
    ):
        """Resample detached runtime-noise buffers in-place.

        This hot path intentionally runs under ``torch.no_grad()`` and preserves
        buffer identity. Gradients do not flow through runtime-noise distribution
        parameters.
        """
        if not self.runtime_noises:
            return False
        selected = (
            tuple(self.runtime_noises) if not names else tuple(str(n) for n in names)
        )
        phase_norm = None if phase is None else str(phase).strip().lower()
        any_sampled = False
        with torch.no_grad():
            for name in selected:
                if name not in self.runtime_noises:
                    raise KeyError(
                        f"Unknown runtime noise {name!r}. Available runtime noises: "
                        f"{tuple(self.runtime_noises)}."
                    )
                spec = self.runtime_noises[name]
                if phase_norm is not None and spec.phase != phase_norm:
                    continue
                buffer = getattr(self, name)
                rng = getattr(self, spec.effective_rng_name)
                if isinstance(rng, RNGModule):
                    rng.init(buffer.device)
                distribution_params = {
                    p: getattr(self, spec.full_parameter_name(p)) for p in spec.params
                }
                sample = sample_runtime_noise(
                    spec,
                    distribution_params,
                    rng,
                    tuple(buffer.shape),
                    device=buffer.device,
                    dtype=buffer.dtype,
                    dt=dt,
                )
                buffer.copy_(sample.to(device=buffer.device, dtype=buffer.dtype))
                any_sampled = True
        return any_sampled

    def resample_runtime_noise(self, *names, dt=None, phase=None):
        """Explicitly resample detached runtime-noise buffers."""
        self.sample_runtime_noises_(*names, dt=dt, phase=phase, force=True)
        return self

    def make_contiguous(self):
        """
        Ensure all parameter buffers are contiguous in memory.
        """
        for name in self.__class__._global.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.__class__._range.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.__class__._batch.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.__class__._global_p.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.__class__._range_p.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.__class__._batch_p.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.__class__._global_n.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.__class__._range_n.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.__class__._batch_n.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.random_parameters.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())
        for name in self.runtime_noises.keys():
            b = getattr(self, name)
            if torch.is_tensor(b) and not b.is_contiguous():
                setattr(self, name, b.contiguous())

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

    def parameters_dict(self, clone=True, trainable_only=False):
        """
        Returns a dictionary of all parameters in the model.
        """
        with torch.no_grad():
            if clone:
                if trainable_only:
                    dct = {
                        name: param.clone()
                        for name, param in self.named_parameters()
                        if param.requires_grad
                    }
                else:
                    dct = {
                        name: param.clone() for name, param in self.named_parameters()
                    }
            else:
                if trainable_only:
                    dct = {
                        name: param
                        for name, param in self.named_parameters()
                        if param.requires_grad
                    }
                else:
                    dct = {name: param for name, param in self.named_parameters()}
        return dct

    def load_parameters_dict(self, parameters, strict=True):
        """
        Load parameters from a dictionary.

        Parameters
        ----------
        parameters : dict
            Mapping of parameter names to tensors. The tensors are copied into the model's parameters.
        strict : bool, optional
            If True, raises an error if a parameter in the model is not found in the dictionary.
        """
        named_parameters = dict(self.named_parameters())
        if strict:
            unexpected = sorted(set(parameters) - set(named_parameters))
            if unexpected:
                raise KeyError(
                    f"Unexpected parameter name(s) in provided dictionary: {unexpected}."
                )
            missing = sorted(set(named_parameters) - set(parameters))
            if missing:
                raise KeyError(
                    f"Parameter name(s) not found in the provided dictionary: {missing}."
                )

        updates = []
        for name, param in named_parameters.items():
            if name not in parameters:
                continue
            value = parameters[name]
            if not torch.is_tensor(value):
                raise TypeError(f"Parameter {name!r} must be loaded from a tensor.")
            if value.shape != param.shape:
                raise ValueError(
                    f"Parameter {name!r} has shape {tuple(param.shape)}, but the "
                    f"provided tensor has shape {tuple(value.shape)}."
                )
            # Stage an independent source snapshot before mutating any target.
            # This makes cross-parameter swaps deterministic even when callers
            # pass live model parameters as the input dictionary values.
            updates.append((param, value.detach().clone()))

        with torch.no_grad():
            backups = [(param, param.clone()) for param, _ in updates]
            try:
                for param, value in updates:
                    param.copy_(value)
            except Exception:
                for param, backup in backups:
                    param.copy_(backup)
                for module in self.modules():
                    if isinstance(module, cacheable):
                        module.clear_cache()
                raise
        for module in self.modules():
            if isinstance(module, cacheable):
                module.clear_cache()

    @classmethod
    def normalize_random_kwargs(cls, kwargs):
        """Normalize insertion kwargs for random distribution parameters.

        Canonical names such as ``rvar_mu`` and ``rvar_sigma`` are left
        unchanged. Bare distribution-parameter names such as ``mu`` or
        ``sigma`` are accepted only when they uniquely identify one declared
        random variable and do not collide with an ordinary declared parameter.
        """
        if not kwargs:
            return dict(kwargs or {})
        specs = []
        specs.extend(getattr(cls, "_random_parameters", {}).values())
        specs.extend(getattr(cls, "_runtime_noises", {}).values())
        if not specs:
            return dict(kwargs or {})
        ordinary_names = set(cls.all_parameter_names())
        bare_to_full = {}
        for spec in specs:
            for param_name in spec.params:
                bare_to_full.setdefault(param_name, []).append(
                    spec.full_parameter_name(param_name)
                )
        out = {}
        for key, value in dict(kwargs).items():
            if key in ordinary_names or key not in bare_to_full:
                out[key] = value
                continue
            full_names = bare_to_full[key]
            if len(full_names) != 1:
                raise ValueError(
                    f"Ambiguous random distribution override {key!r}. Use one of {full_names!r}."
                )
            full_name = full_names[0]
            if full_name in out:
                raise ValueError(
                    f"Both {key!r} and {full_name!r} were provided; use only the canonical random-parameter name."
                )
            out[full_name] = value
        return out

    @classmethod
    def all_parameter_names(cls):
        """
        Returns a list of all parameter names in the model.
        """
        ordered_names = itertools.chain(
            cls._params.keys(),
            cls._params_p.keys(),
            cls._params_n.keys(),
            cls._flags.keys(),
            cls._global.keys(),
            cls._global_p.keys(),
            cls._global_n.keys(),
            cls._batch.keys(),
            cls._batch_p.keys(),
            cls._batch_n.keys(),
            cls._range.keys(),
            cls._range_p.keys(),
            cls._range_n.keys(),
        )
        return list(dict.fromkeys(ordered_names))

    @classmethod
    def random_parameter_names(cls):
        """Return sampled random-parameter buffer names declared by this class."""
        return list(getattr(cls, "_random_parameters", {}).keys())

    @classmethod
    def random_distribution_parameter_names(cls):
        """Return generated distribution-parameter names for random/RUNTIME buffers."""
        names = []
        for spec in getattr(cls, "_random_parameters", {}).values():
            names.extend(spec.parameter_names)
        for spec in getattr(cls, "_runtime_noises", {}).values():
            names.extend(spec.parameter_names)
        return list(dict.fromkeys(names))

    @classmethod
    def runtime_noise_names(cls):
        """Return runtime-noise buffer names declared by this class."""
        return list(getattr(cls, "_runtime_noises", {}).keys())

    def has_random_parameters(self):
        """Whether this object owns any sampled random-parameter buffers."""
        return bool(getattr(self, "random_parameters", None))

    def has_runtime_noises(self):
        """Whether this object owns detached runtime-noise buffers."""
        return bool(getattr(self, "runtime_noises", None))

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
    meth = compile_generated_function(
        func_code,
        func_name=f"{func_name}_with_table",
        filename_prefix=f"dendra.parametric.{func_name}_with_table",
        global_ns=globals(),
    )
    setattr(obj, f"{func_name}_original", getattr(obj, func_name))
    setattr(obj, func_name, MethodType(meth, obj))
