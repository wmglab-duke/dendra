from __future__ import annotations

import copy
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import nn

OutsideMode = Literal["clamp", "zero", "fill"]
OutsideMode3D = Literal["none", "zero", "fill"]
KNNBackend = Literal["auto", "torch", "pytorch3d", "torch_cluster", "faiss"]
BoundaryMode = Literal["replicate", "zero", "one-sided", "valid"]
IndexLike = Union[int, Sequence[int], torch.Tensor]


class PreparedInterp1d(nn.Module):
    """
    Prepared 1D piecewise-linear interpolator with optional learnable ordinates.

    This module implements fast, GPU-friendly 1D linear interpolation for either a
    single lookup table (unbatched) or a batch of lookup tables (batched). It
    performs all *static* preparation once at initialization, including optional
    sorting of `x` (and corresponding permutation of `y`) to satisfy
    ``torch.searchsorted`` preconditions.

    Supported regimes (enforced)
    ----------------------------
    Exactly one of the following input regimes is supported:

    1) **Unbatched**:
       ``x`` and ``y`` are both 1D arrays of shape ``(N,)`` representing a single
       tabulated function.

    2) **Batched**:
       ``x`` and ``y`` are both 2D arrays of shape ``(D, N)`` representing a batch
       of ``D`` independent tabulated functions.

    Query points (`x_new`) are accepted in both regimes as either:
    - 1D array of shape ``(P,)``
    - 2D array of shape ``(Q, P)``

    Disambiguation via `indices`
    ----------------------------
    In batched mode, some `x_new` shapes are ambiguous with respect to which LUT row
    should be used. This module resolves ambiguity as follows:

    - If ``x, y`` are ``(D, N)`` and ``x_new`` is ``(D, P)``, then the mapping is
      unambiguous and the default behavior is **row-wise** alignment
      (row ``d`` of `x_new` uses row ``d`` of `x, y`).

    - Otherwise, the user must provide `indices` to specify the mapping:

      * If ``x_new`` is ``(Q, P)``, then ``indices`` must have length ``Q`` and
        maps each query row ``q`` to a LUT row ``indices[q]`` in ``[0, D-1]``.

      * If ``x_new`` is ``(P,)`` (internally treated as ``(1, P)``), then
        ``indices`` selects which LUT rows to evaluate that same query against.
        If ``indices`` has length ``L``, the output will be shaped ``(L, P)``.

    Out-of-bounds behavior
    ----------------------
    Values outside the domain of `x` are handled according to `outside`:

    - ``outside="clamp"``: constant extension using the endpoint values
      (left of domain -> first `y`, right of domain -> last `y`).
    - ``outside="zero"``: out-of-domain values are set to 0.
    - ``outside="fill"``: out-of-domain values are set to `fill_value`.

    Differentiability w.r.t. `y`
    ----------------------------
    If ``learnable_y=True``, `y` is stored as an ``nn.Parameter`` and interpolation
    remains differentiable with respect to `y` (piecewise linear in `y`). In this
    mode, slopes are **not** precomputed as constants; instead, `y[i]` and `y[i+1]`
    are gathered per query and combined with a precomputed inverse `dx` derived
    from `x`.

    Notes
    -----
    - This module relies on ``torch.searchsorted`` for interval selection. The
      selected interval index is a discrete operation; gradients do not propagate
      through changes in the selected interval.
    - If `sort_xy=True`, `x` is sorted along its last dimension at initialization
      and `y` is permuted to match. This ensures `searchsorted` is valid even if
      the user provides unsorted `x`.
    - Duplicate `x` entries are handled by replacing zero-length segments with an
      epsilon in the denominator. This avoids division-by-zero but does not make
      the interpolation well-defined at duplicates; consider de-duplicating if
      needed.

    Examples
    --------
    Unbatched LUT, 1D query:

    >>> x = torch.tensor([0., 1., 2.], device="cuda")
    >>> y = torch.tensor([0., 1., 0.], device="cuda")
    >>> interp = PreparedInterp1d(x, y, outside="clamp", sort_xy=True).cuda()
    >>> xq = torch.tensor([-1., 0.5, 3.0], device="cuda")
    >>> yq = interp(xq)  # shape (3,)

    Batched LUTs, aligned 2D query:

    >>> x = torch.stack([torch.linspace(0, 1, 5), torch.linspace(0, 2, 5)]).cuda()
    >>> y = torch.randn_like(x)
    >>> interp = PreparedInterp1d(x, y, outside="zero").cuda()
    >>> xq = torch.randn(2, 16, device="cuda")  # (D,P) -> aligned
    >>> yq = interp(xq)  # (2,16)

    Batched LUTs, ambiguous query resolved by indices:

    >>> xq = torch.randn(7, 16, device="cuda")       # (Q,P), Q != D
    >>> idx = torch.randint(0, 2, (7,), device="cuda")
    >>> yq = interp(xq, indices=idx)                 # (7,16)

    Learnable y:

    >>> interp = PreparedInterp1d(x, y, learnable_y=True).cuda()
    >>> yq = interp(xq)
    >>> loss = yq.square().mean()
    >>> loss.backward()  # gradients flow to interp.y
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        outside: OutsideMode = "clamp",
        fill_value: float = 0.0,
        learnable_y: bool = False,
        y_requires_grad: bool = True,
        sort_xy: bool = True,
        eps: Optional[float] = None,
        exact_clamp: bool = True,
        # --- uniform fast paths ---
        uniform: Literal["auto", "never", "always"] = "auto",
        uniform_rtol: float = 1e-5,
        uniform_atol: float = 1e-7,
    ):
        """
        Initialize and prepare the interpolator.

        Parameters
        ----------
        x : torch.Tensor
            Abscissae (x-coordinates) of the lookup table(s).

            Must be either:
            - shape ``(N,)`` in unbatched mode, or
            - shape ``(D, N)`` in batched mode.

            If `sort_xy=True`, `x` may be unsorted; it will be sorted internally.
            If `sort_xy=False`, `x` must already be sorted ascending along its last
            dimension for ``torch.searchsorted`` to behave correctly.

        y : torch.Tensor
            Ordinates (y-coordinates) of the lookup table(s), same shape as `x`
            (either ``(N,)`` or ``(D, N)``). If `sort_xy=True`, `y` is permuted to
            remain aligned with sorted `x`.

        outside : {"clamp", "zero", "fill"}, default="clamp"
            Policy for handling query points outside the domain of `x`.

            - "clamp": return endpoint values (constant extension).
            - "zero":  return 0 outside domain.
            - "fill":  return `fill_value` outside domain.

        fill_value : float, default=0.0
            Fill value used when ``outside="fill"``.

        learnable_y : bool, default=False
            If True, `y` is stored as an ``nn.Parameter`` and the forward pass is
            differentiable with respect to `y`. If False, `y` is treated as constant
            and stored as a buffer.

        y_requires_grad : bool, default=True
            Only relevant when `learnable_y=True`. Sets ``requires_grad`` on the
            internally stored parameter.

        sort_xy : bool, default=True
            If True, sort `x` along its last dimension at initialization, and permute
            `y` identically. This guarantees valid behavior of ``torch.searchsorted``.

        eps : float or None, default=None
            Epsilon used to replace zero-length segments when computing ``1/dx``.
            If None, uses ``torch.finfo(dtype).eps`` for `x`'s dtype.

        exact_clamp : bool, default=True
            Only relevant when ``outside="clamp"`` (either as default or overridden
            in `forward`). If True, endpoint clamping is enforced with explicit
            comparisons (``<= x_min`` and ``>= x_max``) to guarantee constant extension
            semantics even in edge cases. If False, `x_new` is clamped into the domain
            prior to interval selection; this is typically sufficient and can save two
            ``torch.where`` operations.

        uniform : {"auto", "never", "always"}, default="auto"
            Whether to check for uniform spacing in `x` and use a fast path if so.
            - "auto": check for uniform spacing and use fast path if detected.
            - "never": always use the general (non-uniform) path.
            - "always": assume `x` is uniformly spaced; raise an error if not.

        uniform_rtol : float, default=1e-5
            Relative tolerance used when checking for uniform spacing in `x` rows.

        uniform_atol : float, default=1e-7
            Absolute tolerance used when checking for uniform spacing in `x` rows.

        Raises
        ------
        ValueError
            If `x` and `y` do not share the same dimensionality (both 1D or both 2D),
            or shapes are inconsistent (e.g., `x` is (D,N) but `y` is not).
        TypeError
            If tensors are not floating-point.
        """
        super().__init__()

        if outside not in ("clamp", "zero", "fill"):
            raise ValueError("outside must be one of: 'clamp', 'zero', 'fill'")

        if x.ndim != y.ndim or x.ndim not in (1, 2):
            raise ValueError(
                "x and y must have the same ndim, and be either both 1D or both 2D."
            )
        if x.device != y.device:
            raise ValueError("x and y must be on the same device.")
        if x.dtype != y.dtype:
            raise ValueError("x and y must have the same dtype.")
        if not torch.is_floating_point(x):
            raise TypeError(
                "x and y must be floating-point tensors for linear interpolation."
            )

        # Enforce shapes
        if x.ndim == 1:
            if x.shape != y.shape:
                raise ValueError("For 1D mode, x and y must have the same shape (N,).")
            self.batched = False
            self.D = 1
            self.N = int(x.shape[0])
        else:
            if x.shape != y.shape:
                raise ValueError("For 2D mode, x and y must have the same shape (D,N).")
            self.batched = True
            self.D = int(x.shape[0])
            self.N = int(x.shape[1])

        if self.N < 2:
            raise ValueError("Need N >= 2 points for interpolation.")

        self.outside = outside
        self.fill_value = float(fill_value)
        self.learnable_y = bool(learnable_y)
        self.exact_clamp = bool(exact_clamp)

        if uniform not in ("auto", "never", "always"):
            raise ValueError("uniform must be 'auto', 'never', or 'always'")

        # ---- sort x and permute y (optional) ----
        if sort_xy:
            if x.ndim == 1:
                x_sorted, perm = torch.sort(x, dim=0)
                y_sorted = y.index_select(0, perm)
            else:
                x_sorted, perm = torch.sort(x, dim=1)
                y_sorted = torch.gather(y, dim=1, index=perm)
        else:
            x_sorted = x.contiguous()
            y_sorted = y.contiguous()

        # ---- store x-derived buffers ----
        if not self.batched:
            self.register_buffer("_x_search", x_sorted.detach())  # (N,)
            self.register_buffer("_x0", x_sorted[:-1].detach())  # (N-1,)
            self.register_buffer("_x_min", x_sorted[0].detach())  # scalar
            self.register_buffer("_x_max", x_sorted[-1].detach())  # scalar
            dx = x_sorted[1:] - x_sorted[:-1]  # (N-1,)
        else:
            self.register_buffer("_x_search", x_sorted.detach())  # (D,N)
            self.register_buffer("_x0", x_sorted[:, :-1].detach())  # (D,N-1)
            self.register_buffer("_x_min", x_sorted[:, :1].detach())  # (D,1)
            self.register_buffer("_x_max", x_sorted[:, -1:].detach())  # (D,1)
            dx = x_sorted[:, 1:] - x_sorted[:, :-1]  # (D,N-1)

        if eps is None:
            eps = float(torch.finfo(dx.dtype).eps)

        # ---- uniform fast-path detection (per row) ----
        self._use_uniform = False
        if uniform != "never":
            if not self.batched:
                dx0 = dx[:1]  # (1,)
                uniform_ok = bool(torch.all(dx0 > 0).item()) and torch.allclose(
                    dx, dx0.expand_as(dx), rtol=uniform_rtol, atol=uniform_atol
                )
                if uniform == "always" and not uniform_ok:
                    raise ValueError("uniform='always' but x is not uniformly spaced.")
                if uniform_ok:
                    self._use_uniform = True
                    safe_dx0 = torch.where(dx0 == 0, torch.full_like(dx0, eps), dx0)
                    inv_dx_u = (1.0 / safe_dx0)[0]  # scalar
                    dx_u = safe_dx0[0]  # scalar
                    self.register_buffer("_inv_dx_uniform", inv_dx_u.detach())
                    self.register_buffer("_dx_uniform", dx_u.detach())
            else:
                dx0 = dx[:, :1]  # (D,1)
                uniform_ok = bool(torch.all(dx0 > 0).item()) and torch.allclose(
                    dx, dx0.expand_as(dx), rtol=uniform_rtol, atol=uniform_atol
                )
                if uniform == "always" and not uniform_ok:
                    raise ValueError(
                        "uniform='always' but x is not uniformly spaced per row."
                    )
                if uniform_ok:
                    self._use_uniform = True
                    safe_dx0 = torch.where(dx0 == 0, torch.full_like(dx0, eps), dx0)
                    inv_dx_u = (1.0 / safe_dx0).contiguous()
                    dx_u = safe_dx0.contiguous()
                    self.register_buffer("_inv_dx_uniform", inv_dx_u.detach())
                    self.register_buffer("_dx_uniform", dx_u.detach())

        # ---- per-segment inv_dx (used for non-uniform path) ----
        safe_dx = torch.where(dx == 0, torch.full_like(dx, eps), dx)
        inv_dx = (1.0 / safe_dx).contiguous()
        self.register_buffer("_inv_dx", inv_dx.detach())
        self._ind_hi = self.N - 2

        # ---- store y and (optionally) precompute tables ----
        if self.learnable_y:
            self.y = nn.Parameter(y_sorted, requires_grad=y_requires_grad)
        else:
            if not self.batched:
                self.register_buffer("_y", y_sorted.detach())  # (N,)
                self.register_buffer("_y0", y_sorted[:-1].detach())  # (N-1,)
                slopes = (y_sorted[1:] - y_sorted[:-1]) * inv_dx  # (N-1,)
                self.register_buffer("_slopes", slopes.contiguous().detach())
                self.register_buffer("_y_first", y_sorted[0].detach())  # scalar
                self.register_buffer("_y_last", y_sorted[-1].detach())  # scalar
            else:
                self.register_buffer("_y", y_sorted.detach())  # (D,N)
                self.register_buffer("_y0", y_sorted[:, :-1].detach())  # (D,N-1)
                slopes = (y_sorted[:, 1:] - y_sorted[:, :-1]) * inv_dx  # (D,N-1)
                self.register_buffer("_slopes", slopes.contiguous().detach())
                self.register_buffer("_y_first", y_sorted[:, :1].detach())  # (D,1)
                self.register_buffer("_y_last", y_sorted[:, -1:].detach())  # (D,1)

        self._ind_cache: Optional[torch.Tensor] = None

    def _as_index_tensor(
        self,
        indices: IndexLike,
        *,
        device: torch.device,
        expected_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Convert and validate `indices` into a 1D LongTensor.

        Parameters
        ----------
        indices : int or sequence of int or torch.Tensor
            Row indices specifying which LUT row(s) should be used for each query
            row in `x_new`. See `forward` for exact semantics by shape regime.
        device : torch.device
            Target device to place the returned tensor on.
        expected_len : int or None, default=None
            If not None, require that `indices` has exactly this length.

        Returns
        -------
        idx : torch.Tensor
            A 1D tensor of dtype ``torch.long`` on `device`.

        Raises
        ------
        ValueError
            If `indices` is empty, not 1D after conversion, has incorrect length
            when `expected_len` is provided, or contains out-of-range values.
        """
        if isinstance(indices, int):
            idx = torch.tensor([indices], device=device, dtype=torch.long)
        else:
            idx = torch.as_tensor(indices, device=device, dtype=torch.long)

        if idx.ndim != 1:
            raise ValueError("indices must be a 1D sequence/tensor (or a single int).")
        if idx.numel() == 0:
            raise ValueError("indices must be non-empty.")
        if expected_len is not None and idx.numel() != expected_len:
            raise ValueError(
                f"indices must have length {expected_len}, got {idx.numel()}."
            )

        if self.batched:
            if torch.any((idx < 0) | (idx >= self.D)):
                raise ValueError(f"indices values must be in [0, {self.D - 1}].")
        else:
            if torch.any(idx != 0):
                raise ValueError("In 1D mode (x,y 1D), indices can only contain 0.")

        return idx

    def forward(
        self,
        x_new: torch.Tensor,
        *,
        indices: Optional[IndexLike] = None,
        outside: Optional[OutsideMode] = None,
        fill_value: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpolate at query points `x_new`.

        Parameters
        ----------
        x_new : torch.Tensor
            Query points at which to evaluate the interpolant(s).

            Accepted shapes:

            - ``(P,)``: a single query vector
            - ``(Q, P)``: a batch of query vectors

            In unbatched LUT mode (x,y are 1D), both shapes are always unambiguous.

            In batched LUT mode (x,y are 2D), shape compatibility determines whether
            an explicit `indices` mapping is required. See `indices`.

        indices : int or sequence of int or torch.Tensor, optional
            Mapping from query rows in `x_new` to LUT rows in `x`/`y` (batched mode).

            - If x,y are 1D: `indices` is optional; if provided, it must be all zeros.

            - If x,y are 2D (D,N):

              * If ``x_new`` is ``(D, P)`` and `indices` is None:
                row-wise alignment is used (row d uses LUT row d).
              * If ``x_new`` is ``(Q, P)`` and `Q != D`:
                `indices` must be length Q, where `indices[q]` selects which LUT row
                to use for query row q.
              * If ``x_new`` is ``(P,)``:
                `indices` must be provided and selects which LUT rows to evaluate the
                same query vector against. If `indices` has length L, output is (L,P).

            Values in `indices` must be within ``[0, D-1]``.

        outside : {"clamp", "zero", "fill"}, optional
            Override the module's default out-of-bounds policy for this call.

        fill_value : float, optional
            Override the module's default fill value for this call when
            ``outside="fill"``.

        out : torch.Tensor, optional
            Optional output tensor to write results into. Must have exactly the same
            shape as the computed output.


        Returns
        -------
        y_new : torch.Tensor
            Interpolated values.

            **Output shape**

            Let:

            - ``P = x_new.shape[-1]``
            - ``D`` be the LUT batch size when ``x, y`` are batched (i.e., ``x, y`` have
              shape ``(D, N)``)
            - If ``x_new`` is 2D, define ``Q = x_new.shape[0]``
            - If ``indices`` is provided, define ``L = len(indices)`` (after conversion
              to a 1D index tensor)

            The output shape is determined as follows:

            **1) Unbatched LUT mode** (``x, y`` are 1D with shape ``(N,)``)

            - If ``x_new`` has shape ``(P,)``, output has shape ``(P,)``

            - If ``x_new`` has shape ``(Q, P)``, output has shape ``(Q, P)``

            (In unbatched mode, `indices` is never required; if provided it must be all zeros.)

            **2) Batched LUT mode** (``x, y`` are 2D with shape ``(D, N)``)

            - If ``x_new`` has shape ``(D, P)`` and ``indices is None``:

              - output has shape ``(D, P)`` (Row-wise alignment: query row ``d`` uses LUT row ``d``.)

            - If ``x_new`` has shape ``(Q, P)`` and ``indices`` is provided with length ``Q``:

              - output has shape ``(Q, P)`` (Query row ``q`` uses LUT row ``indices[q]``.)

            - If ``x_new`` has shape ``(Q, P)`` with ``Q != D`` and ``indices is None``: **error** (ambiguous mapping)

            - If ``x_new`` has shape ``(P,)``:

              - ``indices`` is **required** (otherwise ambiguous)
              - The same query vector is evaluated against each selected LUT row.

              Let ``L = len(indices)``:

              - If ``L == 1``, output has shape ``(P,)`` (the module returns 1D when the input was 1D
                and only one effective query row is evaluated)
              - If ``L > 1``, output has shape ``(L, P)``

            **Forcing a 2D output for a single selected LUT row**

            If you want a 2D output of shape ``(1, P)`` when selecting a single LUT row,
            pass ``x_new`` as 2D (``(1, P)``) rather than 1D (``(P,)``), e.g.::

                y = interp(x_new[None, :], indices=[k])  # -> (1, P)


        Raises
        ------
        ValueError
            If `x_new` is not 1D/2D, has incompatible batch dimension in batched mode
            without `indices`, or if `indices` is missing or malformed in ambiguous cases.
        TypeError
            If `x_new` is not floating-point.
        """
        outside = self.outside if outside is None else outside
        if outside not in ("clamp", "zero", "fill"):
            raise ValueError("outside must be one of: 'clamp', 'zero', 'fill'")

        fill_value = self.fill_value if fill_value is None else float(fill_value)

        if x_new.ndim not in (1, 2):
            raise ValueError("x_new must be 1D (P,) or 2D (Q,P).")
        if x_new.device != self._x_search.device:
            raise ValueError("x_new must be on the same device as the interpolator.")
        if x_new.dtype != self._x_search.dtype:
            raise ValueError(
                "x_new must have the same dtype as x/y used for initialization."
            )
        if not torch.is_floating_point(x_new):
            raise TypeError("x_new must be floating-point.")

        xnew_was_1d = x_new.ndim == 1
        xq = x_new[None, :] if xnew_was_1d else x_new
        Q = int(xq.shape[0])

        # ------------------------
        # Unbatched LUT mode
        # ------------------------
        if not self.batched:
            if indices is not None:
                _ = self._as_index_tensor(indices, device=xq.device, expected_len=Q)

            x_min = self._x_min
            x_max = self._x_max

            xq_used = (
                torch.maximum(torch.minimum(xq, x_max), x_min)
                if outside == "clamp"
                else xq
            )

            if (
                self._ind_cache is None
                or self._ind_cache.shape != xq_used.shape
                or self._ind_cache.device != xq_used.device
            ):
                self._ind_cache = torch.empty(
                    xq_used.shape, device=xq_used.device, dtype=torch.long
                )
            ind = self._ind_cache

            if self._use_uniform:
                # ---- uniform fast-path: no searchsorted ----
                u = (xq_used - x_min) * self._inv_dx_uniform
                ind.copy_(u.to(torch.long))
                ind.clamp_(0, self._ind_hi)

                if self.learnable_y:
                    y = self.y  # (N,)
                    y0 = y[ind]
                    y1 = y[ind + 1]
                    t = u - ind.to(u.dtype)
                    ynew = y0 + (y1 - y0) * t
                else:
                    x0i = x_min + ind.to(xq_used.dtype) * self._dx_uniform
                    y0 = self._y0[ind]
                    m = self._slopes[ind]
                    ynew = y0 + m * (xq_used - x0i)
            else:
                # ---- original path ----
                torch.searchsorted(self._x_search, xq_used.contiguous(), out=ind)
                ind -= 1
                ind.clamp_(0, self._ind_hi)

                x0i = self._x0[ind]
                inv_dxi = self._inv_dx[ind]

                if self.learnable_y:
                    y = self.y  # (N,)
                    y0 = y[ind]
                    y1 = y[ind + 1]
                    t = (xq_used - x0i) * inv_dxi
                    ynew = y0 + (y1 - y0) * t
                else:
                    y0 = self._y0[ind]
                    m = self._slopes[ind]
                    ynew = y0 + m * (xq_used - x0i)

            # Outside policy (same as your original)
            if outside == "clamp":
                if self.exact_clamp:
                    if self.learnable_y:
                        y_first = self.y[0]
                        y_last = self.y[-1]
                    else:
                        y_first = self._y_first
                        y_last = self._y_last
                    ynew = torch.where(xq <= x_min, y_first, ynew)
                    ynew = torch.where(xq >= x_max, y_last, ynew)
            else:
                outside_mask = (xq < x_min) | (xq > x_max)
                if outside == "zero":
                    ynew = ynew.masked_fill(outside_mask, 0.0)
                else:
                    ynew = ynew.masked_fill(outside_mask, fill_value)

        # ------------------------
        # Batched LUT mode
        # ------------------------
        else:
            if xnew_was_1d:
                if indices is None:
                    raise ValueError(
                        "Ambiguous: x,y are (D,N) but x_new is (P,). Provide indices."
                    )
                idx = self._as_index_tensor(
                    indices, device=xq.device, expected_len=None
                )
                Q_eff = int(idx.numel())
                xq = xq.expand(Q_eff, -1)
                Q = Q_eff
            else:
                if indices is None:
                    if Q != self.D:
                        raise ValueError(
                            f"Ambiguous: D={self.D} but Q={Q}. Provide indices of length Q."
                        )
                    idx = None
                else:
                    idx = self._as_index_tensor(
                        indices, device=xq.device, expected_len=Q
                    )

            # Select LUT rows
            if idx is None:
                x_search = self._x_search
                x0 = self._x0
                inv_dx = self._inv_dx
                x_min = self._x_min
                x_max = self._x_max
                if self._use_uniform:
                    inv_dx_u = self._inv_dx_uniform
                    dx_u = self._dx_uniform
                if self.learnable_y:
                    y_sel = self.y
                else:
                    y0_tab = self._y0
                    m_tab = self._slopes
                    y_first = self._y_first
                    y_last = self._y_last
            else:
                x_search = self._x_search.index_select(0, idx)
                x0 = self._x0.index_select(0, idx)
                inv_dx = self._inv_dx.index_select(0, idx)
                x_min = self._x_min.index_select(0, idx)
                x_max = self._x_max.index_select(0, idx)
                if self._use_uniform:
                    inv_dx_u = self._inv_dx_uniform.index_select(0, idx)
                    dx_u = self._dx_uniform.index_select(0, idx)
                if self.learnable_y:
                    y_sel = self.y.index_select(0, idx)
                else:
                    y0_tab = self._y0.index_select(0, idx)
                    m_tab = self._slopes.index_select(0, idx)
                    y_first = self._y_first.index_select(0, idx)
                    y_last = self._y_last.index_select(0, idx)

            xq_used = (
                torch.maximum(torch.minimum(xq, x_max), x_min)
                if outside == "clamp"
                else xq
            )

            if (
                self._ind_cache is None
                or self._ind_cache.shape != xq_used.shape
                or self._ind_cache.device != xq_used.device
            ):
                self._ind_cache = torch.empty(
                    xq_used.shape, device=xq_used.device, dtype=torch.long
                )
            ind = self._ind_cache

            if self._use_uniform:
                u = (xq_used - x_min) * inv_dx_u
                ind.copy_(u.to(torch.long))
                ind.clamp_(0, self._ind_hi)

                if self.learnable_y:
                    y0 = torch.gather(y_sel, 1, ind)
                    y1 = torch.gather(y_sel, 1, ind + 1)
                    t = u - ind.to(u.dtype)
                    ynew = y0 + (y1 - y0) * t
                    if outside == "clamp" and self.exact_clamp:
                        ynew = torch.where(
                            xq <= x_min, y_sel[:, :1].expand_as(ynew), ynew
                        )
                        ynew = torch.where(
                            xq >= x_max, y_sel[:, -1:].expand_as(ynew), ynew
                        )
                else:
                    x0i = x_min + ind.to(xq_used.dtype) * dx_u
                    y0i = torch.gather(y0_tab, 1, ind)
                    mi = torch.gather(m_tab, 1, ind)
                    ynew = y0i + mi * (xq_used - x0i)
                    if outside == "clamp" and self.exact_clamp:
                        ynew = torch.where(xq <= x_min, y_first.expand_as(ynew), ynew)
                        ynew = torch.where(xq >= x_max, y_last.expand_as(ynew), ynew)
            else:
                torch.searchsorted(x_search, xq_used.contiguous(), out=ind)
                ind -= 1
                ind.clamp_(0, self._ind_hi)

                x0i = torch.gather(x0, 1, ind)
                inv_dxi = torch.gather(inv_dx, 1, ind)

                if self.learnable_y:
                    y0 = torch.gather(y_sel, 1, ind)
                    y1 = torch.gather(y_sel, 1, ind + 1)
                    t = (xq_used - x0i) * inv_dxi
                    ynew = y0 + (y1 - y0) * t

                    if outside == "clamp" and self.exact_clamp:
                        ynew = torch.where(
                            xq <= x_min, y_sel[:, :1].expand_as(ynew), ynew
                        )
                        ynew = torch.where(
                            xq >= x_max, y_sel[:, -1:].expand_as(ynew), ynew
                        )
                else:
                    y0i = torch.gather(y0_tab, 1, ind)
                    mi = torch.gather(m_tab, 1, ind)
                    ynew = y0i + mi * (xq_used - x0i)

                    if outside == "clamp" and self.exact_clamp:
                        ynew = torch.where(xq <= x_min, y_first.expand_as(ynew), ynew)
                        ynew = torch.where(xq >= x_max, y_last.expand_as(ynew), ynew)

            if outside in ("zero", "fill"):
                outside_mask = (xq < x_min) | (xq > x_max)
                if outside == "zero":
                    ynew = ynew.masked_fill(outside_mask, 0.0)
                else:
                    ynew = ynew.masked_fill(outside_mask, fill_value)

        if xnew_was_1d and ynew.shape[0] == 1:
            ynew = ynew[0]

        if out is not None:
            if out.shape != ynew.shape:
                raise ValueError(
                    f"out has shape {tuple(out.shape)} but expected {tuple(ynew.shape)}."
                )
            out.copy_(ynew)
            return out

        return ynew


class PreparedInterp1dUniform(nn.Module):
    """
    Prepared 1D uniform-grid linear interpolator.

    Same input/output regimes as PreparedInterp1d, but assumes x is uniformly spaced
    (per row, if batched), enabling O(1) interval selection without searchsorted.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        outside: OutsideMode = "clamp",
        fill_value: float = 0.0,
        learnable_y: bool = False,
        y_requires_grad: bool = True,
        sort_xy: bool = True,
        check_uniform: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        eps: Optional[float] = None,
        exact_clamp: bool = True,
    ):
        super().__init__()

        if outside not in ("clamp", "zero", "fill"):
            raise ValueError("outside must be one of: 'clamp', 'zero', 'fill'")

        if x.ndim != y.ndim or x.ndim not in (1, 2):
            raise ValueError("x and y must both be 1D or both be 2D.")

        if x.device != y.device or x.dtype != y.dtype:
            raise ValueError(
                "x and y must be on the same device and have the same dtype."
            )
        if not torch.is_floating_point(x):
            raise TypeError("x and y must be floating-point.")

        # Enforce shapes
        if x.ndim == 1:
            if x.shape != y.shape:
                raise ValueError("Unbatched: x and y must both be (N,).")
            self.batched = False
            self.D = 1
            self.N = int(x.shape[0])
        else:
            if x.shape != y.shape:
                raise ValueError("Batched: x and y must both be (D,N).")
            self.batched = True
            self.D = int(x.shape[0])
            self.N = int(x.shape[1])

        if self.N < 2:
            raise ValueError("Need N >= 2 points for interpolation.")

        self.outside = outside
        self.fill_value = float(fill_value)
        self.learnable_y = bool(learnable_y)
        self.exact_clamp = bool(exact_clamp)

        # Optional sort + permute
        if sort_xy:
            if not self.batched:
                x_sorted, perm = torch.sort(x, dim=0)
                y_sorted = y.index_select(0, perm)
            else:
                x_sorted, perm = torch.sort(x, dim=1)
                y_sorted = torch.gather(y, dim=1, index=perm)
        else:
            x_sorted = x.contiguous()
            y_sorted = y.contiguous()

        # Uniform-grid parameters per row:
        # dx0 = first spacing; inv_dx = 1/dx0
        if eps is None:
            eps = float(torch.finfo(x_sorted.dtype).eps)

        if not self.batched:
            dx = x_sorted[1:] - x_sorted[:-1]  # (N-1,)
            dx0 = dx[:1]  # (1,)
            if check_uniform:
                ref = dx0.expand_as(dx)
                if not torch.allclose(dx, ref, rtol=rtol, atol=atol):
                    raise ValueError("x is not uniformly spaced (check_uniform=True).")
            if torch.any(dx0 <= 0):
                raise ValueError(
                    "x must be strictly increasing (after sorting, if enabled)."
                )

            x_min = x_sorted[0]
            x_max = x_sorted[-1]
            safe_dx0 = torch.where(dx0 == 0, torch.full_like(dx0, eps), dx0)
            inv_dx = (1.0 / safe_dx0)[0]  # scalar
            self.register_buffer("_x_min", x_min.detach())
            self.register_buffer("_x_max", x_max.detach())
            self.register_buffer("_inv_dx", inv_dx.detach())
        else:
            dx = x_sorted[:, 1:] - x_sorted[:, :-1]  # (D,N-1)
            dx0 = dx[:, :1]  # (D,1)
            if check_uniform:
                ref = dx0.expand_as(dx)
                if not torch.allclose(dx, ref, rtol=rtol, atol=atol):
                    raise ValueError(
                        "x is not uniformly spaced per row (check_uniform=True)."
                    )
            if torch.any(dx0 <= 0):
                raise ValueError(
                    "Each x row must be strictly increasing (after sorting, if enabled)."
                )

            x_min = x_sorted[:, :1]  # (D,1)
            x_max = x_sorted[:, -1:]  # (D,1)
            safe_dx0 = torch.where(dx0 == 0, torch.full_like(dx0, eps), dx0)
            inv_dx = (1.0 / safe_dx0).contiguous()  # (D,1)
            self.register_buffer("_x_min", x_min.detach())
            self.register_buffer("_x_max", x_max.detach())
            self.register_buffer("_inv_dx", inv_dx.detach())

        self._ind_hi = self.N - 2

        # Store y tables
        if self.learnable_y:
            self.y = nn.Parameter(y_sorted.contiguous(), requires_grad=y_requires_grad)
        else:
            if not self.batched:
                self.register_buffer(
                    "_y0", y_sorted[:-1].detach().contiguous()
                )  # (N-1,)
                self.register_buffer(
                    "_dy", (y_sorted[1:] - y_sorted[:-1]).detach().contiguous()
                )  # (N-1,)
                self.register_buffer("_y_first", y_sorted[0].detach())
                self.register_buffer("_y_last", y_sorted[-1].detach())
            else:
                self.register_buffer(
                    "_y0", y_sorted[:, :-1].detach().contiguous()
                )  # (D,N-1)
                self.register_buffer(
                    "_dy", (y_sorted[:, 1:] - y_sorted[:, :-1]).detach().contiguous()
                )  # (D,N-1)
                self.register_buffer("_y_first", y_sorted[:, :1].detach())
                self.register_buffer("_y_last", y_sorted[:, -1:].detach())

    def _as_index_tensor(
        self,
        indices: IndexLike,
        *,
        device: torch.device,
        expected_len: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(indices, int):
            idx = torch.tensor([indices], device=device, dtype=torch.long)
        else:
            idx = torch.as_tensor(indices, device=device, dtype=torch.long)

        if idx.ndim != 1:
            raise ValueError("indices must be a 1D sequence/tensor (or a single int).")
        if idx.numel() == 0:
            raise ValueError("indices must be non-empty.")
        if expected_len is not None and idx.numel() != expected_len:
            raise ValueError(
                f"indices must have length {expected_len}, got {idx.numel()}."
            )

        if self.batched:
            if torch.any((idx < 0) | (idx >= self.D)):
                raise ValueError(f"indices values must be in [0, {self.D - 1}].")
        else:
            if torch.any(idx != 0):
                raise ValueError("In unbatched mode, indices can only contain 0.")

        return idx

    def forward(
        self,
        x_new: torch.Tensor,
        *,
        indices: Optional[IndexLike] = None,
        outside: Optional[OutsideMode] = None,
        fill_value: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpolate at query points `x_new`.

        Parameters
        ----------
        x_new : torch.Tensor
            Query points at which to evaluate the interpolant(s).
            Accepted shapes are ``(P,)`` for unbatched interpolation and
            ``(Q, P)`` for batched interpolation with ``Q`` batches.
        indices : int or sequence of int or torch.Tensor, optional
            Mapping from query rows in `x_new` to LUT rows in `x`/`y` (batched mode).
        outside : {"clamp", "zero", "fill"}, optional
            Override the module's default out-of-bounds policy for this call.
        fill_value : float, optional
            Override the module's default fill value for this call when ``outside="fill"``.
        out : torch.Tensor, optional
            Optional output tensor to write results into. Must have exactly the same
            shape as the computed output.

        Returns
        -------
        y_new : torch.Tensor
            Interpolated values.

        Raises
        ------
        ValueError
            If `x_new` is not 1D/2D, has incompatible batch dimension in batched mode
            without `indices`, or if `indices` is missing or malformed in ambiguous cases.
        TypeError
            If `x_new` is not floating-point.
        """
        outside = self.outside if outside is None else outside
        if outside not in ("clamp", "zero", "fill"):
            raise ValueError("outside must be one of: 'clamp', 'zero', 'fill'")

        fill_value = self.fill_value if fill_value is None else float(fill_value)

        if x_new.ndim not in (1, 2):
            raise ValueError("x_new must be 1D (P,) or 2D (Q,P).")
        if x_new.device != self._x_min.device or x_new.dtype != self._x_min.dtype:
            raise ValueError("x_new must match device and dtype of the interpolator.")
        if not torch.is_floating_point(x_new):
            raise TypeError("x_new must be floating-point.")

        xnew_was_1d = x_new.ndim == 1
        xq = x_new[None, :] if xnew_was_1d else x_new
        Q = int(xq.shape[0])

        # -------- unbatched --------
        if not self.batched:
            if indices is not None:
                _ = self._as_index_tensor(indices, device=xq.device, expected_len=Q)

            x_min = self._x_min
            x_max = self._x_max
            inv_dx = self._inv_dx

            xq_used = torch.clamp(xq, x_min, x_max) if outside == "clamp" else xq

            u = (xq_used - x_min) * inv_dx  # (Q,P)
            ind = u.to(torch.long)
            ind.clamp_(0, self._ind_hi)  # (Q,P)
            t = u - ind.to(u.dtype)  # (Q,P)

            if outside == "clamp":
                # numeric safety: ensure t in [0,1] on the boundary
                t = torch.clamp(t, 0.0, 1.0)

            if self.learnable_y:
                y = self.y
                y0 = y[ind]
                y1 = y[ind + 1]
                ynew = y0 + (y1 - y0) * t
                if outside == "clamp" and self.exact_clamp:
                    ynew = torch.where(xq <= x_min, y[0], ynew)
                    ynew = torch.where(xq >= x_max, y[-1], ynew)
            else:
                y0 = self._y0[ind]
                dy = self._dy[ind]
                ynew = y0 + dy * t
                if outside == "clamp" and self.exact_clamp:
                    ynew = torch.where(xq <= x_min, self._y_first, ynew)
                    ynew = torch.where(xq >= x_max, self._y_last, ynew)

            if outside in ("zero", "fill"):
                outside_mask = (xq < x_min) | (xq > x_max)
                ynew = ynew.masked_fill(
                    outside_mask, 0.0 if outside == "zero" else fill_value
                )

        # -------- batched --------
        else:
            if xnew_was_1d:
                if indices is None:
                    raise ValueError(
                        "Ambiguous: x,y are (D,N) but x_new is (P,). Provide indices."
                    )
                idx = self._as_index_tensor(
                    indices, device=xq.device, expected_len=None
                )
                Q_eff = int(idx.numel())
                xq = xq.expand(Q_eff, -1)
                Q = Q_eff
            else:
                if indices is None:
                    if Q != self.D:
                        raise ValueError(
                            f"Ambiguous: x,y have D={self.D} but x_new has Q={Q}. Provide indices of length Q."
                        )
                    idx = None
                else:
                    idx = self._as_index_tensor(
                        indices, device=xq.device, expected_len=Q
                    )

            if idx is None:
                x_min = self._x_min
                x_max = self._x_max
                inv_dx = self._inv_dx
                if self.learnable_y:
                    y_sel = self.y
                else:
                    y0_tab = self._y0
                    dy_tab = self._dy
                    y_first = self._y_first
                    y_last = self._y_last
            else:
                x_min = self._x_min.index_select(0, idx)
                x_max = self._x_max.index_select(0, idx)
                inv_dx = self._inv_dx.index_select(0, idx)
                if self.learnable_y:
                    y_sel = self.y.index_select(0, idx)
                else:
                    y0_tab = self._y0.index_select(0, idx)
                    dy_tab = self._dy.index_select(0, idx)
                    y_first = self._y_first.index_select(0, idx)
                    y_last = self._y_last.index_select(0, idx)

            xq_used = torch.clamp(xq, x_min, x_max) if outside == "clamp" else xq

            u = (xq_used - x_min) * inv_dx  # (Q,P)
            ind = u.to(torch.long)
            ind.clamp_(0, self._ind_hi)
            t = u - ind.to(u.dtype)

            if outside == "clamp":
                t = torch.clamp(t, 0.0, 1.0)

            if self.learnable_y:
                y0 = torch.gather(y_sel, 1, ind)
                y1 = torch.gather(y_sel, 1, ind + 1)
                ynew = y0 + (y1 - y0) * t
                if outside == "clamp" and self.exact_clamp:
                    ynew = torch.where(xq <= x_min, y_sel[:, :1].expand_as(ynew), ynew)
                    ynew = torch.where(xq >= x_max, y_sel[:, -1:].expand_as(ynew), ynew)
            else:
                y0 = torch.gather(y0_tab, 1, ind)
                dy = torch.gather(dy_tab, 1, ind)
                ynew = y0 + dy * t
                if outside == "clamp" and self.exact_clamp:
                    ynew = torch.where(xq <= x_min, y_first.expand_as(ynew), ynew)
                    ynew = torch.where(xq >= x_max, y_last.expand_as(ynew), ynew)

            if outside in ("zero", "fill"):
                outside_mask = (xq < x_min) | (xq > x_max)
                ynew = ynew.masked_fill(
                    outside_mask, 0.0 if outside == "zero" else fill_value
                )

        # Restore 1D output iff a single query row was evaluated
        if xnew_was_1d and ynew.shape[0] == 1:
            ynew = ynew[0]

        if out is not None:
            if out.shape != ynew.shape:
                raise ValueError(
                    f"out has shape {tuple(out.shape)} but expected {tuple(ynew.shape)}."
                )
            out.copy_(ynew)
            return out

        return ynew


def interp1d(
    x,
    y,
    xnew,
    out=None,
    *,
    outside: str = "zero",
    # --- uniform fast paths ---
    uniform: str = "auto",  # "never" | "auto" | "always"
    uniform_rtol: float = 1e-5,
    uniform_atol: float = 1e-7,
):
    """
    Linear 1D interpolation for PyTorch (CPU/GPU) with batched support.

    This function returns interpolated values of one or more 1D functions
    tabulated on `x` with values `y`, evaluated at query points `xnew`.
    It parallelizes over the leading (batch) dimension when inputs are 2D.

    Notes
    -----
    - `torch.searchsorted` assumes `x` is sorted ascending along its last dimension
      (for each batch row, if batched).
    - Interpolation is piecewise-linear. The segment selection (`searchsorted`)
      is non-differentiable, but values are differentiable w.r.t. `x`, `y`, and
      `xnew` within each segment.

    Parameters
    ----------
    x : torch.Tensor
        Shape (N,) or (D, N). Abscissae (must be sorted ascending along last dim).
    y : torch.Tensor
        Shape (N,) or (D, N). Ordinates. Must match `x` in columns; rows may be
        either equal or one of them may be 1 (broadcast across rows).
    xnew : torch.Tensor
        Shape (P,) or (D, P) (or generally (1, P) broadcastable).
        If `xnew` has a single row, it will be broadcast to the effective batch
        size determined by `x` and `y`.
    out : torch.Tensor, optional
        Optional output buffer. If provided, must have `numel == D_eff * P`.
        The returned tensor will be a view of `out` with shape (D_eff, P).
        If `x` and `y` are both 1D and `xnew` is 1D, output is reshaped back to (P,).
    outside : {"zero", "clamp"}, default="zero"
        Out-of-bounds handling:
        - "zero":  values with xnew < x_min or xnew > x_max are set to 0
        - "clamp": values outside are clamped to endpoint values y_min / y_max
    uniform : {"never", "auto", "always"}, default="auto"
        Whether to use the uniform-grid fast path (O(1) interval selection).
        - "never": always use searchsorted (O(log N) interval selection)
        - "auto":  use uniform fast path if `x` is uniformly spaced (per row)
        - "always": assume `x` is uniformly spaced; raise ValueError if not
    uniform_rtol : float, default=1e-5
        Relative tolerance for uniformity check (when `uniform="auto"`).
    uniform_atol : float, default=1e-7
        Absolute tolerance for uniformity check (when `uniform="auto"`).

    Returns
    -------
    ynew : torch.Tensor
        Interpolated values. Shape matches `xnew` in the common cases:

        - if x,y,xnew are 1D -> (P,)
        - if batched -> typically (D_eff, P)
        - special case: if x,y are single-row and xnew has multiple rows, the
          output is reshaped back to match xnew's original 2D shape.
    """
    if outside not in {"zero", "clamp"}:
        raise ValueError("outside must be one of {'zero', 'clamp'}")
    if uniform not in {"never", "auto", "always"}:
        raise ValueError("uniform must be one of {'never','auto','always'}")

    # --- make inputs at least 2D ---
    is_flat = {}
    require_grad = {}
    v = {}

    for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
        assert vec.ndim <= 2, "interp1d: all inputs must be at most 2-D."
        v[name] = vec[None, :] if vec.ndim == 1 else vec
        is_flat[name] = v[name].shape[0] == 1
        require_grad[name] = vec.requires_grad

    device = x.device
    assert y.device == device and xnew.device == device, (
        "All parameters must be on the same device."
    )

    assert v["x"].shape[1] == v["y"].shape[1] and (
        v["x"].shape[0] == v["y"].shape[0]
        or v["x"].shape[0] == 1
        or v["y"].shape[0] == 1
    ), (
        "x and y must have the same number of columns, and either the same number "
        "of rows or one of them having only one row."
    )

    # Optimization: if x and y are single-row but xnew has multiple rows, flatten xnew
    reshaped_xnew = False
    if (v["x"].shape[0] == 1) and (v["y"].shape[0] == 1) and (v["xnew"].shape[0] > 1):
        original_xnew_shape = v["xnew"].shape
        v["xnew"] = v["xnew"].contiguous().view(1, -1)
        reshaped_xnew = True

    # Effective batch size
    D_eff = max(v["x"].shape[0], v["y"].shape[0], v["xnew"].shape[0])
    P = v["xnew"].shape[1]
    shape_ynew = (D_eff, P)

    # Prepare output buffer
    if out is not None:
        if out.numel() != D_eff * P:
            out = None
        else:
            ybuf = out.reshape(shape_ynew)
    if out is None:
        ybuf = torch.empty(*shape_ynew, device=device, dtype=v["x"].dtype)

    # Broadcast xnew rows if needed
    if v["xnew"].shape[0] == 1 and D_eff > 1:
        v["xnew"] = v["xnew"].expand(D_eff, -1)

    def sel(t: torch.Tensor, flat: bool, ind: torch.Tensor) -> torch.Tensor:
        if flat:
            return t.contiguous().view(-1)[ind]
        return torch.gather(t, 1, ind)

    enable_grad = require_grad["x"] or require_grad["y"] or require_grad["xnew"]
    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()

    with grad_ctx:
        # Decide whether to use uniform fast-path
        use_uniform = False
        if uniform != "never":
            dx = v["x"][:, 1:] - v["x"][:, :-1]  # (Rx,N-1)
            dx0 = dx[:, :1]
            uniform_ok = bool(torch.all(dx0 > 0).item()) and torch.allclose(
                dx, dx0.expand_as(dx), rtol=uniform_rtol, atol=uniform_atol
            )
            if uniform == "always" and not uniform_ok:
                raise ValueError(
                    "uniform='always' but x is not uniformly spaced (per row)."
                )
            use_uniform = uniform_ok

        if use_uniform:
            x_min = v["x"][:, :1]
            x_max = v["x"][:, -1:]
            dx0 = v["x"][:, 1:2] - v["x"][:, 0:1]

            if x_min.shape[0] == 1 and D_eff > 1:
                x_min = x_min.expand(D_eff, 1)
                x_max = x_max.expand(D_eff, 1)
                dx0 = dx0.expand(D_eff, 1)

            eps = torch.finfo(v["x"].dtype).eps
            safe_dx0 = torch.where(dx0 == 0, torch.full_like(dx0, eps), dx0)
            inv_dx0 = 1.0 / safe_dx0

            x_used = (
                torch.clamp(v["xnew"], x_min, x_max)
                if outside == "clamp"
                else v["xnew"]
            )
            u = (x_used - x_min) * inv_dx0
            ind = u.to(torch.long)
            ind.clamp_(0, v["x"].shape[1] - 2)
            t = u - ind.to(u.dtype)
            if outside == "clamp":
                t = torch.clamp(t, 0.0, 1.0)

            dy = v["y"][:, 1:] - v["y"][:, :-1]
            dy_is_flat = dy.shape[0] == 1

            ynew = sel(v["y"], is_flat["y"], ind) + sel(dy, dy_is_flat, ind) * t

            if outside == "clamp":
                y_min = v["y"][:, :1]
                y_max = v["y"][:, -1:]
                if y_min.shape[0] == 1 and D_eff > 1:
                    y_min = y_min.expand(D_eff, 1)
                    y_max = y_max.expand(D_eff, 1)
                ynew = torch.where(v["xnew"] <= x_min, y_min.expand_as(ynew), ynew)
                ynew = torch.where(v["xnew"] >= x_max, y_max.expand_as(ynew), ynew)
            else:
                outside_mask = (v["xnew"] < x_min) | (v["xnew"] > x_max)
                ynew = ynew.masked_fill(outside_mask, 0.0)

        else:
            # Original searchsorted path
            ind = torch.empty(shape_ynew, device=device, dtype=torch.long)
            torch.searchsorted(
                v["x"].contiguous().squeeze(), v["xnew"].contiguous(), out=ind
            )
            ind -= 1
            ind.clamp_(0, v["x"].shape[1] - 2)

            dx = v["x"][:, 1:] - v["x"][:, :-1]
            safe_dx = torch.where(
                dx == 0, torch.full_like(dx, torch.finfo(dx.dtype).eps), dx
            )
            slopes = (v["y"][:, 1:] - v["y"][:, :-1]) / safe_dx
            slopes_is_flat = slopes.shape[0] == 1

            ynew = sel(v["y"], is_flat["y"], ind) + sel(slopes, slopes_is_flat, ind) * (
                v["xnew"] - sel(v["x"], is_flat["x"], ind)
            )

            x_min = v["x"][:, :1]
            x_max = v["x"][:, -1:]

            if outside == "clamp":
                y_min = v["y"][:, :1]
                y_max = v["y"][:, -1:]
                ynew = torch.where(v["xnew"] <= x_min, y_min.expand_as(ynew), ynew)
                ynew = torch.where(v["xnew"] >= x_max, y_max.expand_as(ynew), ynew)
            else:
                outside_mask = (v["xnew"] < x_min) | (v["xnew"] > x_max)
                ynew = ynew.masked_fill(outside_mask, 0.0)

    ybuf.copy_(ynew)

    if reshaped_xnew:
        ybuf = ybuf.view(original_xnew_shape)

    if x.ndim == 1 and y.ndim == 1 and xnew.ndim == 1:
        return ybuf.view(-1)

    return ybuf


def interp1d_uniform(x, y, xnew, out=None, *, outside: str = "zero"):
    """
    Uniform-grid 1D linear interpolation for PyTorch with batched support.

    Same calling convention as interp1d(x,y,xnew,...) but assumes x is uniformly spaced
    per row (if batched). Uses arithmetic indexing, not searchsorted.
    """
    if outside not in {"zero", "clamp"}:
        raise ValueError("outside must be one of {'zero', 'clamp'}")

    is_flat = {}
    require_grad = {}
    v = {}

    for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
        assert vec.ndim <= 2, "interp1d_uniform: all inputs must be at most 2-D."
        v[name] = vec[None, :] if vec.ndim == 1 else vec
        is_flat[name] = v[name].shape[0] == 1
        require_grad[name] = vec.requires_grad

    device = x.device
    assert y.device == device and xnew.device == device, (
        "All parameters must be on the same device."
    )

    assert v["x"].shape[1] == v["y"].shape[1] and (
        v["x"].shape[0] == v["y"].shape[0]
        or v["x"].shape[0] == 1
        or v["y"].shape[0] == 1
    ), (
        "x and y must have the same number of columns, and either the same number "
        "of rows or one of them having only one row."
    )

    # Optimization: if x and y are single-row but xnew has multiple rows, flatten xnew
    reshaped_xnew = False
    if (v["x"].shape[0] == 1) and (v["y"].shape[0] == 1) and (v["xnew"].shape[0] > 1):
        original_xnew_shape = v["xnew"].shape
        v["xnew"] = v["xnew"].contiguous().view(1, -1)
        reshaped_xnew = True

    D_eff = max(v["x"].shape[0], v["y"].shape[0], v["xnew"].shape[0])
    P = v["xnew"].shape[1]
    shape_ynew = (D_eff, P)

    # Prepare output buffer
    if out is not None:
        if out.numel() != D_eff * P:
            out = None
        else:
            ybuf = out.reshape(shape_ynew)
    if out is None:
        ybuf = torch.empty(*shape_ynew, device=device, dtype=v["x"].dtype)

    # Broadcast xnew rows if needed
    if v["xnew"].shape[0] == 1 and D_eff > 1:
        v["xnew"] = v["xnew"].expand(D_eff, -1)

    # Helper to broadcast row-wise scalar params
    def row_param(t: torch.Tensor) -> torch.Tensor:
        # t is (R,N) -> return (D_eff,1) by selecting from first/each row then expanding if needed
        if t.shape[0] == 1 and D_eff > 1:
            return t[:, :1].expand(D_eff, 1)
        return t[:, :1]

    enable_grad = require_grad["x"] or require_grad["y"] or require_grad["xnew"]
    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()

    with grad_ctx:
        # Uniform axis params
        x_min = row_param(v["x"])
        x_max = (
            v["x"][:, -1:] if v["x"].shape[0] != 1 else v["x"][:, -1:].expand(D_eff, 1)
        )
        dx0 = v["x"][:, 1:2] - v["x"][:, 0:1]
        if dx0.shape[0] == 1 and D_eff > 1:
            dx0 = dx0.expand(D_eff, 1)

        eps = torch.finfo(v["x"].dtype).eps
        safe_dx0 = torch.where(dx0 == 0, torch.full_like(dx0, eps), dx0)
        inv_dx = 1.0 / safe_dx0  # (D_eff,1)

        x_used = (
            torch.clamp(v["xnew"], x_min, x_max) if outside == "clamp" else v["xnew"]
        )

        u = (x_used - x_min) * inv_dx  # (D_eff,P)
        ind = u.to(torch.long)
        ind.clamp_(0, v["x"].shape[1] - 2)

        t = u - ind.to(u.dtype)
        if outside == "clamp":
            t = torch.clamp(t, 0.0, 1.0)

        def sel(tensor: torch.Tensor, flat: bool) -> torch.Tensor:
            if flat:
                return tensor.contiguous().view(-1)[ind]
            return torch.gather(tensor, 1, ind)

        dy = v["y"][:, 1:] - v["y"][:, :-1]
        dy_is_flat = dy.shape[0] == 1

        ynew = sel(v["y"], is_flat["y"]) + sel(dy, dy_is_flat) * t

        if outside == "clamp":
            y_min = row_param(v["y"])
            y_max = (
                v["y"][:, -1:]
                if v["y"].shape[0] != 1
                else v["y"][:, -1:].expand(D_eff, 1)
            )
            ynew = torch.where(v["xnew"] <= x_min, y_min.expand_as(ynew), ynew)
            ynew = torch.where(v["xnew"] >= x_max, y_max.expand_as(ynew), ynew)
        else:
            outside_mask = (v["xnew"] < x_min) | (v["xnew"] > x_max)
            ynew = ynew.masked_fill(outside_mask, 0.0)

    ybuf.copy_(ynew)

    if reshaped_xnew:
        ybuf = ybuf.view(original_xnew_shape)

    if x.ndim == 1 and y.ndim == 1 and xnew.ndim == 1:
        return ybuf.view(-1)

    return ybuf


class PreparedInterp3dRect(nn.Module):
    """
    Prepared trilinear interpolator on a rectilinear (tensor-product) grid.

    Non-uniform axes use searchsorted. Uniform axes (detected per row) use arithmetic
    indexing as a fast-path.
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        values: torch.Tensor,
        *,
        outside: OutsideMode = "clamp",
        fill_value: float = 0.0,
        learnable_values: bool = False,
        values_requires_grad: bool = True,
        sort_xyz: bool = True,
        eps: Optional[float] = None,
        # NEW:
        uniform: Literal["auto", "never", "always"] = "auto",
        uniform_rtol: float = 1e-5,
        uniform_atol: float = 1e-7,
    ):
        super().__init__()

        if outside not in ("clamp", "zero", "fill"):
            raise ValueError("outside must be one of: 'clamp', 'zero', 'fill'")
        if uniform not in ("auto", "never", "always"):
            raise ValueError("uniform must be 'auto', 'never', or 'always'")

        if not (x.ndim == y.ndim == z.ndim) or x.ndim not in (1, 2):
            raise ValueError("x, y, z must have the same ndim (all 1D or all 2D).")

        if x.device != y.device or x.device != z.device or x.device != values.device:
            raise ValueError("x, y, z, values must be on the same device.")
        if x.dtype != y.dtype or x.dtype != z.dtype or x.dtype != values.dtype:
            raise ValueError("x, y, z, values must have the same dtype.")
        if not torch.is_floating_point(x):
            raise TypeError("x, y, z, values must be floating-point tensors.")

        self.outside = outside
        self.fill_value = float(fill_value)
        self.learnable_values = bool(learnable_values)

        self.batched = x.ndim == 2

        # Validate shapes vs values
        if not self.batched:
            Nx, Ny, Nz = int(x.shape[0]), int(y.shape[0]), int(z.shape[0])
            if values.ndim not in (3, 4):
                raise ValueError(
                    "Unbatched: values must be (Nx,Ny,Nz) or (Nx,Ny,Nz,C)."
                )
            if values.shape[0] != Nx or values.shape[1] != Ny or values.shape[2] != Nz:
                raise ValueError(
                    "Unbatched: values first 3 dims must match (Nx,Ny,Nz)."
                )
            self.D = 1
        else:
            D = int(x.shape[0])
            if y.shape[0] != D or z.shape[0] != D:
                raise ValueError("Batched: x,y,z must have same leading dim D.")
            Nx, Ny, Nz = int(x.shape[1]), int(y.shape[1]), int(z.shape[1])
            if values.ndim not in (4, 5):
                raise ValueError(
                    "Batched: values must be (D,Nx,Ny,Nz) or (D,Nx,Ny,Nz,C)."
                )
            if (
                values.shape[0] != D
                or values.shape[1] != Nx
                or values.shape[2] != Ny
                or values.shape[3] != Nz
            ):
                raise ValueError(
                    "Batched: values must match (D,Nx,Ny,Nz,...) in its first 4 dims."
                )
            self.D = D

        if Nx < 2 or Ny < 2 or Nz < 2:
            raise ValueError("Need at least 2 samples along each axis.")

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz

        # Normalize values to have explicit channels-last
        self._had_channels = values.ndim == (4 if not self.batched else 5)
        if not self._had_channels:
            values = values.unsqueeze(-1)
        self.C = int(values.shape[-1])

        # Sorting + permute values
        if sort_xyz:
            if not self.batched:
                x_sorted, px = torch.sort(x, dim=0)
                y_sorted, py = torch.sort(y, dim=0)
                z_sorted, pz = torch.sort(z, dim=0)
                vals = (
                    values.index_select(0, px).index_select(1, py).index_select(2, pz)
                )
            else:
                x_sorted, px = torch.sort(x, dim=1)
                y_sorted, py = torch.sort(y, dim=1)
                z_sorted, pz = torch.sort(z, dim=1)

                vals = values
                ix = px[:, :, None, None, None].expand(-1, -1, Ny, Nz, self.C)
                vals = torch.gather(vals, dim=1, index=ix)
                iy = py[:, None, :, None, None].expand(-1, Nx, -1, Nz, self.C)
                vals = torch.gather(vals, dim=2, index=iy)
                iz = pz[:, None, None, :, None].expand(-1, Nx, Ny, -1, self.C)
                vals = torch.gather(vals, dim=3, index=iz)
        else:
            x_sorted, y_sorted, z_sorted = (
                x.contiguous(),
                y.contiguous(),
                z.contiguous(),
            )
            vals = values.contiguous()

        # Store axis searchsorted buffers (always; used for non-uniform fallback)
        if not self.batched:
            self.register_buffer("_x_search", x_sorted.detach())
            self.register_buffer("_x0", x_sorted[:-1].detach())
            self.register_buffer("_x_min", x_sorted[0].detach())
            self.register_buffer("_x_max", x_sorted[-1].detach())
            dx = x_sorted[1:] - x_sorted[:-1]

            self.register_buffer("_y_search", y_sorted.detach())
            self.register_buffer("_y0", y_sorted[:-1].detach())
            self.register_buffer("_y_min", y_sorted[0].detach())
            self.register_buffer("_y_max", y_sorted[-1].detach())
            dy = y_sorted[1:] - y_sorted[:-1]

            self.register_buffer("_z_search", z_sorted.detach())
            self.register_buffer("_z0", z_sorted[:-1].detach())
            self.register_buffer("_z_min", z_sorted[0].detach())
            self.register_buffer("_z_max", z_sorted[-1].detach())
            dz = z_sorted[1:] - z_sorted[:-1]
        else:
            self.register_buffer("_x_search", x_sorted.detach())
            self.register_buffer("_x0", x_sorted[:, :-1].detach())
            self.register_buffer("_x_min", x_sorted[:, :1].detach())
            self.register_buffer("_x_max", x_sorted[:, -1:].detach())
            dx = x_sorted[:, 1:] - x_sorted[:, :-1]

            self.register_buffer("_y_search", y_sorted.detach())
            self.register_buffer("_y0", y_sorted[:, :-1].detach())
            self.register_buffer("_y_min", y_sorted[:, :1].detach())
            self.register_buffer("_y_max", y_sorted[:, -1:].detach())
            dy = y_sorted[:, 1:] - y_sorted[:, :-1]

            self.register_buffer("_z_search", z_sorted.detach())
            self.register_buffer("_z0", z_sorted[:, :-1].detach())
            self.register_buffer("_z_min", z_sorted[:, :1].detach())
            self.register_buffer("_z_max", z_sorted[:, -1:].detach())
            dz = z_sorted[:, 1:] - z_sorted[:, :-1]

        if eps is None:
            eps = float(torch.finfo(x_sorted.dtype).eps)

        def safe_inv(d: torch.Tensor) -> torch.Tensor:
            sd = torch.where(d == 0, torch.full_like(d, eps), d)
            return (1.0 / sd).contiguous()

        self.register_buffer("_inv_dx", safe_inv(dx).detach())
        self.register_buffer("_inv_dy", safe_inv(dy).detach())
        self.register_buffer("_inv_dz", safe_inv(dz).detach())

        self._ix_hi = Nx - 2
        self._iy_hi = Ny - 2
        self._iz_hi = Nz - 2

        # Per-axis uniform detection + params
        self._uniform_x = False
        self._uniform_y = False
        self._uniform_z = False

        def detect_uniform(d: torch.Tensor) -> Tuple[bool, torch.Tensor]:
            # returns (ok, inv_dx0) where inv_dx0 is scalar or (D,1)
            if d.ndim == 1:
                dx0 = d[:1]
                ok = bool(torch.all(dx0 > 0).item()) and torch.allclose(
                    d, dx0.expand_as(d), rtol=uniform_rtol, atol=uniform_atol
                )
                safe_dx0 = torch.where(dx0 == 0, torch.full_like(dx0, eps), dx0)
                inv = (1.0 / safe_dx0)[0]
                return ok, inv
            else:
                dx0 = d[:, :1]
                ok = bool(torch.all(dx0 > 0).item()) and torch.allclose(
                    d, dx0.expand_as(d), rtol=uniform_rtol, atol=uniform_atol
                )
                safe_dx0 = torch.where(dx0 == 0, torch.full_like(dx0, eps), dx0)
                inv = (1.0 / safe_dx0).contiguous()
                return ok, inv

        if uniform != "never":
            okx, invx = detect_uniform(dx)
            oky, invy = detect_uniform(dy)
            okz, invz = detect_uniform(dz)

            if uniform == "always" and not (okx and oky and okz):
                raise ValueError(
                    "uniform='always' but not all axes are uniformly spaced."
                )

            if uniform == "auto":
                self._uniform_x, self._uniform_y, self._uniform_z = okx, oky, okz
            else:  # always
                self._uniform_x = self._uniform_y = self._uniform_z = True

            if self._uniform_x:
                self.register_buffer("_inv_dx0_x", invx.detach())
            if self._uniform_y:
                self.register_buffer("_inv_dy0_y", invy.detach())
            if self._uniform_z:
                self.register_buffer("_inv_dz0_z", invz.detach())

        # Store values
        if self.learnable_values:
            self.values = nn.Parameter(
                vals.contiguous(), requires_grad=values_requires_grad
            )
        else:
            self.register_buffer("_values", vals.detach().contiguous())

        # constant corner offsets for flattened indexing
        stride_x = Ny * Nz
        stride_y = Nz
        offsets = torch.tensor(
            [
                0,
                stride_x,
                stride_y,
                stride_x + stride_y,
                1,
                stride_x + 1,
                stride_y + 1,
                stride_x + stride_y + 1,
            ],
            device=values.device,
            dtype=torch.long,
        )
        self.register_buffer("_corner_offsets", offsets)

        self._ix_cache: Optional[torch.Tensor] = None
        self._iy_cache: Optional[torch.Tensor] = None
        self._iz_cache: Optional[torch.Tensor] = None

    def _as_index_tensor(
        self,
        indices: IndexLike,
        *,
        device: torch.device,
        expected_len: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(indices, int):
            idx = torch.tensor([indices], device=device, dtype=torch.long)
        else:
            idx = torch.as_tensor(indices, device=device, dtype=torch.long)

        if idx.ndim != 1:
            raise ValueError("indices must be a 1D sequence/tensor (or a single int).")
        if idx.numel() == 0:
            raise ValueError("indices must be non-empty.")
        if expected_len is not None and idx.numel() != expected_len:
            raise ValueError(
                f"indices must have length {expected_len}, got {idx.numel()}."
            )

        if self.batched:
            if torch.any((idx < 0) | (idx >= self.D)):
                raise ValueError(f"indices values must be in [0, {self.D - 1}].")
        else:
            if torch.any(idx != 0):
                raise ValueError("Unbatched: indices can only contain 0.")
        return idx

    def forward(
        self,
        xyz_new: torch.Tensor,
        *,
        indices: Optional[IndexLike] = None,
        outside: Optional[OutsideMode] = None,
        fill_value: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpolate at new query points.

        Parameters
        ----------
        xyz_new : torch.Tensor
            Shape (P,3) or (Q,P,3) (or generally (1,P,3) broadcastable).
            If 2D, single grid is used (unbatched). If 3D, Q is the batch size.
        indices : int or Sequence[int], optional
            If the interpolator is batched (D > 1), specifies which grid(s) to use.
            Can be a single int (applied to all points) or a sequence of length Q.
            If None and batched, then Q must equal D and all grids are used.
        outside : {"clamp", "zero", "fill"}, optional
            Out-of-bounds handling override. If None, uses the setting from constructor.
            - "clamp": values outside are clamped to endpoint values
            - "zero":  values outside are set to 0
            - "fill":  values outside are set to `fill_value`
        fill_value : float, optional
            Fill value to use if `outside="fill"`. If None, uses the setting from
            constructor.
        out : torch.Tensor, optional
            Optional output buffer. If provided, must have the correct shape.

        Returns
        -------
        values_new : torch.Tensor
            Interpolated values at `xyz_new`. Shape is (P,C) if unbatched,
            or (Q,P,C) if batched (or generally broadcastable to that).
        """
        outside = self.outside if outside is None else outside
        if outside not in ("clamp", "zero", "fill"):
            raise ValueError("outside must be one of: 'clamp', 'zero', 'fill'")
        fill_value = self.fill_value if fill_value is None else float(fill_value)

        if xyz_new.ndim not in (2, 3) or xyz_new.shape[-1] != 3:
            raise ValueError("xyz_new must have shape (P,3) or (Q,P,3).")
        if xyz_new.device != self._x_min.device or xyz_new.dtype != self._x_min.dtype:
            raise ValueError("xyz_new must match device and dtype of the interpolator.")
        if not torch.is_floating_point(xyz_new):
            raise TypeError("xyz_new must be floating-point.")

        pts_was_2d = xyz_new.ndim == 2
        xyzq = xyz_new[None, :, :] if pts_was_2d else xyz_new  # (Q,P,3)
        Q = int(xyzq.shape[0])
        P = int(xyzq.shape[1])

        xq = xyzq[..., 0]
        yq = xyzq[..., 1]
        zq = xyzq[..., 2]

        # Resolve batched mapping
        if not self.batched:
            if indices is not None:
                _ = self._as_index_tensor(indices, device=xyzq.device, expected_len=Q)
            idx = None
        else:
            if pts_was_2d:
                if indices is None:
                    raise ValueError(
                        "Ambiguous: batched grid but xyz_new is (P,3). Provide indices."
                    )
                idx = self._as_index_tensor(
                    indices, device=xyzq.device, expected_len=None
                )
                Q_eff = int(idx.numel())
                xyzq = xyzq.expand(Q_eff, -1, -1)
                xq = xyzq[..., 0]
                yq = xyzq[..., 1]
                zq = xyzq[..., 2]
                Q = Q_eff
            else:
                if indices is None:
                    if Q != self.D:
                        raise ValueError(
                            f"Ambiguous: D={self.D} but Q={Q}. Provide indices of length Q."
                        )
                    idx = None
                else:
                    idx = self._as_index_tensor(
                        indices, device=xyzq.device, expected_len=Q
                    )

        # Select rows / params
        if not self.batched:
            x_min, x_max = self._x_min, self._x_max
            y_min, y_max = self._y_min, self._y_max
            z_min, z_max = self._z_min, self._z_max
            x_search, y_search, z_search = (
                self._x_search,
                self._y_search,
                self._z_search,
            )
            x0, y0, z0 = self._x0, self._y0, self._z0
            inv_dx_tab, inv_dy_tab, inv_dz_tab = (
                self._inv_dx,
                self._inv_dy,
                self._inv_dz,
            )
            vals = self.values if self.learnable_values else self._values
            if self._uniform_x:
                inv_dx0 = self._inv_dx0_x
            if self._uniform_y:
                inv_dy0 = self._inv_dy0_y
            if self._uniform_z:
                inv_dz0 = self._inv_dz0_z
        else:
            if idx is None:
                x_min, x_max = self._x_min, self._x_max
                y_min, y_max = self._y_min, self._y_max
                z_min, z_max = self._z_min, self._z_max
                x_search, y_search, z_search = (
                    self._x_search,
                    self._y_search,
                    self._z_search,
                )
                x0, y0, z0 = self._x0, self._y0, self._z0
                inv_dx_tab, inv_dy_tab, inv_dz_tab = (
                    self._inv_dx,
                    self._inv_dy,
                    self._inv_dz,
                )
                vals = self.values if self.learnable_values else self._values
                if self._uniform_x:
                    inv_dx0 = self._inv_dx0_x
                if self._uniform_y:
                    inv_dy0 = self._inv_dy0_y
                if self._uniform_z:
                    inv_dz0 = self._inv_dz0_z
            else:
                x_min = self._x_min.index_select(0, idx)
                x_max = self._x_max.index_select(0, idx)
                y_min = self._y_min.index_select(0, idx)
                y_max = self._y_max.index_select(0, idx)
                z_min = self._z_min.index_select(0, idx)
                z_max = self._z_max.index_select(0, idx)

                x_search = self._x_search.index_select(0, idx)
                y_search = self._y_search.index_select(0, idx)
                z_search = self._z_search.index_select(0, idx)

                x0 = self._x0.index_select(0, idx)
                y0 = self._y0.index_select(0, idx)
                z0 = self._z0.index_select(0, idx)

                inv_dx_tab = self._inv_dx.index_select(0, idx)
                inv_dy_tab = self._inv_dy.index_select(0, idx)
                inv_dz_tab = self._inv_dz.index_select(0, idx)

                vals = (
                    self.values if self.learnable_values else self._values
                ).index_select(0, idx)

                if self._uniform_x:
                    inv_dx0 = self._inv_dx0_x.index_select(0, idx)
                if self._uniform_y:
                    inv_dy0 = self._inv_dy0_y.index_select(0, idx)
                if self._uniform_z:
                    inv_dz0 = self._inv_dz0_z.index_select(0, idx)

        # Clamp coords if requested (border semantics)
        if outside == "clamp":
            x_used = torch.clamp(xq, x_min, x_max)
            y_used = torch.clamp(yq, y_min, y_max)
            z_used = torch.clamp(zq, z_min, z_max)
        else:
            x_used, y_used, z_used = xq, yq, zq

        # Ensure caches
        def ensure(cache: Optional[torch.Tensor], shape, device):
            if cache is None or cache.shape != shape or cache.device != device:
                return torch.empty(shape, device=device, dtype=torch.long)
            return cache

        self._ix_cache = ensure(self._ix_cache, x_used.shape, x_used.device)
        self._iy_cache = ensure(self._iy_cache, y_used.shape, y_used.device)
        self._iz_cache = ensure(self._iz_cache, z_used.shape, z_used.device)
        ix, iy, iz = self._ix_cache, self._iy_cache, self._iz_cache

        # Axis helper: compute (ind,t) either uniform or searchsorted
        def axis_ind_t_uniform(ucoord, amin, inv_d0, hi, ind_out):
            u = (ucoord - amin) * inv_d0
            ind_out.copy_(u.to(torch.long))
            ind_out.clamp_(0, hi)
            t = u - ind_out.to(u.dtype)
            if outside == "clamp":
                t = torch.clamp(t, 0.0, 1.0)
            return ind_out, t

        def axis_ind_t_nu(search, ucoord, a0, inv_tab, hi, ind_out, batched_axis: bool):
            torch.searchsorted(search, ucoord.contiguous(), out=ind_out)
            ind_out -= 1
            ind_out.clamp_(0, hi)
            if not batched_axis:
                a0i = a0[ind_out]
                inv_i = inv_tab[ind_out]
            else:
                a0i = torch.gather(a0, 1, ind_out)
                inv_i = torch.gather(inv_tab, 1, ind_out)
            t = (ucoord - a0i) * inv_i
            return ind_out, t

        if not self.batched:
            ix, tx = (
                axis_ind_t_uniform(x_used, x_min, inv_dx0, self._ix_hi, ix)
                if self._uniform_x
                else axis_ind_t_nu(
                    x_search, x_used, x0, inv_dx_tab, self._ix_hi, ix, False
                )
            )
            iy, ty = (
                axis_ind_t_uniform(y_used, y_min, inv_dy0, self._iy_hi, iy)
                if self._uniform_y
                else axis_ind_t_nu(
                    y_search, y_used, y0, inv_dy_tab, self._iy_hi, iy, False
                )
            )
            iz, tz = (
                axis_ind_t_uniform(z_used, z_min, inv_dz0, self._iz_hi, iz)
                if self._uniform_z
                else axis_ind_t_nu(
                    z_search, z_used, z0, inv_dz_tab, self._iz_hi, iz, False
                )
            )
        else:
            ix, tx = (
                axis_ind_t_uniform(x_used, x_min, inv_dx0, self._ix_hi, ix)
                if self._uniform_x
                else axis_ind_t_nu(
                    x_search, x_used, x0, inv_dx_tab, self._ix_hi, ix, True
                )
            )
            iy, ty = (
                axis_ind_t_uniform(y_used, y_min, inv_dy0, self._iy_hi, iy)
                if self._uniform_y
                else axis_ind_t_nu(
                    y_search, y_used, y0, inv_dy_tab, self._iy_hi, iy, True
                )
            )
            iz, tz = (
                axis_ind_t_uniform(z_used, z_min, inv_dz0, self._iz_hi, iz)
                if self._uniform_z
                else axis_ind_t_nu(
                    z_search, z_used, z0, inv_dz_tab, self._iz_hi, iz, True
                )
            )

        # Gather 8 corners via flattened indexing
        base = (ix * self.Ny + iy) * self.Nz + iz  # (Q,P)
        idxs = base.unsqueeze(-1) + self._corner_offsets  # (Q,P,8)

        if not self.batched:
            v_flat = vals.reshape(-1, self.C)  # (M,C)
            idxs_flat = idxs.reshape(Q, -1)  # (Q,8P)
            corners = v_flat[idxs_flat].view(Q, P, 8, self.C)
        else:
            v_flat = vals.reshape(Q, -1, self.C)  # (Q,M,C)
            idxs_flat = idxs.reshape(Q, -1)  # (Q,8P)
            idxs_exp = idxs_flat.unsqueeze(-1).expand(-1, -1, self.C)
            corners = torch.gather(v_flat, 1, idxs_exp).view(Q, P, 8, self.C)

        # Trilinear blending
        txe = tx.unsqueeze(-1)
        tye = ty.unsqueeze(-1)
        tze = tz.unsqueeze(-1)

        v000 = corners[:, :, 0, :]
        v100 = corners[:, :, 1, :]
        v010 = corners[:, :, 2, :]
        v110 = corners[:, :, 3, :]
        v001 = corners[:, :, 4, :]
        v101 = corners[:, :, 5, :]
        v011 = corners[:, :, 6, :]
        v111 = corners[:, :, 7, :]

        v00 = v000 + (v100 - v000) * txe
        v10 = v010 + (v110 - v010) * txe
        v01 = v001 + (v101 - v001) * txe
        v11 = v011 + (v111 - v011) * txe

        v0 = v00 + (v10 - v00) * tye
        v1 = v01 + (v11 - v01) * tye

        ynew = v0 + (v1 - v0) * tze  # (Q,P,C)

        # Outside mask for zero/fill
        if outside in ("zero", "fill"):
            outside_mask = (
                (xq < x_min)
                | (xq > x_max)
                | (yq < y_min)
                | (yq > y_max)
                | (zq < z_min)
                | (zq > z_max)
            )
            ynew = ynew.masked_fill(
                outside_mask.unsqueeze(-1), 0.0 if outside == "zero" else fill_value
            )

        if not self._had_channels:
            ynew = ynew.squeeze(-1)  # (Q,P)

        if pts_was_2d and ynew.shape[0] == 1:
            ynew = ynew[0]

        if out is not None:
            if out.shape != ynew.shape:
                raise ValueError(
                    f"out has shape {tuple(out.shape)} but expected {tuple(ynew.shape)}."
                )
            out.copy_(ynew)
            return out

        return ynew

    def laplacian_values(
        self,
        *,
        boundary: BoundaryMode = "valid",
    ) -> torch.Tensor:
        """
        Compute the discrete (finite-difference) Laplacian of the *grid values*.

        This is NOT the analytic Laplacian of the trilinear interpolant (which is 0
        within each cell). Instead, it computes a second-derivative stencil along each
        axis on the rectilinear grid and returns:

            lap = d2/dx2(values) + d2/dy2(values) + d2/dz2(values)

        Parameters
        ----------
        boundary : {"replicate", "zero", "one-sided", "valid"}, optional
            How to fill the Laplacian at the boundary indices along each axis:
            - "valid" (default): only compute Laplacian at fully interior nodes (requires Nx, Ny, Nz >= 3)
            - "replicate": copy nearest interior value (e.g. lap[0]=lap[1])
            - "zero": leave boundary Laplacian as 0
            - "one-sided": use a 3-point one-sided second-derivative formula at the edges

        Returns
        -------
        lap : torch.Tensor
            Same shape as the original `values` passed to __init__:
            - unbatched: (Nx,Ny,Nz) or (Nx,Ny,Nz,C)
            - batched:   (D,Nx,Ny,Nz) or (D,Nx,Ny,Nz,C)
        """
        if boundary not in ("replicate", "zero", "one-sided", "valid"):
            raise ValueError(
                "boundary must be 'replicate', 'zero', 'one-sided', or 'valid'"
            )

        vals = self.values if self.learnable_values else self._values  # channels-last
        dtype = vals.dtype
        eps = float(torch.finfo(dtype).eps)

        def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
            den = torch.where(den == 0, torch.full_like(den, eps), den)
            return num / den

        if boundary == "valid":
            # Need at least 3 nodes along each axis to have any valid interior
            if self.Nx < 3 or self.Ny < 3 or self.Nz < 3:
                raise ValueError("boundary='valid' requires Nx, Ny, Nz >= 3.")

            if not self.batched:
                f = vals  # (Nx,Ny,Nz,C)
                f0 = f[1:-1, 1:-1, 1:-1, :]  # common center (Nx-2,Ny-2,Nz-2,C)

                # x second derivative at fully interior nodes
                x = self._x_search  # (Nx,)
                h0x = x[1:-1] - x[:-2]  # (Nx-2,)
                h1x = x[2:] - x[1:-1]  # (Nx-2,)
                denx = h0x * h1x * (h0x + h1x)  # (Nx-2,)
                h0x = h0x[:, None, None, None]
                h1x = h1x[:, None, None, None]
                denx = denx[:, None, None, None]

                fxm = f[:-2, 1:-1, 1:-1, :]
                fxp = f[2:, 1:-1, 1:-1, :]
                d2x = _safe_div(2.0 * (h0x * fxp - (h0x + h1x) * f0 + h1x * fxm), denx)

                # y second derivative at fully interior nodes
                y = self._y_search  # (Ny,)
                h0y = y[1:-1] - y[:-2]  # (Ny-2,)
                h1y = y[2:] - y[1:-1]  # (Ny-2,)
                deny = h0y * h1y * (h0y + h1y)  # (Ny-2,)
                h0y = h0y[None, :, None, None]
                h1y = h1y[None, :, None, None]
                deny = deny[None, :, None, None]

                fym = f[1:-1, :-2, 1:-1, :]
                fyp = f[1:-1, 2:, 1:-1, :]
                d2y = _safe_div(2.0 * (h0y * fyp - (h0y + h1y) * f0 + h1y * fym), deny)

                # z second derivative at fully interior nodes
                z = self._z_search  # (Nz,)
                h0z = z[1:-1] - z[:-2]  # (Nz-2,)
                h1z = z[2:] - z[1:-1]  # (Nz-2,)
                denz = h0z * h1z * (h0z + h1z)  # (Nz-2,)
                h0z = h0z[None, None, :, None]
                h1z = h1z[None, None, :, None]
                denz = denz[None, None, :, None]

                fzm = f[1:-1, 1:-1, :-2, :]
                fzp = f[1:-1, 1:-1, 2:, :]
                d2z = _safe_div(2.0 * (h0z * fzp - (h0z + h1z) * f0 + h1z * fzm), denz)

                lap = d2x + d2y + d2z  # (Nx-2,Ny-2,Nz-2,C)

            else:
                f = vals  # (D,Nx,Ny,Nz,C)
                f0 = f[:, 1:-1, 1:-1, 1:-1, :]  # (D,Nx-2,Ny-2,Nz-2,C)

                # x
                x = self._x_search  # (D,Nx)
                h0x = x[:, 1:-1] - x[:, :-2]  # (D,Nx-2)
                h1x = x[:, 2:] - x[:, 1:-1]  # (D,Nx-2)
                denx = h0x * h1x * (h0x + h1x)  # (D,Nx-2)
                h0x = h0x[:, :, None, None, None]
                h1x = h1x[:, :, None, None, None]
                denx = denx[:, :, None, None, None]

                fxm = f[:, :-2, 1:-1, 1:-1, :]
                fxp = f[:, 2:, 1:-1, 1:-1, :]
                d2x = _safe_div(2.0 * (h0x * fxp - (h0x + h1x) * f0 + h1x * fxm), denx)

                # y
                y = self._y_search  # (D,Ny)
                h0y = y[:, 1:-1] - y[:, :-2]  # (D,Ny-2)
                h1y = y[:, 2:] - y[:, 1:-1]  # (D,Ny-2)
                deny = h0y * h1y * (h0y + h1y)  # (D,Ny-2)
                h0y = h0y[:, None, :, None, None]
                h1y = h1y[:, None, :, None, None]
                deny = deny[:, None, :, None, None]

                fym = f[:, 1:-1, :-2, 1:-1, :]
                fyp = f[:, 1:-1, 2:, 1:-1, :]
                d2y = _safe_div(2.0 * (h0y * fyp - (h0y + h1y) * f0 + h1y * fym), deny)

                # z
                z = self._z_search  # (D,Nz)
                h0z = z[:, 1:-1] - z[:, :-2]  # (D,Nz-2)
                h1z = z[:, 2:] - z[:, 1:-1]  # (D,Nz-2)
                denz = h0z * h1z * (h0z + h1z)  # (D,Nz-2)
                h0z = h0z[:, None, None, :, None]
                h1z = h1z[:, None, None, :, None]
                denz = denz[:, None, None, :, None]

                fzm = f[:, 1:-1, 1:-1, :-2, :]
                fzp = f[:, 1:-1, 1:-1, 2:, :]
                d2z = _safe_div(2.0 * (h0z * fzp - (h0z + h1z) * f0 + h1z * fzm), denz)

                lap = d2x + d2y + d2z  # (D,Nx-2,Ny-2,Nz-2,C)

            if not self._had_channels:
                lap = lap.squeeze(-1)

            return lap

        # -----------------------------
        # Unbatched helpers
        # -----------------------------
        def _d2_unbatched_x(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # f: (Nx,Ny,Nz,C), x: (Nx,)
            Nx = f.shape[0]
            if Nx < 3:
                return torch.zeros_like(f)

            d2 = torch.zeros_like(f)

            h0 = x[1:-1] - x[:-2]  # (Nx-2,)
            h1 = x[2:] - x[1:-1]  # (Nx-2,)
            den = h0 * h1 * (h0 + h1)  # (Nx-2,)

            h0b = h0[:, None, None, None]
            h1b = h1[:, None, None, None]
            denb = den[:, None, None, None]

            fm = f[:-2, :, :, :]
            f0 = f[1:-1, :, :, :]
            fp = f[2:, :, :, :]

            d2_inner = _safe_div(
                2.0 * (h0b * fp - (h0b + h1b) * f0 + h1b * fm),
                denb,
            )
            d2[1:-1, :, :, :] = d2_inner

            if boundary == "replicate":
                d2[0, :, :, :] = d2[1, :, :, :]
                d2[-1, :, :, :] = d2[-2, :, :, :]
            elif boundary == "one-sided":
                # left edge i=0 using points 0,1,2
                h0l = x[1] - x[0]
                h1l = x[2] - x[1]
                den01 = torch.where(
                    h0l * h1l == 0, torch.full_like(h0l, eps), h0l * h1l
                )
                den0 = torch.where(
                    h0l * (h0l + h1l) == 0, torch.full_like(h0l, eps), h0l * (h0l + h1l)
                )
                den2 = torch.where(
                    h1l * (h0l + h1l) == 0, torch.full_like(h1l, eps), h1l * (h0l + h1l)
                )
                c0 = 2.0 / den0
                c1 = -2.0 / den01
                c2 = 2.0 / den2
                d2[0, :, :, :] = (
                    c0 * f[0, :, :, :] + c1 * f[1, :, :, :] + c2 * f[2, :, :, :]
                )

                # right edge i=Nx-1 using points Nx-3, Nx-2, Nx-1
                h0r = x[-2] - x[-3]
                h1r = x[-1] - x[-2]
                den01 = torch.where(
                    h0r * h1r == 0, torch.full_like(h0r, eps), h0r * h1r
                )
                den0 = torch.where(
                    h0r * (h0r + h1r) == 0, torch.full_like(h0r, eps), h0r * (h0r + h1r)
                )
                den2 = torch.where(
                    h1r * (h0r + h1r) == 0, torch.full_like(h1r, eps), h1r * (h0r + h1r)
                )
                c0 = 2.0 / den0
                c1 = -2.0 / den01
                c2 = 2.0 / den2
                d2[-1, :, :, :] = (
                    c0 * f[-3, :, :, :] + c1 * f[-2, :, :, :] + c2 * f[-1, :, :, :]
                )

            # boundary == "zero": leave as zeros
            return d2

        def _d2_unbatched_y(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # f: (Nx,Ny,Nz,C), y: (Ny,)
            Ny = f.shape[1]
            if Ny < 3:
                return torch.zeros_like(f)

            d2 = torch.zeros_like(f)

            h0 = y[1:-1] - y[:-2]  # (Ny-2,)
            h1 = y[2:] - y[1:-1]  # (Ny-2,)
            den = h0 * h1 * (h0 + h1)  # (Ny-2,)

            h0b = h0[None, :, None, None]
            h1b = h1[None, :, None, None]
            denb = den[None, :, None, None]

            fm = f[:, :-2, :, :]
            f0 = f[:, 1:-1, :, :]
            fp = f[:, 2:, :, :]

            d2_inner = _safe_div(
                2.0 * (h0b * fp - (h0b + h1b) * f0 + h1b * fm),
                denb,
            )
            d2[:, 1:-1, :, :] = d2_inner

            if boundary == "replicate":
                d2[:, 0, :, :] = d2[:, 1, :, :]
                d2[:, -1, :, :] = d2[:, -2, :, :]
            elif boundary == "one-sided":
                h0l = y[1] - y[0]
                h1l = y[2] - y[1]
                den01 = torch.where(
                    h0l * h1l == 0, torch.full_like(h0l, eps), h0l * h1l
                )
                den0 = torch.where(
                    h0l * (h0l + h1l) == 0, torch.full_like(h0l, eps), h0l * (h0l + h1l)
                )
                den2 = torch.where(
                    h1l * (h0l + h1l) == 0, torch.full_like(h1l, eps), h1l * (h0l + h1l)
                )
                c0 = 2.0 / den0
                c1 = -2.0 / den01
                c2 = 2.0 / den2
                d2[:, 0, :, :] = (
                    c0 * f[:, 0, :, :] + c1 * f[:, 1, :, :] + c2 * f[:, 2, :, :]
                )

                h0r = y[-2] - y[-3]
                h1r = y[-1] - y[-2]
                den01 = torch.where(
                    h0r * h1r == 0, torch.full_like(h0r, eps), h0r * h1r
                )
                den0 = torch.where(
                    h0r * (h0r + h1r) == 0, torch.full_like(h0r, eps), h0r * (h0r + h1r)
                )
                den2 = torch.where(
                    h1r * (h0r + h1r) == 0, torch.full_like(h1r, eps), h1r * (h0r + h1r)
                )
                c0 = 2.0 / den0
                c1 = -2.0 / den01
                c2 = 2.0 / den2
                d2[:, -1, :, :] = (
                    c0 * f[:, -3, :, :] + c1 * f[:, -2, :, :] + c2 * f[:, -1, :, :]
                )

            return d2

        def _d2_unbatched_z(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            # f: (Nx,Ny,Nz,C), z: (Nz,)
            Nz = f.shape[2]
            if Nz < 3:
                return torch.zeros_like(f)

            d2 = torch.zeros_like(f)

            h0 = z[1:-1] - z[:-2]  # (Nz-2,)
            h1 = z[2:] - z[1:-1]  # (Nz-2,)
            den = h0 * h1 * (h0 + h1)  # (Nz-2,)

            h0b = h0[None, None, :, None]
            h1b = h1[None, None, :, None]
            denb = den[None, None, :, None]

            fm = f[:, :, :-2, :]
            f0 = f[:, :, 1:-1, :]
            fp = f[:, :, 2:, :]

            d2_inner = _safe_div(
                2.0 * (h0b * fp - (h0b + h1b) * f0 + h1b * fm),
                denb,
            )
            d2[:, :, 1:-1, :] = d2_inner

            if boundary == "replicate":
                d2[:, :, 0, :] = d2[:, :, 1, :]
                d2[:, :, -1, :] = d2[:, :, -2, :]
            elif boundary == "one-sided":
                h0l = z[1] - z[0]
                h1l = z[2] - z[1]
                den01 = torch.where(
                    h0l * h1l == 0, torch.full_like(h0l, eps), h0l * h1l
                )
                den0 = torch.where(
                    h0l * (h0l + h1l) == 0, torch.full_like(h0l, eps), h0l * (h0l + h1l)
                )
                den2 = torch.where(
                    h1l * (h0l + h1l) == 0, torch.full_like(h1l, eps), h1l * (h0l + h1l)
                )
                c0 = 2.0 / den0
                c1 = -2.0 / den01
                c2 = 2.0 / den2
                d2[:, :, 0, :] = (
                    c0 * f[:, :, 0, :] + c1 * f[:, :, 1, :] + c2 * f[:, :, 2, :]
                )

                h0r = z[-2] - z[-3]
                h1r = z[-1] - z[-2]
                den01 = torch.where(
                    h0r * h1r == 0, torch.full_like(h0r, eps), h0r * h1r
                )
                den0 = torch.where(
                    h0r * (h0r + h1r) == 0, torch.full_like(h0r, eps), h0r * (h0r + h1r)
                )
                den2 = torch.where(
                    h1r * (h0r + h1r) == 0, torch.full_like(h1r, eps), h1r * (h0r + h1r)
                )
                c0 = 2.0 / den0
                c1 = -2.0 / den01
                c2 = 2.0 / den2
                d2[:, :, -1, :] = (
                    c0 * f[:, :, -3, :] + c1 * f[:, :, -2, :] + c2 * f[:, :, -1, :]
                )

            return d2

        # -----------------------------
        # Batched helpers
        # -----------------------------
        def _d2_batched_x(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # f: (D,Nx,Ny,Nz,C), x: (D,Nx)
            Nx = f.shape[1]
            if Nx < 3:
                return torch.zeros_like(f)

            d2 = torch.zeros_like(f)

            h0 = x[:, 1:-1] - x[:, :-2]  # (D,Nx-2)
            h1 = x[:, 2:] - x[:, 1:-1]  # (D,Nx-2)
            den = h0 * h1 * (h0 + h1)  # (D,Nx-2)

            h0b = h0[:, :, None, None, None]
            h1b = h1[:, :, None, None, None]
            denb = den[:, :, None, None, None]

            fm = f[:, :-2, :, :, :]
            f0 = f[:, 1:-1, :, :, :]
            fp = f[:, 2:, :, :, :]

            d2_inner = _safe_div(
                2.0 * (h0b * fp - (h0b + h1b) * f0 + h1b * fm),
                denb,
            )
            d2[:, 1:-1, :, :, :] = d2_inner

            if boundary == "replicate":
                d2[:, 0, :, :, :] = d2[:, 1, :, :, :]
                d2[:, -1, :, :, :] = d2[:, -2, :, :, :]
            elif boundary == "one-sided":
                # left boundary (i=0): use points 0,1,2
                h0l = x[:, 1] - x[:, 0]  # (D,)
                h1l = x[:, 2] - x[:, 1]  # (D,)
                den01 = torch.where(
                    h0l * h1l == 0, torch.full_like(h0l, eps), h0l * h1l
                )
                den0 = torch.where(
                    h0l * (h0l + h1l) == 0, torch.full_like(h0l, eps), h0l * (h0l + h1l)
                )
                den2 = torch.where(
                    h1l * (h0l + h1l) == 0, torch.full_like(h1l, eps), h1l * (h0l + h1l)
                )
                c0 = (2.0 / den0)[:, None, None, None]
                c1 = (-2.0 / den01)[:, None, None, None]
                c2 = (2.0 / den2)[:, None, None, None]
                d2[:, 0, :, :, :] = (
                    c0 * f[:, 0, :, :, :]
                    + c1 * f[:, 1, :, :, :]
                    + c2 * f[:, 2, :, :, :]
                )

                # right boundary (i=Nx-1): use points Nx-3, Nx-2, Nx-1
                h0r = x[:, -2] - x[:, -3]
                h1r = x[:, -1] - x[:, -2]
                den01 = torch.where(
                    h0r * h1r == 0, torch.full_like(h0r, eps), h0r * h1r
                )
                den0 = torch.where(
                    h0r * (h0r + h1r) == 0, torch.full_like(h0r, eps), h0r * (h0r + h1r)
                )
                den2 = torch.where(
                    h1r * (h0r + h1r) == 0, torch.full_like(h1r, eps), h1r * (h0r + h1r)
                )
                c0 = (2.0 / den0)[:, None, None, None]
                c1 = (-2.0 / den01)[:, None, None, None]
                c2 = (2.0 / den2)[:, None, None, None]
                d2[:, -1, :, :, :] = (
                    c0 * f[:, -3, :, :, :]
                    + c1 * f[:, -2, :, :, :]
                    + c2 * f[:, -1, :, :, :]
                )

            return d2

        def _d2_batched_y(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # f: (D,Nx,Ny,Nz,C), y: (D,Ny)
            Ny = f.shape[2]
            if Ny < 3:
                return torch.zeros_like(f)

            d2 = torch.zeros_like(f)

            h0 = y[:, 1:-1] - y[:, :-2]  # (D,Ny-2)
            h1 = y[:, 2:] - y[:, 1:-1]  # (D,Ny-2)
            den = h0 * h1 * (h0 + h1)  # (D,Ny-2)

            h0b = h0[:, None, :, None, None]
            h1b = h1[:, None, :, None, None]
            denb = den[:, None, :, None, None]

            fm = f[:, :, :-2, :, :]
            f0 = f[:, :, 1:-1, :, :]
            fp = f[:, :, 2:, :, :]

            d2_inner = _safe_div(
                2.0 * (h0b * fp - (h0b + h1b) * f0 + h1b * fm),
                denb,
            )
            d2[:, :, 1:-1, :, :] = d2_inner

            if boundary == "replicate":
                d2[:, :, 0, :, :] = d2[:, :, 1, :, :]
                d2[:, :, -1, :, :] = d2[:, :, -2, :, :]
            elif boundary == "one-sided":
                h0l = y[:, 1] - y[:, 0]
                h1l = y[:, 2] - y[:, 1]
                den01 = torch.where(
                    h0l * h1l == 0, torch.full_like(h0l, eps), h0l * h1l
                )
                den0 = torch.where(
                    h0l * (h0l + h1l) == 0, torch.full_like(h0l, eps), h0l * (h0l + h1l)
                )
                den2 = torch.where(
                    h1l * (h0l + h1l) == 0, torch.full_like(h1l, eps), h1l * (h0l + h1l)
                )
                c0 = (2.0 / den0)[:, None, None, None]
                c1 = (-2.0 / den01)[:, None, None, None]
                c2 = (2.0 / den2)[:, None, None, None]
                d2[:, :, 0, :, :] = (
                    c0 * f[:, :, 0, :, :]
                    + c1 * f[:, :, 1, :, :]
                    + c2 * f[:, :, 2, :, :]
                )

                h0r = y[:, -2] - y[:, -3]
                h1r = y[:, -1] - y[:, -2]
                den01 = torch.where(
                    h0r * h1r == 0, torch.full_like(h0r, eps), h0r * h1r
                )
                den0 = torch.where(
                    h0r * (h0r + h1r) == 0, torch.full_like(h0r, eps), h0r * (h0r + h1r)
                )
                den2 = torch.where(
                    h1r * (h0r + h1r) == 0, torch.full_like(h1r, eps), h1r * (h0r + h1r)
                )
                c0 = (2.0 / den0)[:, None, None, None]
                c1 = (-2.0 / den01)[:, None, None, None]
                c2 = (2.0 / den2)[:, None, None, None]
                d2[:, :, -1, :, :] = (
                    c0 * f[:, :, -3, :, :]
                    + c1 * f[:, :, -2, :, :]
                    + c2 * f[:, :, -1, :, :]
                )

            return d2

        def _d2_batched_z(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
            # f: (D,Nx,Ny,Nz,C), z: (D,Nz)
            Nz = f.shape[3]
            if Nz < 3:
                return torch.zeros_like(f)

            d2 = torch.zeros_like(f)

            h0 = z[:, 1:-1] - z[:, :-2]  # (D,Nz-2)
            h1 = z[:, 2:] - z[:, 1:-1]  # (D,Nz-2)
            den = h0 * h1 * (h0 + h1)  # (D,Nz-2)

            h0b = h0[:, None, None, :, None]
            h1b = h1[:, None, None, :, None]
            denb = den[:, None, None, :, None]

            fm = f[:, :, :, :-2, :]
            f0 = f[:, :, :, 1:-1, :]
            fp = f[:, :, :, 2:, :]

            d2_inner = _safe_div(
                2.0 * (h0b * fp - (h0b + h1b) * f0 + h1b * fm),
                denb,
            )
            d2[:, :, :, 1:-1, :] = d2_inner

            if boundary == "replicate":
                d2[:, :, :, 0, :] = d2[:, :, :, 1, :]
                d2[:, :, :, -1, :] = d2[:, :, :, -2, :]
            elif boundary == "one-sided":
                h0l = z[:, 1] - z[:, 0]
                h1l = z[:, 2] - z[:, 1]
                den01 = torch.where(
                    h0l * h1l == 0, torch.full_like(h0l, eps), h0l * h1l
                )
                den0 = torch.where(
                    h0l * (h0l + h1l) == 0, torch.full_like(h0l, eps), h0l * (h0l + h1l)
                )
                den2 = torch.where(
                    h1l * (h0l + h1l) == 0, torch.full_like(h1l, eps), h1l * (h0l + h1l)
                )
                c0 = (2.0 / den0)[:, None, None, None]
                c1 = (-2.0 / den01)[:, None, None, None]
                c2 = (2.0 / den2)[:, None, None, None]
                d2[:, :, :, 0, :] = (
                    c0 * f[:, :, :, 0, :]
                    + c1 * f[:, :, :, 1, :]
                    + c2 * f[:, :, :, 2, :]
                )

                h0r = z[:, -2] - z[:, -3]
                h1r = z[:, -1] - z[:, -2]
                den01 = torch.where(
                    h0r * h1r == 0, torch.full_like(h0r, eps), h0r * h1r
                )
                den0 = torch.where(
                    h0r * (h0r + h1r) == 0, torch.full_like(h0r, eps), h0r * (h0r + h1r)
                )
                den2 = torch.where(
                    h1r * (h0r + h1r) == 0, torch.full_like(h1r, eps), h1r * (h0r + h1r)
                )
                c0 = (2.0 / den0)[:, None, None, None]
                c1 = (-2.0 / den01)[:, None, None, None]
                c2 = (2.0 / den2)[:, None, None, None]
                d2[:, :, :, -1, :] = (
                    c0 * f[:, :, :, -3, :]
                    + c1 * f[:, :, :, -2, :]
                    + c2 * f[:, :, :, -1, :]
                )

            return d2

        # -----------------------------
        # Compute Laplacian
        # -----------------------------
        if not self.batched:
            # sorted axes are stored in _x_search/_y_search/_z_search
            d2x = _d2_unbatched_x(vals, self._x_search)
            d2y = _d2_unbatched_y(vals, self._y_search)
            d2z = _d2_unbatched_z(vals, self._z_search)
            lap = d2x + d2y + d2z
        else:
            d2x = _d2_batched_x(vals, self._x_search)
            d2y = _d2_batched_y(vals, self._y_search)
            d2z = _d2_batched_z(vals, self._z_search)
            lap = d2x + d2y + d2z

        # match original "had channels" convention
        if not self._had_channels:
            lap = lap.squeeze(-1)

        return lap


class PreparedInterp3dRectUniform(nn.Module):
    """
    Prepared trilinear interpolator on a uniform rectilinear (tensor-product) grid.

    Same regimes as the non-uniform 3D rectilinear interpolator, but assumes each
    axis is uniformly spaced (per row, if batched).
    """

    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        values: torch.Tensor,
        *,
        outside: OutsideMode = "clamp",
        fill_value: float = 0.0,
        learnable_values: bool = False,
        values_requires_grad: bool = True,
        sort_xyz: bool = True,
        check_uniform: bool = False,
        rtol: float = 1e-5,
        atol: float = 1e-7,
        eps: Optional[float] = None,
    ):
        super().__init__()

        if outside not in ("clamp", "zero", "fill"):
            raise ValueError("outside must be one of: 'clamp', 'zero', 'fill'")

        if not (x.ndim == y.ndim == z.ndim) or x.ndim not in (1, 2):
            raise ValueError(
                "x, y, z must have the same ndim, either all 1D or all 2D."
            )

        if x.device != y.device or x.device != z.device or x.device != values.device:
            raise ValueError("x, y, z, values must be on the same device.")
        if x.dtype != y.dtype or x.dtype != z.dtype or x.dtype != values.dtype:
            raise ValueError("x, y, z, values must have the same dtype.")
        if not torch.is_floating_point(x):
            raise TypeError("x, y, z, values must be floating-point.")

        self.outside = outside
        self.fill_value = float(fill_value)
        self.learnable_values = bool(learnable_values)

        self.batched = x.ndim == 2

        # Validate shapes vs values
        if not self.batched:
            Nx, Ny, Nz = int(x.shape[0]), int(y.shape[0]), int(z.shape[0])
            if values.ndim not in (3, 4):
                raise ValueError("Unbatched values must be (Nx,Ny,Nz) or (Nx,Ny,Nz,C).")
            if values.shape[0] != Nx or values.shape[1] != Ny or values.shape[2] != Nz:
                raise ValueError(
                    "Unbatched values must match (Nx,Ny,Nz) in its first 3 dims."
                )
            self.D = 1
        else:
            D = int(x.shape[0])
            if y.shape[0] != D or z.shape[0] != D:
                raise ValueError("Batched: x,y,z must share leading dim D.")
            Nx, Ny, Nz = int(x.shape[1]), int(y.shape[1]), int(z.shape[1])
            if values.ndim not in (4, 5):
                raise ValueError(
                    "Batched values must be (D,Nx,Ny,Nz) or (D,Nx,Ny,Nz,C)."
                )
            if (
                values.shape[0] != D
                or values.shape[1] != Nx
                or values.shape[2] != Ny
                or values.shape[3] != Nz
            ):
                raise ValueError(
                    "Batched values must match (D,Nx,Ny,Nz,...) in its first 4 dims."
                )
            self.D = D

        if Nx < 2 or Ny < 2 or Nz < 2:
            raise ValueError("Need at least 2 samples along each axis.")

        self.Nx, self.Ny, self.Nz = Nx, Ny, Nz

        # Normalize values to have explicit channels-last
        self._had_channels = values.ndim == (4 if not self.batched else 5)
        if not self._had_channels:
            values = values.unsqueeze(-1)
        self.C = int(values.shape[-1])

        # Optional sort + permute values accordingly
        if sort_xyz:
            if not self.batched:
                x_sorted, px = torch.sort(x, dim=0)
                y_sorted, py = torch.sort(y, dim=0)
                z_sorted, pz = torch.sort(z, dim=0)

                vals = values.index_select(0, px)
                vals = vals.index_select(1, py)
                vals = vals.index_select(2, pz)
            else:
                x_sorted, px = torch.sort(x, dim=1)
                y_sorted, py = torch.sort(y, dim=1)
                z_sorted, pz = torch.sort(z, dim=1)

                vals = values
                ix = px[:, :, None, None, None].expand(-1, -1, Ny, Nz, self.C)
                vals = torch.gather(vals, dim=1, index=ix)
                iy = py[:, None, :, None, None].expand(-1, Nx, -1, Nz, self.C)
                vals = torch.gather(vals, dim=2, index=iy)
                iz = pz[:, None, None, :, None].expand(-1, Nx, Ny, -1, self.C)
                vals = torch.gather(vals, dim=3, index=iz)
        else:
            x_sorted, y_sorted, z_sorted = (
                x.contiguous(),
                y.contiguous(),
                z.contiguous(),
            )
            vals = values.contiguous()

        # Uniform-axis params per row
        if eps is None:
            eps = float(torch.finfo(x_sorted.dtype).eps)

        def axis_params(axis: torch.Tensor, name: str):
            if axis.ndim == 1:
                d = axis[1:] - axis[:-1]
                d0 = d[:1]
                if check_uniform:
                    if not torch.allclose(d, d0.expand_as(d), rtol=rtol, atol=atol):
                        raise ValueError(
                            f"{name} is not uniformly spaced (check_uniform=True)."
                        )
                if torch.any(d0 <= 0):
                    raise ValueError(
                        f"{name} must be strictly increasing (after sorting, if enabled)."
                    )
                a_min = axis[0]
                a_max = axis[-1]
                safe = torch.where(d0 == 0, torch.full_like(d0, eps), d0)
                inv = (1.0 / safe)[0]
                return a_min.detach(), a_max.detach(), inv.detach()
            else:
                d = axis[:, 1:] - axis[:, :-1]  # (D,N-1)
                d0 = d[:, :1]  # (D,1)
                if check_uniform:
                    if not torch.allclose(d, d0.expand_as(d), rtol=rtol, atol=atol):
                        raise ValueError(
                            f"{name} is not uniformly spaced per row (check_uniform=True)."
                        )
                if torch.any(d0 <= 0):
                    raise ValueError(
                        f"{name} rows must be strictly increasing (after sorting, if enabled)."
                    )
                a_min = axis[:, :1]
                a_max = axis[:, -1:]
                safe = torch.where(d0 == 0, torch.full_like(d0, eps), d0)
                inv = (1.0 / safe).contiguous()
                return a_min.detach(), a_max.detach(), inv.detach()

        x_min, x_max, inv_dx = axis_params(x_sorted, "x")
        y_min, y_max, inv_dy = axis_params(y_sorted, "y")
        z_min, z_max, inv_dz = axis_params(z_sorted, "z")

        self.register_buffer("_x_min", x_min)
        self.register_buffer("_x_max", x_max)
        self.register_buffer("_inv_dx", inv_dx)
        self.register_buffer("_y_min", y_min)
        self.register_buffer("_y_max", y_max)
        self.register_buffer("_inv_dy", inv_dy)
        self.register_buffer("_z_min", z_min)
        self.register_buffer("_z_max", z_max)
        self.register_buffer("_inv_dz", inv_dz)

        self._ix_hi = Nx - 2
        self._iy_hi = Ny - 2
        self._iz_hi = Nz - 2

        # Store values
        if self.learnable_values:
            self.values = nn.Parameter(
                vals.contiguous(), requires_grad=values_requires_grad
            )
        else:
            self.register_buffer("_values", vals.detach().contiguous())

        # Corner offsets for linear indexing into flattened (Nx*Ny*Nz) volume
        stride_x = Ny * Nz
        stride_y = Nz
        offsets = torch.tensor(
            [
                0,
                stride_x,
                stride_y,
                stride_x + stride_y,
                1,
                stride_x + 1,
                stride_y + 1,
                stride_x + stride_y + 1,
            ],
            device=values.device,
            dtype=torch.long,
        )
        self.register_buffer("_corner_offsets", offsets)

    def _as_index_tensor(
        self,
        indices: IndexLike,
        *,
        device: torch.device,
        expected_len: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(indices, int):
            idx = torch.tensor([indices], device=device, dtype=torch.long)
        else:
            idx = torch.as_tensor(indices, device=device, dtype=torch.long)

        if idx.ndim != 1:
            raise ValueError("indices must be a 1D sequence/tensor (or a single int).")
        if idx.numel() == 0:
            raise ValueError("indices must be non-empty.")
        if expected_len is not None and idx.numel() != expected_len:
            raise ValueError(
                f"indices must have length {expected_len}, got {idx.numel()}."
            )

        if self.batched:
            if torch.any((idx < 0) | (idx >= self.D)):
                raise ValueError(f"indices values must be in [0, {self.D - 1}].")
        else:
            if torch.any(idx != 0):
                raise ValueError("In unbatched mode, indices can only contain 0.")

        return idx

    def forward(
        self,
        xyz_new: torch.Tensor,
        *,
        indices: Optional[IndexLike] = None,
        outside: Optional[OutsideMode] = None,
        fill_value: Optional[float] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpolate values at query points xyz_new.

        Parameters
        ----------
        xyz_new : (Q,P,3) or (P,3) float tensor
            Query points.
        indices : Optional[IndexLike], optional
            Indices for batched interpolation, by default None
        outside : Optional[OutsideMode], optional
            How to handle points outside the interpolation domain, by default None
        fill_value : Optional[float], optional
            Value to use for points outside the domain if outside='fill', by default None
        out : Optional[torch.Tensor], optional
            Optional output tensor to write results into, by default None

        Returns
        -------
        torch.Tensor
            Interpolated values at query points.
        """
        outside = self.outside if outside is None else outside
        if outside not in ("clamp", "zero", "fill"):
            raise ValueError("outside must be one of: 'clamp', 'zero', 'fill'")
        fill_value = self.fill_value if fill_value is None else float(fill_value)

        if xyz_new.ndim not in (2, 3) or xyz_new.shape[-1] != 3:
            raise ValueError("xyz_new must be (P,3) or (Q,P,3).")
        if xyz_new.device != self._x_min.device or xyz_new.dtype != self._x_min.dtype:
            raise ValueError("xyz_new must match device and dtype of the interpolator.")
        if not torch.is_floating_point(xyz_new):
            raise TypeError("xyz_new must be floating-point.")

        pts_was_2d = xyz_new.ndim == 2
        xyzq = xyz_new[None, :, :] if pts_was_2d else xyz_new  # (Q,P,3)
        Q = int(xyzq.shape[0])
        P = int(xyzq.shape[1])

        xq = xyzq[..., 0]
        yq = xyzq[..., 1]
        zq = xyzq[..., 2]

        # Resolve mapping in batched mode
        if not self.batched:
            if indices is not None:
                _ = self._as_index_tensor(indices, device=xyzq.device, expected_len=Q)
            idx = None
        else:
            if pts_was_2d:
                if indices is None:
                    raise ValueError(
                        "Ambiguous: batched grid (D,...) but xyz_new is (P,3). Provide indices."
                    )
                idx = self._as_index_tensor(
                    indices, device=xyzq.device, expected_len=None
                )
                Q_eff = int(idx.numel())
                xyzq = xyzq.expand(Q_eff, -1, -1)
                xq = xyzq[..., 0]
                yq = xyzq[..., 1]
                zq = xyzq[..., 2]
                Q = Q_eff
            else:
                if indices is None:
                    if Q != self.D:
                        raise ValueError(
                            f"Ambiguous: batched grid has D={self.D} but xyz_new has Q={Q}. Provide indices of length Q."
                        )
                    idx = None
                else:
                    idx = self._as_index_tensor(
                        indices, device=xyzq.device, expected_len=Q
                    )

        # Select per-query-row params/values
        if not self.batched:
            x_min, x_max, inv_dx = self._x_min, self._x_max, self._inv_dx
            y_min, y_max, inv_dy = self._y_min, self._y_max, self._inv_dy
            z_min, z_max, inv_dz = self._z_min, self._z_max, self._inv_dz
            vals = (
                self.values if self.learnable_values else self._values
            )  # (Nx,Ny,Nz,C)
        else:
            if idx is None:
                x_min, x_max, inv_dx = self._x_min, self._x_max, self._inv_dx
                y_min, y_max, inv_dy = self._y_min, self._y_max, self._inv_dy
                z_min, z_max, inv_dz = self._z_min, self._z_max, self._inv_dz
                vals = (
                    self.values if self.learnable_values else self._values
                )  # (D,Nx,Ny,Nz,C)
            else:
                x_min = self._x_min.index_select(0, idx)
                x_max = self._x_max.index_select(0, idx)
                inv_dx = self._inv_dx.index_select(0, idx)
                y_min = self._y_min.index_select(0, idx)
                y_max = self._y_max.index_select(0, idx)
                inv_dy = self._inv_dy.index_select(0, idx)
                z_min = self._z_min.index_select(0, idx)
                z_max = self._z_max.index_select(0, idx)
                inv_dz = self._inv_dz.index_select(0, idx)
                vals = (
                    self.values if self.learnable_values else self._values
                ).index_select(0, idx)

        # Coordinate clamp if requested
        if outside == "clamp":
            x_used = torch.clamp(xq, x_min, x_max)
            y_used = torch.clamp(yq, y_min, y_max)
            z_used = torch.clamp(zq, z_min, z_max)
        else:
            x_used, y_used, z_used = xq, yq, zq

        # Uniform indexing (no searchsorted)
        ux = (x_used - x_min) * inv_dx
        ix = ux.to(torch.long)
        ix.clamp_(0, self._ix_hi)
        tx = ux - ix.to(ux.dtype)

        uy = (y_used - y_min) * inv_dy
        iy = uy.to(torch.long)
        iy.clamp_(0, self._iy_hi)
        ty = uy - iy.to(uy.dtype)

        uz = (z_used - z_min) * inv_dz
        iz = uz.to(torch.long)
        iz.clamp_(0, self._iz_hi)
        tz = uz - iz.to(uz.dtype)

        if outside == "clamp":
            # numeric safety at upper boundary
            tx = torch.clamp(tx, 0.0, 1.0)
            ty = torch.clamp(ty, 0.0, 1.0)
            tz = torch.clamp(tz, 0.0, 1.0)

        # Flattened corner gather
        base = (ix * self.Ny + iy) * self.Nz + iz  # (Q,P)
        idxs = base.unsqueeze(-1) + self._corner_offsets  # (Q,P,8)

        if not self.batched:
            v_flat = vals.reshape(-1, self.C)  # (M,C)
            idxs_flat = idxs.reshape(Q, -1)  # (Q,8P)
            corners = v_flat[idxs_flat].view(Q, P, 8, self.C)  # (Q,P,8,C)
        else:
            v_flat = vals.reshape(Q, -1, self.C)  # (Q,M,C)
            idxs_flat = idxs.reshape(Q, -1)  # (Q,8P)
            idxs_exp = idxs_flat.unsqueeze(-1).expand(-1, -1, self.C)
            corners = torch.gather(v_flat, 1, idxs_exp).view(Q, P, 8, self.C)

        # Trilinear blending
        txe = tx.unsqueeze(-1)
        tye = ty.unsqueeze(-1)
        tze = tz.unsqueeze(-1)

        v000 = corners[:, :, 0, :]
        v100 = corners[:, :, 1, :]
        v010 = corners[:, :, 2, :]
        v110 = corners[:, :, 3, :]
        v001 = corners[:, :, 4, :]
        v101 = corners[:, :, 5, :]
        v011 = corners[:, :, 6, :]
        v111 = corners[:, :, 7, :]

        v00 = v000 + (v100 - v000) * txe
        v10 = v010 + (v110 - v010) * txe
        v01 = v001 + (v101 - v001) * txe
        v11 = v011 + (v111 - v011) * txe

        v0 = v00 + (v10 - v00) * tye
        v1 = v01 + (v11 - v01) * tye

        ynew = v0 + (v1 - v0) * tze  # (Q,P,C)

        # Outside masking for zero/fill
        if outside in ("zero", "fill"):
            outside_mask = (
                (xq < x_min)
                | (xq > x_max)
                | (yq < y_min)
                | (yq > y_max)
                | (zq < z_min)
                | (zq > z_max)
            )  # (Q,P)
            ynew = ynew.masked_fill(
                outside_mask.unsqueeze(-1), 0.0 if outside == "zero" else fill_value
            )

        # Drop channel dim if input had no channels
        if not self._had_channels:
            ynew = ynew.squeeze(-1)  # (Q,P)

        if pts_was_2d and ynew.shape[0] == 1:
            ynew = ynew[0]

        if out is not None:
            if out.shape != ynew.shape:
                raise ValueError(
                    f"out has shape {tuple(out.shape)} but expected {tuple(ynew.shape)}."
                )
            out.copy_(ynew)
            return out

        return ynew


class PreparedInterp3dScattered(nn.Module):
    """
    Scattered-data interpolator in 3D using kNN-based local methods.

    Required inputs
    ---------------
    points : (N,3) float tensor
    values : (N,P) or (N,) float tensor

    Query
    -----
    xq : (...,3) -> (...,P) (or (...) if values were scalar)

    Interpolation methods
    ---------------------
    - nearest : nearest neighbor (k ignored; uses k=1)
    - idw     : inverse-distance weighted average over k neighbors
    - mls     : moving least squares local affine fit (reproduces linear fields)

    Optional accelerated kNN backends
    ---------------------------------
    knn_backend="auto" chooses based on device & availability:

      CUDA:
        1) pytorch3d (knn_points) if installed
        2) torch_cluster if installed
        3) torch fallback

      CPU:
        1) faiss (IndexFlatL2) if installed
        2) torch fallback

    Notes
    -----
    - Neighbor selection is discrete, so gradients do not flow through changes in the
      neighbor set. Gradients do flow to `values` and (optionally) to `xq` inside a
      fixed neighbor set when `recompute_d2=True`.
    """

    def __init__(
        self,
        points: torch.Tensor,
        values: torch.Tensor,
        *,
        method: Literal["nearest", "idw", "mls"] = "idw",
        k: int = 8,
        # kNN backend
        knn_backend: KNNBackend = "auto",
        recompute_d2: bool = True,
        # IDW params
        power: float = 2.0,
        # MLS params
        mls_reg: float = 1e-6,
        # Optional distance cutoff
        radius: Optional[float] = None,
        outside: OutsideMode3D = "none",
        fill_value: float = 0.0,
        # Performance
        chunk_size: int = 4096,
        # Numerical stability
        eps: Optional[float] = None,
        # Learnability
        learnable_values: bool = False,
        values_requires_grad: bool = True,
        # FAISS options (CPU-only in this implementation)
        faiss_factory: Optional[
            str
        ] = None,  # None -> IndexFlatL2; or e.g. "IVF1024,PQ16"
        faiss_nprobe: int = 16,
    ):
        super().__init__()

        if method not in ("nearest", "idw", "mls"):
            raise ValueError("method must be one of {'nearest','idw','mls'}")
        if outside not in ("none", "zero", "fill"):
            raise ValueError("outside must be one of {'none','zero','fill'}")
        if knn_backend not in ("auto", "torch", "pytorch3d", "torch_cluster", "faiss"):
            raise ValueError(
                "knn_backend must be one of {'auto','torch','pytorch3d','torch_cluster','faiss'}"
            )

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N,3).")
        if values.ndim == 1:
            values = values[:, None]
            self._scalar_values = True
        elif values.ndim == 2:
            self._scalar_values = False
        else:
            raise ValueError("values must have shape (N,) or (N,P).")

        if points.shape[0] != values.shape[0]:
            raise ValueError(
                f"points has N={points.shape[0]} but values has N={values.shape[0]}."
            )

        if points.device != values.device:
            raise ValueError("points and values must be on the same device.")
        if points.dtype != values.dtype:
            raise ValueError("points and values must have the same dtype.")
        if not torch.is_floating_point(points):
            raise TypeError("points and values must be floating-point tensors.")

        N = int(points.shape[0])
        if k < 1 or k > N:
            raise ValueError(f"k must be in [1, N]; got k={k}, N={N}.")
        if method == "mls" and k < 4:
            raise ValueError("MLS in 3D requires k >= 4 (prefer k>=8).")

        self.method = method
        self.k = int(k)
        self.knn_backend = knn_backend
        self.recompute_d2 = bool(recompute_d2)

        self.power = float(power)
        self.mls_reg = float(mls_reg)

        self.radius = None if radius is None else float(radius)
        self.outside = outside
        self.fill_value = float(fill_value)

        self.chunk_size = int(chunk_size)
        if eps is None:
            eps = float(torch.finfo(points.dtype).eps)
        self.eps = float(eps)

        self.faiss_factory = faiss_factory
        self.faiss_nprobe = int(faiss_nprobe)

        # Store points + norms (used by torch fallback and distance recomputation)
        self.register_buffer("_points", points.contiguous())  # (N,3)
        self.register_buffer(
            "_points_norm", (points * points).sum(dim=1).contiguous()
        )  # (N,)

        # Values
        if learnable_values:
            self.values = nn.Parameter(
                values.contiguous(), requires_grad=values_requires_grad
            )
        else:
            self.register_buffer("_values", values.contiguous())

        # Lazy FAISS state (CPU only here)
        self._faiss_index = None
        self._faiss_index_dim = 3

    def _get_values(self) -> torch.Tensor:
        return self.values if hasattr(self, "values") else self._values

    # -----------------------------
    # Backend availability helpers
    # -----------------------------
    @staticmethod
    def _has_pytorch3d() -> bool:
        try:
            import pytorch3d  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _has_torch_cluster() -> bool:
        try:
            import torch_cluster  # noqa: F401

            return True
        except Exception:
            return False

    @staticmethod
    def _has_faiss() -> bool:
        try:
            import faiss  # noqa: F401

            return True
        except Exception:
            return False

    def _select_backend(self, device: torch.device) -> KNNBackend:
        if self.knn_backend != "auto":
            return self.knn_backend

        if device.type == "cuda":
            if self._has_pytorch3d():
                return "pytorch3d"
            if self._has_torch_cluster():
                return "torch_cluster"
            return "torch"

        # CPU
        if self._has_faiss():
            return "faiss"
        return "torch"

    # -----------------------------
    # kNN implementations
    # -----------------------------
    def _knn_torch(self, xq: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact kNN via dense distances in chunks.
        Returns idx (Q,k) and d2 (Q,k).
        """
        Q = int(xq.shape[0])

        idx_out = torch.empty((Q, k), device=xq.device, dtype=torch.long)
        d2_out = torch.empty((Q, k), device=xq.device, dtype=xq.dtype)

        pts = self._points
        pts_norm = self._points_norm  # (N,)

        cs = self.chunk_size if self.chunk_size > 0 else Q
        for s in range(0, Q, cs):
            e = min(Q, s + cs)
            q = xq[s:e]  # (Qc,3)

            q_norm = (q * q).sum(dim=1, keepdim=True)  # (Qc,1)
            prod = q @ pts.t()  # (Qc,N)
            d2 = q_norm + pts_norm.unsqueeze(0) - 2.0 * prod
            d2 = torch.clamp(d2, min=0.0)

            d2k, idxk = torch.topk(d2, k=k, dim=1, largest=False, sorted=True)
            idx_out[s:e] = idxk
            d2_out[s:e] = d2k

        return idx_out, d2_out

    def _knn_pytorch3d(
        self, xq: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Exact kNN via pytorch3d.ops.knn_points.
        Returns idx (Q,k) and d2 (Q,k) (squared distances).
        """
        from pytorch3d.ops import knn_points  # type: ignore

        Q = int(xq.shape[0])

        idx_out = torch.empty((Q, k), device=xq.device, dtype=torch.long)
        d2_out = torch.empty((Q, k), device=xq.device, dtype=xq.dtype)

        p2 = self._points.unsqueeze(0)  # (1,N,3)

        cs = self.chunk_size if self.chunk_size > 0 else Q
        for s in range(0, Q, cs):
            e = min(Q, s + cs)
            p1 = xq[s:e].unsqueeze(0)  # (1,Qc,3)

            knn = knn_points(p1, p2, K=k, return_sorted=True)
            # knn is a NamedTuple-like with fields .idx and .dists in most versions
            idx = knn.idx if hasattr(knn, "idx") else knn[1]
            d2 = knn.dists if hasattr(knn, "dists") else knn[0]

            # (1,Qc,k) -> (Qc,k)
            idx_out[s:e] = idx[0]
            d2_out[s:e] = d2[0]

        return idx_out, d2_out

    def _knn_torch_cluster(
        self, xq: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Exact kNN via torch_cluster.knn (CUDA extension when tensors are CUDA).
        Returns idx (Q,k). Distances are computed later in Torch.
        """
        try:
            from torch_cluster import knn as tc_knn  # type: ignore
        except Exception as e:
            raise RuntimeError("torch_cluster not available") from e

        Q = int(xq.shape[0])
        x = self._points.contiguous()
        y = xq.contiguous()

        # batch vectors (single batch)
        batch_x = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        batch_y = torch.zeros(y.shape[0], device=y.device, dtype=torch.long)

        edge_index = tc_knn(x, y, k, batch_x, batch_y)  # (2, Q*k) typically
        row, col = edge_index[0], edge_index[1]  # row in [0,Q), col in [0,N)

        # Group edges by query row
        order = torch.argsort(row)
        row_s = row[order]
        col_s = col[order]

        # Fast path: expect exactly k per query in row-major groups
        if col_s.numel() != Q * k:
            raise RuntimeError(
                f"torch_cluster.knn returned {col_s.numel()} edges, expected {Q * k}."
            )
        row_view = row_s.view(Q, k)
        if not torch.equal(
            row_view[:, 0], torch.arange(Q, device=row_s.device, dtype=row_s.dtype)
        ):
            raise RuntimeError(
                "torch_cluster.knn output not groupable into (Q,k) in a simple reshape."
            )

        idx = col_s.view(Q, k).contiguous()
        return idx, None

    def _ensure_faiss_index(self) -> None:
        """
        Build a FAISS index on CPU for the current points (CPU only in this implementation).
        """
        if self._faiss_index is not None:
            return

        if self._points.device.type != "cpu":
            raise RuntimeError(
                "FAISS backend in this implementation is CPU-only. Move module/points to CPU or use knn_backend='pytorch3d'/'torch_cluster' on CUDA."
            )

        import faiss  # type: ignore

        pts = self._points.detach().contiguous().to(dtype=torch.float32).cpu().numpy()

        d = 3
        if self.faiss_factory is None:
            index = faiss.IndexFlatL2(d)
        else:
            # Index factory string, e.g. "IVF1024,PQ16"
            index = faiss.index_factory(d, self.faiss_factory)

        if not index.is_trained:
            index.train(pts)
        index.add(pts)

        # Optional: IVF indexes use nprobe
        try:
            index.nprobe = self.faiss_nprobe
        except Exception:
            pass

        self._faiss_index = index

    def _knn_faiss(
        self, xq: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        kNN via FAISS (CPU-only here). Returns idx (Q,k).
        Distances are recomputed in Torch unless recompute_d2=False.
        """
        self._ensure_faiss_index()
        index = self._faiss_index

        q = xq.detach().contiguous().to(dtype=torch.float32).cpu().numpy()
        d2, idx = index.search(q, k)  # d2: (Q,k) squared L2, idx: (Q,k)
        idx_t = torch.from_numpy(idx).to(device=xq.device, dtype=torch.long)

        if self.recompute_d2:
            return idx_t, None
        d2_t = torch.from_numpy(d2).to(device=xq.device, dtype=xq.dtype)
        return idx_t, d2_t

    def _knn(
        self, xq: torch.Tensor, k: int, *, need_d2: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified kNN: returns (idx, d2) with d2 squared distances.
        """
        # Keep empty query batches well-defined for every backend.  In particular,
        # the torch fallback otherwise derives a zero chunk size when
        # ``chunk_size <= 0`` and calls ``range(..., step=0)``.
        if xq.shape[0] == 0:
            return (
                torch.empty((0, k), device=xq.device, dtype=torch.long),
                torch.empty((0, k), device=xq.device, dtype=xq.dtype),
            )

        backend = self._select_backend(xq.device)

        if backend == "torch":
            idx, d2 = self._knn_torch(xq, k)
            return idx, d2

        if backend == "pytorch3d":
            idx, d2b = self._knn_pytorch3d(xq, k)
        elif backend == "torch_cluster":
            idx, d2b = self._knn_torch_cluster(xq, k)
        elif backend == "faiss":
            idx, d2b = self._knn_faiss(xq, k)
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        # If we don't need distances and don't plan to use radius/weights, we can skip d2.
        # But for simplicity/robustness, we always return d2 (computed cheaply from idx).
        if (d2b is None) or self.recompute_d2 or need_d2:
            # Recompute d2 in Torch from gathered neighbors; cheap O(Q*k*3)
            pts = self._points
            pnn = pts.index_select(0, idx.reshape(-1)).view(xq.shape[0], k, 3)
            d2 = ((pnn - xq[:, None, :]) ** 2).sum(dim=-1)
        else:
            d2 = d2b

        return idx, d2

    # -----------------------------
    # Forward interpolation
    # -----------------------------
    def forward(
        self, xq: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Interpolate values at query points xq.

        Parameters
        ----------
        xq : (...,3) float tensor
            Query points.
        out : (...,P) or (...) float tensor, optional
            Optional output tensor to write results into.

        Returns
        -------
        y : (...,P) or (...) float tensor
            Interpolated values at query points.
            If input values were scalar, shape is (...); otherwise (...,P).
        """
        if xq.ndim < 2 or xq.shape[-1] != 3:
            raise ValueError("xq must have shape (...,3).")
        if xq.device != self._points.device:
            raise ValueError("xq must be on the same device as points/values.")
        if xq.dtype != self._points.dtype:
            raise ValueError("xq must have the same dtype as points/values.")

        orig_shape = xq.shape[:-1]
        xq_flat = xq.reshape(-1, 3)
        Q = int(xq_flat.shape[0])

        vals = self._get_values()  # (N,P)
        Pdim = int(vals.shape[1])

        k_eff = 1 if self.method == "nearest" else self.k
        need_d2 = (self.method in ("idw", "mls")) or (self.radius is not None)

        idx, d2 = self._knn(xq_flat, k_eff, need_d2=need_d2)  # (Q,k), (Q,k)

        # Gather neighbor values -> (Q,k,P)
        v = vals.index_select(0, idx.reshape(-1)).view(Q, k_eff, Pdim)

        if self.method == "nearest" or k_eff == 1:
            y = v[:, 0, :]  # (Q,P)

        elif self.method == "idw":
            p = self.power
            w = 1.0 / (d2.clamp_min(0.0).pow(0.5 * p) + self.eps)  # (Q,k)

            # exact hit => take nearest exactly
            hit = d2[:, 0] <= (self.eps * self.eps)
            if hit.any():
                w = torch.where(hit[:, None], torch.zeros_like(w), w)
                w[:, 0] = torch.where(hit, torch.ones_like(w[:, 0]), w[:, 0])

            w = w / w.sum(dim=1, keepdim=True).clamp_min(self.eps)
            y = (w.unsqueeze(-1) * v).sum(dim=1)
            if hit.any():
                y = torch.where(hit[:, None], v[:, 0, :], y)

        else:  # MLS
            # neighbor coords (Q,k,3)
            pnn = self._points.index_select(0, idx.reshape(-1)).view(Q, k_eff, 3)
            dx = pnn - xq_flat[:, None, :]  # (Q,k,3)
            ones = torch.ones((Q, k_eff, 1), device=xq.device, dtype=xq.dtype)
            A = torch.cat([ones, dx], dim=2)  # (Q,k,4)

            # weight kernel using farthest neighbor as scale
            sigma2 = d2[:, -1].clamp_min(self.eps)  # (Q,)
            w = torch.exp(-0.5 * d2 / sigma2[:, None])  # (Q,k)

            AtWA = torch.einsum("qki,qkj,qk->qij", A, A, w)  # (Q,4,4)
            if self.mls_reg > 0:
                I_ = torch.eye(4, device=xq.device, dtype=xq.dtype).unsqueeze(0)
                AtWA = AtWA + self.mls_reg * I_
            AtWy = torch.einsum("qki,qk,qkp->qip", A, w, v)  # (Q,4,P)

            theta = torch.linalg.solve(AtWA, AtWy)  # (Q,4,P)
            y = theta[:, 0, :]  # value at query (since features are [1,0,0,0])

            hit = d2[:, 0] <= (self.eps * self.eps)
            if hit.any():
                y = torch.where(hit[:, None], v[:, 0, :], y)

        # Radius-based outside handling (optional)
        if self.radius is not None and self.outside != "none":
            outside_mask = d2[:, 0] > (self.radius * self.radius)
            if outside_mask.any():
                if self.outside == "zero":
                    y = torch.where(outside_mask[:, None], torch.zeros_like(y), y)
                else:
                    y = torch.where(
                        outside_mask[:, None], torch.full_like(y, self.fill_value), y
                    )

        y = y.view(*orig_shape, Pdim)
        if self._scalar_values:
            y = y.squeeze(-1)

        if out is not None:
            if out.shape != y.shape:
                raise ValueError(
                    f"out has shape {tuple(out.shape)} but expected {tuple(y.shape)}."
                )
            out.copy_(y)
            return out

        return y


FEMDataKind = Literal["node", "element"]
FEMElementMethod = Literal["assign", "linear"]
FEMPreparedKind = Literal[
    "node",
    "element_assign",
    "element_linear_discontinuous",
]


class PreparedInterp3dFEM(nn.Module):
    """
    Mesh-guided tetrahedral FEM interpolator compatible with SimNIBS-style
    ``NodeData.interpolate_scattered`` and ``ElementData.interpolate_scattered``.

    This module deliberately uses the mesh's own tetrahedron point-location routine
    during ``forward`` and then performs the value gathering/blending in PyTorch.
    That preserves the high-quality mesh-guided behavior of the original methods:

    - node data: barycentric interpolation inside the containing tetrahedron;
    - element data, ``method="assign"``: assign the containing tetrahedron's value;
    - element data, ``method="linear", continuous=True``: recover element data to
      nodes once with ``ElementData.elm_data2node_data`` and then use node-data
      barycentric interpolation;
    - element data, ``method="linear", continuous=False``: perform the same
      tag-wise superconvergent patch recovery as ``ElementData.interpolate_scattered``
      and interpolate with the recovered nodal field of the containing tag.

    Parameters
    ----------
    mesh : object
        A mesh object implementing the relevant ``mesh_io.Msh`` API:
        ``find_tetrahedron_with_points``, ``nodes.find_closest_node``,
        ``find_closest_element``, ``elm.node_number_list``, ``elm.tag1``, and
        ``elm.tetrahedra``.
    values : array-like or torch.Tensor
        Nodal values for ``kind="node"`` or element values for
        ``kind="element_assign"``. For ``kind="element_linear_discontinuous"`` this
        is used only for the outside ``out_fill="nearest"`` behavior; tag-wise
        recovered nodal values must be provided via ``tag_states``.
    kind : {"node", "element_assign", "element_linear_discontinuous"}
        Prepared interpolation regime.
    out_fill : float or "nearest", default=np.nan
        Outside-volume policy, matching the ``mesh_io.py`` methods.
    th_indices : array-like, optional
        One-based tetrahedron/element numbers to consider as the valid volume.
        Tetrahedra outside this set are treated as outside.
    squeeze : bool, default=True
        Match the ``squeeze`` behavior of the original methods.
    learnable_values : bool, default=False
        Store ``values`` as a parameter. Supported for ``kind="node"`` and
        ``kind="element_assign"``. For tag-wise element-linear interpolation, use
        the classmethod defaults unless you intentionally want only outside-nearest
        element values to be learnable.

    Notes
    -----
    Gradients flow to stored values through the PyTorch gather/blend operations.
    Gradients do not flow through the discrete tetrahedron search, and barycentric
    coordinates are computed by the mesh's NumPy/Cython locator, matching the source
    implementation rather than providing differentiability with respect to query
    coordinates.
    """

    def __init__(
        self,
        mesh: Any,
        values: Union[np.ndarray, torch.Tensor],
        *,
        kind: FEMPreparedKind = "node",
        out_fill: Union[float, str] = np.nan,
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]] = None,
        squeeze: bool = True,
        field_name: str = "",
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        learnable_values: bool = False,
        values_requires_grad: bool = True,
        tag_states: Optional[Sequence[dict[str, Any]]] = None,
    ):
        super().__init__()

        if kind not in ("node", "element_assign", "element_linear_discontinuous"):
            raise ValueError(
                "kind must be one of {'node', 'element_assign', "
                "'element_linear_discontinuous'}."
            )

        self.mesh = mesh
        self.kind = kind
        self.out_fill = out_fill
        self.squeeze = bool(squeeze)
        self.field_name = field_name
        self.learnable_values = bool(learnable_values)

        th_np = self._normalize_th_indices(th_indices)
        self._default_th_indices_np = th_np

        values_t, input_was_1d = self._coerce_values(values, dtype=dtype, device=device)
        self._input_was_1d = bool(input_was_1d)
        self._n_components = int(values_t.shape[1])

        if self.learnable_values:
            self.values = nn.Parameter(values_t, requires_grad=values_requires_grad)
        else:
            self.register_buffer("_values", values_t)

        # Original element tags are needed to select the tag-local recovered nodal
        # field in the discontinuous ElementData linear branch.
        self._element_tags_np = np.asarray(mesh.elm.tag1).copy()

        self._n_tag_states = 0
        self._tag_state_meta: list[dict[str, Any]] = []
        if kind == "element_linear_discontinuous":
            if tag_states is None:
                raise ValueError(
                    "tag_states are required for kind='element_linear_discontinuous'. "
                    "Use PreparedInterpolate3dFEM.from_ElementData(..., "
                    "method='linear', continuous=False)."
                )
            self._register_tag_states(
                tag_states, dtype=values_t.dtype, device=values_t.device
            )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_NodeData(
        cls,
        node_data: Any,
        *,
        out_fill: Union[float, str] = np.nan,
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]] = None,
        squeeze: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        learnable_values: bool = False,
        values_requires_grad: bool = True,
    ) -> "PreparedInterp3dFEM":
        """
        Build an interpolator from a ``mesh_io.NodeData``-like object.

        The returned module reproduces ``node_data.interpolate_scattered(points,
        out_fill=out_fill, squeeze=squeeze, th_indices=th_indices)`` for the same
        mesh and points, up to normal floating-point roundoff.
        """
        cls._test_data_mesh(node_data)
        return cls(
            node_data.mesh,
            node_data.value,
            kind="node",
            out_fill=out_fill,
            th_indices=th_indices,
            squeeze=squeeze,
            field_name=getattr(node_data, "field_name", ""),
            dtype=dtype,
            device=device,
            learnable_values=learnable_values,
            values_requires_grad=values_requires_grad,
        )

    @classmethod
    def from_ElementData(
        cls,
        element_data: Any,
        *,
        out_fill: Union[float, str] = np.nan,
        method: FEMElementMethod = "linear",
        continuous: bool = False,
        squeeze: bool = True,
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        learnable_values: bool = False,
        values_requires_grad: bool = True,
    ) -> "PreparedInterp3dFEM":
        """
        Build an interpolator from a ``mesh_io.ElementData``-like object.

        This mirrors ``ElementData.interpolate_scattered``:

        - ``method='assign'`` stores element values and assigns containing-tet
          values in ``forward``.
        - ``method='linear', continuous=True`` performs the same global
          ``elm_data2node_data`` recovery once and then returns a node-data FEM
          interpolator using the recovered nodal field.
        - ``method='linear', continuous=False`` performs tag-wise recovery once
          and selects the containing tetrahedron's tag-specific nodal field in
          ``forward``.
        """
        cls._test_data_mesh(element_data)

        if method not in ("assign", "linear"):
            raise ValueError("method must be 'assign' or 'linear'.")

        mesh = element_data.mesh
        if len(mesh.elm.tetrahedra) == 0:
            raise ValueError("Mesh has no volume elements.")

        if method == "assign":
            return cls(
                mesh,
                element_data.value,
                kind="element_assign",
                out_fill=out_fill,
                th_indices=th_indices,
                squeeze=squeeze,
                field_name=getattr(element_data, "field_name", ""),
                dtype=dtype,
                device=device,
                learnable_values=learnable_values,
                values_requires_grad=values_requires_grad,
            )

        if continuous:
            if learnable_values:
                raise ValueError(
                    "learnable_values=True is not supported for ElementData "
                    "method='linear', continuous=True because values are first "
                    "converted to recovered nodal values by the mesh_io routine. "
                    "Use method='assign', or build from the recovered NodeData."
                )
            recovered = element_data.elm_data2node_data()
            return cls(
                mesh,
                recovered.value,
                kind="node",
                out_fill=out_fill,
                th_indices=th_indices,
                squeeze=squeeze,
                field_name=getattr(element_data, "field_name", ""),
                dtype=dtype,
                device=device,
                learnable_values=False,
                values_requires_grad=False,
            )

        if learnable_values:
            raise ValueError(
                "learnable_values=True is not supported for ElementData "
                "method='linear', continuous=False in this faithful wrapper, "
                "because tag-wise superconvergent patch recovery is performed "
                "with the mesh_io implementation at preparation time."
            )

        tag_states = cls._build_discontinuous_tag_states(element_data)
        return cls(
            mesh,
            element_data.value,
            kind="element_linear_discontinuous",
            out_fill=out_fill,
            th_indices=th_indices,
            squeeze=squeeze,
            field_name=getattr(element_data, "field_name", ""),
            dtype=dtype,
            device=device,
            learnable_values=False,
            values_requires_grad=False,
            tag_states=tag_states,
        )

    @staticmethod
    def _test_data_mesh(data: Any) -> None:
        if getattr(data, "mesh", None) is None:
            raise ValueError("Cannot prepare FEM interpolation if data.mesh is None.")

    @staticmethod
    def _normalize_th_indices(
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]],
    ) -> Optional[np.ndarray]:
        if th_indices is None:
            return None
        if isinstance(th_indices, torch.Tensor):
            th = th_indices.detach().cpu().numpy()
        else:
            th = np.asarray(th_indices)
        th = th.astype(np.int64, copy=False).reshape(-1)
        if np.any(th <= 0):
            raise ValueError(
                "th_indices must contain one-based positive element numbers."
            )
        return th

    @staticmethod
    def _coerce_values(
        values: Union[np.ndarray, torch.Tensor],
        *,
        dtype: Optional[torch.dtype],
        device: Optional[Union[str, torch.device]],
    ) -> tuple[torch.Tensor, bool]:
        if isinstance(values, torch.Tensor):
            t = values.detach().clone() if not values.is_leaf else values
            if dtype is not None or device is not None:
                t = t.to(dtype=dtype or t.dtype, device=device or t.device)
        else:
            t = torch.as_tensor(values, dtype=dtype, device=device)

        if not torch.is_floating_point(t):
            raise TypeError(
                "FEM interpolation values must be floating-point tensors/arrays."
            )
        if t.ndim == 1:
            return t.contiguous().unsqueeze(-1), True
        if t.ndim == 2:
            return t.contiguous(), False
        raise ValueError("values must have shape (N,) or (N,C).")

    @classmethod
    def _build_discontinuous_tag_states(cls, element_data: Any) -> list[dict[str, Any]]:
        """Precompute the tag-wise ElementData->NodeData recovery used by mesh_io."""
        mesh = element_data.mesh
        field_name = getattr(element_data, "field_name", "")
        ed_cls = element_data.__class__

        # Work on a copy exactly as ElementData.interpolate_scattered does. This
        # lets crop_mesh carry the element values onto each tag-local mesh without
        # mutating the caller's mesh.
        msh_work = copy.deepcopy(mesh)
        msh_work.elmdata = [ed_cls(element_data.value, field_name, mesh=msh_work)]

        tet_numbers = np.asarray(msh_work.elm.tetrahedra, dtype=np.int64)
        tet_tags = np.asarray(msh_work.elm.tag1[tet_numbers - 1])
        tag_states: list[dict[str, Any]] = []

        for tag in np.unique(tet_tags):
            # This mirrors the source branch:
            #   msh_tag = msh.crop_mesh(tags=t)
            #   nd = msh_tag.elmdata[0].elm_data2node_data()
            #   msh_with_t = msh.elm.elm_number[msh.elm.get_tags(t)]
            msh_tag = msh_work.crop_mesh(tags=int(tag))
            nd = msh_tag.elmdata[0].elm_data2node_data()

            orig_elms = np.asarray(
                msh_work.elm.elm_number[msh_work.elm.get_tags(int(tag))], dtype=np.int64
            )
            local_nodes = np.asarray(msh_tag.elm.node_number_list, dtype=np.int64) - 1

            tag_states.append(
                {
                    "tag": int(tag),
                    "orig_elms": orig_elms,
                    "local_nodes": local_nodes,
                    "node_values": nd.value,
                }
            )
        return tag_states

    def _register_tag_states(
        self,
        tag_states: Sequence[dict[str, Any]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._n_tag_states = len(tag_states)
        self._tag_state_meta = []

        for i, state in enumerate(tag_states):
            tag = int(state["tag"])
            orig_elms_np = np.asarray(state["orig_elms"], dtype=np.int64)
            self._tag_state_meta.append({"tag": tag, "orig_elms_np": orig_elms_np})

            node_values, input_was_1d = self._coerce_values(
                state["node_values"], dtype=dtype, device=device
            )
            if int(node_values.shape[1]) != self._n_components:
                # Scalar ElementData recovery can squeeze (N,1) to (N,), so the
                # number of components should still be one. Anything else is a
                # genuine shape mismatch.
                if not (input_was_1d and self._n_components == 1):
                    raise ValueError(
                        "Recovered tag-local node values have incompatible component count."
                    )

            local_nodes = torch.as_tensor(
                np.asarray(state["local_nodes"], dtype=np.int64),
                device=device,
                dtype=torch.long,
            ).contiguous()

            self.register_buffer(f"_tag_{i}_node_values", node_values.contiguous())
            self.register_buffer(f"_tag_{i}_local_nodes", local_nodes)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _get_values(self) -> torch.Tensor:
        return self.values if hasattr(self, "values") else self._values

    def _points_to_numpy(self, points: torch.Tensor) -> tuple[np.ndarray, torch.Size]:
        if points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3).")
        if not torch.is_floating_point(points):
            raise TypeError("points must be floating-point.")
        orig_shape = points.shape[:-1]
        pts_np = (
            points.detach().reshape(-1, 3).cpu().numpy().astype(np.float64, copy=False)
        )
        return pts_np, orig_shape

    def _locate_points(
        self,
        points_np: np.ndarray,
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        th_np = (
            self._default_th_indices_np
            if th_indices is None
            else self._normalize_th_indices(th_indices)
        )
        if points_np.shape[0] == 0:
            empty_th = np.empty((0,), dtype=np.int64)
            empty_bar = np.empty((0, 4), dtype=np.float64)
            empty_inside = np.empty((0,), dtype=bool)
            return empty_th, empty_bar, empty_inside

        th_with_points, bary = self.mesh.find_tetrahedron_with_points(
            points_np, compute_baricentric=True
        )
        th_with_points = np.asarray(th_with_points, dtype=np.int64)
        bary = np.asarray(bary)

        if th_np is not None:
            th_with_points[~np.isin(th_with_points, th_np)] = -1

        inside = th_with_points != -1
        return th_with_points, bary, inside

    def _empty_output(self, Q: int, values: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (Q, self._n_components), device=values.device, dtype=values.dtype
        )

    def _fill_outside(
        self,
        y: torch.Tensor,
        points_np: np.ndarray,
        outside: np.ndarray,
        values: torch.Tensor,
        *,
        out_fill: Union[float, str],
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]],
        nearest_kind: Literal["node", "element"],
    ) -> None:
        if not np.any(outside):
            return

        outside_idx = torch.as_tensor(
            np.where(outside)[0], device=y.device, dtype=torch.long
        )

        if out_fill == "nearest":
            pts_out = points_np[outside]
            th_np = (
                self._default_th_indices_np
                if th_indices is None
                else self._normalize_th_indices(th_indices)
            )

            if nearest_kind == "node":
                if th_np is None:
                    _, nearest = self.mesh.nodes.find_closest_node(
                        pts_out, return_index=True
                    )
                    nearest0 = np.asarray(nearest, dtype=np.int64) - 1
                else:
                    # Equivalent to adding the NodeData to the mesh, cropping to
                    # th_indices, finding the closest cropped node, and reading the
                    # cropped NodeData value.
                    elm_nodes = np.asarray(
                        self.mesh.elm.node_number_list[th_np - 1], dtype=np.int64
                    )
                    valid_nodes = np.unique(elm_nodes[elm_nodes > 0])
                    if valid_nodes.size == 0:
                        raise ValueError(
                            "th_indices selects no valid nodes for nearest fill."
                        )
                    coords = self.mesh.nodes.node_coord[valid_nodes - 1]
                    import scipy.spatial

                    _, nearest_local = scipy.spatial.cKDTree(coords).query(pts_out)
                    nearest0 = (
                        valid_nodes[np.asarray(nearest_local, dtype=np.int64)] - 1
                    )

                idx_t = torch.as_tensor(nearest0, device=y.device, dtype=torch.long)
                fill = values.index_select(0, idx_t)

            else:  # nearest element
                if th_np is None:
                    _, nearest = self.mesh.find_closest_element(
                        pts_out, return_index=True
                    )
                else:
                    _, nearest = self.mesh.find_closest_element(
                        pts_out, return_index=True, elements_of_interest=th_np
                    )
                nearest0 = np.asarray(nearest, dtype=np.int64) - 1
                idx_t = torch.as_tensor(nearest0, device=y.device, dtype=torch.long)
                fill = values.index_select(0, idx_t)

            y.index_copy_(0, outside_idx, fill)
        else:
            fill_value = float(out_fill)
            fill = torch.full(
                (outside_idx.numel(), self._n_components),
                fill_value,
                device=y.device,
                dtype=y.dtype,
            )
            y.index_copy_(0, outside_idx, fill)

    def _forward_node_like(
        self,
        points_np: np.ndarray,
        th_with_points: np.ndarray,
        bary: np.ndarray,
        inside: np.ndarray,
        values: torch.Tensor,
        *,
        out_fill: Union[float, str],
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]],
        nearest_kind: Literal["node", "element"] = "node",
    ) -> torch.Tensor:
        Q = int(points_np.shape[0])
        y = self._empty_output(Q, values)

        if np.any(inside):
            node_ids = (
                np.asarray(
                    self.mesh.elm.node_number_list[th_with_points[inside] - 1],
                    dtype=np.int64,
                )
                - 1
            )
            node_ids_t = torch.as_tensor(
                node_ids, device=values.device, dtype=torch.long
            )
            weights = torch.as_tensor(
                bary[inside], device=values.device, dtype=values.dtype
            )
            gathered = values.index_select(0, node_ids_t.reshape(-1)).view(
                -1, 4, self._n_components
            )
            yi = torch.einsum("ik,ikj->ij", weights, gathered)
            inside_idx = torch.as_tensor(
                np.where(inside)[0], device=values.device, dtype=torch.long
            )
            y.index_copy_(0, inside_idx, yi)

        self._fill_outside(
            y,
            points_np,
            ~inside,
            values,
            out_fill=out_fill,
            th_indices=th_indices,
            nearest_kind=nearest_kind,
        )
        return y

    def _forward_element_assign(
        self,
        points_np: np.ndarray,
        th_with_points: np.ndarray,
        inside: np.ndarray,
        values: torch.Tensor,
        *,
        out_fill: Union[float, str],
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]],
    ) -> torch.Tensor:
        Q = int(points_np.shape[0])
        y = self._empty_output(Q, values)

        if np.any(inside):
            elem0 = torch.as_tensor(
                th_with_points[inside] - 1, device=values.device, dtype=torch.long
            )
            yi = values.index_select(0, elem0)
            inside_idx = torch.as_tensor(
                np.where(inside)[0], device=values.device, dtype=torch.long
            )
            y.index_copy_(0, inside_idx, yi)

        self._fill_outside(
            y,
            points_np,
            ~inside,
            values,
            out_fill=out_fill,
            th_indices=th_indices,
            nearest_kind="element",
        )
        return y

    def _forward_element_linear_discontinuous(
        self,
        points_np: np.ndarray,
        th_with_points: np.ndarray,
        bary: np.ndarray,
        inside: np.ndarray,
        values: torch.Tensor,
        *,
        out_fill: Union[float, str],
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]],
    ) -> torch.Tensor:
        Q = int(points_np.shape[0])
        y = self._empty_output(Q, values)

        if np.any(inside):
            inside_positions = np.where(inside)[0]
            inside_th = th_with_points[inside]
            inside_tags = self._element_tags_np[inside_th - 1]

            for i, meta in enumerate(self._tag_state_meta):
                tag = meta["tag"]
                is_tag = inside_tags == tag
                if not np.any(is_tag):
                    continue

                pos_np = inside_positions[is_tag]
                th_tag = inside_th[is_tag]
                orig_elms = meta["orig_elms_np"]
                local_idx_np = np.searchsorted(orig_elms, th_tag).astype(
                    np.int64, copy=False
                )

                local_nodes_all = getattr(self, f"_tag_{i}_local_nodes")
                node_values = getattr(self, f"_tag_{i}_node_values")

                local_idx = torch.as_tensor(
                    local_idx_np, device=values.device, dtype=torch.long
                )
                local_nodes = local_nodes_all.index_select(0, local_idx)[:, :4]
                weights = torch.as_tensor(
                    bary[pos_np], device=values.device, dtype=node_values.dtype
                )
                gathered = node_values.index_select(0, local_nodes.reshape(-1)).view(
                    -1, 4, self._n_components
                )
                yi = torch.einsum("ik,ikj->ij", weights, gathered)
                pos_t = torch.as_tensor(pos_np, device=values.device, dtype=torch.long)
                y.index_copy_(0, pos_t, yi)

        self._fill_outside(
            y,
            points_np,
            ~inside,
            values,
            out_fill=out_fill,
            th_indices=th_indices,
            nearest_kind="element",
        )
        return y

    def _finish_output(
        self,
        y: torch.Tensor,
        orig_shape: torch.Size,
        *,
        squeeze: bool,
        out: Optional[torch.Tensor],
    ) -> torch.Tensor:
        y = y.view(*tuple(orig_shape), self._n_components)
        if self._input_was_1d:
            y = y.squeeze(-1)
        if squeeze:
            y = y.squeeze()

        if out is not None:
            if out.shape != y.shape:
                raise ValueError(
                    f"out has shape {tuple(out.shape)} but expected {tuple(y.shape)}."
                )
            out.copy_(y)
            return out
        return y

    def forward(
        self,
        points: torch.Tensor,
        *,
        out_fill: Optional[Union[float, str]] = None,
        squeeze: Optional[bool] = None,
        th_indices: Optional[Union[np.ndarray, Sequence[int], torch.Tensor]] = None,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Interpolate at query points.

        Parameters
        ----------
        points : torch.Tensor
            Query coordinates with shape ``(..., 3)``. They may live on CPU or GPU;
            point-location is performed by the mesh implementation on a detached CPU
            copy, and the output is returned on the module value buffers' device.
        out_fill : float or "nearest", optional
            Per-call outside policy override. Defaults to the constructor setting.
        squeeze : bool, optional
            Per-call squeeze override. Defaults to the constructor setting.
        th_indices : array-like, optional
            Per-call valid tetrahedron/element subset override. Defaults to the
            constructor setting. Passing this argument does not mutate the prepared
            object.
        out : torch.Tensor, optional
            Optional output tensor.
        """
        values = self._get_values()
        points_np, orig_shape = self._points_to_numpy(points)
        th_with_points, bary, inside = self._locate_points(points_np, th_indices)

        out_fill_eff = self.out_fill if out_fill is None else out_fill
        squeeze_eff = self.squeeze if squeeze is None else bool(squeeze)

        if self.kind == "node":
            y = self._forward_node_like(
                points_np,
                th_with_points,
                bary,
                inside,
                values,
                out_fill=out_fill_eff,
                th_indices=th_indices,
                nearest_kind="node",
            )
        elif self.kind == "element_assign":
            y = self._forward_element_assign(
                points_np,
                th_with_points,
                inside,
                values,
                out_fill=out_fill_eff,
                th_indices=th_indices,
            )
        else:
            y = self._forward_element_linear_discontinuous(
                points_np,
                th_with_points,
                bary,
                inside,
                values,
                out_fill=out_fill_eff,
                th_indices=th_indices,
            )

        return self._finish_output(y, orig_shape, squeeze=squeeze_eff, out=out)
