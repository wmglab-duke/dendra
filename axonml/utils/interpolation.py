from __future__ import annotations

from typing import Literal, Optional, Sequence, Union

import torch
from torch import nn

OutsideMode = Literal["clamp", "zero", "fill"]
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

        # Cached index tensor to reduce allocations in repeated calls with same x_new shape
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

            - If ``x_new`` has shape ``(P,)``:
              - output has shape ``(P,)``

            - If ``x_new`` has shape ``(Q, P)``:
              - output has shape ``(Q, P)``

            (In unbatched mode, `indices` is never required; if provided it must be all zeros.)

            **2) Batched LUT mode** (``x, y`` are 2D with shape ``(D, N)``)

            - If ``x_new`` has shape ``(D, P)`` and ``indices is None``:
              - output has shape ``(D, P)``
              - (Row-wise alignment: query row ``d`` uses LUT row ``d``.)

            - If ``x_new`` has shape ``(Q, P)`` and ``indices`` is provided with length ``Q``:
              - output has shape ``(Q, P)``
              - (Query row ``q`` uses LUT row ``indices[q]``.)

            - If ``x_new`` has shape ``(Q, P)`` with ``Q != D`` and ``indices is None``:
              - **error** (ambiguous mapping)

            - If ``x_new`` has shape ``(P,)``:
              - ``indices`` is **required** (otherwise ambiguous)
              - The same query vector is evaluated against each selected LUT row.

              Let ``L = len(indices)``:
              - If ``L == 1``:
                - output has shape ``(P,)`` (the module returns 1D when the input was 1D
                  and only one effective query row is evaluated)
              - If ``L > 1``:
                - output has shape ``(L, P)``

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

        if fill_value is None:
            fill_value = self.fill_value
        else:
            fill_value = float(fill_value)

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

        # Normalize to 2D
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

            if outside == "clamp":
                xq_used = torch.maximum(torch.minimum(xq, x_max), x_min)
            else:
                xq_used = xq

            if (
                self._ind_cache is None
                or self._ind_cache.shape != xq_used.shape
                or self._ind_cache.device != xq_used.device
            ):
                self._ind_cache = torch.empty(
                    xq_used.shape, device=xq_used.device, dtype=torch.long
                )
            ind = self._ind_cache

            torch.searchsorted(self._x_search, xq_used, out=ind)
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

            # Outside policy
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
                        "Ambiguous: x,y are (D,N) but x_new is (P,). "
                        "Provide `indices` to specify which LUT rows to use."
                    )
                idx = self._as_index_tensor(
                    indices, device=xq.device, expected_len=None
                )
                Q_eff = int(idx.numel())
                xq = xq.expand(
                    Q_eff, -1
                )  # evaluate same x_new against selected LUT rows
                Q = Q_eff
            else:
                if indices is None:
                    if Q != self.D:
                        raise ValueError(
                            f"Ambiguous: x,y are (D,N) with D={self.D} but x_new is (Q,P) with Q={Q}. "
                            "Provide `indices` of length Q to map each x_new row to an (x,y) row."
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

                if self.learnable_y:
                    y_sel = self.y.index_select(0, idx)
                else:
                    y0_tab = self._y0.index_select(0, idx)
                    m_tab = self._slopes.index_select(0, idx)
                    y_first = self._y_first.index_select(0, idx)
                    y_last = self._y_last.index_select(0, idx)

            if outside == "clamp":
                xq_used = torch.maximum(torch.minimum(xq, x_max), x_min)
            else:
                xq_used = xq

            if (
                self._ind_cache is None
                or self._ind_cache.shape != xq_used.shape
                or self._ind_cache.device != xq_used.device
            ):
                self._ind_cache = torch.empty(
                    xq_used.shape, device=xq_used.device, dtype=torch.long
                )
            ind = self._ind_cache

            torch.searchsorted(x_search, xq_used, out=ind)
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
                    y_first_dyn = y_sel[:, :1]
                    y_last_dyn = y_sel[:, -1:]
                    ynew = torch.where(xq <= x_min, y_first_dyn.expand_as(ynew), ynew)
                    ynew = torch.where(xq >= x_max, y_last_dyn.expand_as(ynew), ynew)
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

        # Restore 1D output only when there is exactly one query row
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


def interp1d(x, y, xnew, out=None, *, outside: str = "zero"):
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

    # --- make inputs at least 2D ---
    is_flat = {}
    require_grad = {}
    v = {}

    for name, vec in {"x": x, "y": y, "xnew": xnew}.items():
        assert vec.ndim <= 2, "interp1d: all inputs must be at most 2-D."
        v[name] = vec[None, :] if vec.ndim == 1 else vec
        is_flat[name] = v[name].shape[0] == 1
        require_grad[name] = vec.requires_grad

    # --- device consistency ---
    device = x.device
    assert y.device == device and xnew.device == device, (
        "All parameters must be on the same device."
    )

    # --- shape checks ---
    assert v["x"].shape[1] == v["y"].shape[1] and (
        v["x"].shape[0] == v["y"].shape[0]
        or v["x"].shape[0] == 1
        or v["y"].shape[0] == 1
    ), (
        "x and y must have the same number of columns, and either the same number "
        "of rows or one of them having only one row."
    )

    # Optimization: if x and y are single-row but xnew has multiple rows, flatten xnew
    # into a single long row, interpolate once, then reshape back.
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

    # Allocate indices
    ind = torch.empty(shape_ynew, device=device, dtype=torch.long)

    # searchsorted:
    # - if x is (1,N), squeeze -> (N,) and searchsorted works for (D_eff,P)
    # - if x is (D_eff,N), squeeze keeps 2D and requires xnew to be (D_eff,P)
    torch.searchsorted(v["x"].contiguous().squeeze(), v["xnew"].contiguous(), out=ind)

    # Convert insertion index to left-interval index
    ind -= 1
    ind.clamp_(0, v["x"].shape[1] - 2)  # [0, N-2]

    def sel(t: torch.Tensor, flat: bool) -> torch.Tensor:
        # Select along last dimension using `ind`.
        # If flat (1, M): advanced indexing broadcasts across (D_eff,P).
        if flat:
            return t.contiguous().view(-1)[ind]
        return torch.gather(t, 1, ind)

    # Enable grad only if any input needs it; otherwise disable for speed.
    enable_grad = require_grad["x"] or require_grad["y"] or require_grad["xnew"]
    grad_ctx = torch.enable_grad() if enable_grad else torch.no_grad()

    with grad_ctx:
        # Compute slopes
        dx = v["x"][:, 1:] - v["x"][:, :-1]
        safe_dx = torch.where(
            dx == 0, torch.full_like(dx, torch.finfo(dx.dtype).eps), dx
        )
        slopes = (v["y"][:, 1:] - v["y"][:, :-1]) / safe_dx

        # IMPORTANT: slopes "flatness" depends on slopes itself, not on x
        slopes_is_flat = slopes.shape[0] == 1

        # Linear interpolation
        ynew = sel(v["y"], is_flat["y"]) + sel(slopes, slopes_is_flat) * (
            v["xnew"] - sel(v["x"], is_flat["x"])
        )

        # Bounds (assumes x sorted along last dim per row)
        x_min = v["x"][:, :1]
        x_max = v["x"][:, -1:]

        if outside == "clamp":
            y_min = v["y"][:, :1]
            y_max = v["y"][:, -1:]
            ynew = torch.where(v["xnew"] <= x_min, y_min.expand_as(ynew), ynew)
            ynew = torch.where(v["xnew"] >= x_max, y_max.expand_as(ynew), ynew)
        else:  # outside == "zero"
            outside_mask = (v["xnew"] < x_min) | (v["xnew"] > x_max)
            ynew = ynew.masked_fill(outside_mask, 0.0)

    # Write into output buffer (keeps semantics simple and makes `out` actually work)
    ybuf.copy_(ynew)

    if reshaped_xnew:
        ybuf = ybuf.view(original_xnew_shape)

    # If all inputs were 1D, return 1D
    if x.ndim == 1 and y.ndim == 1 and xnew.ndim == 1:
        return ybuf.view(-1)

    return ybuf
