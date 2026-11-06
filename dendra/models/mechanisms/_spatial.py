"""Reusable one-dimensional spatial linear operators for Dendra.

The initial operators here target Dendra's own unbranched Axon classes, where
all compartments are ordered along the final tensor dimension.  They are kept
ion/material agnostic so voltage, material diffusion, and future cable-like
processes can share the same tridiagonal solve path.
"""

from __future__ import annotations

import warnings
from collections import deque
from collections.abc import Mapping
from functools import partial
from math import prod
from numbers import Number
from typing import Literal

import numpy as np
import torch

from dendra.models.integrators._native import _cpu_topology, _mps_solver_available

try:  # Registers torch.ops.dendra_solvers when available.
    import dendra_solvers  # noqa: F401

    DENDRA_SOLVERS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on local installation
    dendra_solvers = None
    DENDRA_SOLVERS_AVAILABLE = False

_FUNCTIONAL_SCALAR_SPD = getattr(dendra_solvers, "solve_tri_spd", None)
if not getattr(_FUNCTIONAL_SCALAR_SPD, "_DENDRA_FUNC_COMPATIBLE", False):
    _FUNCTIONAL_SCALAR_SPD = None

try:
    from dendra.models.integrators.tridiag import (
        pcr_solve_parallel_t,
        pcr_solve_t,
    )
except Exception:  # pragma: no cover - import robustness for partial installs
    pcr_solve_parallel_t = None
    pcr_solve_t = None

try:
    from dendra.models.integrators.triton import (
        dhs_solve_cuda,
        pcr_solve_cuda_t,
        thomas_solve_cuda_t,
    )
except Exception:  # pragma: no cover - triton may be unavailable
    dhs_solve_cuda = None
    pcr_solve_cuda_t = None
    thomas_solve_cuda_t = None


def _as_tensor_like(value, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=like.device, dtype=like.dtype)
    if isinstance(value, Number):
        return like.new_full((), value)
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _broadcast_to(value, target: torch.Tensor) -> torch.Tensor:
    value = _as_tensor_like(value, target)
    if value.ndim == 0:
        return value.expand_as(target)
    while value.ndim < target.ndim:
        value = value.unsqueeze(0)
    return value.expand_as(target)


def _broadcast_mask_to(value, target: torch.Tensor) -> torch.Tensor:
    """Broadcast a structural boolean mask without adopting target dtype."""
    if isinstance(value, bool):
        value = target.new_full((), value, dtype=torch.bool)
    elif torch.is_tensor(value):
        value = value.to(device=target.device, dtype=torch.bool)
    else:
        value = torch.as_tensor(value, device=target.device, dtype=torch.bool)
    if value.ndim == 0:
        return value.expand(target.shape)
    while value.ndim < target.ndim:
        value = value.unsqueeze(0)
    return value.expand(target.shape)


def _broadcast_edge_to(value, target: torch.Tensor) -> torch.Tensor:
    """Broadcast an edge quantity against a node-shaped spatial field."""
    edge_shape = tuple(target.shape[:-1]) + (max(int(target.shape[-1]) - 1, 0),)
    value = _as_tensor_like(value, target)
    if value.ndim == 0:
        return value.expand(edge_shape)
    while value.ndim < len(edge_shape):
        value = value.unsqueeze(0)
    try:
        return value.expand(edge_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"edge quantity with shape {tuple(value.shape)} cannot broadcast to "
            f"spatial edge shape {edge_shape}."
        ) from exc


def _broadcast_tree_edge_to(
    value, target: torch.Tensor, edge_count: int
) -> torch.Tensor:
    """Broadcast a compact Tree-edge quantity against a node-shaped field.

    Tree edge values use the population's stable material-edge order rather
    than the DHS solver order.  For a connected Tree ``edge_count == C - 1``;
    keeping it explicit also makes the single-compartment ``E == 0`` case
    unambiguous.
    """
    edge_shape = tuple(target.shape[:-1]) + (int(edge_count),)
    value = _as_tensor_like(value, target)
    if value.ndim == 0:
        return value.expand(edge_shape)
    while value.ndim < len(edge_shape):
        value = value.unsqueeze(0)
    try:
        return value.expand(edge_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Tree edge quantity with shape {tuple(value.shape)} cannot "
            f"broadcast to compact material-edge shape {edge_shape}."
        ) from exc


def _flatten_last(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    shape = tuple(x.shape)
    if x.ndim < 1:
        raise ValueError("spatial operators require at least one spatial dimension")
    # Explicit leading-size arithmetic also works under a zero-lane vmap;
    # reshape(-1, K) otherwise asks PyTorch to infer a dimension from 0 / 0.
    return x.reshape(prod(shape[:-1]), shape[-1]), shape


def _dense_tridiagonal_solve(a, b, c, rhs):
    """Debug/fallback batched tridiagonal solve.

    a, c: (..., K-1), b/rhs: (..., K).  The solve is over the final dimension.
    """
    rhs2, shape = _flatten_last(rhs)
    K = rhs2.shape[-1]
    B = rhs2.shape[0]
    a2 = a.reshape(B, max(K - 1, 0))
    b2 = b.reshape(B, K)
    c2 = c.reshape(B, max(K - 1, 0))
    A = torch.diag_embed(b2)
    if K > 1:
        A = A + torch.diag_embed(a2, offset=-1) + torch.diag_embed(c2, offset=1)
    out = torch.linalg.solve(A, rhs2.unsqueeze(-1)).squeeze(-1)
    return out.reshape(shape)


def _normalize_solver_name(solver: str | None) -> str:
    mode = str(solver or "auto").lower()
    aliases = {
        "inv": "thomas",
        "custom": "thomas",
        "cuda": "thomas",
        "cpu": "thomas",
        "torch": "dense",
        "linalg": "dense",
    }
    return aliases.get(mode, mode)


def select_tridiagonal_solver(solver: str | None, device: torch.device | str):
    """Select a tridiagonal solver using Dendra voltage-integrator conventions.

    The policy mirrors the unbranched voltage integrator:

    - ``solver='pcr'`` explicitly requests PCR: Triton on CUDA, vectorized
      pure PyTorch on MPS, and the historical pure-PyTorch Thomas fallback on
      CPU.
    - ``solver='spd'`` uses the CPU SPD tridiagonal solver when available and
      falls back to Thomas-style solvers otherwise.
    - ``solver='thomas'``/``'auto'`` uses Triton Thomas on CUDA, Dendra's CPU
      extension on CPU when available, vectorized pure-PyTorch PCR on MPS, and
      the pure-PyTorch Thomas fallback on CPU when needed.
    """
    mode = _normalize_solver_name(solver)
    dev_type = torch.device(device).type

    if mode in {"dense", "debug"}:
        return None, "dense"

    if mode == "pcr":
        if dev_type == "cuda":
            if pcr_solve_cuda_t is None:
                raise ImportError(
                    "solver='pcr' on CUDA requires dendra.models.integrators.triton.pcr_solve_cuda_t."
                )
            return pcr_solve_cuda_t, "pcr_cuda"
        if dev_type == "mps":
            if pcr_solve_parallel_t is None:
                raise ImportError(
                    "solver='pcr' on MPS requires "
                    "dendra.models.integrators.tridiag.pcr_solve_parallel_t."
                )
            return pcr_solve_parallel_t, "pcr_mps"
        if pcr_solve_t is None:
            raise ImportError(
                "solver='pcr' requires dendra.models.integrators.tridiag.pcr_solve_t."
            )
        return pcr_solve_t, "pcr_cpu"

    if mode == "spd":
        if dev_type == "cuda":
            warnings.warn(
                "Material diffusion solver='spd' is not implemented on CUDA; falling back to Thomas."
            )
            mode = "thomas"
        elif dev_type == "mps":
            warnings.warn(
                "Material diffusion solver='spd' is not implemented on MPS; "
                "falling back to parallel cyclic reduction."
            )
            mode = "thomas"
        else:
            try:
                from dendra_solvers import solve_tri_spd
            except ImportError:
                warnings.warn(
                    "Material diffusion solver='spd' requested on CPU but dendra_solvers.solve_tri_spd "
                    "is unavailable; falling back to Thomas/PCR."
                )
                mode = "thomas"
            else:
                return solve_tri_spd, "spd_cpu"

    if mode not in {"auto", "thomas"}:
        warnings.warn(
            f"Unknown material tridiagonal solver {solver!r}; falling back to 'thomas'."
        )
        mode = "thomas"

    if dev_type == "cuda":
        if thomas_solve_cuda_t is not None:
            return thomas_solve_cuda_t, "thomas_cuda"
        if pcr_solve_cuda_t is not None:
            warnings.warn(
                "Triton Thomas solver unavailable; falling back to CUDA PCR solver."
            )
            return pcr_solve_cuda_t, "pcr_cuda"
        if mode == "thomas":
            raise ImportError(
                "solver='thomas' on CUDA requires dendra.models.integrators.triton.thomas_solve_cuda_t."
            )
        return None, "dense"

    if dev_type == "mps":
        if pcr_solve_parallel_t is not None:
            return pcr_solve_parallel_t, "pcr_mps"
        return None, "dense"

    if dev_type == "cpu":
        if DENDRA_SOLVERS_AVAILABLE:
            return torch.ops.dendra_solvers.thomas_solve_t, "thomas_cpu"
        if pcr_solve_t is not None:
            warnings.warn(
                "Using material diffusion on CPU without dendra-solvers installed; "
                "falling back to the PyTorch PCR/Thomas solver. Install the native "
                "package with `python -m pip install --upgrade "
                '--only-binary=dendra-solvers "dendra-solvers>=0.5.1"`.'
            )
            return pcr_solve_t, "pcr_cpu"
        return None, "dense"

    return None, "dense"


def solve_tridiagonal_1d(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    rhs: torch.Tensor,
    *,
    solver: Literal[
        "auto", "custom", "torch", "dense", "thomas", "pcr", "spd", "inv"
    ] = "auto",
) -> torch.Tensor:
    """Solve a batched tridiagonal system along the final dimension.

    This function remains as a public/debug helper.  Hot material-process paths
    should prefer :class:`SpatialOperator1D`, which selects the solver and
    precomputes static bands outside the timestep loop.
    """
    if rhs.shape[-1] == 1:
        return rhs / b

    rhs2, shape = _flatten_last(rhs)
    K = rhs2.shape[-1]
    B = rhs2.shape[0]
    a2 = a.reshape(B, K - 1).contiguous()
    b2 = b.reshape(B, K).contiguous()
    c2 = c.reshape(B, K - 1).contiguous()
    d2 = rhs2.contiguous()

    mode = _normalize_solver_name(solver)
    if mode in {"dense", "torch", "linalg", "debug"}:
        return _dense_tridiagonal_solve(a, b, c, rhs)

    solve_fn, _ = select_tridiagonal_solver(mode, d2.device)
    if solve_fn is None:
        return _dense_tridiagonal_solve(a, b, c, rhs)
    out = solve_fn(a2, b2, c2, d2)
    return out.reshape(shape)


class SpatialOperator1D(torch.nn.Module):
    """Finite-volume helper for 1D compartment chains.

    The operator assumes the final tensor dimension is the compartment axis; all
    leading dimensions are independent batches/fibers.  Geometry is expected in
    micrometers.  Diffusivities are therefore expected in ``um^2 / ms`` by
    default, and the implicit solve advances concentration-like fields in ms.

    Unlike the convenience :func:`solve_tridiagonal_1d`, this module follows the
    same split used by Dendra's voltage integrators: solver selection and static
    geometry/band construction happen in ``configure_diffusion(...)``; the
    per-step ``diffuse_*_configured(...)`` methods only assemble the RHS and call
    the preselected solver.
    """

    def __init__(
        self,
        *,
        solver: str = "auto",
        boundary: str = "sealed",
        interface_scheme: str = "arithmetic",
    ):
        super().__init__()
        self.solver = _normalize_solver_name(solver)
        self.boundary = str(boundary or "sealed").lower()
        if self.boundary not in {"sealed", "no_flux", "noflux"}:
            raise NotImplementedError(
                "SpatialOperator1D MVP supports only sealed/no-flux boundaries."
            )
        interface_scheme = str(interface_scheme or "arithmetic").lower()
        interface_aliases = {
            "mean": "arithmetic",
            "legacy": "arithmetic",
            "harmonic": "series",
            "resistance": "series",
            "resistive": "series",
        }
        self.interface_scheme = interface_aliases.get(
            interface_scheme, interface_scheme
        )
        if self.interface_scheme not in {"arithmetic", "series"}:
            raise ValueError(
                "interface_scheme must be 'arithmetic' or 'series'; got "
                f"{interface_scheme!r}."
            )
        self.configured = False
        self.solver_name = "unconfigured"
        self._solve = None
        self.base_shape: tuple[int, ...] | None = None
        self.B = 0
        self.K = 0

    def _set_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    @staticmethod
    def cylinder_cross_section_um2(diam_um: torch.Tensor) -> torch.Tensor:
        return torch.pi * (diam_um * 0.5) ** 2

    @staticmethod
    def cylinder_volume_um3(diam_um: torch.Tensor, dx_um: torch.Tensor) -> torch.Tensor:
        return SpatialOperator1D.cylinder_cross_section_um2(diam_um) * dx_um

    def finite_volume_geometry(
        self,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compartment volumes and interface areas for a 1D cable."""
        dx_um = _broadcast_to(dx_um, diam_um)
        area = self.cylinder_cross_section_um2(diam_um)
        volume = area * dx_um * _broadcast_to(volume_fraction, diam_um)
        edge_area = 0.5 * (area[..., :-1] + area[..., 1:])
        edge_area = edge_area * _broadcast_to(area_fraction, edge_area)
        return volume, edge_area

    def edge_conductance(
        self,
        D_um2_per_ms,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
        interface_scheme: str | None = None,
        _validate_values: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(volume, g_edge)`` for finite-volume diffusion."""
        volume, edge_area = self.finite_volume_geometry(
            diam_um,
            dx_um,
            volume_fraction=volume_fraction,
            area_fraction=area_fraction,
        )
        dx_um = _broadcast_to(dx_um, diam_um)
        D_full = _broadcast_to(D_um2_per_ms, diam_um)
        scheme = str(interface_scheme or self.interface_scheme).lower()
        scheme = {
            "mean": "arithmetic",
            "legacy": "arithmetic",
            "harmonic": "series",
            "resistance": "series",
            "resistive": "series",
        }.get(scheme, scheme)
        if scheme == "arithmetic":
            edge_len = 0.5 * (dx_um[..., :-1] + dx_um[..., 1:])
            D_edge = 0.5 * (D_full[..., :-1] + D_full[..., 1:])
            tiny = torch.finfo(edge_len.dtype).tiny
            g_edge = D_edge * edge_area / edge_len.clamp_min(tiny)
        elif scheme == "series":
            if _validate_values and bool(torch.any(D_full < 0).item()):
                raise ValueError(
                    "series-interface diffusion requires non-negative diffusivity."
                )
            area = self.cylinder_cross_section_um2(diam_um)
            capacity = D_full * area
            tiny = torch.finfo(capacity.dtype).tiny
            infinity = torch.full_like(capacity, torch.inf)
            half_resistance = torch.where(
                capacity > 0,
                0.5 * dx_um / capacity.clamp_min(tiny),
                infinity,
            )
            resistance = half_resistance[..., :-1] + half_resistance[..., 1:]
            valid = torch.isfinite(resistance) & (resistance > 0)
            g_edge = torch.where(
                valid,
                resistance.clamp_min(tiny).reciprocal(),
                torch.zeros_like(resistance),
            )
            g_edge = g_edge * _broadcast_to(area_fraction, g_edge)
        else:
            raise ValueError(
                "interface_scheme must be 'arithmetic' or 'series'; got "
                f"{interface_scheme!r}."
            )
        return volume, g_edge

    def configure_diffusion(
        self,
        c_like: torch.Tensor,
        dt,
        D_um2_per_ms,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
        interface_scheme: str | None = None,
        node_mask=None,
        solver: str | None = None,
        volume=None,
        edge_factor=None,
        edge_area=None,
        edge_distance=None,
        D_location: str = "node",
        _validate_values: bool = True,
        _select_solver: bool = True,
    ) -> "SpatialOperator1D":
        """Precompute geometry, solver choice, and timestep-scaled bands.

        ``node_mask`` selects the induced diffusion subgraph.  An interface is
        active only when both of its endpoint compartments are selected, so
        omitted compartments impose exact sealed boundaries.  The full field
        shape is retained: disconnected selected regions therefore form
        independent diagonal blocks in the same solve.
        """
        if c_like.ndim < 1:
            raise ValueError("diffusion fields must have a final compartment dimension")
        if c_like.shape[-1] <= 0:
            raise ValueError("diffusion fields must have at least one compartment")

        c_ref = c_like
        dt_t = _as_tensor_like(dt, c_ref)
        if node_mask is None:
            active = torch.ones_like(c_ref, dtype=torch.bool)
        else:
            active = _broadcast_mask_to(node_mask, c_ref)

        D_location = str(D_location or "node").lower()
        if D_location not in {"node", "edge"}:
            raise ValueError("D_location must be 'node' or 'edge'.")
        has_factor = edge_factor is not None
        has_area_distance = edge_area is not None or edge_distance is not None
        explicit_geometry = volume is not None or has_factor or has_area_distance
        if explicit_geometry and volume is None:
            raise ValueError("Explicit finite-volume geometry requires volume.")
        if has_factor and has_area_distance:
            raise ValueError(
                "Specify edge_factor or edge_area/edge_distance, not both."
            )
        if (
            explicit_geometry
            and not has_factor
            and not (edge_area is not None and edge_distance is not None)
        ):
            raise ValueError(
                "Explicit finite-volume geometry requires edge_factor or both "
                "edge_area and edge_distance."
            )
        if D_location == "edge" and not explicit_geometry:
            raise ValueError("Edge-located diffusivity requires explicit geometry.")

        edge_active = active[..., :-1] & active[..., 1:]
        if explicit_geometry:
            volume = _broadcast_to(volume, c_ref)
            if edge_factor is not None:
                edge_factor = _broadcast_edge_to(edge_factor, c_ref)
            else:
                edge_area = _broadcast_edge_to(edge_area, c_ref)
                edge_distance = _broadcast_edge_to(edge_distance, c_ref)
                if _validate_values:
                    selected_area = edge_area[edge_active]
                    selected_distance = edge_distance[edge_active]
                    if bool(torch.any(~torch.isfinite(selected_area)).item()) or bool(
                        torch.any(selected_area < 0).item()
                    ):
                        raise ValueError(
                            "Active finite-volume interfaces require finite "
                            "non-negative edge_area."
                        )
                    if bool(
                        torch.any(~torch.isfinite(selected_distance)).item()
                    ) or bool(torch.any(selected_distance <= 0).item()):
                        raise ValueError(
                            "Active finite-volume interfaces require finite positive "
                            "edge_distance."
                        )
                safe_area = torch.where(
                    edge_active, edge_area, torch.zeros_like(edge_area)
                )
                safe_distance = torch.where(
                    edge_active, edge_distance, torch.ones_like(edge_distance)
                )
                edge_factor = safe_area / safe_distance
            if _validate_values:
                active_volume = volume[active]
                if bool(torch.any(~torch.isfinite(active_volume)).item()) or bool(
                    torch.any(active_volume <= 0).item()
                ):
                    raise ValueError(
                        "Active finite-volume compartments require finite positive volume."
                    )
                active_edge_factor = edge_factor[edge_active]
                if bool(torch.any(~torch.isfinite(active_edge_factor)).item()) or bool(
                    torch.any(active_edge_factor < 0).item()
                ):
                    raise ValueError(
                        "Active finite-volume interfaces require finite non-negative "
                        "edge_factor."
                    )

            if D_location == "edge":
                D_edge = _broadcast_edge_to(D_um2_per_ms, c_ref)
            else:
                D_full = _broadcast_to(D_um2_per_ms, c_ref)
                D_full = torch.where(active, D_full, torch.zeros_like(D_full))
                if _validate_values:
                    participates = torch.zeros_like(active)
                    participates[..., :-1] |= edge_active
                    participates[..., 1:] |= edge_active
                    active_node_D = D_full[participates]
                    if bool(torch.any(~torch.isfinite(active_node_D)).item()) or bool(
                        torch.any(active_node_D < 0).item()
                    ):
                        raise ValueError(
                            "Nodes incident to active finite-volume interfaces "
                            "require finite non-negative diffusivity."
                        )
                D_edge = 0.5 * (D_full[..., :-1] + D_full[..., 1:])
            if _validate_values:
                active_D = D_edge[edge_active]
                if bool(torch.any(~torch.isfinite(active_D)).item()) or bool(
                    torch.any(active_D < 0).item()
                ):
                    raise ValueError(
                        "Active finite-volume interfaces require finite non-negative "
                        "diffusivity."
                    )
            # Mask before multiplication so NaN/sentinel values on excluded
            # interfaces cannot contaminate the induced transport subgraph.
            safe_D = torch.where(edge_active, D_edge, torch.zeros_like(D_edge))
            safe_factor = torch.where(
                edge_active, edge_factor, torch.zeros_like(edge_factor)
            )
            g_edge = safe_D * safe_factor
        else:
            diam_um = _broadcast_to(diam_um, c_ref)
            dx_um = _broadcast_to(dx_um, c_ref)
            D_full = _broadcast_to(D_um2_per_ms, c_ref)
            # Excluded values have no physical meaning in a regional process.
            # Clear them before interface validation/geometry so a sentinel,
            # NaN, or negative value outside support cannot affect the induced
            # subgraph. Crossing edges are sealed below in all cases.
            D_effective = torch.where(active, D_full, torch.zeros_like(D_full))
            volume, g_edge = self.edge_conductance(
                D_effective,
                diam_um,
                dx_um,
                volume_fraction=volume_fraction,
                area_fraction=area_fraction,
                interface_scheme=interface_scheme,
                _validate_values=_validate_values,
            )

        vol2, base_shape = _flatten_last(volume)
        active2, _ = _flatten_last(active)
        K = int(vol2.shape[-1])
        B = int(vol2.shape[0])
        self.B = B
        self.K = K
        self.base_shape = base_shape

        dt_t = dt_t.to(device=c_ref.device, dtype=c_ref.dtype)
        if dt_t.ndim != 0:
            # Broadcastable tensor dt is allowed, but store it in flattened solve
            # layout so the compiled hot path never repeats shape normalization.
            dt_t = _broadcast_to(dt_t, c_ref).reshape(B, K)
        self._set_buffer("dt", dt_t)
        self._set_buffer("node_mask", active2.contiguous())
        self._set_buffer("relax_rhs", vol2.new_zeros((B, K)))
        self._has_relaxation = False

        if K == 1:
            self._set_buffer("volume", vol2)
            safe_volume = torch.where(active2, vol2, torch.ones_like(vol2))
            self._set_buffer(
                "inv_volume",
                torch.where(
                    active2,
                    1.0 / safe_volume.clamp_min(torch.finfo(vol2.dtype).tiny),
                    torch.zeros_like(vol2),
                ),
            )
            self._set_buffer("g_edge", vol2.new_empty((B, 0)))
            self._set_buffer("lower", vol2.new_empty((B, 0)))
            self._set_buffer("upper", vol2.new_empty((B, 0)))
            solve_volume = safe_volume
            self._set_buffer("solve_volume", solve_volume.contiguous())
            self._set_buffer("main", solve_volume.contiguous())
            self._solve = None
            self.solver_name = "trivial"
            self.configured = True
            return self

        g2 = g_edge.reshape(B, K - 1)
        edge_active = active2[:, :-1] & active2[:, 1:]
        # Apply the regional boundary after endpoint diffusivities have been
        # averaged.  Setting D=0 outside the region alone would leave a
        # half-strength crossing edge under arithmetic endpoint averaging.
        g2 = torch.where(edge_active, g2, torch.zeros_like(g2)).contiguous()
        zeros = vol2.new_zeros((B, 1))
        left_g = torch.cat((zeros, g2), dim=-1)
        right_g = torch.cat((g2, zeros), dim=-1)

        solve_volume = torch.where(active2, vol2, torch.ones_like(vol2))
        if dt_t.ndim == 0:
            # A rank-two scalar view avoids PyTorch's empty-vmap scalar
            # dispatch special case without materializing a broadcast plane.
            dt_math = dt_t.reshape(1, 1)
            lower = -dt_math * g2
            upper = lower
            main = solve_volume + dt_math * (left_g + right_g)
        else:
            # A node-wise timestep scales each finite-volume balance row.  The
            # two entries for one physical edge are consequently not symmetric:
            # the upper entry belongs to its left row and the lower entry to its
            # right row. Row scaling preserves every constant field exactly.
            upper = -dt_t[:, :-1] * g2
            lower = -dt_t[:, 1:] * g2
            main = solve_volume + dt_t * (left_g + right_g)

        self._set_buffer("volume", vol2.contiguous())
        self._set_buffer("solve_volume", solve_volume.contiguous())
        self._set_buffer(
            "inv_volume",
            torch.where(
                active2,
                1.0 / solve_volume.clamp_min(torch.finfo(vol2.dtype).tiny),
                torch.zeros_like(vol2),
            ).contiguous(),
        )
        self._set_buffer("g_edge", g2)
        self._set_buffer("left_g", left_g.contiguous())
        self._set_buffer("right_g", right_g.contiguous())
        self._set_buffer("lower", lower.contiguous())
        self._set_buffer("upper", upper.contiguous())
        self._set_buffer("main", main.contiguous())

        if _select_solver:
            solve_fn, solver_name = select_tridiagonal_solver(
                solver or self.solver, c_ref.device
            )
            self._solve = solve_fn
            self.solver_name = solver_name
        self.configured = True
        return self

    def configure_relaxation(self, rate, target=0.0, *, where=None):
        """Add ``-rate * (c - target)`` to the configured implicit system.

        Multiple calls accumulate, allowing independent linear reservoirs to
        couple to the same transported field without an additional split step.
        """
        self._require_configured()
        updates = self._prepare_relaxation(
            self._buffers, rate, target, where=where, validate_values=True
        )
        self._set_buffer("main", updates["main"])
        self._set_buffer("relax_rhs", updates["relax_rhs"])
        self._has_relaxation = True
        return self

    def _prepare_relaxation(self, workspace, rate, target, *, where, validate_values):
        like = workspace["volume"].reshape(self.base_shape)
        active = workspace["node_mask"].reshape(self.base_shape)
        if where is not None:
            active = active & _broadcast_mask_to(where, like)

        rate_full = _broadcast_to(rate, like)
        if validate_values:
            selected_rate = rate_full[active]
            if bool(torch.any(~torch.isfinite(selected_rate)).item()) or bool(
                torch.any(selected_rate < 0).item()
            ):
                raise ValueError("Relaxation rates must be finite and non-negative.")
        rate_full = torch.where(active, rate_full, torch.zeros_like(rate_full))

        target_full = _broadcast_to(target, like)
        coupled = active & (rate_full != 0)
        if validate_values and bool(
            torch.any(~torch.isfinite(target_full[coupled])).item()
        ):
            raise ValueError("Relaxation targets must be finite where rate is nonzero.")
        target_full = torch.where(coupled, target_full, torch.zeros_like(target_full))

        rate2 = rate_full.reshape(self.B, self.K)
        target2 = target_full.reshape(self.B, self.K)
        active2 = active.reshape(self.B, self.K)
        safe_volume = torch.where(
            active2,
            workspace["volume"],
            torch.zeros_like(workspace["volume"]),
        )
        mass_rate = safe_volume * rate2
        dt_t = workspace["dt"]
        dt_math = dt_t.reshape(1, 1) if dt_t.ndim == 0 else dt_t
        reaction_main = dt_math * mass_rate
        reaction_rhs = reaction_main * target2
        return dict(
            workspace,
            main=workspace["main"] + reaction_main,
            relax_rhs=workspace["relax_rhs"] + reaction_rhs,
        )

    def _prepared_buffer_names(self) -> tuple[str, ...]:
        """Inventory the shared numerical export without cloning its tensors."""
        self._require_configured()
        return tuple(self._buffers)

    def prepared_buffers(self) -> dict[str, torch.Tensor]:
        """Export an independent, differentiable numerical workspace."""
        return {
            name: self._buffers[name].clone() for name in self._prepared_buffer_names()
        }

    def prepare_diffusion(self, c_like, dt, D_um2_per_ms, diam_um, dx_um, **kwargs):
        """Build explicit tensor workspace without changing this operator.

        Structural checks remain active. Numerical inputs must satisfy the
        same physical-domain contract checked by :meth:`configure_diffusion`;
        no tensor-value reads or solver selection occur in this transformable
        builder. Configure and validate the owning plan before lowering it.
        """
        self._require_configured()
        if (
            kwargs.get("solver") is not None
            and _normalize_solver_name(kwargs["solver"]) != self.solver
        ):
            raise ValueError("Prepared diffusion cannot change its configured solver.")
        staged = type(self)(
            solver=self.solver,
            boundary=self.boundary,
            interface_scheme=self.interface_scheme,
        )
        staged.configure_diffusion(
            c_like,
            dt,
            D_um2_per_ms,
            diam_um,
            dx_um,
            **kwargs,
            _validate_values=False,
            _select_solver=False,
        )
        if staged.base_shape != self.base_shape:
            raise ValueError(
                "Prepared diffusion must retain its configured field shape."
            )
        return staged.prepared_buffers()

    def prepare_relaxation(self, workspace, rate, target=0.0, *, where=None):
        """Return updated implicit coefficients for a validated reservoir."""
        self._require_configured()
        return self._prepare_relaxation(
            workspace, rate, target, where=where, validate_values=False
        )

    def _functional_solver(self):
        """Bind AD-compatible facades over the selected, unchanged solver."""
        self._require_configured()
        solver = self._solve
        if solver is None:
            return None
        if solver is _FUNCTIONAL_SCALAR_SPD:
            return solver
        if solver is thomas_solve_cuda_t:
            from dendra.models.integrators._cuda_functional import thomas_solve

            return thomas_solve
        if solver is pcr_solve_cuda_t:
            from dendra.models.integrators._cuda_functional import pcr_solve

            return pcr_solve
        name = getattr(solver, "__name__", None)
        if getattr(solver, "__module__", None) == "torch._ops.dendra_solvers":
            facade = getattr(dendra_solvers, name, None)
            if name in {"thomas_solve_t", "pcr_solve_t"} and callable(facade):
                return facade
        if solver in {pcr_solve_t, pcr_solve_parallel_t}:
            return solver
        raise RuntimeError(
            f"Material diffusion solver {self.solver_name!r} is not "
            "transform-compatible; use a dendra-solvers build with pure SPD "
            "facades, or select the native Thomas, PCR, or dense solver."
        )

    def _require_configured(self) -> None:
        if not self.configured:
            raise RuntimeError(
                "SpatialOperator1D has not been configured. Call configure_diffusion(...) "
                "from the owning MaterialProcess.set_dt(...) before advancing."
            )

    def diffuse_implicit_configured(self, c: torch.Tensor) -> torch.Tensor:
        """Backward-Euler sealed-boundary update using precomputed bands."""
        self._require_configured()
        return self.diffuse_implicit_prepared(c, self._buffers, solve=self._solve)

    def diffuse_implicit_prepared(
        self, c: torch.Tensor, workspace: Mapping[str, torch.Tensor], *, solve
    ) -> torch.Tensor:
        """Advance from explicit coefficients using the configured solver plan."""
        # Only the bound workspace shares the template's reservoir declaration.
        # A caller may add relaxation to an independent prepared workspace.
        if self.K <= 1 and workspace is self._buffers and not self._has_relaxation:
            return c
        c2 = c.reshape(self.B, self.K).contiguous()
        rhs = workspace["solve_volume"] * c2 + workspace["relax_rhs"]
        if self.K <= 1:
            out = rhs / workspace["main"]
        elif solve is None:
            out = _dense_tridiagonal_solve(
                workspace["lower"],
                workspace["main"],
                workspace["upper"],
                rhs,
            ).reshape(self.B, self.K)
        else:
            out = solve(
                workspace["lower"],
                workspace["main"],
                workspace["upper"],
                rhs,
            )
        out = out.reshape(self.base_shape)
        active = workspace["node_mask"].reshape(self.base_shape)
        return torch.where(active, out, c)

    def diffuse_explicit_configured(self, c: torch.Tensor) -> torch.Tensor:
        """Explicit sealed-boundary finite-volume update using precomputed geometry."""
        self._require_configured()
        return self.diffuse_explicit_prepared(c, self._buffers)

    def diffuse_explicit_prepared(
        self, c: torch.Tensor, workspace: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Advance explicit diffusion without reading bound numerical buffers."""
        if self.K <= 1:
            return c
        c2 = c.reshape(self.B, self.K)
        edge_flux = workspace["g_edge"] * (c2[:, 1:] - c2[:, :-1])
        padding = torch.zeros_like(edge_flux[:, :1])
        net = torch.cat((edge_flux, padding), -1) - torch.cat((padding, edge_flux), -1)
        dt_t = workspace["dt"]
        dt_math = dt_t.reshape(1, 1) if dt_t.ndim == 0 else dt_t
        out = (c2 + dt_math * net * workspace["inv_volume"]).reshape(self.base_shape)
        active = workspace["node_mask"].reshape(self.base_shape)
        return torch.where(active, out, c)

    # ------------------------------------------------------------------
    # Backward-compatible convenience methods.  These recompute coefficients
    # and should not be used in the hot MaterialProcess path.
    # ------------------------------------------------------------------
    def diffuse_implicit(
        self,
        c: torch.Tensor,
        dt,
        D_um2_per_ms,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
        interface_scheme: str | None = None,
        node_mask=None,
        solver: str | None = None,
    ) -> torch.Tensor:
        self.configure_diffusion(
            c,
            dt,
            D_um2_per_ms,
            diam_um,
            dx_um,
            volume_fraction=volume_fraction,
            area_fraction=area_fraction,
            interface_scheme=interface_scheme,
            node_mask=node_mask,
            solver=solver,
        )
        return self.diffuse_implicit_configured(c)

    def diffuse_explicit(
        self,
        c: torch.Tensor,
        dt,
        D_um2_per_ms,
        diam_um: torch.Tensor,
        dx_um: torch.Tensor,
        *,
        volume_fraction=1.0,
        area_fraction=1.0,
        interface_scheme: str | None = None,
        node_mask=None,
    ) -> torch.Tensor:
        self.configure_diffusion(
            c,
            dt,
            D_um2_per_ms,
            diam_um,
            dx_um,
            volume_fraction=volume_fraction,
            area_fraction=area_fraction,
            interface_scheme=interface_scheme,
            node_mask=node_mask,
            solver=self.solver,
        )
        return self.diffuse_explicit_configured(c)


# ---------------------------------------------------------------------------
# Tree / DHS finite-volume operators
# ---------------------------------------------------------------------------

THREADS_PER_WARP = 32


def _build_tree_morphology(
    parent_idx: list[int],
) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:
    """Build DHS/Hines forest metadata from a parent-index vector.

    This mirrors :func:`dendra.models.integrators.tree.build_morphology` but is
    kept local to avoid importing the full voltage-integrator module from the
    mechanism/material namespace.
    """
    K = len(parent_idx)
    children: list[list[int]] = [[] for _ in range(K)]
    roots: list[int] = []
    for i, p in enumerate(parent_idx):
        p = int(p)
        if p == -1:
            roots.append(i)
        else:
            if p < 0 or p >= K:
                raise ValueError(f"invalid parent index {p} for node {i}")
            children[p].append(i)

    if not roots:
        raise ValueError("tree morphology must have at least one root with parent=-1")

    depth = torch.zeros(K, dtype=torch.int32)
    visited = torch.zeros(K, dtype=torch.bool)
    visited[roots] = True
    q = deque(roots)
    while q:
        u = q.popleft()
        for child in children[u]:
            if visited[child]:
                raise ValueError("tree morphology contains a cycle")
            visited[child] = True
            depth[child] = depth[u] + 1
            q.append(child)

    if not bool(torch.all(visited)):
        raise ValueError(
            "tree morphology must be connected to a root; unreachable nodes may form a cycle"
        )

    return torch.as_tensor(parent_idx, dtype=torch.int32), children, depth


def _build_dhs_layers(
    depth: torch.Tensor, threads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build DHS elimination order/layer pointers, deepest nodes first."""
    depth_cpu = depth.detach().cpu().numpy()
    max_d = int(depth_cpu.max()) if depth_cpu.size else 0

    order: list[int] = []
    layer_ptr: list[int] = [0]
    bins: list[list[int]] = [[] for _ in range(max_d + 1)]
    for i, d in enumerate(depth_cpu):
        bins[int(d)].append(i)

    for d in range(max_d, -1, -1):
        bucket = bins[d]
        for start in range(0, len(bucket), threads):
            order.extend(bucket[start : start + threads])
            layer_ptr.append(len(order))

    return torch.as_tensor(order, dtype=torch.int32), torch.as_tensor(
        layer_ptr, dtype=torch.int32
    )


def _edge_diff_geom_from_graph(graph, parent, child) -> float:
    """Return edge diffusion geometry in µm from a graph edge.

    Preferred graph metadata is ``edge['diff_geom_um']`` generated by Dendra's
    NEURON import path.  The resistance-based fallback is only for older or
    hand-written graphs and assumes a single effective axial resistivity.
    """
    edge = graph.edges[parent, child]
    if "diff_geom_um" in edge:
        return float(edge["diff_geom_um"])

    R_ohm = edge.get("R_ohm", None)
    if R_ohm is not None and float(R_ohm) != 0.0:
        parent_Ra = graph.nodes[parent].get("Ra", None)
        child_Ra = graph.nodes[child].get("Ra", None)
        if parent_Ra is None and child_Ra is None:
            raise KeyError(
                "Edge is missing diff_geom_um and cannot recover it from R_ohm "
                "because neither endpoint has Ra."
            )
        if parent_Ra is None:
            rhoa = float(child_Ra)
        elif child_Ra is None:
            rhoa = float(parent_Ra)
        else:
            rhoa = 0.5 * (float(parent_Ra) + float(child_Ra))
        return float(rhoa * 1e4 / float(R_ohm))

    # Last-resort stylized fallback, useful for minimal tests.
    L_edge = float(edge.get("L", 0.0) or 0.0)
    if L_edge <= 0.0:
        return 0.0
    d_parent = float(graph.nodes[parent].get("diam", 0.0) or 0.0)
    d_child = float(graph.nodes[child].get("diam", 0.0) or 0.0)
    if d_parent <= 0.0 or d_child <= 0.0:
        return 0.0
    a_parent = float(np.pi * (0.5 * d_parent) ** 2)
    a_child = float(np.pi * (0.5 * d_child) ** 2)
    integral = 0.5 * L_edge / a_parent + 0.5 * L_edge / a_child
    return float(1.0 / integral) if integral > 0.0 else 0.0


def graph_to_tree_topology(graphs):
    """Convert graph-backed morphology into stable Tree transport topology.

    Returns
    -------
    parent_idx : torch.Tensor
        Length-K parent vector in solver/topological order.
    node_order : list
        Mapping from solver index to model-storage compartment index.
    edge_parent, edge_child : torch.Tensor
        Compact length-E material-edge map in ascending child storage order.

    Notes
    -----
    The DHS solver may reorder compartments topologically, while public named
    geometry is expressed in model storage order.  Keeping these maps separate
    lets every chemical geometry reuse the morphology adjacency without also
    inheriting its native diffusion coefficients.
    """
    if not isinstance(graphs, (list, tuple)):
        graphs = [graphs]
    if len(graphs) == 0:
        raise ValueError("at least one graph is required")

    # Import networkx lazily so non-tree material diffusion does not require it.
    import networkx as nx  # noqa: PLC0415

    nodes = list(nx.topological_sort(graphs[0]))
    K = len(nodes)
    try:
        storage_nodes = [int(node) for node in nodes]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Tree diffusion graph nodes must be integer model-storage indices."
        ) from exc
    if sorted(storage_nodes) != list(range(K)) or any(
        node != storage for node, storage in zip(nodes, storage_nodes)
    ):
        raise ValueError(
            "Tree diffusion graph nodes must be the model-storage indices 0..C-1."
        )

    idx_of = {node: i for i, node in enumerate(nodes)}
    parent_idx = np.full(K, -1, dtype=np.int32)
    edge_signature = None
    storage_parent = {}
    for gi, graph in enumerate(graphs):
        graph_nodes = list(nx.topological_sort(graph))
        if graph_nodes != nodes:
            raise AssertionError(
                "all tree diffusion graphs must share the same topological order"
            )
        signature = []
        for child in nodes:
            preds = list(graph.predecessors(child))
            if not preds:
                signature.append((child, None))
                continue
            if len(preds) > 1:
                raise ValueError(
                    f"Node {child!r} has {len(preds)} parents; material diffusion requires a rooted tree."
                )
            parent = preds[0]
            signature.append((child, parent))
            if gi == 0:
                parent_idx[idx_of[child]] = idx_of[parent]
                storage_parent[int(child)] = int(parent)

        if edge_signature is None:
            edge_signature = signature
        elif signature != edge_signature:
            raise AssertionError(
                "all tree diffusion graphs must share the same parent/child topology"
            )

    edge_child = sorted(storage_parent)
    edge_parent = [storage_parent[child] for child in edge_child]
    return (
        torch.as_tensor(parent_idx, dtype=torch.int32),
        nodes,
        torch.as_tensor(edge_parent, dtype=torch.long),
        torch.as_tensor(edge_child, dtype=torch.long),
    )


def parent_index_to_tree_topology(parent_index):
    """Build solver and compact-edge maps from storage-order parents.

    ``parent_index[j]`` is the parent of model-storage compartment ``j`` or
    ``-1`` at each root. This registered tensor is the authoritative compiled
    topology for named Tree geometry; mutable interoperability graph views are
    deliberately not consulted.
    """
    storage_parent = torch.as_tensor(
        parent_index, dtype=torch.long, device="cpu"
    ).reshape(-1)
    _, _, storage_depth = _build_tree_morphology(storage_parent.tolist())
    storage_order = torch.argsort(storage_depth, stable=True).to(torch.long)
    inverse_order = torch.argsort(storage_order)
    parent_solver = torch.full_like(storage_parent, -1)
    parent_in_order = storage_parent.index_select(0, storage_order)
    has_parent = parent_in_order >= 0
    parent_solver[has_parent] = inverse_order.index_select(
        0, parent_in_order[has_parent]
    )
    edge_child = torch.nonzero(storage_parent >= 0, as_tuple=False).flatten()
    edge_parent = storage_parent.index_select(0, edge_child)
    return (
        parent_solver.to(torch.int32),
        storage_order,
        edge_parent,
        edge_child,
    )


def graph_to_parent_and_diffusion(graphs, dtype: torch.dtype = torch.float32):
    """Convert tree graph(s) into DHS parent indices and native geometry.

    Returns
    -------
    parent_idx : torch.Tensor
        Length-K parent vector in solver/topological node order.
    diff_geom_um : torch.Tensor
        Shape ``(n_graphs, K)``. Entry ``[..., i]`` is the diffusion geometry of
        the edge connecting node ``i`` to its parent. The root entry is zero.
    node_order : list
        Mapping from solver index to original graph node label.
    """
    if not isinstance(graphs, (list, tuple)):
        graphs = [graphs]
    parent_idx, nodes, _, _ = graph_to_tree_topology(graphs)
    idx_of = {node: i for i, node in enumerate(nodes)}
    K = len(nodes)
    diff_geom = np.zeros((len(graphs), K), dtype=np.float64)
    for gi, graph in enumerate(graphs):
        for child in nodes:
            preds = list(graph.predecessors(child))
            if not preds:
                continue
            parent = preds[0]
            i = idx_of[child]
            diff_geom[gi, i] = _edge_diff_geom_from_graph(graph, parent, child)

    return (
        parent_idx,
        torch.as_tensor(diff_geom, dtype=dtype),
        nodes,
    )


def _select_dhs_solver(solver: str | None, device: torch.device | str, *, threads: int):
    # Resolve tree-specific automatic aliases before the generic tridiagonal
    # aliases: the latter maps ``custom`` to an explicit Thomas request.
    mode = str(solver or "auto").lower()
    aliases = {"dhs": "auto", "hines": "auto", "tree": "auto", "custom": "auto"}
    mode = aliases.get(mode, _normalize_solver_name(mode))
    dev_type = torch.device(device).type

    if mode in {"dense", "debug", "torch", "linalg"}:
        return None, "dense"
    if mode not in {"auto", "thomas"}:
        warnings.warn(
            f"Unknown tree material solver {solver!r}; falling back to DHS auto solver."
        )
        mode = "auto"

    if dev_type == "cuda":
        if dhs_solve_cuda is None:
            if mode == "thomas":
                raise ImportError(
                    "Tree material diffusion on CUDA requires dendra.models.integrators.triton.dhs_solve_cuda."
                )
            return None, "dense"
        return partial(dhs_solve_cuda, threads=threads), "dhs_cuda"

    if dev_type == "cpu":
        if DENDRA_SOLVERS_AVAILABLE:
            return torch.ops.dendra_solvers.dhs_solve, "dhs_cpu"
        if mode == "thomas":
            raise ImportError(
                "Tree material diffusion solver='thomas' on CPU requires the "
                "dendra-solvers package. Install it with `python -m pip install "
                "--upgrade --only-binary=dendra-solvers "
                '"dendra-solvers>=0.5.1"`.'
            )
        warnings.warn(
            "Using tree material diffusion on CPU without dendra-solvers installed; "
            "falling back to dense torch.linalg.solve. Install the native package "
            "with `python -m pip install --upgrade "
            '--only-binary=dendra-solvers "dendra-solvers>=0.5.1"`.'
        )
        return None, "dense"

    if dev_type == "mps":
        if DENDRA_SOLVERS_AVAILABLE and _mps_solver_available(dendra_solvers):
            return torch.ops.dendra_solvers.dhs_solve, "dhs_mps"
        if mode == "thomas":
            raise ImportError(
                "Tree material diffusion on MPS requires an MPS-enabled "
                "dendra-solvers installation and an available Apple GPU."
            )
        return None, "dense"

    return None, "dense"


def _dense_tree_solve(
    d_mem: torch.Tensor,
    a_geom: torch.Tensor,
    rhs: torch.Tensor,
    parent_idx: torch.Tensor,
) -> torch.Tensor:
    """Dense debug fallback for a DHS/Hines tree system.

    ``d_mem`` is the mass/membrane diagonal term and ``a_geom[..., child]`` is
    the parent-child coupling for each child.  The final matrix diagonal is
    ``d_mem + sum(edge couplings incident to node)`` with off-diagonal entries
    ``-g``.
    """
    B, K = rhs.shape
    # Assemble from explicit endpoint maps rather than exporting one parent
    # index per Python iteration. This is also usable under vmap/FX/compile.
    child = torch.arange(K, device=rhs.device)
    parent = parent_idx.clamp_min(0)
    edge = torch.where(parent_idx.unsqueeze(0) >= 0, a_geom, 0.0)
    parent_diagonal = torch.zeros_like(d_mem).scatter_add(
        -1, parent.unsqueeze(0).expand(B, -1), edge
    )
    A = torch.diag_embed(d_mem + edge + parent_diagonal).reshape(B, K * K)
    forward_index = (child * K + parent).unsqueeze(0).expand(B, -1)
    reverse_index = (parent * K + child).unsqueeze(0).expand(B, -1)
    A = A.scatter_add(-1, forward_index, -edge)
    A = A.scatter_add(-1, reverse_index, -edge).reshape(B, K, K)
    return torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)


def _functional_dhs_plan_solve(main, edge, rhs, _parent, _order, _layers, *, plan):
    """Use one independently owned native topology snapshot in functional AD."""
    return dendra_solvers.dhs_solve_plan(main, edge, rhs, plan)


class SpatialOperatorTree(torch.nn.Module):
    """Finite-volume implicit diffusion operator for branched tree morphologies.

    This operator mirrors Dendra's DHS voltage integrator.  The same tree
    topology/elimination layout is used, but the matrix represents material
    conservation rather than membrane voltage:

    ``d_mem = volume / dt`` and ``a_geom = D * diff_geom_um``.

    Branchpoints with zero volume are handled naturally by the implicit solve as
    algebraic flux-balance junctions.  Explicit diffusion rejects zero-volume
    nodes because they would require division by zero.
    """

    _FUNCTIONAL_TOPOLOGY_BUFFERS = frozenset(
        {
            "solver_order",
            "inv_solver_order",
            "parent_idx",
            "order",
            "layer_ptr",
        }
    )

    def __init__(
        self, *, solver: str = "auto", boundary: str = "sealed", threads: int = 16
    ):
        super().__init__()
        threads = int(threads)
        if (
            threads <= 0
            or threads > THREADS_PER_WARP
            or THREADS_PER_WARP % threads != 0
        ):
            raise ValueError(
                "threads must be positive, divide 32, and be <= 32 for DHS-style tree solves"
            )
        self.solver = _normalize_solver_name(solver)
        self.boundary = str(boundary or "sealed").lower()
        if self.boundary not in {"sealed", "no_flux", "noflux"}:
            raise NotImplementedError(
                "SpatialOperatorTree MVP supports only sealed/no-flux boundaries."
            )
        self.threads = threads
        self.configured = False
        self.solver_name = "unconfigured"
        self._solve = None
        self.base_shape: tuple[int, ...] | None = None
        self.B = 0
        self.K = 0

    def _set_buffer(self, name: str, value: torch.Tensor) -> None:
        if name in self._buffers:
            self._buffers[name] = value
        else:
            self.register_buffer(name, value)

    @staticmethod
    def _domain_volume(model, domain: str | None):
        domain = str(domain or "intracellular").lower()
        if hasattr(model, "material_volume"):
            return model.material_volume(domain)
        if domain in {
            "i",
            "inside",
            "intracellular",
            "cytosol",
            "cytoplasm",
        } and hasattr(model, "volume_i"):
            return model.volume_i
        if domain in {"o", "outside", "extracellular"} and hasattr(model, "volume_o"):
            return model.volume_o
        if domain in {"membrane", "surface"} and hasattr(model, "area"):
            return model.area
        if hasattr(model, "volume"):
            return model.volume
        raise AttributeError(
            "Tree material diffusion requires a model volume buffer. Apply the Tree material-geometry patch first."
        )

    def configure_diffusion(
        self,
        c_like: torch.Tensor,
        dt,
        D_um2_per_ms,
        model,
        *,
        domain: str | None = "intracellular",
        node_mask=None,
        solver: str | None = None,
        volume=None,
        edge_factor=None,
        edge_area=None,
        edge_distance=None,
        D_location: str = "node",
        parent_index=None,
        material_edge_index=None,
        native_edge_factor=None,
        _validate_values: bool = True,
        _select_solver: bool = True,
        _topology_plan=None,
    ) -> "SpatialOperatorTree":
        """Precompute DHS topology, mass term, diffusion couplings, and solver.

        ``node_mask`` selects an induced forest of the full morphology.  Parent-
        child coupling is retained only when both endpoints are selected.  The
        original Hines topology can then solve all selected connected components
        as independent diagonal blocks without rebuilding or flattening the
        morphology graph.

        Passing ``volume`` plus either ``edge_factor`` or
        ``edge_area``/``edge_distance`` replaces the morphology's native
        chemical geometry while retaining its topology. ``volume`` is
        node-aligned. Edge quantities and edge-located diffusivity use the
        compact material-edge order returned by ``Tree.material_edge_index``.
        """
        if c_like.ndim < 1:
            raise ValueError("diffusion fields must have a final compartment dimension")
        K = int(c_like.shape[-1])
        if K <= 0:
            raise ValueError("diffusion fields must have at least one compartment")

        has_factor = edge_factor is not None
        has_area_distance = edge_area is not None or edge_distance is not None
        explicit_geometry = volume is not None or has_factor or has_area_distance
        if explicit_geometry and volume is None:
            raise ValueError("Explicit Tree finite-volume geometry requires volume.")
        if has_factor and has_area_distance:
            raise ValueError(
                "Specify Tree edge_factor or edge_area/edge_distance, not both."
            )
        if (
            explicit_geometry
            and not has_factor
            and not (edge_area is not None and edge_distance is not None)
        ):
            raise ValueError(
                "Explicit Tree finite-volume geometry requires edge_factor or "
                "both edge_area and edge_distance."
            )
        D_location = str(D_location or "node").lower()
        if D_location not in {"node", "edge"}:
            raise ValueError("D_location must be 'node' or 'edge'.")
        if D_location == "edge" and not explicit_geometry:
            raise ValueError(
                "Edge-located Tree diffusivity requires explicit geometry."
            )

        if _topology_plan is not None:
            (
                parents,
                nodes,
                edge_parents,
                edge_children,
                elimination_order,
                elimination_layers,
            ) = _topology_plan
            parent_idx_t = torch.tensor(parents, dtype=torch.int32)
            node_order = nodes
            edge_parent_t = torch.tensor(edge_parents, dtype=torch.long)
            edge_child_t = torch.tensor(edge_children, dtype=torch.long)
            diff_geom_t = None
        elif parent_index is not None:
            parent_idx_t, solver_order_t, edge_parent_t, edge_child_t = (
                parent_index_to_tree_topology(parent_index)
            )
            node_order = solver_order_t.tolist()
            if material_edge_index is not None:
                supplied_edges = torch.as_tensor(
                    material_edge_index, dtype=torch.long, device="cpu"
                )
                expected_edges = torch.stack((edge_parent_t, edge_child_t), dim=0)
                if supplied_edges.shape != expected_edges.shape or not torch.equal(
                    supplied_edges, expected_edges
                ):
                    raise ValueError(
                        "Tree material_edge_index is inconsistent with the "
                        "registered parent_index topology."
                    )
            diff_geom_t = None
        else:
            graph = getattr(model, "graph", None)
            if graph is None:
                try:
                    graph = model.assemble_graphs()
                except Exception as err:  # pragma: no cover - model dependent
                    raise ValueError(
                        "Tree material diffusion requires registered parent_index "
                        "topology, `model.graph`, or `model.assemble_graphs()`."
                    ) from err
            graph_arg = graph if isinstance(graph, (list, tuple)) else [graph]
        if _topology_plan is None and explicit_geometry and parent_index is None:
            parent_idx_t, node_order, edge_parent_t, edge_child_t = (
                graph_to_tree_topology(graph_arg)
            )
            diff_geom_t = None
        elif _topology_plan is None and not explicit_geometry and parent_index is None:
            parent_idx_t, diff_geom_t, node_order = graph_to_parent_and_diffusion(
                graph_arg, dtype=c_like.dtype
            )
            _, _, edge_parent_t, edge_child_t = graph_to_tree_topology(graph_arg)
        if len(node_order) != K:
            raise ValueError(
                f"tree graph has {len(node_order)} compartments, but field has final dimension {K}."
            )

        if _topology_plan is None:
            parent_idx, _, depth = _build_tree_morphology(parent_idx_t.tolist())
            order, layer_ptr = _build_dhs_layers(depth, self.threads)
            self._topology_plan = (
                tuple(parent_idx.tolist()),
                tuple(node_order),
                tuple(edge_parent_t.tolist()),
                tuple(edge_child_t.tolist()),
                tuple(order.tolist()),
                tuple(layer_ptr.tolist()),
            )
        else:
            parent_idx = parent_idx_t
            order = torch.tensor(elimination_order, dtype=torch.int32)
            layer_ptr = torch.tensor(elimination_layers, dtype=torch.int32)
            self._topology_plan = _topology_plan

        device = c_like.device
        dtype = c_like.dtype
        solver_order = torch.as_tensor(node_order, dtype=torch.long, device=device)
        inv_solver_order = torch.argsort(solver_order, dim=0)

        if volume is None:
            volume = self._domain_volume(model, domain)
        volume = _as_tensor_like(volume, c_like)
        try:
            volume = _broadcast_to(volume, c_like)
        except RuntimeError as exc:
            raise ValueError(
                f"Tree node volume with shape {tuple(volume.shape)} cannot "
                f"broadcast to material-field shape {tuple(c_like.shape)}."
            ) from exc
        volume2, base_shape = _flatten_last(volume)
        B = int(volume2.shape[0])
        self.B = B
        self.K = K
        self.base_shape = base_shape

        if node_mask is None:
            active = torch.ones_like(c_like, dtype=torch.bool)
        else:
            active = _broadcast_mask_to(node_mask, c_like)
        active2 = active.reshape(B, K)

        # Align volume and field-space quantities to solver/topological order.
        volume_s = volume2.index_select(-1, solver_order).contiguous()
        active_s = active2.index_select(-1, solver_order).contiguous()

        parent_idx_dev = parent_idx.to(dtype=torch.long, device=device)
        has_parent = parent_idx_dev >= 0
        parent_safe = parent_idx_dev.clamp_min(0)
        parent_active_s = active_s.index_select(-1, parent_safe)
        edge_active_s = (
            active_s & parent_active_s & has_parent.unsqueeze(0)
        ).contiguous()

        if _validate_values:
            active_volume = volume_s[active_s]
            if bool(torch.any(~torch.isfinite(active_volume)).item()) or bool(
                torch.any(active_volume < 0).item()
            ):
                raise ValueError(
                    "Active Tree finite-volume nodes require finite non-negative volume."
                )

        if D_location == "edge":
            E = int(edge_child_t.numel())
            D_compact = _broadcast_tree_edge_to(D_um2_per_ms, c_like, E).reshape(B, E)
            edge_child = edge_child_t.to(device=device)
            child_solver = inv_solver_order.index_select(0, edge_child)
            scatter_index = child_solver.unsqueeze(0).expand(B, -1)
            D_edge_s = D_compact.new_zeros((B, K)).scatter(1, scatter_index, D_compact)
        else:
            D_value = _as_tensor_like(D_um2_per_ms, c_like)
            try:
                D_full = _broadcast_to(D_value, c_like).reshape(B, K)
            except RuntimeError as exc:
                raise ValueError(
                    f"Tree node diffusivity with shape {tuple(D_value.shape)} "
                    "cannot broadcast to material-field shape "
                    f"{tuple(c_like.shape)}."
                ) from exc
            # Values outside the selected induced forest are immaterial and may
            # deliberately contain sentinels. Clear them before endpoint
            # averaging so crossing edges cannot contaminate active transport.
            D_full = torch.where(active2, D_full, torch.zeros_like(D_full))
            D_s = D_full.index_select(-1, solver_order)
            # Mark both endpoints of every conductive edge. This is a static,
            # configure-time topology walk; keeping it boolean avoids making
            # validation depend on scatter-reduction dtype/backend support.
            if _validate_values:
                participates_s = edge_active_s.clone()
                for child in range(K):
                    parent = int(parent_idx_dev[child].item())
                    if parent >= 0:
                        participates_s[:, parent] |= edge_active_s[:, child]
                active_node_D = D_s[participates_s]
                if bool(torch.any(~torch.isfinite(active_node_D)).item()) or bool(
                    torch.any(active_node_D < 0).item()
                ):
                    raise ValueError(
                        "Nodes incident to active Tree finite-volume interfaces "
                        "require finite non-negative diffusivity."
                    )
            D_parent_s = D_s.index_select(-1, parent_safe)
            D_edge_s = torch.where(
                has_parent.unsqueeze(0),
                0.5 * (D_s + D_parent_s),
                torch.zeros_like(D_s),
            )

        if explicit_geometry:
            E = int(edge_child_t.numel())
            if edge_factor is not None:
                factor_compact = _broadcast_tree_edge_to(
                    edge_factor, c_like, E
                ).reshape(B, E)
            else:
                area_compact = _broadcast_tree_edge_to(edge_area, c_like, E).reshape(
                    B, E
                )
                distance_compact = _broadcast_tree_edge_to(
                    edge_distance, c_like, E
                ).reshape(B, E)
                edge_parent = edge_parent_t.to(device=device)
                edge_child = edge_child_t.to(device=device)
                compact_active = active2.index_select(
                    -1, edge_parent
                ) & active2.index_select(-1, edge_child)
                if _validate_values:
                    selected_area = area_compact[compact_active]
                    selected_distance = distance_compact[compact_active]
                    if bool(torch.any(~torch.isfinite(selected_area)).item()) or bool(
                        torch.any(selected_area < 0).item()
                    ):
                        raise ValueError(
                            "Active Tree finite-volume interfaces require finite "
                            "non-negative edge_area."
                        )
                    if bool(
                        torch.any(~torch.isfinite(selected_distance)).item()
                    ) or bool(torch.any(selected_distance <= 0).item()):
                        raise ValueError(
                            "Active Tree finite-volume interfaces require finite "
                            "positive edge_distance."
                        )
                safe_area = torch.where(
                    compact_active, area_compact, torch.zeros_like(area_compact)
                )
                safe_distance = torch.where(
                    compact_active,
                    distance_compact,
                    torch.ones_like(distance_compact),
                )
                factor_compact = safe_area / safe_distance
            edge_child = edge_child_t.to(device=device)
            child_solver = inv_solver_order.index_select(0, edge_child)
            scatter_index = child_solver.unsqueeze(0).expand(B, -1)
            diff_geom_s = factor_compact.new_zeros((B, K)).scatter(
                1, scatter_index, factor_compact
            )
        else:
            if native_edge_factor is not None:
                native_factor = _as_tensor_like(native_edge_factor, c_like)
                try:
                    native_factor = _broadcast_to(native_factor, c_like).reshape(B, K)
                except RuntimeError as exc:
                    raise ValueError(
                        "Native Tree edge factor with shape "
                        f"{tuple(native_factor.shape)} cannot broadcast to "
                        f"material-field shape {tuple(c_like.shape)}."
                    ) from exc
                diff_geom_s = native_factor.index_select(-1, solver_order)
            else:
                diff_geom_s = diff_geom_t.to(device=device, dtype=dtype)
                # graph_to_parent_and_diffusion returns one row per graph. Most
                # Tree models have one graph and B population rows. If multiple
                # graph rows are supplied and they match B, retain them.
                if diff_geom_s.shape[0] == 1:
                    diff_geom_s = diff_geom_s.expand(B, -1)
                elif diff_geom_s.shape[0] != B:
                    raise ValueError(
                        f"diffusion graph batch has {diff_geom_s.shape[0]} rows, "
                        f"but material field has {B} solve rows."
                    )

        if _validate_values:
            active_D = D_edge_s[edge_active_s]
            if bool(torch.any(~torch.isfinite(active_D)).item()) or bool(
                torch.any(active_D < 0).item()
            ):
                raise ValueError(
                    "Active Tree finite-volume interfaces require finite "
                    "non-negative diffusivity."
                )
            active_factor = diff_geom_s[edge_active_s]
            if bool(torch.any(~torch.isfinite(active_factor)).item()) or bool(
                torch.any(active_factor < 0).item()
            ):
                raise ValueError(
                    "Active Tree finite-volume interfaces require finite "
                    "non-negative edge_factor."
                )
        safe_D = torch.where(edge_active_s, D_edge_s, torch.zeros_like(D_edge_s))
        safe_factor = torch.where(
            edge_active_s, diff_geom_s, torch.zeros_like(diff_geom_s)
        )
        g_diff_s = (safe_D * safe_factor).contiguous()

        # A zero-volume junction is a valid algebraic node when it is coupled
        # to at least one positive-volume compartment.  An induced component
        # made entirely of zero-volume nodes, however, has neither storage nor
        # an anchoring mass term and leaves a singular pure-Laplacian block.
        # Validate that every active component reaches positive material mass.
        if _validate_values:
            reachable_mass = active_s & (volume_s > 0)
            conductive_edge_s = edge_active_s & (g_diff_s != 0)
            # Topological order permits two linear sweeps. This physical-domain
            # preflight runs before functional lowering, never in a scan body.
            for child in range(K - 1, -1, -1):
                p = int(parent_idx[child].item())
                if p >= 0:
                    reachable_mass[:, p] |= (
                        conductive_edge_s[:, child] & reachable_mass[:, child]
                    )
            for child in range(K):
                p = int(parent_idx[child].item())
                if p >= 0:
                    reachable_mass[:, child] |= (
                        conductive_edge_s[:, child] & reachable_mass[:, p]
                    )
            unanchored = active_s & ~reachable_mass
            if bool(torch.any(unanchored).item()):
                first_row, first_slot = torch.nonzero(unanchored, as_tuple=False)[
                    0
                ].tolist()
                raise ValueError(
                    "Each active Tree diffusion component must contain positive "
                    "material volume; found a zero-volume component at solve row "
                    f"{first_row}, compartment slot {first_slot}."
                )

        dt_t = _as_tensor_like(dt, c_like)
        if dt_t.ndim != 0:
            dt_t = (
                _broadcast_to(dt_t, c_like).reshape(B, K).index_select(-1, solver_order)
            )
        self._set_buffer("dt", dt_t.to(device=device, dtype=dtype))

        tiny = torch.finfo(dtype).tiny
        safe_volume_s = torch.where(active_s, volume_s, torch.ones_like(volume_s))
        if dt_t.ndim == 0:
            dmem = safe_volume_s / dt_t.clamp_min(tiny).reshape(1, 1)
        else:
            safe_dt_t = torch.where(active_s, dt_t, torch.ones_like(dt_t))
            dmem = safe_volume_s / safe_dt_t.clamp_min(tiny)
        # An excluded zero-volume branchpoint would otherwise leave a singular
        # all-zero row after its incident edges are masked.  A unit identity row
        # is numerically benign and, together with the final where, makes every
        # excluded compartment an exact no-op.
        dmem = torch.where(active_s, dmem, torch.ones_like(dmem))

        self._set_buffer("volume", volume2.contiguous())
        self._set_buffer("volume_solver", volume_s.contiguous())
        positive_volume = active2 & (volume2 > 0)
        safe_volume2 = torch.where(positive_volume, volume2, torch.ones_like(volume2))
        self._set_buffer(
            "inv_volume",
            torch.where(
                positive_volume,
                1.0 / safe_volume2,
                torch.zeros_like(volume2),
            ).contiguous(),
        )
        self._set_buffer("node_mask", active2.contiguous())
        self._set_buffer("node_mask_solver", active_s.contiguous())
        self._set_buffer("storage", dmem.contiguous())
        self._set_buffer("dmem", dmem.contiguous())
        self._set_buffer("relax_rhs", dmem.new_zeros((B, K)))
        self._has_relaxation = False
        self._set_buffer("a_geom", g_diff_s.contiguous())
        if not explicit_geometry:
            self._set_buffer(
                "native_edge_factor",
                diff_geom_s.index_select(-1, inv_solver_order).contiguous(),
            )
        self._set_buffer("solver_order", solver_order)
        self._set_buffer("inv_solver_order", inv_solver_order)
        self._set_buffer("parent_idx", parent_idx_dev)
        self._set_buffer("order", order.to(dtype=torch.long, device=device))
        self._set_buffer("layer_ptr", layer_ptr.to(dtype=torch.long, device=device))

        if _select_solver:
            solve_fn, solver_name = _select_dhs_solver(
                solver or self.solver, device, threads=self.threads
            )
            self._solve = solve_fn
            self.solver_name = solver_name
        self.configured = True
        return self

    def configure_relaxation(self, rate, target=0.0, *, where=None):
        """Add a linear reservoir term to the configured tree system."""
        self._require_configured()
        updates = self._prepare_relaxation(
            self._buffers, rate, target, where=where, validate_values=True
        )
        self._set_buffer("dmem", updates["dmem"])
        self._set_buffer("relax_rhs", updates["relax_rhs"])
        self._has_relaxation = True
        return self

    def _prepare_relaxation(self, workspace, rate, target, *, where, validate_values):
        like = workspace["volume"].reshape(self.base_shape)
        active = workspace["node_mask"].reshape(self.base_shape)
        if where is not None:
            active = active & _broadcast_mask_to(where, like)

        rate_full = _broadcast_to(rate, like)
        if validate_values:
            selected_rate = rate_full[active]
            if bool(torch.any(~torch.isfinite(selected_rate)).item()) or bool(
                torch.any(selected_rate < 0).item()
            ):
                raise ValueError("Relaxation rates must be finite and non-negative.")
        rate_full = torch.where(active, rate_full, torch.zeros_like(rate_full))

        target_full = _broadcast_to(target, like)
        coupled = active & (rate_full != 0)
        if validate_values and bool(
            torch.any(~torch.isfinite(target_full[coupled])).item()
        ):
            raise ValueError("Relaxation targets must be finite where rate is nonzero.")
        target_full = torch.where(coupled, target_full, torch.zeros_like(target_full))

        order = self._buffers["solver_order"]
        rate_s = rate_full.reshape(self.B, self.K).index_select(-1, order)
        target_s = target_full.reshape(self.B, self.K).index_select(-1, order)
        active_s = active.reshape(self.B, self.K).index_select(-1, order)
        safe_volume_s = torch.where(
            active_s,
            workspace["volume_solver"],
            torch.zeros_like(workspace["volume_solver"]),
        )
        mass_rate = safe_volume_s * rate_s
        return dict(
            workspace,
            dmem=workspace["dmem"] + mass_rate,
            relax_rhs=workspace["relax_rhs"] + mass_rate * target_s,
        )

    def _prepared_buffer_names(self) -> tuple[str, ...]:
        """Inventory numerical coefficients independently of fixed topology."""
        self._require_configured()
        return tuple(
            name
            for name in self._buffers
            if name not in self._FUNCTIONAL_TOPOLOGY_BUFFERS
            and name != "native_edge_factor"
        )

    def prepared_buffers(self) -> dict[str, torch.Tensor]:
        """Export numerical coefficients, excluding fixed topology metadata."""
        return {
            name: self._buffers[name].clone() for name in self._prepared_buffer_names()
        }

    def prepare_diffusion(self, c_like, dt, D_um2_per_ms, model, **kwargs):
        """Build coefficients on this operator's immutable, validated topology.

        This shares the imperative coefficient builder, excluding its eager
        value preflights and topology export. The caller must retain the
        configured physical-domain contract while changing numerical inputs.
        No caller, bound buffer, or solver plan is mutated.
        """
        self._require_configured()
        if (
            kwargs.get("parent_index") is not None
            or kwargs.get("material_edge_index") is not None
        ):
            raise ValueError(
                "Prepared Tree diffusion reuses its configured topology; "
                "do not supply parent_index or material_edge_index."
            )
        if (
            kwargs.get("solver") is not None
            and _normalize_solver_name(kwargs["solver"]) != self.solver
        ):
            raise ValueError("Prepared diffusion cannot change its configured solver.")
        staged = type(self)(
            solver=self.solver, boundary=self.boundary, threads=self.threads
        )
        if (
            not any(
                kwargs.get(name) is not None
                for name in ("volume", "edge_factor", "edge_area", "edge_distance")
            )
            and kwargs.get("native_edge_factor") is None
        ):
            kwargs["native_edge_factor"] = self._buffers["native_edge_factor"]
        staged.configure_diffusion(
            c_like,
            dt,
            D_um2_per_ms,
            model,
            **kwargs,
            _validate_values=False,
            _select_solver=False,
            _topology_plan=self._topology_plan,
        )
        if staged.base_shape != self.base_shape:
            raise ValueError(
                "Prepared diffusion must retain its configured field shape."
            )
        return staged.prepared_buffers()

    def prepare_relaxation(self, workspace, rate, target=0.0, *, where=None):
        """Return updated coefficients for a validated implicit reservoir."""
        self._require_configured()
        return self._prepare_relaxation(
            workspace, rate, target, where=where, validate_values=False
        )

    def _functional_solver(self):
        """Bind transform-compatible execution without changing the algorithm."""
        self._require_configured()
        solver = self._solve
        if solver is None:
            return None
        if (
            isinstance(solver, partial)
            and solver.func is dhs_solve_cuda
            and not solver.args
            and solver.keywords == {"threads": self.threads}
        ):
            from dendra.models.integrators._cuda_functional import dhs_solve

            return partial(dhs_solve, threads=self.threads)
        if DENDRA_SOLVERS_AVAILABLE and solver is torch.ops.dendra_solvers.dhs_solve:
            if callable(getattr(dendra_solvers, "DHSPlan", None)):
                plan = dendra_solvers.DHSPlan(
                    *_cpu_topology(
                        tuple(
                            self._buffers[name]
                            for name in ("parent_idx", "order", "layer_ptr")
                        )
                    )
                )
                return partial(_functional_dhs_plan_solve, plan=plan)
            facade = getattr(dendra_solvers, "dhs_solve", None)
            if callable(facade):
                return facade
        raise RuntimeError(
            f"Tree material diffusion solver {self.solver_name!r} is not yet "
            "transform-compatible; use the native DHS or dense solver."
        )

    def validate_explicit(self) -> None:
        """Reject unsupported zero-volume explicit dynamics before lowering."""
        self._require_configured()
        if self.K > 1 and bool(
            torch.any(
                (self._buffers["volume_solver"] <= 0)
                & self._buffers["node_mask_solver"]
            ).item()
        ):
            raise RuntimeError(
                "Explicit tree diffusion is undefined for zero-volume nodes/branchpoints; use implicit diffusion."
            )

    def _require_configured(self) -> None:
        if not self.configured:
            raise RuntimeError(
                "SpatialOperatorTree has not been configured. Call configure_diffusion(...) "
                "from the owning MaterialProcess.set_dt(...) before advancing."
            )

    def diffuse_implicit_configured(self, c: torch.Tensor) -> torch.Tensor:
        """Backward-Euler sealed-boundary diffusion update on a tree."""
        self._require_configured()
        return self.diffuse_implicit_prepared(c, self._buffers, solve=self._solve)

    def diffuse_implicit_prepared(
        self, c: torch.Tensor, workspace: Mapping[str, torch.Tensor], *, solve
    ) -> torch.Tensor:
        """Advance with explicit numerical coefficients and a static solve plan."""
        # Independent workspaces can contain a reservoir absent from the
        # configured template, including when this forest has just one node.
        if self.K <= 1 and workspace is self._buffers and not self._has_relaxation:
            return c
        c2 = c.reshape(self.B, self.K).contiguous()
        c_s = c2.index_select(-1, self._buffers["solver_order"])
        rhs_s = workspace["storage"] * c_s + workspace["relax_rhs"]

        if self.K <= 1:
            out_s = rhs_s / workspace["dmem"]
        elif solve is None:
            out_s = _dense_tree_solve(
                workspace["dmem"],
                workspace["a_geom"],
                rhs_s,
                self._buffers["parent_idx"],
            )
        else:
            out_s = solve(
                workspace["dmem"],
                workspace["a_geom"],
                rhs_s,
                self._buffers["parent_idx"],
                self._buffers["order"],
                self._buffers["layer_ptr"],
            )

        out = out_s.index_select(-1, self._buffers["inv_solver_order"])
        out = out.reshape(self.base_shape)
        active = workspace["node_mask"].reshape(self.base_shape)
        return torch.where(active, out, c)

    def diffuse_explicit_configured(self, c: torch.Tensor) -> torch.Tensor:
        """Explicit tree diffusion update for debug use.

        This rejects zero-volume nodes, so it is not appropriate for Tree graphs
        containing branchpoints.  Use implicit diffusion for imported branched
        morphologies.
        """
        self._require_configured()
        self.validate_explicit()
        return self.diffuse_explicit_prepared(c, self._buffers)

    def diffuse_explicit_prepared(
        self, c: torch.Tensor, workspace: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Tensor-only explicit flux update on a prevalidated tree plan."""
        if self.K <= 1:
            return c
        c2 = c.reshape(self.B, self.K).contiguous()
        c_s = c2.index_select(-1, self._buffers["solver_order"])
        volume_s = workspace["volume_solver"]
        active_s = workspace["node_mask_solver"]
        parent_idx = self._buffers["parent_idx"]
        parent_safe = parent_idx.clamp_min(0)
        flux = workspace["a_geom"] * (c_s.index_select(-1, parent_safe) - c_s)
        flux = torch.where(parent_idx.unsqueeze(0) >= 0, flux, 0.0)
        net = flux - torch.zeros_like(c_s).scatter_add(
            -1, parent_safe.unsqueeze(0).expand(self.B, -1), flux
        )

        dt_t = workspace["dt"]
        dt_math = dt_t.reshape(1, 1) if dt_t.ndim == 0 else dt_t
        safe_volume_s = torch.where(active_s, volume_s, torch.ones_like(volume_s))
        c_next_s = c_s + dt_math * net / safe_volume_s
        c_next = c_next_s.index_select(-1, self._buffers["inv_solver_order"])
        c_next = c_next.reshape(self.base_shape)
        active = workspace["node_mask"].reshape(self.base_shape)
        return torch.where(active, c_next, c)

    # Backward-compatible convenience wrappers.
    def diffuse_implicit(
        self,
        c: torch.Tensor,
        dt,
        D_um2_per_ms,
        model,
        *,
        domain: str | None = "intracellular",
        node_mask=None,
        solver: str | None = None,
    ) -> torch.Tensor:
        self.configure_diffusion(
            c,
            dt,
            D_um2_per_ms,
            model,
            domain=domain,
            node_mask=node_mask,
            solver=solver,
        )
        return self.diffuse_implicit_configured(c)

    def diffuse_explicit(
        self,
        c: torch.Tensor,
        dt,
        D_um2_per_ms,
        model,
        *,
        domain: str | None = "intracellular",
        node_mask=None,
    ) -> torch.Tensor:
        self.configure_diffusion(
            c,
            dt,
            D_um2_per_ms,
            model,
            domain=domain,
            node_mask=node_mask,
            solver=self.solver,
        )
        return self.diffuse_explicit_configured(c)
