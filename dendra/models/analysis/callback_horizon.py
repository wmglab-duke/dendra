"""Check whether an activation callback's selected voltage peak is observed.

An ``Active`` decision can be correct for its finite check window even when
the voltage is still rising as that window ends. This diagnostic distinguishes
that valid time-limited decision from a trace-gradient probe whose event peak
has not been observed. It makes no claim about the post-spike recovery tail.
The caller supplies the actual solver step and any stronger protocol-specific
post-peak clearance requirements; no model or stimulus horizon is assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from ..callbacks import ActiveAL, ActiveALCount, Recorder, _Active
from .callback_margin import (
    CallbackMargin,
    _checked_nodes,
    _recorder_trace,
    callback_signed_margin,
)


@dataclass(frozen=True)
class CallbackPeakHorizonDiagnostic:
    """Observation status for the callback margin's selected site and step.

    Tensor fields have the same batch/population/partition shape as ``margin``.
    ``peak_observed`` requires at least one later *checked* voltage sample and
    a drop from the selected peak by more than ``required_peak_drop_mv``.
    ``required_post_peak_ms`` can demand additional observed time. A true
    value establishes only this stated trace condition, not a complete spike
    waveform or a correct hard-threshold parameter gradient.
    """

    margin: CallbackMargin
    last_checked_step_index: int
    last_checked_frame_is_final_trace_frame: bool
    post_peak_steps: torch.Tensor
    post_peak_ms: torch.Tensor
    terminal_drop_from_peak_mv: torch.Tensor
    peak_observed: torch.Tensor

    @property
    def horizon_censored(self) -> torch.Tensor:
        return ~self.peak_observed


def callback_peak_horizon_diagnostic(
    callback: ActiveAL | ActiveALCount | _Active,
    recorded_voltage: torch.Tensor | Recorder,
    *,
    dt_ms: float,
    required_post_peak_ms: float = 0.0,
    required_peak_drop_mv: float = 0.0,
    partition: Sequence[int] | None = None,
    selected_nodes: bool = False,
    n_compartments: int | None = None,
) -> CallbackPeakHorizonDiagnostic:
    """Assess the selected activation event inside the *hard callback window*.

    The trace includes the initial frame and every post-step frame, as for
    :func:`callback_signed_margin`. The callback's selected event is the
    maximum checked voltage for ``Active``/``ActiveAL(at_least=1)`` and the
    kth-largest checked site peak for ``ActiveAL(at_least=k)``. The diagnostic
    uses the last frame that the callback actually checks, which may precede
    the simulation's final frame. ``dt_ms`` must match the callback's step so
    its check-window indices have the same physical meaning as the replay.

    A final-frame maximum, including a flat maximum that reaches the last
    checked frame, is always marked censored. An earlier peak with a later
    decline passes the default minimal observation check. A protocol that
    requires more of the falling phase should specify positive clearance
    requirements. This does not alter the finite-horizon hard decision.
    """
    for name, value, positive in (
        ("dt_ms", dt_ms, True),
        ("required_post_peak_ms", required_post_peak_ms, False),
        ("required_peak_drop_mv", required_peak_drop_mv, False),
    ):
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            bound = "positive" if positive else "nonnegative"
            raise ValueError(f"{name} must be finite and {bound}.")
    if not math.isclose(float(callback.dt), dt_ms, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("callback.dt must match the replay's dt_ms.")

    margin = callback_signed_margin(
        callback,
        recorded_voltage,
        partition=partition,
        selected_nodes=selected_nodes,
        n_compartments=n_compartments,
    )
    if isinstance(recorded_voltage, Recorder):
        trace, selected_nodes = _recorder_trace(
            callback, recorded_voltage, n_compartments=n_compartments
        )
    else:
        trace = recorded_voltage
    n_steps = int(trace.shape[0]) - 1
    stop = min(n_steps, callback.ind_end)
    last_checked_step = stop - 1
    terminal = trace[stop]
    if not selected_nodes:
        nodes = _checked_nodes(callback, int(trace.shape[-1])).to(trace.device)
        terminal = terminal.index_select(-1, nodes)
    if partition is None:
        terminal_selected = terminal.gather(
            -1, margin.checked_site_index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        terminal_selected = terminal.gather(-1, margin.checked_site_index)

    post_peak_steps = last_checked_step - margin.step_index
    post_peak_ms = post_peak_steps.to(dtype=torch.float64) * dt_ms
    terminal_drop = margin.margin + callback.threshold - terminal_selected
    peak_observed = (
        (post_peak_steps >= 1)
        & (post_peak_ms >= required_post_peak_ms)
        & (terminal_drop > required_peak_drop_mv)
    )
    return CallbackPeakHorizonDiagnostic(
        margin=margin,
        last_checked_step_index=last_checked_step,
        last_checked_frame_is_final_trace_frame=(stop == n_steps),
        post_peak_steps=post_peak_steps,
        post_peak_ms=post_peak_ms,
        terminal_drop_from_peak_mv=terminal_drop,
        peak_observed=peak_observed,
    )
