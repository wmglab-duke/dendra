"""Signed margins for supported threshold callbacks on complete voltage traces.

``Active`` and ``ActiveAL`` initialize every checked site's crossing cache to
``True``. Consequently a site has a nonzero callback count if and only if one
of its *checked post-step* voltages reaches ``callback.threshold``. For
``ActiveAL(at_least=k)``, the kth largest site peak minus the threshold is
therefore an exact signed margin: ``margin >= 0`` is the hard decision.

The caller supplies a time-first trace containing the initial frame followed
by every post-step frame. A :class:`~dendra.models.callbacks.Recorder` can be
passed directly; its configuration is checked to prevent accidental temporal
downsampling or voltage reduction. The adapter is independent of the model,
stimulus, amplitude units, and number of batch/population axes.

The callback must start from reset state for this trace. A prior run's
latched activity cannot be reconstructed from a new trace alone.

The same margin applies to legacy ``_Active`` and
``ActiveALCount(at_least=1)``. Higher total-count targets, ``APCount``,
``Raster``, and inverted activity are not supported. In particular, the number
of rising edges can change when a trace touches the threshold, and negating a
margin changes the callback's inclusive ``>=`` boundary to ``<=``. Such cases
need a separately validated margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import torch

from ..callbacks import (
    Active,
    ActiveAL,
    ActiveALCount,
    Recorder,
    ThresholdCallback,
    _Active,
)


@dataclass(frozen=True)
class CallbackMargin:
    """Margin and its selected event, with the callback's output shape.

    ``step_index`` uses the callback's zero-based post-step counter ``i``;
    ``checked_site_index`` indexes ``callback.node_check`` in its given order.
    These indices define a local branch and are recomputed on every call.
    """

    margin: torch.Tensor
    step_index: torch.Tensor
    checked_site_index: torch.Tensor

    @property
    def decision(self) -> torch.Tensor:
        """The exact supported hard decision, including equality at threshold."""
        return self.margin >= 0

    @property
    def branch_id(self) -> tuple[int, int]:
        """Hashable event identity for a single output lane."""
        if self.margin.numel() != 1:
            raise ValueError("Select one output lane before requesting a branch ID.")
        return (int(self.step_index.item()), int(self.checked_site_index.item()))

    @property
    def branch_ids(self) -> tuple[tuple[int, int], ...]:
        """Event identities flattened in row-major output order."""
        steps = self.step_index.detach().reshape(-1).tolist()
        sites = self.checked_site_index.detach().reshape(-1).tolist()
        return tuple((int(step), int(site)) for step, site in zip(steps, sites))


def _checked_nodes(callback: ThresholdCallback, compartment_count: int) -> torch.Tensor:
    nodes = torch.as_tensor(callback.node_check, dtype=torch.long).reshape(-1)
    if nodes.numel() == 0:
        raise ValueError("The activity callback checks no compartments.")
    if bool(((nodes < -compartment_count) | (nodes >= compartment_count)).any()):
        raise IndexError("callback.node_check is outside the voltage trace.")
    return torch.where(nodes < 0, nodes + compartment_count, nodes)


def _partition_lengths(
    partition: Sequence[int] | None, n_checked: int, at_least: int
) -> tuple[int, ...]:
    if partition is None:
        if n_checked < at_least:
            raise ValueError("at_least exceeds the number of checked sites.")
        return (n_checked,)
    lengths = torch.as_tensor(partition, dtype=torch.long, device="cpu")
    if lengths.ndim != 1:
        raise ValueError("partition must be a 1D sequence of integers.")
    if lengths.numel() == 0:
        raise ValueError("partition must be non-empty.")
    values = tuple(int(value) for value in lengths)
    if sum(values) != n_checked:
        raise ValueError("sum(partition) must equal the number of checked sites.")
    if min(values) < at_least:
        raise ValueError("all partition values must be at least at_least.")
    return values


def _recorder_trace(
    callback: ThresholdCallback,
    recorder: Recorder,
    *,
    n_compartments: int | None,
) -> tuple[torch.Tensor, bool]:
    if "v" not in recorder.states:
        raise ValueError("Recorder must include the full membrane-voltage state 'v'.")
    if recorder.max_only or recorder.sliding_window is not None:
        raise ValueError("Recorder voltage must not be reduced or smoothed.")
    if recorder.save_every not in (None, 1):
        raise ValueError("Recorder must save every solver step.")
    if recorder.cache_with_hdf5:
        raise ValueError("An HDF5-cached Recorder does not retain a complete trace.")
    if not recorder.indexed:
        return recorder.stack("v"), False

    recorded = torch.as_tensor(recorder.node_indices, dtype=torch.long).reshape(-1)
    if bool((recorded < 0).any()):
        raise ValueError("Indexed Recorder requires nonnegative node_indices.")
    checked = torch.as_tensor(callback.node_check, dtype=torch.long).reshape(-1)
    if n_compartments is None:
        # Equal raw encodings are sufficient. If the callback still contains
        # negative indices, the caller must supply the full model width.
        same = torch.equal(recorded.cpu(), checked.cpu())
    else:
        if (
            isinstance(n_compartments, bool)
            or not isinstance(n_compartments, Integral)
            or n_compartments < 1
        ):
            raise ValueError("n_compartments must be a positive integer.")
        checked = _checked_nodes(callback, n_compartments)
        if bool(((recorded < -n_compartments) | (recorded >= n_compartments)).any()):
            raise IndexError("Recorder.node_indices is outside the model.")
        recorded = torch.where(recorded < 0, recorded + n_compartments, recorded)
        same = torch.equal(recorded.cpu(), checked.cpu())
    if not same:
        raise ValueError(
            "Indexed Recorder nodes must equal callback.node_check in order; "
            "supply n_compartments when comparing normalized and negative indices."
        )
    return recorder.stack("v"), True


def callback_signed_margin(
    callback: ActiveAL | ActiveALCount | _Active,
    recorded_voltage: torch.Tensor | Recorder,
    *,
    partition: Sequence[int] | None = None,
    selected_nodes: bool = False,
    n_compartments: int | None = None,
) -> CallbackMargin:
    """Return a margin matching the supported callback's hard decision exactly.

    ``recorded_voltage`` has shape ``(time, *batch, population, compartments)``
    and includes the initial frame at index 0. The callback checks post-step
    frame ``i+1`` iff ``ind_start <= i < ind_end``; the initial frame is never
    checked. ``selected_nodes=True`` means its final axis already contains
    precisely ``callback.node_check`` in the same order. For an indexed
    ``Recorder`` that fact is verified automatically. ``n_compartments`` is
    needed only to compare normalized recorder indices with negative callback
    indices before a run. Dendra's indexed Recorder itself requires
    nonnegative indices because it uses ``index_select``.

    ``partition`` has the same meaning as ``callback.is_active(partition)``:
    contiguous groups of checked sites, each requiring ``callback.at_least``
    sites to cross. For ``ActiveALCount(at_least=1)``, one crossing at any site
    is enough. Legacy ``_Active`` accepts no partition. The returned margin
    has the same batch/population/partition axes. A full trace with no checked
    post-step frame is rejected because it cannot define a finite root.
    """
    if type(callback) in (Active, ActiveAL):
        at_least = callback.at_least
    elif type(callback) is ActiveALCount and callback.at_least == 1:
        at_least = 1
    elif type(callback) is _Active:
        if partition is not None:
            raise ValueError("Legacy _Active does not support partitions.")
        at_least = 1
    else:
        raise NotImplementedError(
            "Only Active, ActiveAL, ActiveALCount(at_least=1), and legacy "
            "_Active have a validated signed-margin adapter."
        )
    if callback.inv:
        raise NotImplementedError(
            "Inverted activity has a strict boundary and no inclusive "
            "negative-of-peak margin."
        )
    if not isinstance(at_least, int) or at_least < 1:
        raise ValueError("at_least must be a positive integer.")

    if isinstance(recorded_voltage, Recorder):
        if selected_nodes:
            raise ValueError("selected_nodes is inferred for a Recorder.")
        trace, selected_nodes = _recorder_trace(
            callback, recorded_voltage, n_compartments=n_compartments
        )
    else:
        trace = recorded_voltage
    if not isinstance(trace, torch.Tensor) or trace.ndim < 3:
        raise ValueError(
            "Voltage must be a time-first tensor with population and site axes."
        )
    if not trace.is_floating_point() or not bool(torch.isfinite(trace).all()):
        raise ValueError("Voltage must be finite and floating point.")

    if selected_nodes:
        node_count = len(torch.as_tensor(callback.node_check).reshape(-1))
        if trace.shape[-1] != node_count:
            raise ValueError("Selected voltage width differs from callback.node_check.")
        checked_voltage = trace
    else:
        nodes = _checked_nodes(callback, int(trace.shape[-1])).to(trace.device)
        checked_voltage = trace.index_select(-1, nodes)
        node_count = int(nodes.numel())
    lengths = _partition_lengths(partition, node_count, at_least)

    n_steps = trace.shape[0] - 1
    first = max(0, callback.ind_start)
    stop = min(n_steps, callback.ind_end)
    if first >= stop:
        raise ValueError(
            "No post-step voltage frame falls in the callback check window."
        )
    # The first frame is the initial state. Callback i=0 reads frame 1.
    window = checked_voltage[first + 1 : stop + 1]
    site_peaks, peak_offsets = window.max(dim=0)
    peak_steps = peak_offsets + first

    margins = []
    steps = []
    sites = []
    site_offset = 0
    for length in lengths:
        scores = site_peaks[..., site_offset : site_offset + length]
        selected = torch.topk(scores, at_least, dim=-1).indices[..., -1]
        margin = (
            scores.gather(-1, selected.unsqueeze(-1)).squeeze(-1) - callback.threshold
        )
        margins.append(margin)
        steps.append(
            peak_steps[..., site_offset : site_offset + length]
            .gather(-1, selected.unsqueeze(-1))
            .squeeze(-1)
        )
        sites.append(selected + site_offset)
        site_offset += length

    if partition is None:
        return CallbackMargin(margins[0], steps[0], sites[0])
    return CallbackMargin(
        torch.stack(margins, dim=-1),
        torch.stack(steps, dim=-1),
        torch.stack(sites, dim=-1),
    )
