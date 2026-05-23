from typing import Any, Dict

import torch

from ..parametric import Referency
from .netstim import NetStim
from .spiking import update_active, update_active_diff
from .utils import make_getattr


class ContinuousCon(Referency):
    """Continuous analog projection between a presynaptic variable and a postsynaptic mechanism.

    ``ContinuousCon`` is the analog counterpart of :class:`NetCon`: it gathers a
    presynaptic variable, applies optional weighting and integer delay, scatters
    the resulting values into the target synapse's local shape, and calls
    ``syn.continuous_receive(...)``.  It performs no thresholding and no event
    detection.

    The hot path is specialized at construction / delay-rebuild time.  Static
    delay masks, selected post indices, flat delay-buffer offsets, and dispatch
    mode are precomputed once so ``advance()`` does not repeatedly build masks or
    check for empty delayed subsets.
    """

    def __init__(
        self,
        pre,
        pre_idx,
        post,
        post_idx,
        post_syn,
        weight,
        delay,
        dt,
        pre_var=None,
        input=None,
        reduce="sum",
        max_delay=None,
        transform=None,
        reset_inputs=False,
    ):
        super().__init__()
        self.weight = weight
        self.delay_ms = delay
        self.syn = post_syn
        self.pre = pre
        self.post = post
        self.input = input
        self.reduce = reduce
        self.transform = transform
        self.max_delay = max_delay

        # reset_inputs is accepted for compatibility with older internal
        # builders, but input reset is now owned by Network.step so every
        # ContinuousSynapse target is reset exactly once before all analog
        # deliveries.  ContinuousCon therefore has no per-step reset branch.
        self._reset_inputs = False

        self._refresh_peer_devices()
        self.weight = self.weight.to(device=self.device)
        self.delay_ms = self.delay_ms.to(device=self.device)
        if isinstance(self.transform, torch.nn.Module):
            self.transform = self.transform.to(device=self.device, dtype=self.dtype)

        self.dt = torch.tensor(dt, device=self.device, dtype=torch.float32)
        if pre_var is None:
            pre_var = "v"
        self.get_pre_var = make_getattr(pre_var)
        self.pre_var = pre_var
        self._has_transform = transform is not None

        self.register_buffer(
            "pre_idx", pre_idx.flatten().to(self.pre_device, dtype=torch.long)
        )
        self.register_buffer(
            "post_idx", post_idx.flatten().to(self.device, dtype=torch.long)
        )
        self.register_buffer(
            "syn_numel",
            torch.prod(torch.tensor(self.syn.shape_f, device=self.device)).to(
                self.device, dtype=torch.long
            ),
        )
        self._syn_numel = int(self.syn_numel.item())
        self.register_buffer(
            "n",
            torch.tensor(self.pre_idx.numel(), device=self.device, dtype=torch.long),
        )
        self._n_conn = int(self.pre_idx.numel())
        self.register_buffer(
            "current_time_step", torch.tensor([0], device=self.device, dtype=torch.long)
        )
        self.register_buffer(
            "global_step", torch.tensor([0], device=self.device, dtype=torch.long)
        )

        delay_values = self.delay_ms.init().w
        delay_steps = (delay_values / dt).round().long()
        self.register_buffer("delay_steps", delay_steps.flatten().to(self.device))
        self.max_delay_steps = self._compute_max_delay_steps()
        self._delivery_numel = int(self.max_delay_steps * self._syn_numel)
        self.register_buffer(
            "delivery_buffer",
            torch.zeros(
                (self.max_delay_steps, self._syn_numel),
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self.register_buffer(
            "con_range",
            torch.arange(self._n_conn, device=self.device, dtype=torch.long),
        )

        # Delay/layout metadata buffers are registered in _rebuild_delay_metadata().
        self._rebuild_delay_metadata()
        self._align_buffer_devices()
        self._configure_advance_impl()

    # ------------------------------------------------------------------
    # Device / dtype bookkeeping
    # ------------------------------------------------------------------

    def _refresh_peer_devices(self):
        pre_device = self.pre.device()
        pre_dtype = self.pre.dtype()
        post_device = self.post.device()
        post_dtype = self.post.dtype()
        syn_device = self.syn.device() if hasattr(self.syn, "device") else post_device

        self.pre_device = pre_device
        self.pre_dtype = pre_dtype
        self.post_device = post_device
        self.device = syn_device
        self.dtype = post_dtype

    def _move_buffer(self, name, device, dtype=None):
        if not hasattr(self, name):
            return
        buf = getattr(self, name)
        target_dtype = dtype if dtype is not None else buf.dtype
        if buf.device != device or buf.dtype != target_dtype:
            setattr(self, name, buf.to(device=device, dtype=target_dtype))

    def _set_buffer(self, name, value):
        if name in self._buffers:
            setattr(self, name, value)
        else:
            self.register_buffer(name, value)

    def _align_buffer_devices(self):
        self._move_buffer("pre_idx", self.pre_device)
        for name in (
            "post_idx",
            "syn_numel",
            "n",
            "delay_steps",
            "current_time_step",
            "global_step",
            "con_range",
            # Precomputed continuous-delay metadata.
            "zero_con_idx",
            "nonzero_con_idx",
            "post_idx_zero",
            "post_idx_nz",
            "delay_steps_nz",
            "flat_delay_offsets_nz",
        ):
            self._move_buffer(name, self.device)
        self._move_buffer("delivery_buffer", self.device, self.dtype)
        self.dt = self.dt.to(device=self.device, dtype=torch.float32)
        self.weight = self.weight.to(device=self.device, dtype=self.dtype)
        self.delay_ms = self.delay_ms.to(device=self.device, dtype=self.dtype)

    def to(self, *args, **kwargs):  # type: ignore[override]
        super().to(*args, **kwargs)
        self._refresh_peer_devices()
        self._align_buffer_devices()
        if isinstance(self.transform, torch.nn.Module):
            self.transform = self.transform.to(device=self.device, dtype=self.dtype)
        self._configure_advance_impl()
        return self

    # ------------------------------------------------------------------
    # Delay metadata / specialization
    # ------------------------------------------------------------------

    def _compute_max_delay_steps(self):
        if self.max_delay is not None:
            return max(1, int(self.max_delay / self.dt.item()) + 1)
        if self.delay_steps.numel() == 0:
            return 1
        return max(1, int(self.delay_steps.max().item()) + 1)

    def _rebuild_delay_metadata(self):
        """Precompute masks/indices/offsets used by the continuous hot path."""
        delay_steps = self.delay_steps.flatten().to(self.device, dtype=torch.long)
        n_conn = int(delay_steps.numel())
        self._n_conn = n_conn
        self._has_connections = n_conn > 0

        if n_conn == 0:
            empty = torch.empty(0, device=self.device, dtype=torch.long)
            self._set_buffer("zero_con_idx", empty)
            self._set_buffer("nonzero_con_idx", empty)
            self._set_buffer("post_idx_zero", empty)
            self._set_buffer("post_idx_nz", empty)
            self._set_buffer("delay_steps_nz", empty)
            self._set_buffer("flat_delay_offsets_nz", empty)
            self._has_zero_delay = False
            self._has_nonzero_delay = False
            self._all_zero_delay = False
            self._all_nonzero_delay = False
            self._nonzero_delay_uniform = False
            self._direct_all = False
            return

        zero_mask = delay_steps == 0
        nonzero_mask = ~zero_mask
        zero_idx = torch.nonzero(zero_mask, as_tuple=False).flatten().to(self.device)
        nonzero_idx = (
            torch.nonzero(nonzero_mask, as_tuple=False).flatten().to(self.device)
        )

        self._has_zero_delay = bool(zero_idx.numel() > 0)
        self._has_nonzero_delay = bool(nonzero_idx.numel() > 0)
        self._all_zero_delay = bool(
            self._has_zero_delay and not self._has_nonzero_delay
        )
        self._all_nonzero_delay = bool(
            self._has_nonzero_delay and not self._has_zero_delay
        )

        post_idx_zero = (
            self.post_idx.index_select(0, zero_idx) if zero_idx.numel() else zero_idx
        )
        post_idx_nz = (
            self.post_idx.index_select(0, nonzero_idx)
            if nonzero_idx.numel()
            else nonzero_idx
        )
        delay_steps_nz = (
            delay_steps.index_select(0, nonzero_idx)
            if nonzero_idx.numel()
            else nonzero_idx
        )
        flat_offsets = delay_steps_nz * int(self._syn_numel) + post_idx_nz

        self._set_buffer("zero_con_idx", zero_idx)
        self._set_buffer("nonzero_con_idx", nonzero_idx)
        self._set_buffer("post_idx_zero", post_idx_zero)
        self._set_buffer("post_idx_nz", post_idx_nz)
        self._set_buffer("delay_steps_nz", delay_steps_nz)
        self._set_buffer("flat_delay_offsets_nz", flat_offsets)

        if delay_steps_nz.numel() > 0:
            first = delay_steps_nz[0]
            self._nonzero_delay_uniform = bool(
                torch.all(delay_steps_nz == first).item()
            )
            self._uniform_delay_step = (
                int(first.item()) if self._nonzero_delay_uniform else None
            )
        else:
            self._nonzero_delay_uniform = False
            self._uniform_delay_step = None

        # Fast path for dense one-to-one all-immediate projections: avoid scatter.
        if n_conn == self._syn_numel:
            expected = torch.arange(
                self._syn_numel, device=self.device, dtype=torch.long
            )
            self._direct_all = bool(
                torch.equal(self.post_idx.to(self.device), expected)
            )
        else:
            self._direct_all = False

    def _configure_advance_impl(self):
        if not getattr(self, "_has_connections", False):
            self.advance = self._advance_empty
        elif self._all_zero_delay:
            self.advance = self._advance_all_immediate
        elif self._all_nonzero_delay:
            if self._nonzero_delay_uniform:
                self.advance = self._advance_all_delayed_uniform
            else:
                self.advance = self._advance_all_delayed_mixed
        else:
            if self._nonzero_delay_uniform:
                self.advance = self._advance_mixed_uniform
            else:
                self.advance = self._advance_mixed

    def _rebuild_delay_buffers(self):
        with torch.no_grad():
            delay_values = self.delay_ms()
            delay_steps = (delay_values / self.dt.to(self.dtype)).round().long()
            self.delay_steps = delay_steps.flatten().to(self.device)
            self.max_delay_steps = self._compute_max_delay_steps()
            self._delivery_numel = int(self.max_delay_steps * self._syn_numel)
            self.delivery_buffer = torch.zeros(
                (self.max_delay_steps, self._syn_numel),
                device=self.device,
                dtype=self.dtype,
            )
            self._rebuild_delay_metadata()
            self._align_buffer_devices()
            self._configure_advance_impl()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def zero(self, clear_deliveries=True):
        self.current_time_step.fill_(0)
        self.global_step.fill_(0)
        if clear_deliveries:
            self.delivery_buffer.zero_()

    def detach(self):
        for n, b in self.named_buffers():
            setattr(self, n, b.detach())

    def initialize(
        self, reinit_weights=True, reinit_delays=True, clear_deliveries=True
    ):
        self.zero(clear_deliveries=clear_deliveries)
        self.weight.init(reinit=reinit_weights)
        self.delay_ms.init(reinit=reinit_delays)
        if reinit_delays:
            self._rebuild_delay_buffers()
        if hasattr(self, "t"):
            step = torch.round(
                self.t.to(self.device, dtype=self.dtype) / self.dt.to(self.dtype)
            ).long()
            self.global_step.copy_(step.reshape_as(self.global_step))
        self.detach()
        self._configure_advance_impl()
        return self

    def n_connections(self):
        return int(self.n.item())

    def state_dict_for_checkpoint(self):
        return {
            "delivery_buffer": self.delivery_buffer,
            "current_time_step": self.current_time_step,
            "global_step": self.global_step,
        }

    def restore_dict_from_checkpoint(self, state_dict):
        self.delivery_buffer = state_dict["delivery_buffer"]
        self.current_time_step = state_dict["current_time_step"]
        self.global_step = state_dict["global_step"]
        return self

    # ------------------------------------------------------------------
    # Hot-path helpers
    # ------------------------------------------------------------------

    def _maybe_reset_inputs(self):
        # Deprecated compatibility hook.  Continuous input resets are performed
        # by Network before advancing continuous projections.
        return None

    def _advance_counters(self):
        self.current_time_step.add_(1).remainder_(self.max_delay_steps)
        self.global_step.add_(1)

    def _pre_value(self):
        # The presynaptic variable should already live on the source device; only
        # move if necessary.  This avoids redundant no-op .to(...) dispatches in
        # the common single-device path.
        x_full = self.get_pre_var(self.pre)
        if x_full.device != self.pre_device or x_full.dtype != self.pre_dtype:
            x_full = x_full.to(device=self.pre_device, dtype=self.pre_dtype)
        x = x_full.reshape(-1).index_select(0, self.pre_idx)
        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(device=self.device, dtype=self.dtype)
        if self._has_transform:
            x = self.transform(x)
        return x

    def _weighted_pre_value(self):
        return self.weight() * self._pre_value()

    def _zeros_delivery(self):
        return torch.zeros(self._syn_numel, device=self.device, dtype=self.dtype)

    def _scatter_all(self, values):
        if self._direct_all:
            return values
        out = self._zeros_delivery()
        out.index_add_(0, self.post_idx, values)
        return out

    def _scatter_zero_subset(self, values):
        out = self._zeros_delivery()
        vals = values.index_select(0, self.zero_con_idx)
        out.index_add_(0, self.post_idx_zero, vals)
        return out

    def _read_and_clear_current_row(self):
        cur_idx = self.current_time_step
        delayed_delivery = self.delivery_buffer.index_select(0, cur_idx).squeeze(0)
        self.delivery_buffer.index_fill_(0, cur_idx, 0.0)
        return cur_idx, delayed_delivery

    def _deliver(self, flat_delivery):
        self.syn.continuous_receive(
            flat_delivery.view(*self.syn.shape_f),
            self,
            input=self.input,
            reduce=self.reduce,
        )

    def _schedule_all_delayed_uniform(self, cur_idx, weighted):
        future = (cur_idx + int(self._uniform_delay_step)).remainder(
            self.max_delay_steps
        )
        flat = future * self._syn_numel + self.post_idx
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), weighted)

    def _schedule_selected_delayed_uniform(self, cur_idx, weighted):
        vals = weighted.index_select(0, self.nonzero_con_idx)
        future = (cur_idx + int(self._uniform_delay_step)).remainder(
            self.max_delay_steps
        )
        flat = future * self._syn_numel + self.post_idx_nz
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), vals)

    def _schedule_all_delayed_mixed(self, cur_idx, weighted):
        base = cur_idx * self._syn_numel
        flat = (base + self.flat_delay_offsets_nz).remainder(self._delivery_numel)
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), weighted)

    def _schedule_selected_delayed_mixed(self, cur_idx, weighted):
        vals = weighted.index_select(0, self.nonzero_con_idx)
        base = cur_idx * self._syn_numel
        flat = (base + self.flat_delay_offsets_nz).remainder(self._delivery_numel)
        self.delivery_buffer.view(-1).index_add_(0, flat.reshape(-1), vals)

    # ------------------------------------------------------------------
    # Specialized advance paths
    # ------------------------------------------------------------------

    def _advance_empty(self):
        self._advance_counters()

    def _advance_all_immediate(self):
        weighted = self._weighted_pre_value()
        self._deliver(self._scatter_all(weighted))
        # All delays are zero, so current_time_step remains zero modulo one.
        self.global_step.add_(1)

    def _advance_all_delayed_uniform(self):
        cur_idx, delayed_delivery = self._read_and_clear_current_row()
        self._deliver(delayed_delivery)
        weighted = self._weighted_pre_value()
        self._schedule_all_delayed_uniform(cur_idx, weighted)
        self._advance_counters()

    def _advance_all_delayed_mixed(self):
        cur_idx, delayed_delivery = self._read_and_clear_current_row()
        self._deliver(delayed_delivery)
        weighted = self._weighted_pre_value()
        self._schedule_all_delayed_mixed(cur_idx, weighted)
        self._advance_counters()

    def _advance_mixed_uniform(self):
        cur_idx, delayed_delivery = self._read_and_clear_current_row()
        weighted = self._weighted_pre_value()
        immediate = self._scatter_zero_subset(weighted)
        self._deliver(delayed_delivery + immediate)
        self._schedule_selected_delayed_uniform(cur_idx, weighted)
        self._advance_counters()

    def _advance_mixed(self):
        cur_idx, delayed_delivery = self._read_and_clear_current_row()
        weighted = self._weighted_pre_value()
        immediate = self._scatter_zero_subset(weighted)
        self._deliver(delayed_delivery + immediate)
        self._schedule_selected_delayed_mixed(cur_idx, weighted)
        self._advance_counters()


class NetCon(Referency):
    """
    Event-based connectivity wrapper between a pre-synaptic source and a
    post-synaptic synapse mechanism.

    Overview
    --------
    A ``NetCon`` instance mediates communication between:

    * a **pre-synaptic source** (population module or :class:`NetStim`),
    * a **post-synaptic synapse mechanism** (module exposing ``net_receive``), and
    * an implicit **post population** that owns the synapse state.

    Conceptually, each time step performs three operations:

    1. **Event detection**:

       - Observe a pre-synaptic variable (by default ``pre.v`` or an explicit
         gate on a :class:`NetStim`).
       - Detect intrinsic events (hard or surrogate spiking) via threshold
         crossings, or directly consume externally scheduled events.

    2. **Weighting and delay**:

       - Combine intrinsic and scheduled events into a scalar gate per
         connection.
       - Multiply by a per-connection weight (optionally differentiable).
       - Write the resulting payload into a per-synapse delay line
         (delivery buffer) according to per-connection delays
         (integer or differentiable).

    3. **Delivery**:

       - At each step, read the current row of the delay buffer.
       - Call ``syn.net_receive(payload, netcon=self)`` on the synapse
         mechanism, with payload reshaped to ``syn.shape_f``.

    The class is designed to:

    * Support heterogeneous device placement for pre, post, and synapse modules.
    * Provide differentiable pathways for:

      - weights,
      - delays,
      - spiking / event detection,
      - and optionally scheduled event times.

    * Provide explicit *scheduled* events that can be trained by reference.

    Construction and lifecycle
    --------------------------
    ``NetCon`` objects are normally constructed by
    :class:`dendra.models.networks.Network` during ``Network.build()``.
    Initialization and stepping are managed by the network's simulation loop.

    1. **Construction** (usually internal):

       .. code-block:: python

           nc = NetCon(
               pre=pre_pop,
               pre_idx=pre_indices,
               thresholds=thresholds,
               post=post_pop,
               post_idx=post_indices,
               post_syn=synapse_module,
               weight=weight_param,
               delay=delay_param,
               dt=dt,
               pre_var="v",         # optional
               max_delay=None,      # optional
           )

    2. **Differentiable configuration (optional but required for training)**:

       Call :meth:`set_diff_config` once before the first differentiable
       simulation step to configure which components are treated as
       differentiable:

       .. code-block:: python

           nc.set_diff_config(
               diff_weights=True,
               diff_delays=True,
               diff_spiking=True,
               diff_scheduled_times=True,
               # taps, sigma, tau, sched_width as appropriate
           )

    3. **Initialization**:

       Call :meth:`initialize` before stepping:

       .. code-block:: python

           nc.train()         # if using standard PyTorch training mode
           nc.set_diff_config(...)  # must be called before first advance_diff
           nc.initialize(
               reinit_weights=True,
               reinit_delays=True,
               clear_deliveries=True,
           )

       ``initialize`` will:

       * select ``nc.advance`` to point to either :meth:`advance_diff` or
         :meth:`advance_non_diff` depending on ``nc.training``,
       * (re)initialize weight and delay parameter modules,
       * rebuild internal delay buffers if delays changed,
       * reset the internal timing state, and
       * detach buffers from any pre-existing computation graph
         (fresh forward pass).

    4. **Simulation loop**:

       .. code-block:: python

           for step in range(T):
               nc.advance()  # calls advance_diff or advance_non_diff

       At each call, ``NetCon`` delivers events to ``post_syn`` via
       ``post_syn.net_receive(...)``.

    Scheduling: value- vs reference-mode
    ------------------------------------
    In addition to intrinsic spiking, ``NetCon`` supports *scheduled* events.
    These are particularly useful for:

    * injecting known spike trains or stimulus patterns,
    * training event **weights** and/or **times** as separate learnable tensors,
    * conditioning the synaptic drive on external controllers or task structure.

    There are two orthogonal axes:

    1. **Value mode** (direct tensors):

       - You provide explicit event weights and/or times (as numbers or Tensors).
       - Gradients flow into those tensors if they require gradients and are
         wired into your optimization loop.

       Value-mode APIs:

       * :meth:`schedule`
       * :meth:`schedule_time_ref` with ``weight`` specified as a Tensor.

    2. **Reference mode** (indices into bound sources):

       - You bind a *source* tensor once via :meth:`bind_weight_source` or
         :meth:`bind_time_source`.
       - You then schedule events by storing integer indices into the source;
         the values are looked up on-the-fly during the simulation.
       - Gradients flow into the bound source tensors, not the indices.

       Reference-mode APIs:

       * :meth:`schedule_ref` for reference-mode weights,
       * :meth:`schedule_time_ref` for reference-mode times.

    Mixed use is allowed: you can combine intrinsic spiking, value-mode
    scheduled events, and reference-mode scheduled events in the same ``NetCon``.

    Order-of-operations for training scheduled **weights**
    ------------------------------------------------------
    To make scheduled event weights trainable by reference:

    1. **Create a trainable weight source**:

       .. code-block:: python

           # Example: one scalar weight per planned event
           sched_w = torch.nn.Parameter(
               torch.zeros(num_events, device=device, dtype=dtype)
           )

    2. **Bind the weight source once**:

       .. code-block:: python

           nc.bind_weight_source(sched_w)

       This does not copy data; it stores a reference to ``sched_w``. The
       tensor must live on the same device as the synapse.

    3. **Schedule events by index** using :meth:`schedule_ref`:

       .. code-block:: python

           # con_indices or pre_indices must map to concrete connections
           nc.schedule_ref(
               con_indices=con_idx,   # or pre_indices=...
               times_ms=t_ms,         # float times in ms
               weight_idx=w_idx,      # Long tensor of indices into sched_w
               allow_past=False,
           )

       On each time step:

       * ``NetCon`` will look up per-event weights as
         ``sched_w[weight_idx]``.
       * These contribute to the per-connection gates, and hence to the
         synaptic deliveries.
       * Gradients from the loss will flow back into ``sched_w``.

    Order-of-operations for training scheduled **times**
    ----------------------------------------------------
    To make scheduled event **times** trainable by reference:

    1. **Create a trainable time source**:

       .. code-block:: python

           # Example: one scalar time per event, in ms
           sched_t = torch.nn.Parameter(
               torch.zeros(num_events, device=device, dtype=dtype)
           )

    2. **Bind the time source once**:

       .. code-block:: python

           nc.bind_time_source(sched_t)

    3. **Schedule events by time index** using :meth:`schedule_time_ref`:

       .. code-block:: python

           nc.schedule_time_ref(
               con_indices=con_idx,   # or pre_indices=...
               time_idx=t_idx,        # Long tensor of indices into sched_t
               weight=1.0,            # scalar or tensor (value mode)
               allow_past=False,
           )

       On each time step:

       * ``NetCon`` will read the *current* values from ``sched_t[time_idx]``.
       * The resulting times are interpreted in one of two ways:

         - If ``diff_scheduled_times=True`` in :meth:`set_diff_config`,
           events are smeared with a differentiable triangular kernel over
           time steps, so gradients can move event times.
         - If ``diff_scheduled_times=False``, times are rounded to the
           nearest integer step and treated as exact.

       In both cases, gradients flow into ``sched_t`` when using the
       differentiable configuration.

    Interactions with global time
    -----------------------------
    Many scheduling methods accept an ``allow_past`` flag:

    * When ``allow_past=False`` (default), events whose scheduled steps are
      **earlier** than the current internal ``global_step`` are silently
      dropped.
    * When ``allow_past=True``, events are kept regardless of their scheduled
      step index.

    In practice, this means that you usually want to:

    * Call :meth:`initialize` (which also sets ``global_step``) before
      scheduling events, or
    * Explicitly pass ``allow_past=True`` when scheduling relative to a
      fixed origin (e.g., step 0) while reusing a ``NetCon``.

    Notes
    -----
    * ``NetCon`` automatically tracks and realigns its internal buffers when
      moved across devices via ``.to(...)`` or when the peer modules move.
    * The class is intentionally conservative about in-loop memory allocation:
      buffers are pre-allocated where possible and reused across steps.
    """

    def __init__(
        self,
        pre,
        pre_idx,
        thresholds,
        post,
        post_idx,
        post_syn,
        weight,
        delay,
        dt,
        pre_var=None,
        max_delay=None,
    ):
        """
        Parameters
        ----------
        pre : :class:`~dendra.models.core.Population` or :class:`~dendra.models.networks.NetStim`
            Pre-synaptic population module or :class:`NetStim`. The module must
            implement ``device()`` and ``dtype()``; when ``pre`` is not a
            :class:`NetStim`, it is also expected to expose the attribute named
            by ``pre_var`` (e.g., ``v``) which holds the pre-synaptic state
            used for spiking.
        pre_idx : torch.Tensor
            1-D Long tensor of indices into the flattened pre-synaptic variable
            (e.g., ``pre.v.view(-1)[pre_idx]``). Length defines the number of
            managed connections.
        thresholds : torch.Tensor
            Per-connection spiking thresholds in the same shape as ``pre_idx``.
            Entries may be NaN to indicate:

            * all-NaN: skip thresholding and treat the selected pre variable as
              a continuous gate;
            * mixed: apply thresholding for finite entries, and use the raw
              pre variable wherever the threshold is NaN.

        post : :class:`~dendra.models.core.Population`
            Post-synaptic population module that owns the synapse mechanism.
            Must implement ``device()`` and ``dtype()`` interfaces and is used
            mainly for device alignment.
        post_idx : torch.Tensor
            1-D Long tensor of indices into the flattened synapse output space;
            used to scatter connection events into the synapse's delivery buffer.
        post_syn : :class:`dendra.models.mechanisms.Synapse`
            Synapse mechanism module. Must expose:

            * ``shape_f``: final tensor shape of per-synapse payloads, and
            * ``net_receive(payload, netcon)``: function invoked each step with
              delivery buffer slice reshaped to ``shape_f``.

        weight : torch.nn.Module
            Parameter-like module representing per-connection weights. Must support:

            * ``weight.init(reinit: bool)``,
            * ``weight()`` returning the current weight tensor, and
            * ``weight.w`` exposing the underlying tensor.

        delay : torch.nn.Module
            Parameter-like module representing per-connection delays in ms. Must
            support:

            * ``delay.init(reinit: bool)`` and
            * ``delay()`` returning the current delays tensor.

        dt : float
            Simulation time step in milliseconds.
        pre_var : str, optional
            Name of the attribute on ``pre`` to use as the spiking variable when
            ``pre`` is not a :class:`NetStim`. Defaults to ``"v"``.
        max_delay : float, optional
            Optional maximum delay (ms). If provided, the internal delay buffer
            depth is set to ``int(max_delay / dt) + 1``; otherwise it is inferred
            from the maximum delay in ``delay``.

        Notes
        -----
        This constructor is typically invoked by
        :class:`dendra.models.networks.Network` during ``Network.build()``; user
        code usually interacts with :class:`NetCon` via high-level network
        building APIs rather than constructing it directly.
        """
        super().__init__()

        self.weight = weight
        self.syn = post_syn
        self.pre = pre
        self.post = post

        # Track peer devices/dtypes and keep NetCon buffers aligned with them.
        self._refresh_peer_devices()

        # Ensure parameter modules live on the delivery (post/synapse) device.
        self.weight = self.weight.to(device=self.device)
        delay = delay.to(device=self.device)

        self.dt = torch.tensor(dt, device=self.device, dtype=torch.float32)
        self.max_delay = max_delay

        self.delay_ms = delay
        delay = delay.init().w

        if pre_var is None:
            pre_var = "v"

        self.get_pre_var = make_getattr(pre_var)

        if isinstance(pre, NetStim):
            self.determine_spiking = self.determine_spiking_ns
        else:
            self.determine_spiking = self.determine_spiking_var

        self.register_buffer(
            "pre_idx", pre_idx.flatten().to(self.pre_device, dtype=torch.long)
        )
        self.register_buffer(
            "post_idx", post_idx.flatten().to(self.device, dtype=torch.long)
        )
        self.register_buffer(
            "threshold",
            torch.nan_to_num(thresholds)
            .flatten()
            .to(self.pre_device, dtype=self.pre_dtype),
        )

        nan_thresh = torch.isnan(thresholds).to(self.pre_device)
        self.register_buffer("thresh_is_nan", nan_thresh)

        self.skip_thresholding = bool(nan_thresh.all())
        self.apply_masking = bool(nan_thresh.any()) and not self.skip_thresholding

        self.register_buffer(
            "syn_numel",
            torch.prod(torch.tensor(self.syn.shape_f, device=self.device)).to(
                self.device, dtype=torch.long
            ),
        )
        self._syn_numel = int(self.syn_numel.item())  # cache for performance

        self.register_buffer(
            "n",
            torch.tensor(self.pre_idx.numel()).to(self.device, dtype=torch.long),
        )
        self.register_buffer(
            "pre_range",
            torch.arange(self.n.item(), device=self.pre_device, dtype=torch.long),
        )
        n_pre = self.n.item()
        assert len(self.threshold) == n_pre, (
            "Thresholds must match the number of pre synaptic locations."
        )
        assert delay.shape[0] == self.pre_idx.shape[0], (
            "Delay tensor shape must match pre_idx."
        )

        self.register_buffer(
            "has_spiked", torch.zeros(n_pre, device=self.pre_device, dtype=torch.bool)
        )
        # Store a *gate* in the same dtype as the rest of the module (float)
        self.register_buffer(
            "is_spiking",
            torch.zeros(n_pre, device=self.pre_device, dtype=self.pre_dtype),
        )

        # --- Delay Handling Logic (Compiler-Friendly) ---
        delay_steps = (delay / dt).round().long()
        self.register_buffer("delay_steps", delay_steps.flatten().to(self.device))

        self.max_delay_steps = self._compute_max_delay_steps()

        buffer_shape = (self.max_delay_steps, self.syn_numel.item())
        self.register_buffer(
            "delivery_buffer",
            torch.zeros(buffer_shape, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "event_queue",
            torch.zeros(
                (self.max_delay_steps, n_pre), device=self.device, dtype=torch.int32
            ),
        )
        self.register_buffer(
            "events", torch.zeros(n_pre, device=self.device, dtype=torch.int32)
        )

        self.register_buffer(
            "current_time_step", torch.tensor([0], device=self.device, dtype=torch.long)
        )

        # --- Pre-computed tensor for masking, avoids creating tensors in the loop ---
        self.register_buffer(
            "time_indices", torch.arange(self.max_delay_steps, device=self.device)
        )

        self.train_flags = None

        # scheduled events
        # Python-side calendar: abs_step -> list of (pre_idx:int, weight:float)
        self._sched_pre = {}

        # flat calendar (initially empty)
        self.register_buffer(
            "sched_pre_idx", torch.empty(0, device=self.device, dtype=torch.long)
        )
        self.register_buffer(
            "sched_abs_step", torch.empty(0, device=self.device, dtype=torch.long)
        )

        # ---- Value mode (snapshot) ----
        # store the actual per-event weights; NOT a Parameter, NOT a buffer (to keep graph)
        self.sched_weight = torch.empty(0, device=self.device, dtype=self.dtype)

        self.register_buffer(
            "global_step", torch.tensor([0], device=self.device, dtype=torch.long)
        )
        self._n_conn: int = int(self.pre_idx.numel())  # Python int
        self.register_buffer(
            "con_range",
            torch.arange(self.pre_idx.numel(), device=self.device, dtype=torch.long),
        )

        # connection-schedule storage (fixed-length vectors; all per-connection)
        self.register_buffer(
            "sched_con_idx", torch.empty(0, device=self.device, dtype=torch.long)
        )
        self.register_buffer(
            "sched_abs_step", torch.empty(0, device=self.device, dtype=torch.long)
        )
        # value-mode per-event weights (kept as a Tensor to preserve autograd to producers)
        self.sched_weight = torch.empty(0, device=self.device, dtype=self.dtype)
        # reference-mode source + indices
        self._sched_w_source = None
        self.register_buffer(
            "sched_weight_idx", torch.empty(0, device=self.device, dtype=torch.long)
        )

        self.sched_time_ms = torch.empty(
            0, device=self.device, dtype=self.dtype
        )  # NOT a buffer

        # optional reference-mode source + indices for times
        self._sched_t_source = None
        self.register_buffer(
            "sched_time_idx", torch.empty(0, device=self.device, dtype=torch.long)
        )

        # optional introspection (per-connection)
        self.register_buffer(
            "sched_wsum",
            torch.zeros(self._n_conn, device=self.device, dtype=self.dtype),
        )
        self.register_buffer(
            "sched_counts",
            torch.zeros(self._n_conn, device=self.device, dtype=torch.int32),
        )

        # ---- build a pre→conn CSR map once (Python side; outside compiled step) ----
        with torch.no_grad():
            # sort connections by pre id
            vals, order = torch.sort(self.pre_idx)  # both [n_conn]
            # unique pre ids and run-lengths
            uniq, counts = torch.unique_consecutive(vals, return_counts=True)
            starts = torch.cat(
                [
                    torch.zeros(1, device=self.device, dtype=torch.long),
                    counts.cumsum(0)[:-1],
                ],
                dim=0,
            )
            # store CSR pieces (all tensors on device for convenience)
            self._csr_pre_ids = (
                uniq  # [n_pre_used] sorted pre ids (maybe a subset of 0..max)
            )
            self._csr_starts = starts  # [n_pre_used]
            self._csr_counts = counts  # [n_pre_used]
            self._csr_conidx_sorted = order  # [n_conn]

        # Final sanity alignment (important if builder later calls .to()).
        self._align_buffer_devices()

    def _move_buffer(self, name: str, device: torch.device, dtype=None):
        buf = getattr(self, name)
        target_dtype = dtype if dtype is not None else buf.dtype
        if buf.device != device or buf.dtype != target_dtype:
            setattr(
                self,
                name,
                buf.to(device=device, dtype=target_dtype),
            )

    def _refresh_peer_devices(self):
        """
        Inspect the pre/post/synapse modules and record their devices and dtypes.

        This method is called at construction and by :meth:`to` to ensure that
        ``NetCon`` keeps an up-to-date view of:

        * the device/dtype of the pre-synaptic population,
        * the device/dtype of the post-synaptic population, and
        * the device/dtype of the synapse mechanism (which defines the main
          computation device for delivery buffers).

        Returns
        -------
        pre_changed : bool
            True if the pre-synaptic device or dtype has changed since the last
            call.
        post_changed : bool
            True if the synapse/post device or dtype has changed since the last
            call.
        """
        pre_device = self.pre.device()
        pre_dtype = self.pre.dtype()
        post_device = self.post.device()
        post_dtype = self.post.dtype()
        syn_device = self.syn.device() if hasattr(self.syn, "device") else post_device

        pre_changed = (
            getattr(self, "pre_device", None) != pre_device
            or getattr(self, "pre_dtype", None) != pre_dtype
        )
        post_changed = (
            getattr(self, "device", None) != syn_device
            or getattr(self, "dtype", None) != post_dtype
        )

        self.pre_device = pre_device
        self.pre_dtype = pre_dtype
        self.post_device = post_device
        self.device = syn_device
        self.dtype = post_dtype

        return pre_changed, post_changed

    def _align_buffer_devices(self):
        """
        Move internal buffers to the appropriate devices/dtypes.

        This is called at construction time, at the end of ``__init__``, and
        after any external ``.to(...)`` call via :meth:`to`. The method keeps:

        * all buffers that interact with pre-synaptic state on ``pre_device``,
        * all delivery-path buffers (delay lines, queues, scheduling metadata)
          on ``self.device`` (the synapse/post device),
        * parameter modules (weights and delays) and ``dt`` on ``self.device``.

        It also ensures that auxiliary CSR structures used to expand pre indices
        to connection indices are kept on the pre-synaptic device.
        """
        # Keep buffers that interact with pre-synaptic state on the pre device/dtype.
        self._move_buffer("pre_idx", self.pre_device)
        self._move_buffer("threshold", self.pre_device, self.pre_dtype)
        self._move_buffer("thresh_is_nan", self.pre_device)
        self._move_buffer("has_spiked", self.pre_device)
        self._move_buffer("is_spiking", self.pre_device, self.pre_dtype)
        self._move_buffer("pre_range", self.pre_device)

        # Buffers that feed the post-synaptic delivery path live with the synapse.
        for name in (
            "post_idx",
            "syn_numel",
            "n",
            "delay_steps",
            "event_queue",
            "events",
            "current_time_step",
            "time_indices",
            "sched_pre_idx",
            "sched_abs_step",
            "sched_con_idx",
            "sched_weight_idx",
            "sched_time_idx",
            "sched_counts",
            "global_step",
            "con_range",
        ):
            self._move_buffer(name, self.device)

        self._move_buffer("delivery_buffer", self.device, self.dtype)
        self._move_buffer("sched_wsum", self.device, self.dtype)
        self.sched_weight = self.sched_weight.to(device=self.device, dtype=self.dtype)
        self.sched_time_ms = self.sched_time_ms.to(device=self.device, dtype=self.dtype)

        # Keep helper caches with the pre buffers.
        if hasattr(self, "_csr_pre_ids"):
            self._csr_pre_ids = self._csr_pre_ids.to(device=self.pre_device)
            self._csr_starts = self._csr_starts.to(device=self.pre_device)
            self._csr_counts = self._csr_counts.to(device=self.pre_device)
            self._csr_conidx_sorted = self._csr_conidx_sorted.to(device=self.pre_device)

        # dt and parameter modules should follow the synapse device.
        self.dt = self.dt.to(device=self.device, dtype=torch.float32)
        self.weight = self.weight.to(device=self.device, dtype=self.dtype)
        self.delay_ms = self.delay_ms.to(device=self.device, dtype=self.dtype)

    def to(self, *args, **kwargs):  # type: ignore[override]
        """
        Move this ``NetCon`` and its internal state to a new device/dtype.

        This is a thin wrapper around ``super().to(...)`` which also:

        * refreshes peer devices/dtypes via :meth:`_refresh_peer_devices`, and
        * realigns internal buffers via :meth:`_align_buffer_devices`.

        All scheduling metadata and delay buffers are preserved, but they are
        moved to the target device/dtype as appropriate.

        Returns
        -------
        NetCon
            The same instance, for chaining.
        """
        super().to(*args, **kwargs)
        self._refresh_peer_devices()
        self._align_buffer_devices()
        return self

    def _compute_max_delay_steps(self):
        """
        Compute the depth (in steps) of the internal delay line.

        If ``max_delay`` is provided at construction, this method simply
        computes ``int(max_delay / dt) + 1``. Otherwise it uses the maximum
        integer delay in ``self.delay_steps`` if present, defaulting to 1
        when no delays exist.

        Returns
        -------
        int
            Maximum number of steps in the delay buffer.
        """
        if self.max_delay is not None:
            return int(self.max_delay / self.dt.item()) + 1
        else:
            return (
                int(self.delay_steps.max().item()) + 1
                if len(self.delay_steps) > 0
                else 1
            )

    def _rebuild_delay_buffers(self):
        """
        Rebuild delay-related buffers from the current delay parameters.

        This is called by :meth:`initialize` when ``reinit_delays=True``.
        It:

        * recomputes integer delay steps from ``delay_ms()`` and ``dt``,
        * recomputes ``max_delay_steps``,
        * reallocates the delivery buffer and event queue to match the new
          depth, and
        * re-aligns all relevant buffers to the correct devices via
          :meth:`_align_buffer_devices`.
        """
        with torch.no_grad():
            delay_steps = (self.delay_ms() / self.dt.to(self.dtype)).round().long()
            self.delay_steps.copy_(delay_steps.flatten().to(self.device))

            self.max_delay_steps = self._compute_max_delay_steps()

            buffer_shape = (self.max_delay_steps, self.syn_numel.item())
            self.delivery_buffer = torch.zeros(
                buffer_shape, device=self.device, dtype=self.dtype
            )
            self.event_queue = torch.zeros(
                (self.max_delay_steps, self.n.item()),
                device=self.device,
                dtype=torch.int32,
            )
            self.time_indices = torch.arange(
                self.max_delay_steps, device=self.device, dtype=torch.long
            )
            self._align_buffer_devices()

    def set_diff_config(
        self,
        diff_weights: bool = True,
        diff_delays: bool = True,
        diff_spiking: bool = True,
        taps: int = 2,
        sigma: float = 0.35,  # used for taps=3 (in steps),
        tau: float = 0.1,  # temperature for surrogate spiking
        diff_scheduled_times: bool = True,
        sched_width: float = 1.0,  # kernel half-width in *steps*
    ):
        """
        Configure differentiable behavior for training.

        This method must be called before the first call to :meth:`advance_diff`
        (via ``NetCon.advance`` in training mode). It controls which aspects of
        the connection are treated as differentiable and how.

        Parameters
        ----------
        diff_weights : bool, optional
            If True (default), synaptic weights are treated as differentiable
            parameters. If False, the weight tensor returned by ``self.weight()``
            is detached before being used to compute deliveries, freezing its
            contribution during training.
        diff_delays : bool, optional
            If True (default), delays are interpreted as continuous values in
            milliseconds and used with a soft, differentiable interpolation
            scheme over time steps (2-tap linear or 3-tap Gaussian, see
            ``taps`` and ``sigma``). If False, delays are rounded to integer
            steps and used as hard delay lines.
        diff_spiking : bool, optional
            If True (default), pre-synaptic spiking is computed via a surrogate
            gradient function (see :func:`update_active_diff`), allowing
            gradients to flow through threshold crossings. If False, hard
            thresholding via :func:`update_active` is used, yielding non-
            differentiable (boolean) spikes.
        taps : int, optional
            Number of interpolation taps to use when ``diff_delays=True``.

                * ``2`` (default): 2-tap linear interpolation between adjacent steps.
                * ``3``: 3-tap Gaussian-like interpolation weighted by ``sigma``.
        sigma : float, optional
            Standard deviation (in steps) for the 3-tap Gaussian interpolation
            when ``taps=3``. Ignored for ``taps=2``.
        tau : float, optional
            Temperature parameter for surrogate spiking, passed through to
            :func:`update_active_diff` as ``tau``. Lower values yield a steeper
            surrogate; higher values produce smoother gates.
        diff_scheduled_times : bool, optional
            If True (default), scheduled event times (including those provided
            via :meth:`schedule_time_ref`) are handled via a differentiable
            triangular kernel over time steps. This allows gradients to move
            event times when they come from value- or reference-mode tensors.
            If False, scheduled times are rounded to integer steps, and the
            timing behavior is non-differentiable.
        sched_width : float, optional
            Half-width of the triangular kernel (in steps) when
            ``diff_scheduled_times=True``. A value of 1.0 yields contributions
            spread over approximately 2 steps around the nominal event time.

        Notes
        -----
        * ``set_diff_config`` does not itself change ``training`` mode; it only
          records configuration flags. ``NetCon.advance`` will dispatch to
          :meth:`advance_diff` when ``self.training = True`` and to
          :meth:`advance_non_diff` otherwise.
        """
        self.train_flags = (
            diff_weights,
            diff_delays,
            diff_spiking,
            taps,
            sigma,
            tau,
            diff_scheduled_times,
            float(sched_width),
        )

    @property
    def w(self):
        """
        Raw synaptic weight tensor for this ``NetCon``.

        Returns
        -------
        torch.Tensor
            The underlying weight tensor managed by the ``weight`` parameter
            module (i.e., ``self.weight.w``). This tensor has one entry per
            connection and is typically used for:

            * inspection and diagnostics,
            * direct regularization or constraints,
            * manual initialization or export.

        Notes
        -----
        To obtain the *current* weight values used during simulation steps
        (which may include functional transformations), use ``self.weight()``
        instead of accessing ``w`` directly.
        """
        return self.weight.w

    def _expand_pre_to_con(self, pre_indices: torch.Tensor | list[int]) -> torch.Tensor:
        """
        Expand pre-synaptic indices to connection indices using the CSR map.

        This helper is used by the scheduling APIs (e.g. :meth:`schedule`,
        :meth:`schedule_ref`, :meth:`schedule_time_ref`) when the caller
        specifies pre-synaptic indices instead of explicit connection
        indices.

        Parameters
        ----------
        pre_indices : array-like of int
            Pre-synaptic indices into the original population. These must be
            compatible with the pre index space used to construct ``pre_idx``.

        Returns
        -------
        torch.Tensor
            1-D Long tensor of connection indices containing all connections
            whose ``pre_idx`` entry matches one of the requested ``pre_indices``.
            The tensor may be empty if no connections originate from the
            specified pre indices.
        """
        pres = torch.as_tensor(
            pre_indices, device=self.pre_device, dtype=torch.long
        ).view(-1)
        if pres.numel() == 0:
            return pres

        # Find each pre id in CSR unique list with searchsorted
        pos = torch.searchsorted(self._csr_pre_ids, pres)
        valid = (pos < self._csr_pre_ids.numel()) & (
            self._csr_pre_ids.index_select(0, pos) == pres
        )
        if not bool(valid.any()):
            return torch.empty(0, device=self.pre_device, dtype=torch.long)

        pos = pos[valid]
        starts = self._csr_starts.index_select(0, pos)
        lens = self._csr_counts.index_select(0, pos)

        # Gather slices of the sorted connection-index array and concat
        con_chunks = []
        for s, L in zip(starts.tolist(), lens.tolist()):
            if L > 0:
                con_chunks.append(self._csr_conidx_sorted.narrow(0, s, L))
        if len(con_chunks) == 0:
            return torch.empty(0, device=self.pre_device, dtype=torch.long)
        return torch.cat(con_chunks, dim=0)  # [n_con_from_these_pres]

    def schedule(
        self,
        *,
        con_indices=None,
        pre_indices=None,
        times_ms=None,
        weight=1.0,
        allow_past: bool = False,
    ):
        """
        Schedule value-mode events on specific connections or pre indices.

        This API attaches *value-mode* scheduled events, where weights and
        times are stored directly as tensors owned by the ``NetCon``. It is
        suitable when you want event-specific weights/times that are not
        shared by reference with other structures, or when you are fine
        managing these tensors directly.

        Exactly one of ``con_indices`` or ``pre_indices`` must be provided.

        Parameters
        ----------
        con_indices : Sequence[int] or torch.Tensor, optional
            Explicit connection indices at which to schedule events.
            Must be in ``[0, num_connections)``. Mutually exclusive with
            ``pre_indices``.
        pre_indices : Sequence[int] or torch.Tensor, optional
            Pre-synaptic indices. These are expanded internally to a set of
            connection indices using the pre→connection CSR map (see
            :meth:`_expand_pre_to_con`). Mutually exclusive with
            ``con_indices``.
        times_ms : Sequence[float] or torch.Tensor
            Event times in milliseconds. Must be broadcast-compatible with
            the selected connections *after* filtering out past events (see
            ``allow_past``). Times are converted to steps via
            ``round(times_ms / dt)`` for non-differentiable scheduling, and
            interpreted as continuous values when ``diff_scheduled_times=True``.
        weight : float or torch.Tensor, optional
            Scalar or per-event weight *multiplier* applied on top of the base
            connection weights returned by ``self.weight()``. Accepted shapes:

            * scalar (single value): broadcast to all scheduled events;
            * tensor of length ``E_before`` (events before past-event filtering):
              will be sliced by the filter; or
            * tensor of length ``E_after`` (events after filtering): used as is.

            When provided as a tensor requiring gradients, and when
            differentiable scheduling is enabled, gradients will flow into
            this tensor.
        allow_past : bool, optional
            If False (default), events whose scheduled integer step is smaller
            than the current ``global_step`` are dropped. If True, no such
            filtering is applied.

        Notes
        -----
        * This is a *value-mode* scheduling API: weights and times are stored
          directly in ``self.sched_weight`` and ``self.sched_time_ms``.
        * For *reference-mode* weights or times (where event metadata lives in
          a separate trainable tensor), use :meth:`schedule_ref` and
          :meth:`schedule_time_ref` after binding sources with
          :meth:`bind_weight_source` and :meth:`bind_time_source`.
        * This method can be called multiple times; new events are appended.
        """
        if (con_indices is None) == (pre_indices is None):
            raise ValueError("Provide exactly one of con_indices or pre_indices")
        if times_ms is None:
            raise ValueError("times_ms is required")

        device, dtype = self.device, self.dtype

        # → connection indices (raw, before any filtering)
        if pre_indices is not None:
            con_idx_raw = self._expand_pre_to_con(pre_indices).to(device)
        else:
            con_idx_raw = torch.as_tensor(
                con_indices, device=device, dtype=torch.long
            ).view(-1)

        tms_raw = torch.as_tensor(times_ms, device=device, dtype=torch.float32).view(-1)

        if con_idx_raw.numel() != tms_raw.numel():
            raise ValueError("con_indices/pre_indices and times_ms must match length")
        if (con_idx_raw < 0).any() or (con_idx_raw >= self.pre_idx.numel()).any():
            raise IndexError("connection index out of range")

        # legacy step field (used when diff_scheduled_times=False)
        steps_raw = torch.round(tms_raw / self.dt.to(tms_raw.dtype)).to(torch.long)

        # optional filtering of past events
        if not allow_past:
            cur = self.global_step.view(())
            keep = steps_raw >= cur
            if not bool(keep.any()):
                return
            con_idx = con_idx_raw[keep]
            tms = tms_raw[keep]
            steps = steps_raw[keep]
        else:
            con_idx = con_idx_raw
            tms = tms_raw
            steps = steps_raw

        E_before = con_idx_raw.numel()
        E_after = con_idx.numel()

        # Normalize weight (supports scalar, pre-filter length, or post-filter length)
        if torch.is_tensor(weight):
            w_raw = weight.to(device=device, dtype=dtype).view(-1)
            if w_raw.numel() == 1:
                w = w_raw.expand(E_after)  # scalar → broadcast
            elif w_raw.numel() == E_before:
                w = (
                    w_raw[keep] if not allow_past else w_raw
                )  # per-event (pre-filter) → slice if filtered
            elif w_raw.numel() == E_after:
                w = w_raw  # per-event (post-filter) already aligned
            else:
                raise ValueError(
                    f"weight tensor length must be 1, {E_before} (pre-filter), or {E_after} (post-filter)"
                )
        else:
            w = torch.full((E_after,), float(weight), device=device, dtype=dtype)

        # append (outside compiled loop: changing E is fine)
        self.sched_con_idx = torch.cat([self.sched_con_idx, con_idx], dim=0)
        self.sched_abs_step = torch.cat([self.sched_abs_step, steps], dim=0)
        self.sched_time_ms = torch.cat([self.sched_time_ms, tms], dim=0)
        self.sched_weight = torch.cat([self.sched_weight, w], dim=0)

        # keep ref-index arrays aligned (sentinel -1)
        filler = torch.full((E_after,), -1, device=device, dtype=torch.long)
        self.sched_weight_idx = torch.cat([self.sched_weight_idx, filler], dim=0)
        self.sched_time_idx = torch.cat([self.sched_time_idx, filler], dim=0)

    def bind_weight_source(self, source: torch.Tensor):
        """
        Bind a tensor providing per-event weights for reference-mode scheduling.

        This method enables *reference-mode* scheduled weights used by
        :meth:`schedule_ref`. Instead of storing per-event weights directly
        inside ``NetCon``, you bind a tensor (usually a parameter) and then
        refer to entries in that tensor by integer indices.

        Parameters
        ----------
        source : torch.Tensor
            Tensor containing candidate weights for scheduled events. It must:

            * reside on the same device as the synapse (``self.device``), and
            * have a size at least as large as any ``weight_idx`` used in
              :meth:`schedule_ref`.

            Typically this is a ``torch.nn.Parameter`` or another tensor that
            participates in the training loop.

        Notes
        -----
        Order of operations:

            1. Construct the weight source tensor (e.g. ``nn.Parameter``).
            2. Call ``netcon.bind_weight_source(weight_source)`` exactly once (or
               whenever you want to replace the source).
            3. Use :meth:`schedule_ref` with ``weight_idx`` values indexing into
               the bound tensor.

        * Calling this replaces any previously bound weight source.
        * The tensor is not copied; ``NetCon`` keeps a reference and reads from
          it during simulation.
        * Gradients flow into the bound tensor whenever
          ``diff_weights=True`` and the scheduling path is differentiable.
        """
        if source.device != self.device:
            raise ValueError("weight source must be on same device")
        self._sched_w_source = source

    def bind_time_source(self, source: torch.Tensor):
        """
        Bind a tensor providing per-event times (ms) for reference-mode scheduling.

        This method enables *reference-mode* scheduled times used by
        :meth:`schedule_time_ref`. Instead of storing times directly in
        ``NetCon``, you bind a tensor (usually a parameter) and then reference
        entries in that tensor by integer indices.

        Parameters
        ----------
        source : torch.Tensor
            Tensor containing candidate event times in milliseconds. It must:

            * reside on the same device as the synapse (``self.device``), and
            * have a size at least as large as any ``time_idx`` used in
              :meth:`schedule_time_ref`.

            Typically this is a ``torch.nn.Parameter`` or another tensor that
            participates in the training loop.

        Notes
        -----
        Order of operations:

            1. Construct the time source tensor (e.g. ``nn.Parameter`` in ms).
            2. Call ``netcon.bind_time_source(time_source)``.
            3. Use :meth:`schedule_time_ref` with ``time_idx`` values indexing into
               the bound tensor.

        * Calling this replaces any previously bound time source.
        * The tensor is not copied; ``NetCon`` reads from it at runtime.
        * When used with :meth:`set_diff_config(diff_scheduled_times=True)`,
          and when the bound tensor requires gradients, event times become
          differentiable via the triangular timing kernel.
        """
        if source.device != self.device:
            raise ValueError("time source must be on same device")
        self._sched_t_source = source

    def schedule_ref(
        self,
        *,
        con_indices=None,
        pre_indices=None,
        times_ms=None,
        weight_idx=None,
        allow_past: bool = False,
    ):
        """
        Schedule reference-mode events with weights drawn from a bound source.

        This API is the reference-mode counterpart to :meth:`schedule`. Instead
        of passing explicit weights, you pass integer indices into a tensor
        that has previously been bound via :meth:`bind_weight_source`.

        Exactly one of ``con_indices`` or ``pre_indices`` must be provided,
        and :meth:`bind_weight_source` must have been called beforehand.

        Parameters
        ----------
        con_indices : Sequence[int] or torch.Tensor, optional
            Explicit connection indices at which to schedule events.
            Mutually exclusive with ``pre_indices``.
        pre_indices : Sequence[int] or torch.Tensor, optional
            Pre-synaptic indices to expand to connection indices via the CSR
            map (see :meth:`_expand_pre_to_con`). Mutually exclusive with
            ``con_indices``.
        times_ms : Sequence[float] or torch.Tensor
            Event times in milliseconds, one per event. Converted to integer
            steps via rounding for non-differentiable behavior; used as
            continuous times when ``diff_scheduled_times=True``.
        weight_idx : Sequence[int] or torch.Tensor
            Integer indices into the tensor bound by :meth:`bind_weight_source`.
            Must have the same length as ``times_ms`` and ``con_indices`` /
            expanded connections.
        allow_past : bool, optional
            If False (default), events scheduled before the current
            ``global_step`` are discarded. If True, no filtering is applied.

        Notes
        -----
        *Reference-mode* scheduling is designed for scenarios where you want
        scheduled weights to be:

        * shared across multiple events or :class:`NetCon` instances,
        * trained by a separate module or optimizer, or
        * constrained/regularized jointly (e.g., via an external loss).

        Because the weight values are read from a shared tensor at runtime,
        gradients from all events that reference a given index accumulate into
        that entry.

        **Order of operations**:

            1. Bind a weight source tensor:

               .. code-block:: python

                   nc.bind_weight_source(weight_source)

            2. Schedule reference-mode events:

               .. code-block:: python

                    nc.schedule_ref(
                        con_indices=...,
                        times_ms=...,
                        weight_idx=...,
                    )

            3. Run simulation (``nc.advance()`` in a loop).

        * If :meth:`bind_weight_source` has not been called, this method raises
          a ``RuntimeError``.
        * The value-mode slots for weights are still allocated but set to zero;
          only reference-mode contributions are active for these events.
        """
        if self._sched_w_source is None:
            raise RuntimeError("call bind_weight_source(...) before schedule_ref(...)")
        if (con_indices is None) == (pre_indices is None):
            raise ValueError("Provide exactly one of con_indices or pre_indices")
        if times_ms is None or weight_idx is None:
            raise ValueError("times_ms and weight_idx are required")

        device, dtype = self.device, self.dtype
        if pre_indices is not None:
            con_idx = self._expand_pre_to_con(pre_indices).to(device)
        else:
            con_idx = torch.as_tensor(
                con_indices, device=device, dtype=torch.long
            ).view(-1)

        tms = torch.as_tensor(times_ms, device=device, dtype=torch.float32).view(-1)
        widx = torch.as_tensor(weight_idx, device=device, dtype=torch.long).view(-1)
        if not (con_idx.numel() == tms.numel() == widx.numel()):
            raise ValueError("indices, times_ms, and weight_idx must have same length")

        steps = torch.round(tms / self.dt.to(tms.dtype)).to(torch.long)
        if not allow_past:
            cur = self.global_step.view(())
            keep = steps >= cur
            con_idx, steps, widx = con_idx[keep], steps[keep], widx[keep]
            if con_idx.numel() == 0:
                return

        if (con_idx < 0).any() or (con_idx >= self.pre_idx.numel()).any():
            raise IndexError("connection index out of range")
        if (widx < 0).any() or (widx >= self._sched_w_source.numel()).any():
            raise IndexError("weight_idx out of range for bound weight source")

        self.sched_con_idx = torch.cat([self.sched_con_idx, con_idx], dim=0)
        self.sched_abs_step = torch.cat([self.sched_abs_step, steps], dim=0)
        self.sched_weight = torch.cat(
            [
                self.sched_weight,
                torch.zeros(con_idx.numel(), device=device, dtype=dtype),
            ],
            dim=0,
        )
        self.sched_weight_idx = torch.cat([self.sched_weight_idx, widx], dim=0)

    def schedule_time_ref(
        self,
        *,
        con_indices=None,
        pre_indices=None,
        time_idx=None,
        weight=1.0,
        allow_past: bool = False,
    ):
        """
        Schedule events whose times come by reference from a bound tensor.

        This API is the time-reference analogue of :meth:`schedule`. Instead
        of providing explicit event times, you provide integer indices into a
        tensor bound via :meth:`bind_time_source`. Weights are still supplied
        in value mode.

        Exactly one of ``con_indices`` or ``pre_indices`` must be provided, and
        :meth:`bind_time_source` must have been called beforehand.

        Parameters
        ----------
        con_indices : Sequence[int] or torch.Tensor, optional
            Explicit connection indices for the events.
        pre_indices : Sequence[int] or torch.Tensor, optional
            Pre-synaptic indices, expanded to connection indices via the CSR map.
        time_idx : Sequence[int] or torch.Tensor
            Integer indices into the tensor bound via :meth:`bind_time_source`.
            Must match the number of events after expanding ``pre_indices`` if
            used.
        weight : float or torch.Tensor, optional
            Value-mode weight multiplier, as in :meth:`schedule`. May be a
            scalar or a per-event tensor of length equal to the number of
            scheduled events. Gradients can flow into this tensor when
            differentiable scheduling is enabled.
        allow_past : bool, optional
            If False (default), events whose current time (from the bound
            source) would be in the past relative to ``global_step`` are
            dropped. If True, keep all events.

        Notes
        -----
        Time-reference scheduling is designed for scenarios where you want to:

        * train event times explicitly as parameters,
        * share timing parameters across multiple events,
        * or couple event times to other model components (e.g., another network).

        When used with ``diff_scheduled_times=True`` in :meth:`set_diff_config`,
        gradients flow into the bound time tensor and can move events in time.

        **Order of operations**:

            1. Bind a time source tensor:

                .. code-block:: python

                    nc.bind_time_source(time_source)

            2. Schedule events by time index:

                .. code-block:: python

                    nc.schedule_time_ref(
                        con_indices=...,
                        time_idx=...,
                        weight=...,
                    )

            3. Run simulation (``nc.advance()`` in a loop).

        * If :meth:`bind_time_source` has not been called, this method raises a
          ``RuntimeError``.
        * The value-mode time slot in ``sched_time_ms`` is populated with zeros;
          the actual times are obtained at runtime from the bound tensor via
          ``sched_time_idx``.
        """
        if self._sched_t_source is None:
            raise RuntimeError(
                "call bind_time_source(...) before schedule_time_ref(...)"
            )
        if (con_indices is None) == (pre_indices is None):
            raise ValueError("Provide exactly one of con_indices or pre_indices")
        if time_idx is None:
            raise ValueError("time_idx is required")

        device, dtype = self.device, self.dtype

        # -> connection indices
        if pre_indices is not None:
            con_idx = self._expand_pre_to_con(pre_indices).to(device)
        else:
            con_idx = torch.as_tensor(
                con_indices, device=device, dtype=torch.long
            ).view(-1)

        tidx = torch.as_tensor(time_idx, device=device, dtype=torch.long).view(-1)
        if con_idx.numel() != tidx.numel():
            raise ValueError("con_indices/pre_indices and time_idx must match length")

        if (con_idx < 0).any() or (con_idx >= self.pre_idx.numel()).any():
            raise IndexError("connection index out of range")
        if (tidx < 0).any() or (tidx >= self._sched_t_source.numel()).any():
            raise IndexError("time_idx out of range for bound time source")

        # use a zeros value-slot for times, and record the indices in sched_time_idx
        # legacy step field still populated from current source values (for non-diff path)
        tms_now = self._sched_t_source.index_select(0, tidx).to(torch.float32)
        steps = torch.round(tms_now / self.dt.to(torch.float32)).to(torch.long)
        if not allow_past:
            cur = self.global_step.view(())
            keep = steps >= cur
            con_idx, steps, tidx, tms_now = (
                con_idx[keep],
                steps[keep],
                tidx[keep],
                tms_now[keep],
            )
            if con_idx.numel() == 0:
                return

        # weight value-mode (optional)
        if torch.is_tensor(weight):
            w = weight.to(device=device, dtype=dtype).view(-1)
            if w.numel() not in (1, con_idx.numel()):
                raise ValueError("weight must be scalar or same length as con_indices")
            if w.numel() == 1:
                w = w.expand_as(con_idx)
        else:
            w = torch.full(
                (con_idx.numel(),), float(weight), device=device, dtype=dtype
            )

        self.sched_con_idx = torch.cat([self.sched_con_idx, con_idx], dim=0)
        self.sched_abs_step = torch.cat([self.sched_abs_step, steps], dim=0)
        self.sched_time_ms = torch.cat(
            [self.sched_time_ms, torch.zeros_like(tms_now)], dim=0
        )  # value slot 0
        self.sched_time_idx = torch.cat([self.sched_time_idx, tidx], dim=0)
        self.sched_weight = torch.cat([self.sched_weight, w], dim=0)

        # keep weight ref index aligned (no weight-ref here, so -1)
        self.sched_weight_idx = torch.cat(
            [
                self.sched_weight_idx,
                torch.full((con_idx.numel(),), -1, device=device, dtype=torch.long),
            ],
            dim=0,
        )

    def clear_schedule(self):
        """
        Remove all scheduled events and reset scheduling-related buffers.

        This clears both value-mode and reference-mode scheduled events:

        * connection indices (``sched_con_idx``),
        * absolute step indices (``sched_abs_step``),
        * value-mode weights and times (``sched_weight``, ``sched_time_ms``),
        * reference-mode indices (``sched_weight_idx``, ``sched_time_idx``),
        * per-connection introspection statistics (``sched_wsum``,
          ``sched_counts``).

        The bound sources set via :meth:`bind_weight_source` and
        :meth:`bind_time_source` are **not** modified.

        Notes
        -----
        * This is typically called between episodes or trials when you want
          to reuse the same ``NetCon`` instance but with a fresh schedule.
        """
        device, dtype = self.device, self.dtype
        self.sched_con_idx = torch.empty(0, device=device, dtype=torch.long)
        self.sched_abs_step = torch.empty(0, device=device, dtype=torch.long)
        self.sched_weight = torch.empty(0, device=device, dtype=dtype)
        self.sched_weight_idx = torch.empty(0, device=device, dtype=torch.long)
        self.sched_time_ms = torch.empty(0, device=device, dtype=dtype)
        self.sched_time_idx = torch.empty(0, device=device, dtype=torch.long)
        self.sched_wsum.zero_()
        self.sched_counts.zero_()

    def _scheduled_gate_this_step(self, gs_long: torch.Tensor, *, use_tri_kernel: bool):
        """
        Aggregate scheduled contributions for the current absolute step.

        This internal helper constructs:

        * a per-connection scheduled gate amplitude (sum of per-event weights),
        * a per-connection count of active scheduled events (for bookkeeping),

        given the current global step and the configuration flags in
        ``self.train_flags``.

        Parameters
        ----------
        gs_long : torch.Tensor
            Scalar Long tensor with the current global step index.
        use_tri_kernel : bool
            If True, use a differentiable triangular kernel in step space for
            scheduled times (used when ``diff_scheduled_times=True``). If False,
            scheduled events are active only when the rounded step equals
            ``gs_long``.

        Returns
        -------
        sched_wsum_conn : torch.Tensor
            1-D tensor of shape ``[n_conn]`` containing the scheduled gate
            amplitudes per connection for this step.
        sched_counts_conn : torch.Tensor
            1-D int32 tensor of shape ``[n_conn]`` containing the number of
            scheduled events contributing to each connection at this step.
        """
        n_conn = self._n_conn
        device, dtype = self.device, self.dtype

        if self.sched_con_idx.numel() == 0:
            return (
                torch.zeros(n_conn, device=device, dtype=dtype),
                torch.zeros(n_conn, device=device, dtype=torch.int32),
            )

        # --- scheduled weights (value + ref) ---
        w_val = self.sched_weight
        if self._sched_w_source is not None:
            idxw = torch.clamp(self.sched_weight_idx, min=0)
            w_src = self._sched_w_source.index_select(0, idxw)
            w_ref_mask = (self.sched_weight_idx >= 0).to(dtype)
            w_val = w_val + w_src * w_ref_mask  # combine if both present

        # --- scheduled times (value + ref) -> float ms ---
        t_val = self.sched_time_ms  # may be zeros if using time-by-ref only
        if self._sched_t_source is not None:
            idxt = torch.clamp(self.sched_time_idx, min=0)
            t_src = self._sched_t_source.index_select(0, idxt).to(torch.float32)
            t_ref_mask = (self.sched_time_idx >= 0).to(t_src.dtype)
            t_val = t_val + t_src * t_ref_mask

        if use_tri_kernel:
            # Differentiable triangular kernel in *steps*
            lam = t_val.to(dtype) / self.dt.to(dtype)  # [E] float
            # gs_long is int64; cast to float
            x = lam - gs_long.to(dtype).view(())
            # width in steps (half-width). <=0 disables contribution
            _, _, _, _, _, _, _, sched_width = self.train_flags
            tri = (1.0 - (x.abs() / (sched_width + 1e-6))).clamp(
                min=0.0, max=1.0
            )  # [E]
            amp_evt = w_val * tri  # [E]
            # counts (bookkeeping): event "active" if tri>0
            cnt_evt = (tri > 0).to(torch.int32)  # [E]
        else:
            # Step-exact firing: round(lam) == gs
            lam = t_val.to(dtype) / self.dt.to(dtype)
            abs_step = lam.round().to(torch.long)
            now_mask = abs_step == gs_long.view(())
            amp_evt = w_val * now_mask.to(dtype)  # [E]
            cnt_evt = now_mask.to(torch.int32)  # [E]

        # Aggregate to connections
        sched_wsum_conn = torch.zeros(n_conn, device=device, dtype=dtype)
        sched_counts_conn = torch.zeros(n_conn, device=device, dtype=torch.int32)
        sched_wsum_conn.index_add_(0, self.sched_con_idx, amp_evt)
        sched_counts_conn.index_add_(0, self.sched_con_idx, cnt_evt)
        return sched_wsum_conn, sched_counts_conn

    def advance_diff(self):
        """
        Advance the connection state by one time step (differentiable path).

        This method implements the differentiable update kernel used when
        ``self.training`` is True and :meth:`set_diff_config` has been
        called. It performs:

        1. Delivery of the current delay-buffer row to ``syn.net_receive``.
        2. Determination of intrinsic spiking via ``determine_spiking`` with
           surrogate gradients if configured.
        3. Construction of the per-connection gate combining intrinsic and
           scheduled contributions (with optional differentiable timing).
        4. Application of weights (optionally differentiable) and delays
           (optionally differentiable via interpolation) to write into future
           delay-buffer rows.
        5. Update of ``current_time_step`` and ``global_step``.

        Notes
        -----
        * Users typically do not call this directly; instead, call
          ``netcon.initialize(...)`` and then invoke ``netcon.advance()``
          inside a simulation loop.
        * A ``RuntimeError`` is raised if :meth:`set_diff_config` has not been
          called (``train_flags is None``).
        """
        if self.train_flags is None:
            raise RuntimeError(
                "NetCon.train(...) must be called before advance_diff()."
            )
        (
            diff_weights,
            diff_delays,
            diff_spiking,
            taps,
            sigma,
            tau,
            diff_sched_times,
            _,
        ) = self.train_flags
        device, dtype = self.device, self.dtype

        # Snapshot indices for this step (avoid version bumps).
        # clone() is not necessary here; detach is enough because we never mutate
        # these tensors in-place and we treat them as non-differentiable counters.
        cur_idx = self.current_time_step.detach()  # [1], long
        gs = self.global_step.detach()  # [1], long

        # 1) deliver today's payload
        # NOTE: event_queue/events are assumed debug-only (not used for dynamics).
        todays = self.delivery_buffer.index_select(0, cur_idx).squeeze(0)  # [n_syn]
        self.syn.net_receive(todays.view(*self.syn.shape_f), self)

        # 2) intrinsic spiking
        self.determine_spiking(
            self.pre, diff_spiking, tau
        )  # self.is_spiking -> [n_conn] bool
        intrinsic_gate = self.is_spiking.to(device=device, dtype=dtype)  # [n_conn]

        # scheduled additions per connection (constant shapes)
        sched_wsum_conn, sched_counts_conn = self._scheduled_gate_this_step(
            gs, use_tri_kernel=bool(diff_sched_times)
        )

        # expose (non-diff)
        with torch.no_grad():
            self.sched_wsum.copy_(sched_wsum_conn)
            self.sched_counts.copy_(sched_counts_conn)

        # combine gates
        gate = intrinsic_gate + sched_wsum_conn  # [n_conn]

        # weights (optionally detach)
        wvals = self.weight()
        if not diff_weights:
            wvals = wvals.detach()
        weighted_spikes = wvals * gate  # [n_conn]

        # ------------------------------------------------------------------
        # Build next delivery buffer WITHOUT allocating a full-size "contrib"
        # buffer and WITHOUT constructing a full-size dense mask.
        #
        # We clone once (this is the output buffer), clear the delivered row,
        # then scatter-add into it.
        #
        # This reduces per-step peak memory substantially compared to:
        #   contrib=zeros_like(buf_flat) + mask2d + (buf_flat*mask2d + contrib)
        # ------------------------------------------------------------------
        buf_next = self.delivery_buffer.clone()
        buf_next.index_fill_(0, cur_idx, 0.0)  # clear the row we just delivered
        buf_flat = buf_next.view(-1)

        if diff_delays:
            d_ms = self.delay_ms().to(dtype)  # [n_conn]
            lam = d_ms / self.dt.to(dtype)
            k = torch.floor(lam)
            kL = k.to(torch.long)

            if taps == 2:
                alpha = (lam - k).to(dtype)
                idx0 = (cur_idx + kL).remainder(self.max_delay_steps)
                idx1 = (idx0 + 1).remainder(self.max_delay_steps)
                flat0 = idx0 * self.syn_numel + self.post_idx
                flat1 = idx1 * self.syn_numel + self.post_idx
                # Vectorized 2-tap write
                buf_flat.index_add_(0, flat0, weighted_spikes * (1.0 - alpha))
                buf_flat.index_add_(0, flat1, weighted_spikes * alpha)
            else:
                offs = torch.stack([kL - 1, kL, kL + 1], dim=-1)
                centers = offs.to(dtype)
                lam_e = lam.unsqueeze(-1)
                w = torch.softmax(
                    -0.5 * ((lam_e - centers) / (sigma + 1e-6)) ** 2, dim=-1
                )

                idxs = (cur_idx + offs).remainder(self.max_delay_steps)
                # Vectorized 3-tap write
                flat_idx = idxs * self.syn_numel + self.post_idx.unsqueeze(
                    -1
                )  # [n_conn,3]
                vals = weighted_spikes.unsqueeze(-1) * w  # [n_conn,3]
                buf_flat.index_add_(0, flat_idx.reshape(-1), vals.reshape(-1))
        else:
            future_steps = (cur_idx + self.delay_steps).remainder(self.max_delay_steps)
            flat = future_steps * self.syn_numel + self.post_idx
            # Integer delay: single destination per connection
            buf_flat.index_add_(0, flat, weighted_spikes)

        # Commit next buffer (already cleared + updated)
        self.delivery_buffer = buf_next

        # advance counters (no grad)
        with torch.no_grad():
            # Rebind buffers (out-of-place) to avoid version bumps on saved tensors
            self.current_time_step = (
                (self.current_time_step + 1).remainder(self.max_delay_steps).detach()
            )
            self.global_step = (self.global_step + 1).detach()

    def advance_non_diff(self):
        """
        Advance the connection state by one time step (non-differentiable path).

        This kernel is used when ``self.training`` is False. It implements the
        same logical operations as :meth:`advance_diff`, but:

        * uses hard threshold spiking (no surrogate gradients),
        * uses integer delay steps only,
        * uses exact step times for scheduled events (no triangular kernel),
        * performs in-place updates where convenient.

        Notes
        -----
        * Users typically call ``netcon.advance()`` after a call to
          :meth:`initialize` rather than invoking this method directly.
        """
        cur_idx = self.current_time_step
        todays_delivery = self.delivery_buffer.index_select(0, cur_idx)
        self.events = self.event_queue.index_select(0, cur_idx).squeeze(0)
        self.syn.net_receive(todays_delivery.squeeze(0).view(*self.syn.shape_f), self)

        # clear current row
        self.delivery_buffer.index_fill_(0, cur_idx, 0.0)
        self.event_queue.index_fill_(0, cur_idx, 0)

        # intrinsic spikes (hard)
        self.determine_spiking(self.pre, diff_spiking=False)
        intrinsic_gate = self.is_spiking.to(
            device=self.device, dtype=self.dtype
        )  # [n_conn]

        # scheduled contributions (constant shape)
        gs = self.global_step
        sched_wsum_conn, sched_counts_conn = self._scheduled_gate_this_step(
            gs, use_tri_kernel=False
        )
        with torch.no_grad():
            self.sched_wsum.copy_(sched_wsum_conn)
            self.sched_counts.copy_(sched_counts_conn)

        gate = intrinsic_gate + sched_wsum_conn
        weighted_spikes = self.weight() * gate  # [n_conn]

        future_steps = (cur_idx + self.delay_steps).remainder(self.max_delay_steps)
        flat = future_steps * self.syn_numel + self.post_idx

        self.delivery_buffer.view(-1).index_add_(0, flat, weighted_spikes)
        flat_e = future_steps * self.n + self.con_range
        self.event_queue.view(-1).index_add_(
            0,
            flat_e,
            (intrinsic_gate > 0).to(self.sched_counts.dtype) + sched_counts_conn,
        )

        self.current_time_step.add_(1).remainder_(self.max_delay_steps)
        self.global_step.add_(1)

    def determine_spiking_ns(self, pre: NetStim, diff_spiking: bool = True, tau=None):
        """
        Determine spiking when the pre-synaptic source is a :class:`NetStim`.

        For :class:`NetStim` sources, the pre-synaptic event signal is already
        represented as ``spikes`` (hard) or ``spike_gate`` (soft/surrogate).
        This method:

        * extracts the relevant entries using ``pre_idx``,
        * selects hard or soft gates depending on ``diff_spiking``, and
        * stores the result in ``self.is_spiking`` as a float tensor.

        Parameters
        ----------
        pre : NetStim
            Pre-synaptic :class:`NetStim` providing ``spikes`` and
            ``spike_gate`` attributes.
        diff_spiking : bool, optional
            If True, use ``pre.spike_gate`` directly (assumed differentiable).
            If False, use ``pre.spikes`` and cast to float gates {0, 1}.
        tau : float, optional
            Unused for :class:`NetStim` sources; present for API symmetry with
            :meth:`determine_spiking_var`.
        """
        # If the pre-synaptic source is a NetStim, we can directly use its spikes
        # (or spikes_gate for differentiable spiking)
        if diff_spiking:
            # surrogate / soft gate, already float
            gate = (
                pre.spike_gate.to(self.pre_device)
                .view(-1)
                .index_select(0, self.pre_idx)
            )
        else:
            # hard spikes; usually bool → cast to float gate {0, 1}
            gate = (
                pre.spikes.to(self.pre_device)
                .view(-1)
                .index_select(0, self.pre_idx)
                .to(self.pre_dtype)
            )

        # Ensure dtype is always self.dtype
        self.is_spiking = gate.to(device=self.pre_device, dtype=self.pre_dtype)

    def determine_spiking_var(self, pre, diff_spiking: bool = True, tau=0.1):
        """
        Determine spiking from a generic pre-synaptic variable (non-NetStim).

        This method is used when the pre-synaptic source is not a
        :class:`NetStim`. It:

        1. Extracts the relevant entries from the configured pre variable
           (e.g., ``pre.v``) using ``pre_idx``.
        2. Applies either surrogate-gradient or hard thresholding depending on
           ``diff_spiking`` and the presence of finite thresholds.
        3. Handles NaN thresholds by optionally using the raw variable as a
           continuous gate for those connections.

        Parameters
        ----------
        pre :
            Pre-synaptic module exposing the attribute named by
            ``self.get_pre_var`` (usually ``"v"``).
        diff_spiking : bool, optional
            If True, use :func:`update_active_diff` with a surrogate gradient
            defined by ``tau``. If False, use :func:`update_active` with hard
            thresholding.
        tau : float, optional
            Temperature parameter passed through to :func:`update_active_diff`.
            Smaller values yield steeper surrogate activation functions.

        Notes
        -----
        * If all thresholds are NaN (``skip_thresholding=True``), the method
          simply mirrors the pre variable as a continuous gate.
        * If some thresholds are NaN and others are finite, the method applies
          thresholding where finite and uses raw pre values where thresholds
          are NaN.
        """
        # Always work in the module's float dtype
        v_selected = (
            self.get_pre_var(pre)
            .to(self.pre_device)
            .view(-1)
            .index_select(0, self.pre_idx)
            .to(self.pre_dtype)
        )

        if self.skip_thresholding:
            # Just mirror v as a continuous "gate"
            self.is_spiking = v_selected.to(
                device=self.pre_device, dtype=self.pre_dtype
            )
            return

        if diff_spiking:
            # Surrogate / differentiable spiking
            self.has_spiked, _, gate = update_active_diff(
                self.has_spiked, v_selected, self.threshold, tau=tau
            )
            # gate is typically float already, but enforce dtype
            gate = gate.to(self.pre_dtype)
        else:
            # Hard threshold spiking; usually returns Bool spikes
            self.has_spiked, spikes = update_active(
                self.has_spiked, v_selected, self.threshold
            )
            # Turn {False,True} into {0.0,1.0} float gate
            gate = spikes.to(self.pre_dtype)

        if self.apply_masking:
            # thresh_is_nan: bool mask; choose between v_selected and gate (both float)
            gate = torch.where(
                self.thresh_is_nan,
                v_selected,
                gate,
            )

        self.is_spiking = gate.to(device=self.pre_device, dtype=self.pre_dtype)

    def zero(self, clear_delivery_buffers=True):
        """
        Reset the per-step state of the connection.

        This method:

        * sets ``current_time_step`` to 0,
        * optionally clears the delivery buffer and event queue state via
          ``delivery_buffer.zero_()``,
        * resets spike-history buffers (``has_spiked``, ``is_spiking``).

        Parameters
        ----------
        clear_delivery_buffers : bool, optional
            If True (default), zero the delivery buffer and reset spike
            histories. If False, keep existing contents of the delay line and
            event queue, but always reset ``current_time_step`` to 0.

        Notes
        -----
        * This does **not** clear schedules; use :meth:`clear_schedule` for
          that.
        * ``initialize`` calls this internally, so explicit calls are only
          needed when manually resetting state mid-simulation.
        """
        self.current_time_step.fill_(0)
        if clear_delivery_buffers:
            self.delivery_buffer.zero_()
            self.has_spiked.fill_(False)
            self.is_spiking.zero_()  # float buffer, reset to 0.0

    def detach(self):
        """
        Detach internal buffers from the current computation graph.

        This simply calls ``.detach()`` on all registered buffers and rebinds
        them. It is useful when:

        * starting a fresh forward pass between episodes,
        * freezing the internal state for inference,
        * or avoiding backpropagation through previous simulation windows.

        Notes
        -----
        * This does not change the values of buffers; it only affects their
          ``grad_fn`` and autograd history.
        """
        for n, b in self.named_buffers():
            setattr(self, n, b.detach())

    def initialize(
        self, reinit_weights=True, reinit_delays=True, clear_deliveries=True
    ):
        """
        Prepare ``NetCon`` for simulation by initializing buffers and parameters.

        This method must be called before the first call to ``advance`` in a
        new simulation episode. It performs the following steps:

        1. **Select advance kernel**:

           - If ``self.training`` is True, set ``self.advance = self.advance_diff``.
           - Otherwise, set ``self.advance = self.advance_non_diff``.

        2. **Reset per-step state** by calling :meth:`zero` with
           ``clear_deliveries``.

        3. **(Re)initialize parameters**:

           - Call ``self.weight.init(reinit=reinit_weights)``.
           - Call ``self.delay_ms.init(reinit=reinit_delays)``.

        4. **Rebuild delay buffers** if delays were reinitialized via
           :meth:`_rebuild_delay_buffers`.

        5. **Align internal time**:

           - Set ``global_step`` from the external time attribute ``self.t``
             and ``self.dt`` (assumes ``self.t`` is managed by the broader
             simulation).

        6. **Detach state** by calling :meth:`detach`.

        Parameters
        ----------
        reinit_weights : bool, optional
            If True (default), reinitialize weights via the underlying
            parameter module. If False, leave existing weights unchanged.
        reinit_delays : bool, optional
            If True (default), reinitialize delays via the underlying
            parameter module and rebuild delay buffers. If False, keep
            existing delays and the current delay buffers.
        clear_deliveries : bool, optional
            If True (default), zero delivery buffers and spike histories via
            :meth:`zero`. If False, keep existing delivery-buffer contents
            while still resetting ``current_time_step`` to 0.

        Notes
        -----
        * For differentiable training, call :meth:`set_diff_config` before
          calling ``initialize`` so that :meth:`advance_diff` can use the
          correct configuration.
        * ``initialize`` does not clear scheduled events; call
          :meth:`clear_schedule` if you need a fresh schedule.
        * This method is invoked automatically by higher-level simulation
          loops in Dendra (e.g., :class:`dendra.models.networks.Network.initialize`).
        """
        if self.training:
            self.advance = self.advance_diff
        else:
            self.advance = self.advance_non_diff
        self.zero(clear_delivery_buffers=clear_deliveries)
        self.weight.init(reinit=reinit_weights)
        self.delay_ms.init(reinit=reinit_delays)
        if reinit_delays:
            self._rebuild_delay_buffers()
        with torch.no_grad():
            self.global_step.fill_(int(round(float(self.t) / float(self.dt))))
        self.detach()

    def numel(self):
        """
        Return the number of synaptic connections managed by this ``NetCon``.

        Returns
        -------
        int
            Number of connections, equal to ``len(pre_idx)``. This is useful
            for:

            * sanity-checking connectivity sizes,
            * allocating auxiliary tensors,
            * reporting statistics about network scale.
        """
        return int(self.n.item())

    def state_dict_for_checkpoint(self):
        """
        Return a state dictionary suitable for gradient checkpointing / long-run chunking.

        This should include ONLY mutable tensors that affect future dynamics and are
        mutated by advance_*(). The returned structure must be stable across chunks
        (same keys; same shapes).

        Notes
        -----
        - This is *not* a full PyTorch state_dict. It is the *minimal mutable*
          runtime state needed to resume stepping identically mid-simulation.
        - Do NOT include weights/delays/modules here; those are model params
          and remain constant across a forward.
        - Under the assumption that event_queue/events are only for logging,
          we intentionally omit them. TODO: if they become essential to dynamics,
          they should be handled here, in a way that the user can flag.
        """

        sd: Dict[str, Any] = {
            "delivery_buffer": self.delivery_buffer,
            "current_time_step": self.current_time_step,
            "global_step": self.global_step,
        }

        # Threshold-crossing history matters for spike detection when thresholds are used.
        if not self.skip_thresholding:
            sd["has_spiked"] = self.has_spiked

        return sd

    def restore_dict_from_checkpoint(self, state_dict):
        """
        Restore the state of this NetCon from a checkpoint dictionary created by
        state_dict_for_checkpoint().

        Important checkpointing detail:
        - delivery_buffer must be rebound directly (to preserve gradient flow).
        - non-differentiable tensors that NetCon mutates in-place should be cloned
        here to avoid in-place mutation of checkpoint *inputs* inside a chunk.

        This matches the expectation in longrun_checkpointed that restore rebinds
        mutable tensors.
        """
        # IMPORTANT:
        # Restore by *rebinding* tensors, not copy_(), so that:
        # - we preserve autograd history through the state tensors, and
        # - we don't inadvertently sever BPTT across chunk boundaries.
        self.delivery_buffer = state_dict["delivery_buffer"]
        self.current_time_step = state_dict["current_time_step"]
        self.global_step = state_dict["global_step"]

        if not self.skip_thresholding and "has_spiked" in state_dict:
            self.has_spiked = state_dict["has_spiked"]
        return self
