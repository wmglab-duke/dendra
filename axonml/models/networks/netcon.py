from typing import Tuple

import torch

from ..parametric import Referency
from .netstim import NetStim
from .utils import make_getattr


def update_active(has_spiked, vm_new, threshold) -> Tuple[torch.Tensor, torch.Tensor]:
    ge = vm_new >= threshold
    spiked = torch.logical_and(ge, ~has_spiked)
    return ge, spiked


def update_active_diff(
    has_spiked: torch.Tensor,  # previous "ge" (bool)
    vm_new: torch.Tensor,  # float
    threshold: torch.Tensor,  # float
    tau: float = 0.1,  # temperature for the surrogate
):
    """
    Returns:
      ge_hard:     bool  (vm_new >= threshold)
      spiked_hard: bool  (rising edge: ge & ~has_spiked)
      ge_gate:     float in [0,1] with STE (forward==ge_hard, backward==sigmoid)
      spk_gate:    float in [0,1] with STE for *rising edge*
    """
    x = (vm_new - threshold) / tau
    s = torch.sigmoid(x)  # smooth "is-above-threshold"

    ge_hard = vm_new >= threshold  # bool
    spiked_hard = ge_hard & (~has_spiked)  # bool rising edge

    # Straight-through gates:
    # - ge_gate forward equals ge_hard; backward follows s
    ge_gate = ge_hard.to(s.dtype) + (s - s.detach())

    # - rising-edge gate: soft approx is s * (1 - has_spiked)
    s_rise = s * (1.0 - has_spiked.to(s.dtype))
    spk_gate = spiked_hard.to(s.dtype) + (s_rise - s_rise.detach())

    return ge_gate, spk_gate


class NetCon(Referency):
    """
    `NetCon`s are responsible for managing all synaptic connections.
    They are generated automatically by instances of axonml.Network
    when `net.build()` is called.
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
    ):
        super().__init__()

        self.weight = weight
        self.syn = post_syn
        self.pre = pre
        self.post = post
        self.device = self.pre.device()
        self.dtype = self.pre.dtype()
        self.dt = torch.tensor(dt, device=self.device, dtype=torch.float32)

        if pre_var is None:
            pre_var = "v"

        self.get_pre_var = make_getattr(pre_var)

        if isinstance(pre, NetStim):
            self.determine_spiking = self.determine_spiking_ns
        else:
            self.determine_spiking = self.determine_spiking_var

        self.register_buffer(
            "pre_idx", pre_idx.flatten().to(self.device, dtype=torch.long)
        )
        self.register_buffer(
            "post_idx", post_idx.flatten().to(self.device, dtype=torch.long)
        )
        self.register_buffer(
            "threshold", thresholds.flatten().to(self.device, dtype=self.dtype)
        )

        self.register_buffer(
            "syn_numel",
            torch.prod(torch.tensor(self.syn.shape_f)).to(
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
            torch.arange(self.n.item(), device=self.device, dtype=torch.long),
        )
        n_pre = self.n.item()
        assert len(self.threshold) == n_pre, (
            "Thresholds must match the number of pre synaptic locations."
        )
        assert delay.shape[0] == self.pre_idx.shape[0], (
            "Delay tensor shape must match pre_idx."
        )

        self.register_buffer(
            "has_spiked", torch.zeros(n_pre, device=self.device, dtype=torch.bool)
        )
        self.register_buffer(
            "is_spiking", torch.zeros(n_pre, device=self.device, dtype=torch.bool)
        )

        # --- Delay Handling Logic (Compiler-Friendly) ---
        delay_steps = (delay / dt).round().long()
        self.register_buffer("delay_steps", delay_steps.flatten().to(self.device))

        self.max_delay_steps = (
            int(self.delay_steps.max().item()) + 1 if len(self.delay_steps) > 0 else 1
        )

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

    def set_diff_config(
        self,
        diff_weights: bool = True,
        diff_delays: bool = True,
        diff_spiking: bool = True,
        taps: int = 2,
        sigma: float = 0.35,  # used for taps=3 (in steps),
        tau: float = 0.1,  # temperature for surrogate spiking
    ):
        self.train_flags = (
            diff_weights,
            diff_delays,
            diff_spiking,
            taps,
            sigma,
            tau,
        )

    @property
    def w(self):
        """
        Returns the weight tensor, which is a parameter of the synapse.
        This is useful for accessing the synaptic weights directly.
        """
        return self.weight.w

    def _expand_pre_to_con(self, pre_indices: torch.Tensor | list[int]) -> torch.Tensor:
        # NOTE: schedule-time helper; not called inside compiled step.
        pres = torch.as_tensor(pre_indices, device=self.device, dtype=torch.long).view(
            -1
        )
        if pres.numel() == 0:
            return pres

        # Find each pre id in CSR unique list with searchsorted
        pos = torch.searchsorted(self._csr_pre_ids, pres)
        valid = (pos < self._csr_pre_ids.numel()) & (
            self._csr_pre_ids.index_select(0, pos) == pres
        )
        if not bool(valid.any()):
            return torch.empty(0, device=self.device, dtype=torch.long)

        pos = pos[valid]
        starts = self._csr_starts.index_select(0, pos)
        lens = self._csr_counts.index_select(0, pos)

        # Gather slices of the sorted connection-index array and concat
        con_chunks = []
        for s, L in zip(starts.tolist(), lens.tolist()):
            if L > 0:
                con_chunks.append(self._csr_conidx_sorted.narrow(0, s, L))
        if len(con_chunks) == 0:
            return torch.empty(0, device=self.device, dtype=torch.long)
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
        Schedule VALUE-MODE events. Provide exactly one of (con_indices, pre_indices).
        'weight' scales the NetCon's existing per-connection weight for that event.
        """
        if (con_indices is None) == (pre_indices is None):
            raise ValueError("Provide exactly one of con_indices or pre_indices")

        if times_ms is None:
            raise ValueError("times_ms is required")

        device, dtype = self.device, self.dtype
        # -> connection indices
        if pre_indices is not None:
            con_idx = self._expand_pre_to_con(pre_indices)
        else:
            con_idx = torch.as_tensor(
                con_indices, device=device, dtype=torch.long
            ).view(-1)

        tms = torch.as_tensor(times_ms, device=device, dtype=torch.float32).view(-1)
        if con_idx.numel() != tms.numel():
            raise ValueError("con_indices/pre_indices and times_ms must match length")

        if (con_idx < 0).any() or (con_idx >= self.pre_idx.numel()).any():
            raise IndexError("connection index out of range")

        steps = torch.round(tms / self.dt.to(tms.dtype)).to(torch.long)
        if not allow_past:
            cur = self.global_step.view(())
            keep = steps >= cur
            con_idx, steps = con_idx[keep], steps[keep]
            if con_idx.numel() == 0:
                return

        # weights → tensor, keep autograd to producers
        if torch.is_tensor(weight):
            w = weight.to(device=device, dtype=dtype).view(-1)
            if w.numel() not in (1, con_idx.numel()):
                raise ValueError(
                    "weight must be scalar or same length as indices/times"
                )
            if w.numel() == 1:
                w = w.expand_as(con_idx)
        else:
            w = torch.full(
                (con_idx.numel(),), float(weight), device=device, dtype=dtype
            )

        # append (schedule happens outside compiled loop, changing E is fine here)
        self.sched_con_idx = torch.cat([self.sched_con_idx, con_idx], dim=0)
        self.sched_abs_step = torch.cat([self.sched_abs_step, steps], dim=0)
        self.sched_weight = torch.cat([self.sched_weight, w], dim=0)
        self.sched_weight_idx = torch.cat(
            [
                self.sched_weight_idx,
                torch.full((con_idx.numel(),), -1, device=device, dtype=torch.long),
            ],
            dim=0,
        )

    def bind_weight_source(self, source: torch.Tensor):
        if source.device != self.device:
            raise ValueError("weight source must be on same device")
        self._sched_w_source = source

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
        Schedule REFERENCE-MODE events. Provide weight indices into a bound source.
        """
        if self._sched_w_source is None:
            raise RuntimeError("call bind_weight_source(...) before schedule_ref(...)")
        if (con_indices is None) == (pre_indices is None):
            raise ValueError("Provide exactly one of con_indices or pre_indices")
        if times_ms is None or weight_idx is None:
            raise ValueError("times_ms and weight_idx are required")

        device, dtype = self.device, self.dtype
        if pre_indices is not None:
            con_idx = self._expand_pre_to_con(pre_indices)
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

    def clear_schedule(self):
        device, dtype = self.device, self.dtype
        self.sched_con_idx = torch.empty(0, device=device, dtype=torch.long)
        self.sched_abs_step = torch.empty(0, device=device, dtype=torch.long)
        self.sched_weight = torch.empty(0, device=device, dtype=dtype)
        self.sched_weight_idx = torch.empty(0, device=device, dtype=torch.long)
        self.sched_wsum.zero_()
        self.sched_counts.zero_()

    def _scheduled_gate_this_step(self, gs_long: torch.Tensor):
        """
        Build per-connection scheduled amplitude for the current absolute step.
        All shapes are fixed; no boolean indexing or shape-changing ops.
        Returns:
            sched_wsum_conn: [n_conn] float
            sched_count_conn: [n_conn] int32
        """
        n_conn = self._n_conn
        device, dtype = self.device, self.dtype

        if self.sched_con_idx.numel() == 0:
            return (
                torch.zeros(n_conn, device=device, dtype=dtype),
                torch.zeros(n_conn, device=device, dtype=torch.int32),
            )

        # fixed-length mask for "fires now"
        now_mask = self.sched_abs_step == gs_long.view(())  # [E]
        now_f = now_mask.to(dtype)  # [E] float 0/1

        # value-mode part
        w_val = self.sched_weight * now_f  # [E]

        # reference-mode part (no boolean filtering)
        if self._sched_w_source is not None:
            idx_clamped = torch.clamp(self.sched_weight_idx, min=0)  # [E]
            w_src = self._sched_w_source.index_select(0, idx_clamped)  # [E]
            ref_valid = (self.sched_weight_idx >= 0).to(dtype)  # [E]
            w_ref = w_src * now_f * ref_valid  # [E]
            w_e = w_val + w_ref
        else:
            w_e = w_val

        # accumulate to connections (constant target size)
        sched_wsum_conn = torch.zeros(n_conn, device=device, dtype=dtype)
        sched_count_conn = torch.zeros(n_conn, device=device, dtype=torch.int32)
        sched_wsum_conn.index_add_(0, self.sched_con_idx, w_e)
        sched_count_conn.index_add_(0, self.sched_con_idx, now_mask.to(torch.int32))
        return sched_wsum_conn, sched_count_conn

    def advance_diff(self):
        if self.train_flags is None:
            raise RuntimeError(
                "NetCon.train(...) must be called before advance_diff()."
            )
        diff_weights, diff_delays, diff_spiking, taps, sigma, tau = self.train_flags
        device, dtype = self.device, self.dtype

        # snapshot indices for this step (avoid version bumps)
        cur_idx = self.current_time_step.detach().clone()  # [1], long
        gs = self.global_step.detach().clone()  # [1], long

        # 1) deliver today's payload
        todays = self.delivery_buffer.index_select(0, cur_idx)  # [1, n_syn]
        self.events = self.event_queue.index_select(0, cur_idx).squeeze(0)  # [n_conn]
        self.syn.net_receive(todays.squeeze(0).view(*self.syn.shape_f), self)

        # 2) intrinsic spiking
        self.determine_spiking(
            self.pre, diff_spiking, tau
        )  # self.is_spiking -> [n_conn] bool
        intrinsic_gate = self.is_spiking.to(dtype)  # [n_conn]

        # scheduled additions per connection (constant shapes)
        sched_wsum_conn, sched_counts_conn = self._scheduled_gate_this_step(gs)

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

        functional_update = diff_weights or diff_delays
        buf_flat = self.delivery_buffer.view(-1)

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

                contrib = torch.zeros_like(buf_flat)
                contrib.index_add_(0, flat0, weighted_spikes * (1.0 - alpha))
                contrib.index_add_(0, flat1, weighted_spikes * alpha)

                flat_e = (cur_idx + kL).remainder(
                    self.max_delay_steps
                ) * self.n + self.con_range
                self.event_queue.view(-1).index_add_(
                    0,
                    flat_e,
                    (intrinsic_gate > 0).to(self.sched_counts.dtype)
                    + sched_counts_conn,
                )
            else:
                offs = torch.stack([kL - 1, kL, kL + 1], dim=-1)
                centers = offs.to(dtype)
                lam_e = lam.unsqueeze(-1)
                w = torch.softmax(
                    -0.5 * ((lam_e - centers) / (sigma + 1e-6)) ** 2, dim=-1
                )

                idxs = (cur_idx + offs).remainder(self.max_delay_steps)
                contrib = torch.zeros_like(buf_flat)
                for j in range(3):
                    flatj = idxs[:, j] * self.syn_numel + self.post_idx
                    contrib.index_add_(0, flatj, weighted_spikes * w[:, j])

                flat_e = (cur_idx + kL).remainder(
                    self.max_delay_steps
                ) * self.n + self.con_range
                self.event_queue.view(-1).index_add_(
                    0,
                    flat_e,
                    (intrinsic_gate > 0).to(self.sched_counts.dtype)
                    + sched_counts_conn,
                )
        else:
            future_steps = (cur_idx + self.delay_steps).remainder(self.max_delay_steps)
            flat = future_steps * self.syn_numel + self.post_idx

            if functional_update:
                contrib = torch.zeros_like(buf_flat)
                contrib.index_add_(0, flat, weighted_spikes)
                flat_e = future_steps * self.n + self.con_range
                self.event_queue.view(-1).index_add_(
                    0,
                    flat_e,
                    (intrinsic_gate > 0).to(self.sched_counts.dtype)
                    + sched_counts_conn,
                )
            else:
                # (kept for completeness; normally you won't hit this in training)
                self.delivery_buffer.index_fill_(0, cur_idx, 0.0)
                self.event_queue.index_fill_(0, cur_idx, 0)
                self.delivery_buffer.view(-1).index_add_(0, flat, weighted_spikes)
                flat_e = future_steps * self.n + self.con_range
                self.event_queue.view(-1).index_add_(
                    0,
                    flat_e,
                    (intrinsic_gate > 0).to(self.sched_counts.dtype)
                    + sched_counts_conn,
                )
                self.current_time_step.add_(1).remainder_(self.max_delay_steps)
                return

        # commit masked update
        mask1d = torch.ones(self.max_delay_steps, device=device, dtype=dtype)
        mask1d.index_fill_(0, cur_idx, 0.0)
        mask2d = mask1d.view(-1, 1).expand(-1, int(self._syn_numel)).reshape(-1)
        self.delivery_buffer = (buf_flat * mask2d + contrib).view_as(
            self.delivery_buffer
        )

        # advance counters (no grad)
        with torch.no_grad():
            new_cur = (self.current_time_step + 1).remainder(self.max_delay_steps)
            new_gs = self.global_step + 1
            # Rebind buffers (out-of-place) to avoid version bumps on saved tensors
            self.current_time_step = new_cur.detach()
            self.global_step = new_gs.detach()

    def advance_non_diff(self):
        cur_idx = self.current_time_step
        todays_delivery = self.delivery_buffer.index_select(0, cur_idx)
        self.events = self.event_queue.index_select(0, cur_idx).squeeze(0)
        self.syn.net_receive(todays_delivery.squeeze(0).view(*self.syn.shape_f), self)

        # clear current row
        self.delivery_buffer.index_fill_(0, cur_idx, 0.0)
        self.event_queue.index_fill_(0, cur_idx, 0)

        # intrinsic spikes (hard)
        self.determine_spiking(self.pre, diff_spiking=False)
        intrinsic_gate = self.is_spiking.to(self.dtype)  # [n_conn]

        # scheduled contributions (constant shape)
        gs = self.global_step
        sched_wsum_conn, sched_counts_conn = self._scheduled_gate_this_step(gs)
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
        # If the pre-synaptic source is a NetStim, we can directly use its spikes
        # (or spikes_gate for differentiable spiking)
        if diff_spiking:
            self.is_spiking = pre.spike_gate.view(-1).index_select(0, self.pre_idx)
        else:
            self.is_spiking = pre.spikes.view(-1).index_select(0, self.pre_idx)

    def determine_spiking_var(self, pre, diff_spiking: bool = True, tau=0.1):
        # Otherwise, we need to compute spiking based on the pre-synaptic membrane potential
        v_selected = self.get_pre_var(pre).view(-1).index_select(0, self.pre_idx)

        if diff_spiking:
            self.has_spiked, self.is_spiking = update_active_diff(
                self.has_spiked, v_selected, self.threshold, tau=tau
            )
        else:
            self.has_spiked, self.is_spiking = update_active(
                self.has_spiked, v_selected, self.threshold
            )

    def zero(self, clear_delivery_buffers=True):
        """
        Reset the delivery buffer and current time step.
        This is useful for re-initializing the module.
        """
        self.current_time_step.fill_(0)
        if clear_delivery_buffers:
            self.delivery_buffer.zero_()
            self.has_spiked.fill_(False)
            self.is_spiking.fill_(False)

    def detach(self):
        """
        Detach the module from the current computation graph.
        This is useful for inference or when you want to stop tracking gradients.
        """
        for n, b in self.named_buffers():
            setattr(self, n, b.detach())

    def initialize(self, reinit_weights=True, clear_deliveries=True):
        if self.training:
            self.advance = self.advance_diff
        else:
            self.advance = self.advance_non_diff
        self.zero(clear_delivery_buffers=clear_deliveries)
        self.weight.init(reinit=reinit_weights)
        with torch.no_grad():
            self.global_step.fill_(int(round(float(self.t) / float(self.dt))))
        self.detach()

    def numel(self):
        """
        Returns the number of synaptic connections managed by this NetCon.
        This is useful for understanding the scale of the network.
        """
        return int(self.n.item())
