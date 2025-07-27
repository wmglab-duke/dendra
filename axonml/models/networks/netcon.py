from typing import Tuple

import torch


def update_active(has_spiked, vm_new, threshold) -> Tuple[torch.Tensor, torch.Tensor]:
    ge = vm_new >= threshold
    spiked = torch.logical_and(ge, ~has_spiked)
    return ge, spiked


class NetCon(torch.nn.Module):
    def __init__(
        self, pre, pre_idx, thresholds, post, post_idx, post_syn, weight, delay, dt
    ):
        super().__init__()

        self.weight = weight
        self.syn = post_syn
        self.pre = pre
        self.post = post
        self.device = self.pre.device()
        self.dtype = self.pre.dtype()

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
            torch.prod(torch.tensor(self.syn.shape)).to(self.device, dtype=torch.long),
        )

        self.register_buffer(
            "n",
            torch.tensor(self.pre.v.view(-1).index_select(0, self.pre_idx).shape[0]).to(
                self.device, dtype=torch.long
            ),
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
                (self.max_delay_steps, n_pre), device=self.device, dtype=torch.long
            ),
        )
        self.register_buffer(
            "events", torch.zeros(n_pre, device=self.device, dtype=torch.long)
        )

        self.register_buffer(
            "current_time_step", torch.tensor(0, device=self.device, dtype=torch.long)
        )

        # --- Pre-computed tensor for masking, avoids creating tensors in the loop ---
        self.register_buffer(
            "time_indices", torch.arange(self.max_delay_steps, device=self.device)
        )

    @property
    def w(self):
        """
        Returns the weight tensor, which is a parameter of the synapse.
        This is useful for accessing the synaptic weights directly.
        """
        return self.weight.w

    def advance(self):
        # 1. DELIVER:
        # Select the single row. The result has shape [1, num_synapses]
        todays_delivery = self.delivery_buffer.index_select(0, self.current_time_step)
        self.events = self.event_queue.index_select(0, self.current_time_step).squeeze(
            0
        )

        # Unconditionally deliver the payload. If it's all zeros, this has no effect.
        # .squeeze(0) removes the dimension of size 1, matching the synapse shape.
        self.syn.net_receive(todays_delivery.squeeze(0).view(*self.syn.shape), self)

        # Unconditionally clear the buffer row using the mask.
        # This is more compiler-friendly than an in-place `zero_()` on a slice.
        # We need to expand the mask to match the shape of the delivery_buffer for masked_fill_
        self.delivery_buffer.index_fill_(0, self.current_time_step, 0.0)
        self.event_queue.index_fill_(0, self.current_time_step, False)

        # 2. SPIKE & SCHEDULE: Check for new spikes and schedule their future delivery
        v_selected = self.pre.v.view(-1).index_select(0, self.pre_idx)
        self.has_spiked, self.is_spiking = update_active(
            self.has_spiked, v_selected, self.threshold
        )

        # Get the weights of the connections that are currently spiking
        # The float conversion is essential for the multiplication
        weighted_spikes = self.weight() * self.is_spiking.to(self.dtype)

        # Calculate future delivery indices
        future_delivery_steps = self.current_time_step + self.delay_steps
        future_buffer_indices = future_delivery_steps.remainder(self.max_delay_steps)

        # Flatten the buffer and use `index_add_` for an efficient, sparse, and
        # unconditional update. This avoids unnecessary graph breaks.
        flat_indices = future_buffer_indices * self.syn_numel + self.post_idx
        self.delivery_buffer.view(-1).index_add_(0, flat_indices, weighted_spikes)

        flat_indices = future_buffer_indices * self.n + self.pre_idx
        # Update the event buffer to mark where spikes occurred
        self.event_queue.view(-1).index_add_(
            0, flat_indices, self.is_spiking.to(torch.long)
        )

        # 3. INCREMENT TIME: Move to the next time step
        self.current_time_step.add_(1).remainder_(self.max_delay_steps)

    def zero(self):
        """
        Reset the delivery buffer and current time step.
        This is useful for re-initializing the module.
        """
        self.delivery_buffer.zero_()
        self.current_time_step.fill_(0)
        self.has_spiked.fill_(False)
        self.is_spiking.fill_(False)

    def detach(self):
        """
        Detach the module from the current computation graph.
        This is useful for inference or when you want to stop tracking gradients.
        """
        for n, b in self.named_buffers():
            setattr(self, n, b.detach())
