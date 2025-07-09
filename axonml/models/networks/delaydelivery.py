from typing import Tuple

import torch
import numpy as np


def update_active(has_spiked, vm_new, threshold) -> Tuple[torch.Tensor, torch.Tensor]:
    ge = vm_new >= threshold
    spiked = torch.logical_and(ge, ~has_spiked)
    return ~ge, spiked


class VariableDelayDelivery(torch.nn.Module):
    def __init__(
        self,
        pre, pre_idx, thresholds,
        post, post_idx, post_syn,
        weight, delay, dt
    ):
        super().__init__()
        
        self.weight = weight
        self.syn = post_syn
        self.pre = pre
        self.post = post
        self.device = self.pre.device()
        self.dtype = self.pre.dtype()

        self.register_buffer("pre_idx", pre_idx.flatten().to(self.device, dtype=torch.long))
        self.register_buffer("post_idx", post_idx.flatten().to(self.device, dtype=torch.long))
        self.register_buffer("threshold", thresholds.flatten().to(self.device, dtype=self.dtype))

        self.register_buffer(
            "syn_numel", 
            torch.prod(torch.tensor(self.syn.shape)).to(self.device, dtype=torch.long)
        )

        n_pre = self.pre.v.view(-1).index_select(0, self.pre_idx).shape[0]
        assert len(self.threshold) == n_pre, "Thresholds must match the number of pre synaptic locations."
        assert delay.shape[0] == self.pre_idx.shape[0], "Delay tensor shape must match pre_idx."

        self.register_buffer("has_spiked", torch.zeros(n_pre, device=self.device, dtype=torch.bool))
        
        # --- Delay Handling Logic (Compiler-Friendly) ---
        delay_steps = (delay / dt).round().long()
        self.register_buffer("delay_steps", delay_steps.flatten().to(self.device))

        self.max_delay_steps = int(self.delay_steps.max().item()) + 1 if len(self.delay_steps) > 0 else 1
        
        buffer_shape = (self.max_delay_steps, self.syn_numel.item())
        self.register_buffer("delivery_buffer", torch.zeros(buffer_shape, device=self.device, dtype=self.dtype))

        self.register_buffer("current_time_step", torch.tensor(0, device=self.device, dtype=torch.long))
        
        # --- Pre-computed tensor for masking, avoids creating tensors in the loop ---
        self.register_buffer("time_indices", torch.arange(self.max_delay_steps, device=self.device))


    def advance(self):
        # 1. DELIVER: Use a boolean mask to find and deliver scheduled events
        # This avoids dynamic indexing
        current_buffer_idx = self.current_time_step % self.max_delay_steps
        delivery_mask = (self.time_indices == current_buffer_idx) # [False, ..., True, ..., False]

        # Select the single row using the boolean mask. Note the result has shape [1, num_synapses]
        todays_delivery = self.delivery_buffer[delivery_mask, :]
        
        # Unconditionally deliver the payload. If it's all zeros, this has no effect.
        # .squeeze(0) removes the dimension of size 1, matching the synapse shape.
        self.syn.net_receive(todays_delivery.squeeze(0).view(*self.syn.shape))
        
        # Unconditionally clear the buffer row using the mask.
        # This is more compiler-friendly than an in-place `zero_()` on a slice.
        # We need to expand the mask to match the shape of the delivery_buffer for masked_fill_
        self.delivery_buffer.masked_fill_(delivery_mask.unsqueeze(1), 0.0)

        # 2. SPIKE & SCHEDULE: Check for new spikes and schedule their future delivery
        v_selected = self.pre.v.view(-1).index_select(0, self.pre_idx)
        self.has_spiked, is_spiking = update_active(self.has_spiked, v_selected, self.threshold)
        
        # Get the weights of the connections that are currently spiking
        # The float conversion is essential for the multiplication
        weighted_spikes = self.weight() * is_spiking.float()
        
        # Calculate future delivery indices
        future_delivery_steps = self.current_time_step + self.delay_steps
        future_buffer_indices = future_delivery_steps % self.max_delay_steps

        # Flatten the buffer and use `index_add_` for an efficient, sparse, and
        # unconditional update. This avoids the `if is_spiking.any():` graph break.
        flat_indices = future_buffer_indices * self.syn_numel + self.post_idx
        self.delivery_buffer.view(-1).index_add_(0, flat_indices, weighted_spikes)

        # 3. INCREMENT TIME: Move to the next time step
        self.current_time_step += 1
