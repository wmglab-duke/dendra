"""Fit an ordinary Dendra sine wave using cycles per reference duration.

Run from the repository root:
    python examples/scaled_waveform_frequency.py

The same parametrized waveform can be passed to model[slice].inject(waveform).
No functional Population API is needed. The reference duration is fixed during
fitting; it changes the optimizer's coordinate, not the physical waveform.
"""

import math

import torch
from torch.nn.utils import parametrize

import dendra as dn
from dendra.units import Hz


class FrequencyFromCycles(torch.nn.Module):
    """Expose physical kHz while storing cycles per reference duration in ms."""

    def __init__(self, reference_ms):
        super().__init__()
        if not math.isfinite(reference_ms) or reference_ms <= 0:
            raise ValueError("reference_ms must be finite and positive")
        self.register_buffer(
            "reference_ms", torch.tensor(reference_ms, dtype=torch.float64)
        )

    def forward(self, cycles):
        return cycles / self.reference_ms

    def right_inverse(self, frequency):
        return frequency * self.reference_ms


def main():
    reference_ms = 1000.0
    initial_hz, target_hz = 19.8, 20.0
    waveform = dn.sin(freq=torch.tensor(initial_hz * Hz, dtype=torch.float64))
    waveform = waveform.to(dtype=torch.float64)
    transform = FrequencyFromCycles(reference_ms).to(waveform.freq)
    parametrize.register_parametrization(waveform, "freq", transform)

    # right_inverse preserved the starting frequency while changing its stored
    # coordinate from 0.0198 kHz to 19.8 cycles per 1000 ms.
    waveform.requires_grad_(False)
    cycles = waveform.parametrizations.freq.original
    cycles.requires_grad_(True)
    optimizer = torch.optim.Adam([cycles], lr=0.01)

    time = torch.linspace(0.0, reference_ms, 1001, dtype=torch.float64)
    target = torch.sin(2 * torch.pi * (target_hz * Hz) * time)
    initial_loss = (waveform(time) - target).square().mean().detach()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = (waveform(time) - target).square().mean()
        loss.backward()
        optimizer.step()

    final_loss = (waveform(time) - target).square().mean().detach()
    fitted_hz = waveform.freq.detach().item() / Hz
    assert torch.isfinite(final_loss) and final_loss < initial_loss
    assert abs(fitted_hz - target_hz) < 1e-3
    print(f"Frequency: {initial_hz:.4f} -> {fitted_hz:.6f} Hz (target {target_hz})")
    print(f"Stored cycles: {cycles.detach().item():.6f}")
    print(f"Mean squared error: {initial_loss.item():.6g} -> {final_loss.item():.6g}")


if __name__ == "__main__":
    main()
