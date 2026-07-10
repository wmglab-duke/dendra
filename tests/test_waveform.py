import torch

from dendra.models.stim.waveform import mono_rect


def test_repeat_accepts_vector_frequency_sweep():
    t = torch.tensor([0.0, 0.1, 0.3, 0.55, 0.8, 1.05])
    waveform = mono_rect(amp=1.0, pw=0.2, tau=0.01).repeat(
        torch.tensor([1.0, 2.0]), off=1.2
    )

    out = waveform(t)

    expected = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ]
    )
    assert out.shape == (2, t.numel())
    torch.testing.assert_close(out, expected)


def test_repeat_accepts_matrix_frequency_sweep():
    t = torch.tensor([0.0, 0.05, 0.10, 0.27, 0.34, 0.53])
    freq = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    waveform = mono_rect(amp=1.0, pw=0.08, tau=0.01).repeat(freq, off=1.0)

    out = waveform(t)

    period = (1.0 / freq).unsqueeze(-1)
    expected = (torch.fmod(t, period) < 0.08).to(t.dtype)
    assert out.shape == (2, 2, t.numel())
    torch.testing.assert_close(out, expected)


def test_repeat_controls_move_with_waveform_module():
    waveform = mono_rect(amp=1.0, pw=0.2).repeat(torch.tensor([1.0, 2.0]))
    waveform = waveform.to(dtype=torch.float64)

    assert waveform.freq.dtype == torch.float64
    assert waveform.delay.dtype == torch.float64
    assert waveform.off.dtype == torch.float64
    assert waveform(torch.tensor([0.1], dtype=torch.float64)).dtype == torch.float64
