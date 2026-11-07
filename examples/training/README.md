# Training and analysis

Fit model or stimulation parameters with PyTorch optimizers, choose trainable
sources, or differentiate a macroscopic response. Native `run` and functional
execution both support gradients; these examples show when explicit sources
and continuation state are useful.

Run from the repository root:

```bash
python examples/training/scaled_waveform_frequency.py
python examples/training/functional_gradient_descent.py --iterations 5 --tstop 0.5
```

The first fits a sine wave's frequency without a neuronal model. The second
fits HH conductances to a voltage response using `torch.func.grad_and_value`.
For longer fits, `--compiled-step` compiles the numerical step so compilation
can be amortized across optimizer iterations. `--checkpointed` demonstrates
activation checkpointing.

## Choose a workflow

| Example | What it demonstrates |
| --- | --- |
| [`functional_gradient_descent.py`](functional_gradient_descent.py) | Fit HH conductances with a pure objective and a callback that accumulates MSE without retaining every candidate trace. |
| [`fit_sinusoid_voltage.py`](fit_sinusoid_voltage.py) | Fit an injected sine's frequency and timing from native voltage recordings, with optional observation noise and diagnostic plots. |
| [`scaled_waveform_frequency.py`](scaled_waveform_frequency.py) | Use PyTorch parametrization to optimize cycles over a reference interval while exposing physical waveform frequency. |
| [`functional_parameter_selection.py`](functional_parameter_selection.py) | Select trainable sources with glob and regex expressions and inspect the selection. |
| [`functional_source_selection.py`](functional_source_selection.py) | Build optimizer groups or separate optimizers from deduplicated source reports. |
| [`native_source_inspection.py`](native_source_inspection.py) | Inspect native model sources and their aliases without lowering the model. |
| [`functional_raw_bindings.py`](functional_raw_bindings.py) | Reuse raw Population and Network sources across optimizer iterations. |
| [`functional_continuation_detach.py`](functional_continuation_detach.py) | Train successive functional segments while keeping optimizer-owned source tensors. |
| [`shared_callback_state_detach.py`](shared_callback_state_detach.py) | Bound native Population gradient history while continuing pure callbacks. |
| [`shared_network_detach.py`](shared_network_detach.py) | Apply the same segment boundary to a connected native Network. |
| [`functional_thresholding.py`](functional_thresholding.py) | Find activation thresholds with native and compiled Populations. |
| [`macroscopic_descriptor_training.py`](macroscopic_descriptor_training.py) | Optimize repeated HH firing using a firing-frequency descriptor instead of a target trace. |
| [`threshold_descriptor_gradients.py`](threshold_descriptor_gradients.py) | Differentiate activation threshold and chronaxie, then check slopes against hard finite differences. |
| [`distillation.py`](distillation.py) | Reusable helpers for fitting a cable model to extracellular teacher responses; import the helpers rather than expecting a command-line demo. |

## Fit stimulation from recorded voltage

A short passive-membrane fit with observation noise and saved plots:

```bash
python examples/training/fit_sinusoid_voltage.py \
    --iterations 5 --noise-std-mv 0.1 --noise-seed 0 --output-dir voltage-fit
```

For the full active HH example:

```bash
python examples/training/fit_sinusoid_voltage.py \
    --membrane hh --noise-std-mv 0.5 --noise-seed 0 --output-dir voltage-hh
```

Noise defaults to zero. A fixed noisy target is reused across optimizer
iterations and starting guesses. The known cable, membrane parameters,
initial voltage, current amplitude and phase stay fixed while frequency,
delay and cutoff are fitted. Use `--help` to change starting guesses,
recorded compartments, the surrogate width `--tau`, and optimization length.
`--no-jit` disables JIT for debugging.

Keep `tau` wide enough relative to `dt` for nearby samples to receive timing
gradients. Cutoffs within one sampling interval can produce identical voltage
traces; multiple starts and wider surrogates can help explore alternatives.
The fit can settle in a local minimum, so compare a noiseless fit before
attributing parameter error to observation noise. `--output-dir` saves
`summary.json`, `traces.npz`, `voltage_fit.png` and `parameter_fit.png`, including
observed and fitted traces, residuals and parameter trajectories.

## Optimize response descriptors

For a short firing-frequency optimization:

```bash
python examples/training/macroscopic_descriptor_training.py \
    --updates 1 --tstop-ms 60 --objective-start-ms 20 --skip-holdout --no-jit
```

The full default run also evaluates the final parameters in a separate holdout
window. The threshold/chronaxie example runs hard searches at four pulse widths,
constructs voltage/amplitude tangents from a batched replay, and checks the
resulting slopes against independent hard searches:

```bash
python examples/training/threshold_descriptor_gradients.py
```

See the [training and parameters guide](../../docs/advanced/training_and_parameters.md),
the [descriptor training guide](../../docs/advanced/descriptor_training.rst), and
[continuation examples](../continuation/README.md) for the underlying workflows.

[All examples](../README.md)
