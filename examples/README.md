# Examples

Network implementation examples in ./net/

[`fit_sinusoid_voltage.py`](fit_sinusoid_voltage.py) fits an injected sine's
frequency, delay, and duration from simulated voltage recordings, optionally
with observation noise, using ordinary `model.run()` and autograd. Try a
passive membrane with 0.1 mV noise, or active Hodgkin–Huxley channels with
0.5 mV noise:

```bash
python examples/fit_sinusoid_voltage.py --noise-std-mv 0.1 --noise-seed 0 --output-dir voltage-fit
python examples/fit_sinusoid_voltage.py --membrane hh --noise-std-mv 0.5 --noise-seed 0 --output-dir voltage-hh
```

Noise defaults to zero. Gaussian noise is added independently at each recorded
time and compartment, once; that recording is reused for every iteration and
starting guess. The loss uses observed voltage MSE, normalized by the
observed response energy. The best iteration and start are selected using
that loss alone; clean-reference RMSE is calculated afterward for evaluation.

The reference and fitted simulations use the same known cable, membrane
parameters, initial voltage (-65 mV), current amplitude, and phase
(0.4 radians). The five-compartment cable is 250 µm long and 2 µm in diameter;
compartments 0, 2, and 3 are recorded by default. The passive defaults are
0.05 nA, `dt=0.125` ms, and `tau=0.25` ms. `--membrane hh` uses fixed sodium,
potassium, and leak channels at 6.3°C, with 0.2 nA, `dt=0.0625` ms, and
`tau=0.5` ms; its reference trace includes an action potential. Change the
known amplitude with `--amplitude-na`; it is not fitted. JIT is enabled by
default; `--no-jit` disables it for debugging.

The default run uses 180 optimization steps for each of two initial cutoff
guesses. Use `--cutoff off` to fit an absolute cutoff instead of `off_after`,
and `--initial-stop-ms 10.25 12.0` to specify initial absolute stop times in ms,
including when fitting `off_after`; one or more values are accepted.
`--initial-frequency-hz` and `--initial-delay-ms` set the other starting values.
`--iterations`, `--dt`, `--tau`, and `--record-nodes` control optimization
length, time resolution, surrogate width, and observed compartments.
Updates keep frequency between 0.001 Hz and the sampling Nyquist limit,
and stimulation within the 16 ms recording window for at least one time step.

Keep `tau` wide enough relative to `dt` for nearby samples to receive timing
gradients; the applied current's edges stay abrupt. Cutoffs within the same
sampling interval can produce identical voltage traces, so interpret the fitted
cutoff at that resolution. Multiple starts and a wider surrogate, such as
`--tau 0.5` for the passive case, can help explore other intervals. Optimization
can still settle in a local minimum, and these synthetic fits do not guarantee
parameter recovery from noisy recordings or a different membrane model.
Compare with `--noise-std-mv 0` before attributing parameter error to noise: a
noiseless fit can also retain parameter error.

`--output-dir` saves `summary.json`, `traces.npz`, `voltage_fit.png`, and
`parameter_fit.png`. The arrays include clean reference, observed, initial,
and fitted voltages, plus observed-minus-fitted residuals. The plots show those
traces, residuals, and parameter trajectories for comparing the starts.

[`functional_gradient_descent.py`](functional_gradient_descent.py) is the
explicit-state functional counterpart to the
[`04_gradient_descent` tutorial](../docs/basics/04_gradient_descent.ipynb).
Its default path uses `torch.func.grad_and_value` and demonstrates a custom
constant-memory callback that accumulates voltage MSE without retaining each
candidate trace. `Recorder` is used only for the fixed target and optional
plots. `--checkpointed` exercises activation-checkpointed autograd, while
`--compiled-step` (or `--compile-step`) compiles the numerical step and is the
recommended performance-oriented mode for this first-order optimization. It is
the fairer comparison with the imperative tutorial's JIT-enabled step.
The functional API and its current limits are described in
[`A8_functional_populations`](../docs/advanced/A8_functional_populations.rst).
Run a short smoke example with:

```bash
python examples/functional_gradient_descent.py --iterations 5 --tstop 0.5
```

For a longer optimization, amortize compilation with:

```bash
python examples/functional_gradient_descent.py --compiled-step
```

## Train a macroscopic descriptor

[`macroscopic_descriptor_training.py`](macroscopic_descriptor_training.py)
trains repeated firing without a target voltage trace. It differentiates the
span firing frequency with respect to two Hodgkin--Huxley rate coordinates,
takes normalized optimizer steps, and accepts a step only after a complete
simulation confirms that the hard descriptor did not regress.

Run a short smoke example from the repository root:

```bash
python examples/macroscopic_descriptor_training.py \
    --updates 1 \
    --tstop-ms 60 \
    --objective-start-ms 20 \
    --skip-holdout \
    --no-jit
```

The full default run uses a longer observation window and evaluates the final
parameters in a separate holdout window:

```bash
python examples/macroscopic_descriptor_training.py
```

See the
[macroscopic descriptor training guide](../docs/advanced/A9_macroscopic_descriptor_training.rst)
for the common optimization pattern and the other analysis interfaces.

## Differentiate threshold and chronaxie

[`threshold_descriptor_gradients.py`](threshold_descriptor_gradients.py)
uses Dendra's built-in myelinated HH model to show the complete workflow for a
threshold-derived descriptor. It runs hard `Thresholder` searches at four
pulse widths, constructs voltage/amplitude tangents from one batched replay,
and differentiates both the individual thresholds and the fitted chronaxie
with respect to log sodium conductance. Fresh hard searches at perturbed
conductances independently check every reported slope.

Run the example from the repository root:

```bash
python examples/threshold_descriptor_gradients.py
```

The script checks exact hard-forward value parity and fails if a trace-derived
slope differs from its hard finite difference by more than 2%. Its default CPU
run normally finishes in about 15 seconds. The
[threshold-derived descriptor guide](../docs/advanced/A9_macroscopic_descriptor_training.rst#threshold-derived-descriptors)
explains the amplitude JVP, stateful replay ordering, probe checks, and
chronaxie composition used by the script.
