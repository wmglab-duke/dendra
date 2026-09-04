# Examples

Network implementation examples in ./net/

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
