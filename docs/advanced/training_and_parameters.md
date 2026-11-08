(training-and-parameters)=
# Train models and select parameters

Use a PyTorch optimizer to fit a model to a recorded response. Compile with
`inference=False` to keep gradients and use `trainable=` to choose which
parameters to optimize. The optimizer updates the compiled program's own
parameters.

## Fit a recorded response

Fit the passive conductance `g` by comparing a 0.5 ms voltage recording with a
constant target of −61 mV. For your own fit, supply a measured trace or choose
a loss that measures the response you want.

```python
import torch
import dendra as dn
from dendra.models.mod import pas

with dn.ctx(DEVICE="cpu", DTYPE=torch.float64, JIT=0, REQUIRE_GRAD=1):
    model = dn.SingleCompartment(
        N=1, C=1, v_init=-60.0, integrator=dn.bwd_euler_sc(imem=False)
    )
    model.insert(pas.rename("leak"), g=0.001, e=-70.0)
    model.train()
    model.initialize()
    program = dn.compile(
        model, dt=0.025, steps=20,
        inference=False, trainable="leak.g_param",
        callbacks={"trace": dn.Recorder(("v", "t"))},
        backend="eager",
    )
    report = dn.func.inspect_sources(program)
    print(report.select("parameters.*.leak.g_param"))
    optimizer = torch.optim.Adam(report.trainable_leaves, lr=1e-6)
    target_voltage = -61.0  # mV
    for iteration in range(3):
        optimizer.zero_grad(set_to_none=True)
        _, auxiliary = dn.func.run(program)
        voltage = auxiliary["callbacks"]["trace"]["v"]
        loss = (voltage - target_voltage).square().mean()
        loss.backward()
        optimizer.step()
        print(f"Iteration {iteration + 1}, loss {loss.item():.4f}")
```

The report confirms that `trainable="leak.g_param"` selected the conductance of
the mechanism named `leak`. The report groups the full path under `parameters`.

```text
SourceReport
  Sources: 1 named path; 1 unique tensor; 0 alias groups
  Gradients: 1 trainable leaf; 0 nonleaf tensors requiring gradients

parameters (1 path)
  └─ integrator.mech.mechanisms.leak.g_param  [shape=(), float64, cpu; trainable leaf]

constants (0 paths)
  (none)
```

The displayed mean squared error falls from about 2.5997 to 2.5878 mV².
Each call starts from the same initial state and uses the latest conductance.
Use `initialization="fresh"` when updated parameters should also change the
initial state. See {doc}`initialization_and_checkpoints` for other ways to
start a trial. The `eager` backend keeps startup short. Omitting it selects
the default optimizing backend.

## Find the sources you mean to train

`trainable=` accepts exact or unique partial parameter names, glob patterns,
`"re:..."` regular expressions and compiled `re.Pattern` objects. Use
`program.find_parameters(selector)` to preview runner parameter matches before
constructing an optimizer. `program.trainable_parameters` shows the live
runner-default subset with gradients enabled.

`dn.func.inspect_sources` reports names, shapes, and which tensors share a
parameter or require gradients. Use `report.trainable_leaves` to pass each
trainable tensor to the optimizer once, even if several Populations or
components share it.

Report paths begin with `parameters.` or `constants.`. Use
`report.select(...)` to choose which parameters go in each optimizer group.

```{literalinclude} ../../examples/training/functional_source_selection.py
:language: python
:start-after: "# region selection"
:end-before: "# endregion selection"
:dedent: 4
```

The complete
{download}`functional_source_selection.py <../../examples/training/functional_source_selection.py>`
example defines the model and shows how to set different learning rates with
optimizer groups or use separate optimizers. Selection keeps all aliases of
each selected tensor in the report. When using separate optimizers, check
that each parameter belongs to only one optimizer if it should get one update.

Reports do not change `requires_grad`. To change trainability, update the
underlying leaves and inspect again; an existing report retains its captured
flags. A Tensor is a *leaf* when it is a direct optimization input. A source such
as `raw * gain` is a nonleaf expression: inspect its result for ownership, but
optimize the original `raw` and `gain` leaves yourself.

## Keep ownership explicit

Inspect the object whose sources will actually execute:

| Object inspected | Sources described |
| --- | --- |
| Native Population or Network | Model-owned sources. |
| `dn.compile` runner | Independent runner defaults. |
| Returned continuation | Effective sources used by that call, including overrides and initialization replacements. |
| Explicit binding | Its bound sources. |

Shared Tensor aliases remain shared inside a model. When overriding sources,
preserve those aliases; equal numerical values in different Tensor objects do not
create a shared parameter. Replace source mappings by constructing a new raw
binding, or pass the new continuation to a runner.

The lower-level `make_functional` interface has an ownership distinction:
Population extraction exposes native raw parameters, while Network extraction
creates detached snapshots. Use `tensors.independent_parameters(...)` for
independent optimization leaves in either case. Portable restoration also creates
new owned leaves; rebuild its optimizer from the restored report.

## Continue or truncate a training graph

Passing `auxiliary["continuation"]` into the next call retains the simulation
graph across calls. This is useful for differentiating a long response in bounded
execution chunks. Do not update parameters in-place between a forward call and
the backward pass that uses them.

Set `checkpoint_every=H` to recompute spans during backward and reduce saved
activation storage. This preserves full-horizon gradients; it does not truncate
history or remove requested recordings. Keep sources and input/draw tapes
unchanged until backward completes.

Use `dn.func.detach_continuation(program, packet)` between independent backward
passes or for truncated backpropagation. It detaches recurrent history while
preserving effective parameter ownership and aliases, including live LFP field
gradients. It does not reset simulation time or callbacks. External nonleaf source
expressions remain caller-owned and need a fresh graph when reused for another
backward pass.

For independent trials that also reset the model state, follow detachment with
`dn.func.initialize(program, detached)`; the
{download}`functional_packet_initialization.py <../../examples/initialization/functional_packet_initialization.py>`
example demonstrates repeated optimizer updates with continued RNG and fresh
recording.

Native segmented training also has two histories to detach: use `model.detach()`
(or `network.detach()`) for simulation history, and
`callbacks.detach_state(results.state)` for a prepared callback collection.
Resume with the detached callback state through `callback_state=`. Callback-only
detachment leaves model and connection histories intact. See the complete
{download}`shared_network_detach.py <../../examples/training/shared_network_detach.py>`
example.

## Train connections and stimulation

Network training uses the model's synaptic differentiation settings. Configure
`network.set_synaptic_diff_config(...)` before compiling to select weight,
delay, spike and scheduled-time differentiation. Event detection is hard in the
forward simulation; enabled surrogate rules define its training gradients.
Continuous connections accept deterministic functions or modules, including
trainable transform parameters. Their weights and delays participate in source
inspection and optimizer selection. Continuous-connection delays are rounded
to solver steps; event-connection interpolation policies provide differentiable
fractional-delay behavior.

Input current/voltage tapes, waveform parameters and extracellular field tensors
can also receive gradients. Keep them as differentiable Tensor inputs rather than
converting them through NumPy or `.item()`; see {doc}`inputs_and_randomness`.
RAND parameter samples can retain gradients through their distribution
parameters. GLOBAL/BATCH/RANGENOISE values are detached; State SDE drift and
diffusion coefficients retain their pathwise gradients. See
{doc}`inputs_and_randomness` for the distinction. Selecting a parameter for
training does not guarantee that a particular loss depends on it; check its
gradients for your experiment.

## Use explicit pure functions for transforms

The explicit interface is useful when composing `torch.func.grad`, `vmap` or
other tensor transforms. Lower the model with `make_functional`, keep raw sources
and initial state explicit, and put preparation inside the transformed function.
Host runners and automatic sampling do not execute inside those transforms.

The complete
{download}`functional_api_overview.py <../../examples/getting_started/functional_api_overview.py>`
example includes a pure eager objective whose current multiplier is differentiated
and batched, with results checked against ordinary autograd and separate runs:

```console
python examples/getting_started/functional_api_overview.py --section transforms
```

Eager pure functions and compiled loops have different transform support.
`execution="scan"` supports ordinary first-order training; `execution="while"`
is inference-only. A transformed eager fallback does not establish compiled
acceleration or higher-order support. Use the
{doc}`functional API reference <../api/dendra.func>` for the chosen operation's
contract, and {doc}`custom_models` for custom component requirements.
