# Continuation and checkpoints

Continue a simulation without resetting its clock, random streams or recording
phase. A `Continuation` carries simulation state, effective parameters,
constants and callback state together; portable checkpoints let a separately
constructed compatible model resume that boundary.

Run from the repository root:

```bash
python examples/continuation/functional_continuation.py --backend eager
python examples/continuation/functional_continuation_checkpoints.py --backend aot_eager
```

| Example | Use it to… |
| --- | --- |
| [`functional_continuation.py`](functional_continuation.py) | Continue compiled runners or explicitly bound models while retaining recordings and gradients. |
| [`functional_continuation_checkpoints.py`](functional_continuation_checkpoints.py) | Export a portable continuation and restore its trainable sources into a compatible runner. |
| [`functional_explicit_continuation_checkpoints.py`](functional_explicit_continuation_checkpoints.py) | Save and restore a lowered Population or Network without a convenience runner. |
| [`functional_continuation_capture.py`](functional_continuation_capture.py) | Capture a native model's current boundary and continue it through a functional plan or runner. |
| [`functional_native_handoff.py`](functional_native_handoff.py) | Transfer a functional boundary back to native execution, with suffix gradients owned by the receiving model. |
| [`shared_api_conformance.py`](shared_api_conformance.py) | Follow one training simulation through native execution, compiled execution and a portable save. |
| [`shared_handoff_staging.py`](shared_handoff_staging.py) | Preserve shared stimulation parameters and RNG position when handing functional state to native execution. |

Portable saves preserve values and trainability, but omit upstream autograd
graphs and optimizer state. Use an in-memory continuation when gradients must
cross the boundary. See [initialization and checkpoints](../../docs/advanced/initialization_and_checkpoints.md)
for ownership and handoff rules, and [functional execution](../../docs/advanced/functional_execution.md)
for the runner and explicit-plan interfaces.

[All examples](../README.md)
