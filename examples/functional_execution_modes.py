#!/usr/bin/env python3
"""Exercise the public functional execution options on one small HH model.

Run from the repository root::

    python examples/functional_execution_modes.py --mode all
    python examples/functional_execution_modes.py --mode inference --backend inductor
    python examples/functional_execution_modes.py --mode checkpointed --backend inductor
    python examples/functional_execution_modes.py --mode scan --backend inductor

The default aot_eager backend exercises graph capture and AOTAutograd without
native code generation. Use inductor for optimized native kernels. Compiled
transforms are verified with aot_eager; backend support can differ by transform.
Scan modes require Dendra's verified PyTorch 2.14.0 implementation; all skips those modes
on other versions, while explicitly requesting them raises the capability error.

Compiled examples check values/derivatives against ordinary functional execution.
This is a usage example, not a performance benchmark. Initial state is held
fixed; see functional_gradient_descent.py for differentiable initialization.
"""

from __future__ import annotations

import argparse
from functools import partial

import torch

import dendra as dn
from dendra.models.mod import hh

DT = 0.01
STEPS = 8
MODES = (
    "eager",
    "inference",
    "atomic-compile",
    "chunks",
    "checkpointed",
    "scan",
    "scan-checkpointed",
    "transforms",
    "compiled-transforms",
    "fallback",
)


def setup():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0],
            L=2.0,
            dx=1.0,
            v_init=torch.tensor([-64.0, -58.0, -62.0]),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(hh)
        model.initialize()
        model.train()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    inputs = dn.func.RolloutInput(
        ve=torch.linspace(
            -1.0, 1.0, STEPS * model.v.numel(), dtype=torch.float64
        ).reshape(STEPS, *model.shape)
    )
    name = tensors.parameter_name("gnabar", within="model")
    base_parameters = tensors.independent_parameters(trainable=())

    def parameters(gnabar):
        return {**base_parameters, name: gnabar}

    def voltage(gnabar):
        # This atomic function exposes the preparation dependency to transforms
        # and compilation. No opaque prepared object crosses that boundary.
        final, _ = functional.prepare_and_rollout(
            parameters(gnabar), tensors.constants, tensors.state, inputs
        )
        return final["integrator"]["v"]

    def loss(gnabar):
        return voltage(gnabar).square().mean()

    return functional, tensors, inputs, base_parameters[name], parameters, voltage, loss


def check(actual, expected):
    for value in torch.utils._pytree.tree_leaves((actual, expected)):
        assert torch.isfinite(value).all(), "example produced a non-finite value"
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-10)


def value_and_gradient(function, initial):
    # Fresh leaf and preparation for every forward/backward graph.
    parameter = initial.detach().clone().requires_grad_()
    value = function(parameter)
    gradient = torch.autograd.grad(value, parameter)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    return value.detach(), gradient.detach()


def eager_example(functional, tensors, inputs, initial, parameters, voltage, loss):
    local_parameters = parameters(initial)
    prepared = functional.prepare(local_parameters, tensors.constants)
    bound = functional.bind(local_parameters, prepared)
    one, _ = bound(tensors.state, dn.func.StepInput(ve=inputs.ve[0]))
    # Continue from the returned state to demonstrate explicit state threading.
    final, _ = functional.rollout(
        local_parameters, prepared, one, dn.func.RolloutInput(ve=inputs.ve[1:])
    )
    check(final["integrator"]["v"], voltage(initial))
    value, gradient = value_and_gradient(loss, initial)
    check(gradient, torch.func.grad(loss)(initial))
    print(f"eager: loss={value.item():.8g}, dloss/dgnabar={gradient.item():.8g}")


def inference_example(
    functional, tensors, inputs, initial, parameters, voltage, backend
):
    # The convenience wrapper automatically prewarms eligible inference loops.
    kernel = functional.compile_rollout_chunk(STEPS, backend=backend)
    with torch.no_grad():
        local_parameters = parameters(initial)
        prepared = functional.prepare(local_parameters, tensors.constants)
        bound = kernel.bind(local_parameters, prepared)
        final, _ = bound(tensors.state, inputs)
        check(final["integrator"]["v"], voltage(initial))
        assert not final["integrator"]["v"].requires_grad

        # Alternative: include preparation in the user-owned compiled function.
        # Prewarm before entering that boundary; this example has only ve input.
        functional.prewarm_structured_rollout(ve=True, intra=False)
        compiled_voltage = torch.compile(voltage, backend=backend, fullgraph=True)
        check(compiled_voltage(initial), voltage(initial))
    print(f"inference: {kernel.execution_report().strategy}; values match eager")


def atomic_compile_example(loss, initial, backend):
    # With gradients enabled, the fixed time recurrence is unrolled in this graph.
    compiled_loss = torch.compile(loss, backend=backend, fullgraph=True)
    check(value_and_gradient(compiled_loss, initial), value_and_gradient(loss, initial))
    print("atomic-compile: compiled preparation + rollout preserve first derivatives")


def runner_example(functional, tensors, inputs, initial, parameters, mode, backend):
    use_scan = mode.startswith("scan")
    checkpointed = "checkpointed" in mode
    kernel = functional.compile_rollout_chunk(
        STEPS if use_scan else 3,
        execution="scan" if use_scan else "default",
        backend=backend,
    )
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})

    def objective(gnabar, *, runner, compiled, legacy=False):
        local_parameters = parameters(gnabar)
        prepared = functional.prepare(local_parameters, tensors.constants)
        if not legacy:
            bound = (kernel if compiled else functional).bind(
                local_parameters, prepared
            )
            options = {}
            if runner is dn.func.longrun_checkpointed:
                options["checkpoint_every"] = 5
            elif runner is dn.func.longrun:
                options["host_span_steps"] = 5
            final, auxiliary = dn.func.run(
                bound,
                tensors.state,
                inputs=inputs,
                steps=STEPS,
                callbacks=callbacks,
                **options,
            )
        elif runner is dn.func.run:
            step = partial(
                kernel if compiled else functional.step, local_parameters, prepared
            )
            final, auxiliary = runner(
                functional, step, tensors.state, inputs, callbacks=callbacks
            )
        else:
            step = partial(
                kernel if compiled else functional.step, local_parameters, prepared
            )
            final, auxiliary = runner(
                functional,
                step,
                tensors.state,
                tstop=STEPS * DT,
                chunklength=5,
                inputs=inputs,
                callbacks=callbacks,
            )
        trace = auxiliary["callbacks"]["trace"]["v"]
        assert trace.shape[0] == STEPS + 1  # includes the initial frame
        return final["integrator"]["v"].square().mean() + trace.square().mean()

    runner = dn.func.longrun_checkpointed if checkpointed else dn.func.run
    reference = partial(objective, runner=dn.func.run, compiled=False, legacy=True)
    candidate = partial(objective, runner=runner, compiled=True)
    # Retain the kernel across calls; prepare again after a parameter change.
    for current in (initial, initial * 1.01):
        check(
            value_and_gradient(candidate, current),
            value_and_gradient(reference, current),
        )
    if mode == "chunks":
        long = partial(objective, runner=dn.func.longrun, compiled=True)
        check(value_and_gradient(long, initial), value_and_gradient(reference, initial))
        legacy = partial(objective, runner=dn.func.longrun, compiled=True, legacy=True)
        check(value_and_gradient(long, initial), value_and_gradient(legacy, initial))
    report = kernel.execution_report()
    assert report is not None and report.callbacks
    print(f"{mode}: latest kernel={report.strategy}, steps={report.steps}")
    print(
        f"{mode}: callback loss and full-run gradient match eager after parameter replacement"
    )


def transforms_example(voltage, loss, initial, backend=None):
    # These transformations occur before optional compilation.
    lanes = torch.stack((initial * 0.9, initial, initial * 1.1))
    jacobian = torch.func.jacfwd(voltage)(initial)
    transforms = {
        "grad": (torch.func.grad(loss), initial, value_and_gradient(loss, initial)[1]),
        "jacrev": (torch.func.jacrev(voltage), initial, jacobian),
        "hessian": (
            torch.func.hessian(loss),
            initial,
            torch.func.jacrev(torch.func.grad(loss))(initial),
        ),
        "vmap": (
            torch.func.vmap(voltage),
            lanes,
            torch.stack([voltage(lane) for lane in lanes]),
        ),
    }

    def directional_derivative(gnabar):
        return torch.func.jvp(voltage, (gnabar,), (torch.ones_like(gnabar),))

    # The selected conductance is scalar, so the all-ones JVP is its Jacobian.
    assert initial.ndim == 0
    transforms["jvp"] = directional_derivative, initial, (voltage(initial), jacobian)
    for name, (function, argument, expected) in transforms.items():
        actual = (
            function(argument)
            if backend is None
            else torch.compile(function, backend=backend, fullgraph=True)(argument)
        )
        check(actual, expected)
        if name in ("grad", "hessian"):
            assert actual.abs().max() > 0
        print(f"{'compiled-' if backend else ''}transforms: {name} passed")


def fallback_example(functional, tensors, inputs, initial, parameters, loss):
    def unexpected_backend(_graph, _examples):
        raise AssertionError("an active transform should bypass inner compilation")

    kernel = functional.compile_rollout_chunk(STEPS, backend=unexpected_backend)

    def chunk_loss(gnabar):
        local_parameters = parameters(gnabar)
        prepared = functional.prepare(local_parameters, tensors.constants)
        bound = kernel.bind(local_parameters, prepared)
        final, _ = bound(tensors.state, inputs)
        return final["integrator"]["v"].square().mean()

    check(torch.func.grad(chunk_loss)(initial), torch.func.grad(loss)(initial))
    assert kernel.execution_report().strategy == "eager_transform_fallback"
    print("fallback: transforming a compiled binding works through eager execution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument(
        "--backend", choices=("aot_eager", "inductor"), default="aot_eager"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    case = setup()
    functional, tensors, inputs, initial, parameters, voltage, loss = case
    modes = MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        if (
            args.mode == "all"
            and mode.startswith("scan")
            and torch.__version__.split("+")[0] != "2.14.0"
        ):
            print(f"{mode}: skipped; verified PyTorch 2.14.0 is required")
            continue
        if mode == "eager":
            eager_example(*case)
        elif mode == "inference":
            inference_example(*case[:-1], args.backend)
        elif mode == "atomic-compile":
            atomic_compile_example(loss, initial, args.backend)
        elif mode in ("chunks", "checkpointed", "scan", "scan-checkpointed"):
            runner_example(
                functional, tensors, inputs, initial, parameters, mode, args.backend
            )
        elif mode in ("transforms", "compiled-transforms"):
            transforms_example(
                voltage,
                loss,
                initial,
                args.backend if mode.startswith("compiled") else None,
            )
        elif mode == "fallback":
            fallback_example(functional, tensors, inputs, initial, parameters, loss)


if __name__ == "__main__":
    main()
