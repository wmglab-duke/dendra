"""Benchmark the functional Unmyelinated/HH reference transition.

Timing is deliberately outside pytest. The script separates lowering,
preparation, first-call compilation, and warmed forward-only rollout execution.
Every sample restores the same detached state, so timing never grows an
autograd graph or compares different trajectory segments.

Examples
--------
Small smoke benchmark::

    python scripts/benchmark_functional_population.py --batch 2 --size 5 --steps 8

Representative CPU steady-state benchmark::

    python scripts/benchmark_functional_population.py --batch 32 --size 129 --steps 100
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.benchmark import Compare, Timer

import dendra as dn
from dendra.models.mod import hh


def _model(*, batch, size, dtype, method, jit):
    if size < 1 or size % 2 == 0:
        raise ValueError("Unmyelinated benchmark size must be a positive odd integer.")
    with dn.ctx(JIT=jit, REQUIRE_GRAD=0):
        model = dn.Unmyelinated(
            [2.0] * batch,
            L=float(size - 1),
            dx=1.0,
            dtype=dtype,
            integrator=dn.bwd_euler_ub(method=method, imem=False),
        )
        model.insert(hh)
        model.initialize()
        model.eval()
    return model


def _drives(model, steps):
    count = steps * model.v.numel()
    ve = torch.linspace(
        -1.0,
        1.0,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    intra = torch.linspace(
        -1.0e-9,
        1.0e-9,
        count,
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    return ve, intra


def _imperative_rollout(model, dt, ve, intra):
    for index in range(ve.shape[0]):
        model.integrator.step(model, dt, ve[index], intra[index])
        model.t = model.t + dt
    return model.v


def _clone_state(state):
    return torch.utils._pytree.tree_map(
        lambda value: value.detach().clone(),
        state,
    )


def _restore_state_(model, state):
    model.v = state["integrator"]["v"].detach().clone()
    for name in ("m", "h", "n"):
        model.mech.hh._buffers[name] = state["mechanisms"]["hh"][name].detach().clone()
    model.t = state["clock"]["t"].detach().clone()
    model._duration_remainder = state["control"]["duration_remainder"].detach().clone()


def _imperative_sample(model, state, dt, ve, intra):
    _restore_state_(model, state)
    with torch.no_grad():
        return _imperative_rollout(model, dt, ve, intra)


def _compiled_sample(compiled, parameters, constants, state, ve, intra):
    with torch.no_grad():
        return compiled(
            parameters,
            constants,
            _clone_state(state),
            ve,
            intra,
        )


def _measurement(name, description, fn, min_run_time):
    return Timer(
        stmt="fn()",
        globals={"fn": fn},
        label="Unmyelinated HH rollout",
        sub_label=description,
        description=name,
    ).blocked_autorange(min_run_time=min_run_time)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--size", type=int, default=129)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--method", choices=("thomas", "pcr"), default="thomas")
    parser.add_argument("--min-run-time", type=float, default=1.0)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    eager_model = _model(
        batch=args.batch,
        size=args.size,
        dtype=dtype,
        method=args.method,
        jit=0,
    )
    compiled_model = _model(
        batch=args.batch,
        size=args.size,
        dtype=dtype,
        method=args.method,
        jit=1,
    )
    ve, intra = _drives(eager_model, args.steps)
    dt = torch.as_tensor(args.dt, dtype=dtype)
    eager_model.integrator._initialize(eager_model, dt, force=True)
    compiled_model.integrator._initialize(compiled_model, dt, force=True)

    lowering_start = time.perf_counter()
    functional, tensors = dn.func.make_functional(eager_model, dt=args.dt)
    lowering_seconds = time.perf_counter() - lowering_start
    preparation_start = time.perf_counter()
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    preparation_seconds = time.perf_counter() - preparation_start

    def functional_rollout():
        state = _clone_state(tensors.state)
        with torch.no_grad():
            return functional.rollout(
                tensors.parameters,
                prepared,
                state,
                dn.func.RolloutInput(ve=ve, intra=intra),
            )[0]["integrator"]["v"]

    def compiled_functional_body(parameters, constants, state, ve_values, intra_values):
        return functional.prepare_and_rollout(
            parameters,
            constants,
            state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )[0]["integrator"]["v"]

    compiled_functional = torch.compile(compiled_functional_body, fullgraph=True)
    compile_start = time.perf_counter()
    with torch.no_grad():
        compiled_functional(
            tensors.parameters,
            tensors.constants,
            _clone_state(tensors.state),
            ve,
            intra,
        )
    compile_seconds = time.perf_counter() - compile_start

    # Warm the existing per-integrator JIT separately from its timed rollout.
    _imperative_sample(compiled_model, tensors.state, dt, ve[:1], intra[:1])

    eager_reference = _imperative_sample(eager_model, tensors.state, dt, ve, intra)
    torch.testing.assert_close(functional_rollout(), eager_reference)
    torch.testing.assert_close(
        _imperative_sample(compiled_model, tensors.state, dt, ve, intra),
        eager_reference,
    )
    with torch.no_grad():
        compiled_reference = compiled_functional(
            tensors.parameters,
            tensors.constants,
            _clone_state(tensors.state),
            ve,
            intra,
        )
    torch.testing.assert_close(compiled_reference, eager_reference)

    description = (
        f"B={args.batch}, K={args.size}, T={args.steps}, {args.dtype}, {args.method}"
    )
    measurements = [
        _measurement(
            "imperative numerical loop (eager)",
            description,
            lambda: _imperative_sample(
                eager_model,
                tensors.state,
                dt,
                ve,
                intra,
            ),
            args.min_run_time,
        ),
        _measurement(
            "functional eager",
            description,
            functional_rollout,
            args.min_run_time,
        ),
        _measurement(
            "imperative numerical loop (Dendra JIT)",
            description,
            lambda: _imperative_sample(
                compiled_model,
                tensors.state,
                dt,
                ve,
                intra,
            ),
            args.min_run_time,
        ),
        _measurement(
            "functional whole-rollout compile",
            description,
            lambda: _compiled_sample(
                compiled_functional,
                tensors.parameters,
                tensors.constants,
                tensors.state,
                ve,
                intra,
            ),
            args.min_run_time,
        ),
    ]

    print(f"lowering:    {lowering_seconds:.6f} s")
    print(f"preparation: {preparation_seconds:.6f} s")
    print(f"compilation: {compile_seconds:.6f} s")
    Compare(measurements).print()


if __name__ == "__main__":
    main()
