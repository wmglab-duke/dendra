"""Benchmark the functional Unmyelinated/HH transition.

Timing is deliberately outside pytest. The script separates lowering,
preparation, per-signature structured prewarming, first-call compilation, and
warmed rollout execution. Forward-only samples restore the same detached state,
so timing never grows an autograd graph or compares different trajectory
segments.

The forward comparison includes both atomic whole-rollout compilation and
prepared fixed-chunk compilation. Both run under ``torch.no_grad()``. Pass
``--bptt`` for an additional first-order parameter-gradient comparison in which
one differentiable preparation is reused across the Python loop of compiled
chunks.

Examples
--------
Small smoke benchmark::

    python scripts/benchmark_functional_population.py \\
        --batch 2 --size 5 --steps 8

Explicitly replicate that two-cable population into three independent lanes::

    python scripts/benchmark_functional_population.py \\
        --batch 2 --explicit-batch 3 --size 5 --steps 8

Representative CPU steady-state benchmark::

    python scripts/benchmark_functional_population.py \\
        --batch 32 --size 129 --steps 100

Small first-order BPTT benchmark using AOT eager::

    python scripts/benchmark_functional_population.py \\
        --batch 2 --size 5 --steps 16 \\
        --chunk-steps 4 --bptt

Include user-composed activation checkpointing and saved-tensor accounting::

    python scripts/benchmark_functional_population.py \\
        --batch 2 --size 5 --steps 16 \\
        --chunk-steps 4 --bptt --checkpointed-bptt

``--batch`` sets the base Unmyelinated population's cable count, while
an optional ``--explicit-batch R`` adds the leading replica dimension created
by ``Population.batch(R)``. Passing ``1`` deliberately benchmarks a real
singleton explicit axis; omitting the option leaves the Population unbatched.
"""

from __future__ import annotations

import argparse
import time

import torch
from torch.utils.benchmark import Compare, Timer
from torch.utils.checkpoint import checkpoint

import dendra as dn
from dendra.models.mod import hh


def _model(*, batch, explicit_batch, size, dtype, method, jit):
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
        if explicit_batch is not None:
            model.batch(explicit_batch)
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


def _compiled_whole_sample(compiled, parameters, constants, state, ve, intra):
    with torch.no_grad():
        return compiled(
            parameters,
            constants,
            _clone_state(state),
            ve,
            intra,
        )


def _build_chunk_schedule(functional, steps, chunk_steps, **compile_options):
    full_chunks, tail_steps = divmod(steps, chunk_steps)
    chunk = (
        functional.compile_rollout_chunk(chunk_steps, **compile_options)
        if full_chunks
        else None
    )
    tail = (
        functional.compile_rollout_chunk(tail_steps, **compile_options)
        if tail_steps
        else None
    )
    return chunk, tail, full_chunks


def _run_chunks(
    chunk,
    tail,
    full_chunks,
    parameters,
    prepared,
    state,
    ve,
    intra,
    *,
    checkpointed=False,
):
    def advance(active_chunk, current, start, stop):
        ve_chunk = ve[start:stop]
        intra_chunk = intra[start:stop]
        if not checkpointed:
            return active_chunk(
                parameters,
                prepared,
                current,
                dn.func.RolloutInput(ve=ve_chunk, intra=intra_chunk),
            )[0]

        # Non-reentrant checkpointing supports nested tensor PyTrees and
        # differentiable tensors captured by the closure. This keeps Dendra's
        # safe prepared-plan validation at the chunk boundary while discarding
        # activations from the tensor-only transition until backward replay.
        def checkpoint_body(checkpoint_state, checkpoint_ve, checkpoint_intra):
            return active_chunk(
                parameters,
                prepared,
                checkpoint_state,
                dn.func.RolloutInput(
                    ve=checkpoint_ve,
                    intra=checkpoint_intra,
                ),
            )[0]

        return checkpoint(
            checkpoint_body,
            current,
            ve_chunk,
            intra_chunk,
            use_reentrant=False,
        )

    cursor = 0
    if chunk is not None:
        for _ in range(full_chunks):
            next_cursor = cursor + chunk.steps
            state = advance(chunk, state, cursor, next_cursor)
            cursor = next_cursor
    if tail is not None:
        state = advance(tail, state, cursor, ve.shape[0])
    return state["integrator"]["v"]


def _compiled_chunk_sample(
    chunk,
    tail,
    full_chunks,
    parameters,
    prepared,
    state,
    ve,
    intra,
):
    with torch.no_grad():
        return _run_chunks(
            chunk,
            tail,
            full_chunks,
            parameters,
            prepared,
            _clone_state(state),
            ve,
            intra,
        )


def _bptt_sample(
    functional,
    tensors,
    parameter_name,
    ve,
    intra,
    *,
    chunk=None,
    tail=None,
    full_chunks=0,
    checkpointed=False,
):
    parameter = tensors.parameters[parameter_name].detach().clone().requires_grad_()
    parameters = dict(tensors.parameters)
    parameters[parameter_name] = parameter

    # Preparation is part of this forward/backward graph and is shared by all
    # chunks. A fresh prepared object is required for the next timed sample.
    prepared = functional.prepare(parameters, tensors.constants)
    state = _clone_state(tensors.state)
    if chunk is None and tail is None:
        state, _ = functional.rollout(
            parameters,
            prepared,
            state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )
        voltage = state["integrator"]["v"]
    else:
        voltage = _run_chunks(
            chunk,
            tail,
            full_chunks,
            parameters,
            prepared,
            state,
            ve,
            intra,
            checkpointed=checkpointed,
        )
    loss = voltage.square().mean()
    loss.backward()
    if parameter.grad is None:
        raise RuntimeError(f"No gradient was produced for {parameter_name!r}")
    return loss.detach(), parameter.grad.detach()


def _saved_tensor_summary(fn):
    """Run one forward/backward sample and count tensors retained for backward."""
    count = 0
    total_bytes = 0

    def pack(tensor):
        nonlocal count, total_bytes
        count += 1
        total_bytes += tensor.numel() * tensor.element_size()
        return tensor

    def unpack(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        result = fn()
    return result, count, total_bytes


def _measurement(name, description, fn, min_run_time, *, label):
    return Timer(
        stmt="fn()",
        globals={"fn": fn},
        label=label,
        sub_label=description,
        description=name,
    ).blocked_autorange(min_run_time=min_run_time)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="number of cables in the base Unmyelinated population (B)",
    )
    parser.add_argument(
        "--explicit-batch",
        type=int,
        default=None,
        help=(
            "optional leading Population.batch(R) replica count; omit for no "
            "explicit axis"
        ),
    )
    parser.add_argument("--size", type=int, default=129)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--method", choices=("thomas", "pcr"), default="thomas")
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument(
        "--bptt",
        action="store_true",
        help="also compare eager and compiled fixed-chunk first-order BPTT",
    )
    parser.add_argument(
        "--checkpointed-bptt",
        action="store_true",
        help=(
            "add torch.utils.checkpoint composition around compiled BPTT chunks "
            "and report saved-tensor counts (requires --bptt)"
        ),
    )
    parser.add_argument(
        "--bptt-backend",
        choices=("aot_eager", "inductor"),
        default="aot_eager",
        help="torch.compile backend for the optional BPTT comparison",
    )
    parser.add_argument("--min-run-time", type=float, default=1.0)
    args = parser.parse_args()
    if args.explicit_batch is not None and args.explicit_batch < 1:
        parser.error("--explicit-batch must be a positive integer")
    if args.steps < 1:
        parser.error("--steps must be a positive integer")
    if args.chunk_steps < 1:
        parser.error("--chunk-steps must be a positive integer")
    if args.checkpointed_bptt and not args.bptt:
        parser.error("--checkpointed-bptt requires --bptt")

    dtype = getattr(torch, args.dtype)
    eager_model = _model(
        batch=args.batch,
        explicit_batch=args.explicit_batch,
        size=args.size,
        dtype=dtype,
        method=args.method,
        jit=0,
    )
    compiled_model = _model(
        batch=args.batch,
        explicit_batch=args.explicit_batch,
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

    structured_prewarm_start = time.perf_counter()
    structured_available = functional.prewarm_structured_rollout(
        ve=True,
        intra=True,
    )
    structured_prewarm_seconds = time.perf_counter() - structured_prewarm_start
    compiled_whole = torch.compile(compiled_functional_body, fullgraph=True)
    whole_compile_start = time.perf_counter()
    with torch.no_grad():
        compiled_whole(
            tensors.parameters,
            tensors.constants,
            _clone_state(tensors.state),
            ve,
            intra,
        )
    whole_compile_seconds = time.perf_counter() - whole_compile_start

    chunk, tail, full_chunks = _build_chunk_schedule(
        functional,
        args.steps,
        args.chunk_steps,
    )
    chunk_compile_start = time.perf_counter()
    _compiled_chunk_sample(
        chunk,
        tail,
        full_chunks,
        tensors.parameters,
        prepared,
        tensors.state,
        ve,
        intra,
    )
    chunk_compile_seconds = time.perf_counter() - chunk_compile_start

    # Warm the existing per-integrator JIT separately from its timed rollout.
    _imperative_sample(compiled_model, tensors.state, dt, ve[:1], intra[:1])

    eager_reference = _imperative_sample(eager_model, tensors.state, dt, ve, intra)
    torch.testing.assert_close(functional_rollout(), eager_reference)
    torch.testing.assert_close(
        _imperative_sample(compiled_model, tensors.state, dt, ve, intra),
        eager_reference,
    )
    with torch.no_grad():
        compiled_reference = compiled_whole(
            tensors.parameters,
            tensors.constants,
            _clone_state(tensors.state),
            ve,
            intra,
        )
    torch.testing.assert_close(compiled_reference, eager_reference)
    chunk_reference = _compiled_chunk_sample(
        chunk,
        tail,
        full_chunks,
        tensors.parameters,
        prepared,
        tensors.state,
        ve,
        intra,
    )
    torch.testing.assert_close(chunk_reference, eager_reference)

    explicit_batch_label = (
        "none" if args.explicit_batch is None else str(args.explicit_batch)
    )
    description = (
        f"R={explicit_batch_label}, B={args.batch}, K={args.size}, "
        f"T={args.steps}, C={args.chunk_steps}, "
        f"{args.dtype}, {args.method}"
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
            label="Unmyelinated HH forward rollout",
        ),
        _measurement(
            "functional eager",
            description,
            functional_rollout,
            args.min_run_time,
            label="Unmyelinated HH forward rollout",
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
            label="Unmyelinated HH forward rollout",
        ),
        _measurement(
            "functional compiled inference (structured rollout)",
            description,
            lambda: _compiled_whole_sample(
                compiled_whole,
                tensors.parameters,
                tensors.constants,
                tensors.state,
                ve,
                intra,
            ),
            args.min_run_time,
            label="Unmyelinated HH forward rollout",
        ),
        _measurement(
            "functional prepared compiled fixed chunks",
            description,
            lambda: _compiled_chunk_sample(
                chunk,
                tail,
                full_chunks,
                tensors.parameters,
                prepared,
                tensors.state,
                ve,
                intra,
            ),
            args.min_run_time,
            label="Unmyelinated HH forward rollout",
        ),
    ]

    print(f"lowering:                    {lowering_seconds:.6f} s")
    print(f"preparation:                 {preparation_seconds:.6f} s")
    print(
        "structured prewarm:          "
        f"{structured_prewarm_seconds:.6f} s "
        f"({'available' if structured_available else 'unrolled fallback'})"
    )
    print(f"whole first compile + call:  {whole_compile_seconds:.6f} s")
    print(f"chunks first compile + call: {chunk_compile_seconds:.6f} s")
    Compare(measurements).print()

    if args.bptt:
        parameter_name = "integrator.mech.mechanisms.hh.gnabar_param"
        bptt_chunk, bptt_tail, bptt_full_chunks = _build_chunk_schedule(
            functional,
            args.steps,
            args.chunk_steps,
            backend=args.bptt_backend,
        )
        eager_loss, eager_gradient = _bptt_sample(
            functional,
            tensors,
            parameter_name,
            ve,
            intra,
        )
        bptt_compile_start = time.perf_counter()
        compiled_loss, compiled_gradient = _bptt_sample(
            functional,
            tensors,
            parameter_name,
            ve,
            intra,
            chunk=bptt_chunk,
            tail=bptt_tail,
            full_chunks=bptt_full_chunks,
        )
        bptt_compile_seconds = time.perf_counter() - bptt_compile_start
        torch.testing.assert_close(compiled_loss, eager_loss)
        torch.testing.assert_close(compiled_gradient, eager_gradient)

        bptt_description = (
            f"R={explicit_batch_label}, B={args.batch}, K={args.size}, "
            f"T={args.steps}, C={args.chunk_steps}, "
            f"{args.dtype}, {args.method}, {args.bptt_backend}"
        )
        bptt_measurements = [
            _measurement(
                "functional eager BPTT",
                bptt_description,
                lambda: _bptt_sample(
                    functional,
                    tensors,
                    parameter_name,
                    ve,
                    intra,
                ),
                args.min_run_time,
                label="Unmyelinated HH first-order BPTT",
            ),
            _measurement(
                "functional prepared compiled fixed-chunk BPTT",
                bptt_description,
                lambda: _bptt_sample(
                    functional,
                    tensors,
                    parameter_name,
                    ve,
                    intra,
                    chunk=bptt_chunk,
                    tail=bptt_tail,
                    full_chunks=bptt_full_chunks,
                ),
                args.min_run_time,
                label="Unmyelinated HH first-order BPTT",
            ),
        ]
        checkpoint_compile_seconds = None
        ordinary_saved = None
        checkpointed_saved = None
        if args.checkpointed_bptt:
            checkpoint_compile_start = time.perf_counter()
            checkpointed_loss, checkpointed_gradient = _bptt_sample(
                functional,
                tensors,
                parameter_name,
                ve,
                intra,
                chunk=bptt_chunk,
                tail=bptt_tail,
                full_chunks=bptt_full_chunks,
                checkpointed=True,
            )
            checkpoint_compile_seconds = time.perf_counter() - checkpoint_compile_start
            torch.testing.assert_close(checkpointed_loss, eager_loss)
            torch.testing.assert_close(checkpointed_gradient, eager_gradient)

            (ordinary_result, ordinary_count, ordinary_bytes) = _saved_tensor_summary(
                lambda: _bptt_sample(
                    functional,
                    tensors,
                    parameter_name,
                    ve,
                    intra,
                    chunk=bptt_chunk,
                    tail=bptt_tail,
                    full_chunks=bptt_full_chunks,
                )
            )
            (checkpointed_result, checkpointed_count, checkpointed_bytes) = (
                _saved_tensor_summary(
                    lambda: _bptt_sample(
                        functional,
                        tensors,
                        parameter_name,
                        ve,
                        intra,
                        chunk=bptt_chunk,
                        tail=bptt_tail,
                        full_chunks=bptt_full_chunks,
                        checkpointed=True,
                    )
                )
            )
            torch.testing.assert_close(ordinary_result[0], checkpointed_result[0])
            torch.testing.assert_close(ordinary_result[1], checkpointed_result[1])
            ordinary_saved = (ordinary_count, ordinary_bytes)
            checkpointed_saved = (checkpointed_count, checkpointed_bytes)
            bptt_measurements.append(
                _measurement(
                    "functional checkpointed compiled fixed-chunk BPTT",
                    bptt_description,
                    lambda: _bptt_sample(
                        functional,
                        tensors,
                        parameter_name,
                        ve,
                        intra,
                        chunk=bptt_chunk,
                        tail=bptt_tail,
                        full_chunks=bptt_full_chunks,
                        checkpointed=True,
                    ),
                    args.min_run_time,
                    label="Unmyelinated HH first-order BPTT",
                )
            )
        print(
            f"BPTT chunks first compile + loss/backward: {bptt_compile_seconds:.6f} s"
        )
        if checkpoint_compile_seconds is not None:
            print(
                "checkpointed chunks first loss/backward: "
                f"{checkpoint_compile_seconds:.6f} s"
            )
            print(
                "saved tensors (ordinary):    "
                f"{ordinary_saved[0]} / {ordinary_saved[1] / 1024:.3f} KiB"
            )
            print(
                "saved tensors (checkpointed): "
                f"{checkpointed_saved[0]} / "
                f"{checkpointed_saved[1] / 1024:.3f} KiB"
            )
        Compare(bptt_measurements).print()


if __name__ == "__main__":
    main()
