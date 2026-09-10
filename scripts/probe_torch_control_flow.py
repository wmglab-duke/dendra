#!/usr/bin/env python3
"""Small, deterministic control-flow compatibility matrix; no Dendra imports.

Every scan result is compared with the same explicit recurrence. Exceptions are
recorded per case so this script can be rerun under another PyTorch interpreter.
"""

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch
from torch.utils._pytree import tree_flatten
from torch.utils.checkpoint import checkpoint

try:
    from torch._higher_order_ops.scan import scan
except ImportError:
    scan = getattr(torch, "scan", None)

# The 2.12 partitioner explicitly enables DEBUG dumps of every generated body.
# Keep warnings and failures visible without making the matrix log megabytes.
logging.getLogger("torch._higher_order_ops.partitioner").setLevel(logging.WARNING)


torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def data(steps=4, initial_grad=True):
    initial = torch.tensor([-0.13, 0.07, 0.19], requires_grad=initial_grad)
    inputs = torch.linspace(-0.2, 0.3, steps * 3).reshape(steps, 3).clone()
    inputs.requires_grad_()
    parameter = torch.tensor([0.19, -0.11, 0.23], requires_grad=True)
    return initial, inputs, parameter


def rollout(initial, inputs, parameter, *, structured, carried_parameters=False):
    # Both raw shared parameters and a preparation performed before the scan
    # influence every transition. The nonlinearity exercises recurrent Hessians.
    prepared = 0.3 + 0.15 * parameter.sin()

    def transition(state, frame):
        next_state = torch.tanh(
            0.81 * state + frame * prepared + 0.02 * parameter.square()
        )
        # Clone the returned carry as well as the emission: AOT's scan backward
        # may save the transition tensor, making it alias an augmented output.
        return next_state.clone(), (next_state.square() + 0.1 * next_state).clone()

    if structured:
        if scan is None:
            raise RuntimeError("scan is unavailable in this PyTorch installation")
        if carried_parameters:

            def carried_transition(carry, frame):
                state, raw, derived = carry
                next_state = torch.tanh(
                    0.81 * state + frame * derived + 0.02 * raw.square()
                )
                return (next_state.clone(), raw.clone(), derived.clone()), (
                    next_state.square() + 0.1 * next_state
                ).clone()

            (state, _, _), emissions = scan(
                carried_transition, (initial, parameter, prepared), inputs
            )
            return state, emissions
        return scan(transition, initial, inputs)
    state = initial
    emissions = []
    for frame in inputs.unbind(0):
        state, emission = transition(state, frame)
        emissions.append(emission)
    return state, torch.stack(emissions)


def objective(result):
    state, emissions = result
    weights = torch.linspace(0.7, 1.3, emissions.shape[0], device=emissions.device)
    return state.square().sum() + (emissions.square() * weights[:, None]).sum()


def make_fn(structured, backend=None, carried_parameters=False):
    def fn(initial, inputs, parameter):
        return rollout(
            initial,
            inputs,
            parameter,
            structured=structured,
            carried_parameters=carried_parameters,
        )

    return torch.compile(fn, backend=backend, fullgraph=True) if backend else fn


class ParityFailure(AssertionError):
    def __init__(self, failures, errors):
        super().__init__("\n\n".join(failures))
        self.errors = errors


def compare(actual, expected):
    left, left_spec = tree_flatten(actual)
    right, right_spec = tree_flatten(expected)
    if left_spec != right_spec:
        raise AssertionError(f"result pytree mismatch: {left_spec} != {right_spec}")
    errors, failures = [], []
    for index, (value, reference) in enumerate(zip(left, right, strict=True)):
        try:
            torch.testing.assert_close(
                value,
                reference,
                atol=2e-10,
                rtol=2e-8,
                equal_nan=False,
                msg=lambda message: f"leaf {index}: {message}",
            )
        except AssertionError as error:
            failures.append(str(error))
        absolute = (value - reference).abs().max().item() if value.numel() else 0.0
        scale = reference.abs().max().item() if reference.numel() else 0.0
        errors.append(
            {
                "leaf": index,
                "shape": list(value.shape),
                "max_abs_error": absolute,
                "reference_linf": scale,
                "normalized_error": absolute / max(scale, 1e-12),
            }
        )
    if failures:
        raise ParityFailure(failures, errors)
    return errors


def evaluate(
    kind, structured, backend=None, initial_grad=True, carried_parameters=False
):
    fn = make_fn(structured, backend, carried_parameters)
    args = data(initial_grad=initial_grad)
    active = args if initial_grad else args[1:]
    if kind in ("first_order", "checkpoint", "repeated_checkpoint", "hvp", "gradgrad"):
        if kind in ("checkpoint", "repeated_checkpoint"):
            result = checkpoint(fn, *args, use_reentrant=False)
        else:
            result = fn(*args)
        loss = objective(result)
        gradient = torch.autograd.grad(
            loss,
            active,
            create_graph=kind in ("hvp", "gradgrad"),
            retain_graph=kind in ("repeated_checkpoint", "hvp", "gradgrad"),
        )
        if kind == "hvp":
            vectors = tuple(
                torch.linspace(0.2, 0.8, x.numel()).reshape(x.shape) for x in active
            )
            product = sum(
                (grad * vec).sum() for grad, vec in zip(gradient, vectors, strict=True)
            )
            second = torch.autograd.grad(product, active)
            return result, loss, gradient, second
        if kind == "gradgrad":
            second = torch.autograd.grad(
                sum(grad.square().sum() for grad in gradient), active
            )
            return result, loss, gradient, second
        if kind == "repeated_checkpoint":
            repeated = torch.autograd.grad(loss, active)
            return result, loss, gradient, repeated
        return result, loss, gradient

    def loss_fn(initial, inputs, parameter):
        return objective(fn(initial, inputs, parameter))

    if kind == "func_grad":
        return loss_fn(*args), torch.func.grad(loss_fn, argnums=(0, 1, 2))(*args)
    if kind == "func_jacrev":
        return fn(*args), torch.func.jacrev(fn, argnums=(0, 1, 2))(*args)
    if kind == "func_jvp":
        tangents = tuple(torch.full_like(arg, 0.3) for arg in args)
        return torch.func.jvp(fn, args, tangents)
    if kind == "func_hvp":
        tangents = tuple(torch.full_like(arg, 0.3) for arg in args)
        return torch.func.jvp(
            torch.func.grad(loss_fn, argnums=(0, 1, 2)), args, tangents
        )
    if kind in ("vmap", "vmap_grad", "grad_vmap"):
        batched = tuple(
            torch.stack([arg.detach(), arg.detach() + 0.025]) for arg in args
        )
        if kind == "vmap":
            return torch.func.vmap(fn)(*batched)
        if kind == "vmap_grad":
            return torch.func.vmap(torch.func.grad(loss_fn, argnums=(0, 1, 2)))(
                *batched
            )

        def batch_loss(initial, inputs, parameter):
            states, emissions = torch.func.vmap(fn)(initial, inputs, parameter)
            return states.square().sum() + emissions.square().sum()

        return torch.func.grad(batch_loss, argnums=(0, 1, 2))(*batched)
    raise ValueError(kind)


def diagnose(error):
    return {
        "type": type(error).__name__,
        "message": str(error)[:9000],
        "traceback": "".join(traceback.format_exception(error))[-15000:],
    }


def graph_sizes(structured, steps):
    graphs = []

    def counting_backend(graph, example_inputs):
        nested = []
        for name, module in graph.named_modules():
            if isinstance(module, torch.fx.GraphModule):
                nested.append(
                    {
                        "name": name or "<root>",
                        "nodes": len(list(module.graph.nodes)),
                        "scan_nodes": sum(
                            node.op == "call_function" and "scan" in str(node.target)
                            for node in module.graph.nodes
                        ),
                    }
                )
        graphs.append(nested)
        return graph.forward

    fn = make_fn(structured, counting_backend)
    result = fn(*data(steps))
    compare(result, make_fn(False)(*data(steps)))
    return graphs


def while_result(structured, backend=None, gradients=False):
    initial, inputs, parameter = data(initial_grad=gradients)
    if not gradients:
        inputs = inputs.detach()
        parameter = parameter.detach()

    def fn(state, inputs, parameter):
        prepared = 0.3 + 0.15 * parameter.sin()
        if not structured:
            for frame in inputs.unbind(0):
                state = torch.tanh(
                    0.81 * state + frame * prepared + 0.02 * parameter.square()
                )
            return state

        def cond(index, carry):
            return index < inputs.shape[0]

        def body(index, carry):
            frame = torch.index_select(inputs, 0, index.reshape(1)).squeeze(0)
            return index + 1, torch.tanh(
                0.81 * carry + frame * prepared + 0.02 * parameter.square()
            )

        _, state = torch.while_loop(
            cond, body, (torch.zeros((), dtype=torch.int64), state)
        )
        return state

    if backend:
        fn = torch.compile(fn, backend=backend, fullgraph=True)
    result = fn(initial, inputs, parameter)
    if gradients:
        return result, torch.autograd.grad(
            result.square().sum(), (initial, inputs, parameter)
        )
    return result


def mixed_result(
    structured,
    backend=None,
    initial_grad=False,
    emission_kind="trace",
    integer_counter=True,
):
    args = data(initial_grad=initial_grad)

    def fn(initial, inputs, parameter):
        prepared = 0.3 + 0.15 * parameter.sin()
        carry = (
            initial,
            torch.zeros(()),
            torch.zeros((), dtype=torch.int64 if integer_counter else torch.float64),
            initial.square().sum(),
        )

        def body(carry, frame):
            state, clock, frames, square_sum = carry
            next_state = torch.tanh(
                0.81 * state + frame * prepared + 0.02 * parameter.square()
            )
            next_clock, next_frames = clock + 0.005, frames + 1
            next_sum = square_sum + next_state.square().sum()
            next_carry = (
                next_state.clone(),
                next_clock.clone(),
                next_frames.clone(),
                next_sum.clone(),
            )
            if emission_kind == "trace":
                return next_carry, (
                    next_state.clone(),
                    next_clock.clone(),
                    next_frames.clone(),
                )
            return next_carry, None if emission_kind == "none" else ()

        if structured:
            return scan(body, carry, inputs)
        outputs = []
        for frame in inputs.unbind(0):
            carry, emission = body(carry, frame)
            outputs.append(emission)
        emissions = (
            tuple(torch.stack(values) for values in zip(*outputs))
            if emission_kind == "trace"
            else None
            if emission_kind == "none"
            else ()
        )
        return carry, emissions

    if backend:
        fn = torch.compile(fn, backend=backend, fullgraph=True)
    result = fn(*args)
    carry, emissions = result
    loss = carry[0].square().sum() + carry[3] / (carry[2] + 1)
    if emission_kind == "trace":
        loss = loss + emissions[0].square().mean()
    return result, loss, torch.autograd.grad(loss, args if initial_grad else args[1:])


def finite_difference_hvp():
    args = data()
    vectors = tuple(torch.linspace(0.2, 0.8, x.numel()).reshape(x.shape) for x in args)
    gradients = []
    epsilon = 1e-5
    for sign in (-1, 1):
        shifted = tuple(
            (x.detach() + sign * epsilon * v).requires_grad_()
            for x, v in zip(args, vectors, strict=True)
        )
        loss = objective(make_fn(False)(*shifted))
        gradients.append(torch.autograd.grad(loss, shifted))
    return tuple(
        (plus - minus) / (2 * epsilon)
        for plus, minus in zip(gradients[1], gradients[0], strict=True)
    )


def mixed_while_result(
    structured, backend=None, initial_grad=False, kind="first_order"
):
    args = data(initial_grad=initial_grad)

    def fn(initial, inputs, parameter):
        prepared = 0.3 + 0.15 * parameter.sin()
        carry = (
            torch.zeros((), dtype=torch.int64),
            initial,
            torch.zeros(()),
            initial.square().sum(),
        )

        def cond(index, state, clock, square_sum):
            return index < inputs.shape[0]

        def body(index, state, clock, square_sum):
            frame = torch.index_select(inputs, 0, index.reshape(1)).squeeze(0)
            next_state = torch.tanh(
                0.81 * state + frame * prepared + 0.02 * parameter.square()
            )
            return (
                index + 1,
                next_state.clone(),
                clock + 0.005,
                square_sum + next_state.square().sum(),
            )

        if structured:
            return torch.while_loop(cond, body, carry)
        for _ in range(inputs.shape[0]):
            carry = body(*carry)
        return carry

    if backend:
        fn = torch.compile(fn, backend=backend, fullgraph=True)

    def loss_fn(*values):
        count, state, _, square_sum = fn(*values)
        return state.square().sum() + square_sum / (count + 1)

    if kind == "func_grad":
        return torch.func.grad(loss_fn, argnums=(0, 1, 2))(*args)
    if kind == "func_jvp":
        return torch.func.jvp(
            loss_fn, args, tuple(torch.full_like(arg, 0.3) for arg in args)
        )
    if kind == "vmap":
        batched = tuple(
            torch.stack([arg.detach(), arg.detach() + 0.025]) for arg in args
        )
        return torch.func.vmap(fn)(*batched)
    active = args if initial_grad else args[1:]
    result = (
        checkpoint(fn, *args, use_reentrant=False)
        if kind == "checkpoint"
        else fn(*args)
    )
    count, state, _, square_sum = result
    loss = state.square().sum() + square_sum / (count + 1)
    gradient = torch.autograd.grad(loss, active, create_graph=kind == "hvp")
    if kind == "hvp":
        vectors = tuple(
            torch.linspace(0.2, 0.8, x.numel()).reshape(x.shape) for x in active
        )
        product = sum(
            (grad * vec).sum() for grad, vec in zip(gradient, vectors, strict=True)
        )
        second = torch.autograd.grad(product, active)
        return result, loss, gradient, second
    return result, loss, gradient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("control_flow_probe.json"))
    parser.add_argument("--quick", action="store_true", help="Skip Inductor cases")
    parser.add_argument(
        "--filter", default="", help="Run only cases containing this substring"
    )
    options = parser.parse_args()
    report = {
        "metadata": {
            "torch": torch.__version__,
            "torch_git": torch.version.git_version,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python": sys.version,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
            "dtype": "float64",
            "atol": 2e-10,
            "rtol": 2e-8,
        },
        "cases": [],
    }

    def save():
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n")

    def run(name, candidate, reference):
        if options.filter and options.filter not in name:
            return
        torch._dynamo.reset()
        entry = {"name": name}
        start = time.monotonic()
        try:
            with warnings.catch_warnings(record=True) as caught:
                entry["stage"] = "reference"
                expected = reference()
                entry["stage"] = "candidate"
                actual = candidate()
                entry["stage"] = "parity"
                entry["errors"] = compare(actual, expected)
                entry["warnings"] = [str(w.message) for w in caught]
            entry["status"] = "pass"
        except Exception as error:
            entry["status"] = "fail"
            entry["diagnostic"] = diagnose(error)
            if hasattr(error, "errors"):
                entry["errors"] = error.errors
        entry["elapsed_seconds"] = time.monotonic() - start
        report["cases"].append(entry)
        save()
        print(
            json.dumps(
                {
                    "name": name,
                    "status": entry["status"],
                    "error": entry.get("diagnostic", {}).get("message", "")[:250],
                }
            ),
            flush=True,
        )

    backends = [None, "aot_eager"] + ([] if options.quick else ["inductor"])
    for backend in backends:
        for initial_grad in (False, True):
            for carried in (False, True):
                name = f"scan/{backend or 'eager'}/first_order/init_grad={initial_grad}/carried={carried}"
                run(
                    name,
                    lambda b=backend, i=initial_grad, c=carried: evaluate(
                        "first_order", True, b, i, c
                    ),
                    lambda i=initial_grad: evaluate(
                        "first_order", False, initial_grad=i
                    ),
                )
        for kind in ("checkpoint", "repeated_checkpoint"):
            run(
                f"scan/{backend or 'eager'}/{kind}",
                lambda b=backend, k=kind: evaluate(k, True, b),
                lambda k=kind: evaluate(k, False),
            )

    for kind in (
        "hvp",
        "gradgrad",
        "func_grad",
        "func_jacrev",
        "func_jvp",
        "func_hvp",
        "vmap",
        "vmap_grad",
        "grad_vmap",
    ):
        run(
            f"scan/eager/{kind}",
            lambda k=kind: evaluate(k, True),
            lambda k=kind: evaluate(k, False),
        )

    for backend in backends:
        for gradients in (False, True):
            run(
                f"while_loop/{backend or 'eager'}/gradients={gradients}",
                lambda b=backend, g=gradients: while_result(True, b, g),
                lambda g=gradients: while_result(False, gradients=g),
            )

    for backend in backends:
        for initial_grad in (False, True):
            for emission_kind in ("trace", "none", "empty"):
                run(
                    f"scan/{backend or 'eager'}/mixed_carry/init_grad={initial_grad}/emission={emission_kind}",
                    lambda b=backend, i=initial_grad, e=emission_kind: mixed_result(
                        True, b, i, e
                    ),
                    lambda i=initial_grad, e=emission_kind: mixed_result(
                        False, initial_grad=i, emission_kind=e
                    ),
                )
        for emission_kind in ("trace", "none", "empty"):
            run(
                f"scan/{backend or 'eager'}/float_counter/emission={emission_kind}",
                lambda b=backend, e=emission_kind: mixed_result(
                    True, b, False, e, False
                ),
                lambda e=emission_kind: mixed_result(
                    False, initial_grad=False, emission_kind=e, integer_counter=False
                ),
            )

    run(
        "oracle/explicit_hvp_vs_finite_difference",
        lambda: evaluate("hvp", False)[-1],
        finite_difference_hvp,
    )

    for backend in backends:
        for initial_grad in (False, True):
            run(
                f"while_loop/{backend or 'eager'}/mixed_carry/init_grad={initial_grad}",
                lambda b=backend, i=initial_grad: mixed_while_result(True, b, i),
                lambda i=initial_grad: mixed_while_result(False, initial_grad=i),
            )
        run(
            f"while_loop/{backend or 'eager'}/mixed_checkpoint",
            lambda b=backend: mixed_while_result(True, b, kind="checkpoint"),
            lambda: mixed_while_result(False, kind="checkpoint"),
        )
    for kind in ("hvp", "func_grad", "func_jvp", "vmap"):
        run(
            f"while_loop/eager/mixed_{kind}",
            lambda k=kind: mixed_while_result(True, initial_grad=True, kind=k),
            lambda k=kind: mixed_while_result(False, initial_grad=True, kind=k),
        )

    for structured in (False, True):
        for steps in (4, 16):
            name = f"graphs/{'scan' if structured else 'explicit'}/T={steps}"
            if options.filter and options.filter not in name:
                continue
            torch._dynamo.reset()
            entry = {"name": name}
            try:
                entry.update(status="pass", graphs=graph_sizes(structured, steps))
            except Exception as error:
                entry.update(status="fail", diagnostic=diagnose(error))
            report["cases"].append(entry)
            save()
            print(
                json.dumps(
                    {
                        "name": name,
                        "status": entry["status"],
                        "graphs": entry.get("graphs"),
                    }
                ),
                flush=True,
            )

    save()
    print(
        json.dumps(
            {
                "output": str(options.output),
                "passed": sum(case["status"] == "pass" for case in report["cases"]),
                "failed": sum(case["status"] == "fail" for case in report["cases"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
