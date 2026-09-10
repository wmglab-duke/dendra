"""Probe a pure Dendra model/callback recurrence under PyTorch's private scan.

This is an experimental acceptance tool, not a runner or a production dispatch
option. A zero exit status means only the selected configuration matched its
ordinary recurrence oracle. Reports retain failures for comparison across Torch
versions. Run each configuration in a fresh process, especially when a compiled
checkpoint can fail: retained exception tracebacks can retain checkpoint hooks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch._higher_order_ops.scan import scan
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils import _pytree as pytree
from torch.utils.checkpoint import checkpoint

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._callbacks import _update_callback_values
from dendra.models.mod import hh, pas
from dendra.units import nA

DT = 0.01
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"


@contextmanager
def compiler_warnings():
    with torch_compiler_warning_context(), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script.*deprecated.*",
            category=FutureWarning,
            module=r"torch\.jit\._script",
        )
        yield


def clone(tree):
    return pytree.tree_map(
        lambda x: x.clone(memory_format=torch.contiguous_format), tree
    )


def detached(tree):
    return pytree.tree_map(lambda x: x.detach().clone(), tree)


class VoltageEnergy(dn.func.FunctionalCallback):
    """A streaming scalar loss with floating-point carry and no emissions."""

    def initialize(self, state, auxiliary):
        return (state["integrator"]["v"] + 62.0).square().mean(), None

    def update(self, carry, state, auxiliary):
        return carry + (state["integrator"]["v"] + 62.0).square().mean(), None

    def finalize(self, carry, emissions):
        return carry


@dataclass(frozen=True)
class TraceMSE(dn.func.FunctionalCallback):
    """Streaming fitting loss with an explicit experimental counter dtype."""

    target: torch.Tensor
    counter_dtype: torch.dtype = torch.int64

    def initialize(self, state, auxiliary):
        voltage = state["integrator"]["v"]
        return {
            "error": (voltage - self.target[0]).square().mean(),
            "frames": voltage.new_ones((), dtype=self.counter_dtype),
        }, None

    def update(self, carry, state, auxiliary):
        reference = torch.index_select(
            self.target, 0, carry["frames"].to(torch.int64).reshape(1)
        ).squeeze(0)
        return {
            "error": carry["error"]
            + (state["integrator"]["v"] - reference).square().mean(),
            "frames": carry["frames"] + 1,
        }, None

    def finalize(self, carry, emissions):
        return carry["error"] / carry["frames"].to(carry["error"].dtype)


def hh_case(steps):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            v_init=torch.tensor(
                [-64.0, -60.0, -56.0, -59.0, -63.0], dtype=torch.float64
            ),
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(hh)
        model.initialize()
        model.train()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, constants, state = detached(
        (tensors.parameters, tensors.constants, tensors.state)
    )
    count = steps * model.v.numel()
    samples = {
        "ve": torch.linspace(-2.0, 2.0, count, dtype=torch.float64)
        .reshape(steps, *model.shape)
        .requires_grad_(),
        "intra": torch.linspace(-1e-9, 1e-9, count, dtype=torch.float64)
        .reshape(steps, *model.shape)
        .requires_grad_(),
    }
    targets = {
        "parameter": parameters[GNABAR].requires_grad_(),
        "diam": constants["diam"].requires_grad_(),
        "dx": constants["dx"].requires_grad_(),
        "initial_voltage": state["integrator"]["v"].requires_grad_(),
        **samples,
    }
    prepared = functional.prepare(parameters, constants).values
    return functional, parameters, prepared, state, samples, targets


def pulse_case(kind, steps):
    if steps != 8:
        raise ValueError("the narrow bound-pulse fixture requires --steps 8")
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float32):
        model = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=-65.0,
            dtype=torch.float32,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas, g=1e-4, e=-65.0)
        waveform = dn.mono_rect(
            amp=0.2 * nA if kind == "intra" else 1.0, delay=1000.070068359375, pw=0.001
        )
        extra = None
        if kind == "intra":
            model[:, 1].inject(waveform)
        else:
            extra = (
                torch.linspace(-1.0, 1.0, model.v.numel()).reshape(model.shape),
                waveform,
            )
        model.initialize()
        model.t.fill_(1000.0)
        model.train()
    functional, tensors = dn.func.make_functional(model, dt=DT, extra=extra)
    parameters = detached(tensors.parameters)
    name = next(
        name
        for name in parameters
        if name.startswith(f"stimulation.{kind}.") and name.endswith(".amp")
    )
    targets = {"amplitude": parameters[name].requires_grad_()}
    prepared = functional.prepare(parameters, tensors.constants).values
    # The dummy scan axis selects no waveform values. The captured body samples
    # registered waveforms at each recurrent model clock through _step_values.
    samples = {"tick": torch.zeros(steps, dtype=functional.dtype)}
    return functional, parameters, prepared, tensors.state, samples, targets


class CapturedCallbackStep:
    """One pure FX body; all evolving values and prepared dependencies explicit."""

    def __init__(self, functional, callbacks, operands):
        self.functional = functional
        self.callbacks = callbacks

        def authored(parameters, prepared, state, carries, sample):
            return self.authored(parameters, prepared, state, carries, sample)

        with torch.no_grad():
            self.graph = make_fx(authored)(*detached(operands))

    def authored(self, parameters, prepared, state, carries, sample):
        state, auxiliary = self.functional._step_values(
            parameters,
            prepared,
            state,
            dn.func.StepInput(sample.get("ve"), sample.get("intra")),
        )
        carries, emitted = _update_callback_values(
            self.callbacks._plans, carries, state, auxiliary
        )
        # HOPs require independent carry/emission storage, including unchanged
        # clock/remainder leaves and Recorder aliases. Never detach gradients.
        return clone((state, carries)), clone(emitted or {})

    def scanned(self, parameters, prepared, state, carries, samples):
        def combine(carry, sample):
            state, carries = carry
            return self.graph(parameters, prepared, state, carries, sample)

        return scan(combine, clone((state, carries)), samples)

    def ordinary(self, parameters, prepared, state, carries, samples):
        emissions = []
        for index in range(pytree.tree_leaves(samples)[0].shape[0]):
            sample = pytree.tree_map(lambda x: x[index], samples)
            (state, carries), emitted = self.authored(
                parameters, prepared, state, carries, sample
            )
            emissions.append(emitted)
        stacked = pytree.tree_map(lambda *xs: torch.stack(xs), *emissions)
        return (state, carries), stacked


def objective(outputs):
    (state, carries), emissions = outputs
    value = state["integrator"]["v"].square().mean()
    if "mse" in carries:
        mse = carries["mse"]
        value = value + mse["error"] / mse["frames"].to(mse["error"].dtype)
    if "energy" in carries:
        value = value + carries["energy"]
    if "trace" in emissions:
        value = value + emissions["trace"]["v"].square().mean()
    return value


def graph_size(graph):
    return {
        name or "<root>": {
            "nodes": len(list(module.graph.nodes)),
            "scan_nodes": sum(
                node.op == "call_function" and str(node.target) == "scan"
                for node in module.graph.nodes
            ),
        }
        for name, module in graph.named_modules()
        if isinstance(module, torch.fx.GraphModule)
    }


def compare_gradients(actual, expected, names, tolerance):
    metrics = {}
    for name, a, e in zip(names, actual, expected, strict=True):
        scale = e.abs().amax().item()
        if not torch.isfinite(e).all() or scale == 0:
            raise AssertionError(f"gradient oracle {name} must be finite and nonzero")
        error = None if a is None else (a - e).abs().amax().item() / scale
        metrics[name] = {
            "reference_linf": scale,
            "relative_linf_error": error,
            "pass": a is not None
            and bool(torch.isfinite(a).all())
            and error <= tolerance,
        }
    return metrics


def run_probe(args):
    torch.set_num_threads(1)
    result = {
        "torch": torch.__version__,
        "torch_path": torch.__file__,
        "dendra": dn.__version__,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "accepted": False,
    }
    stage = "fixture"
    try:
        with compiler_warnings():
            case = (
                hh_case(args.steps)
                if args.fixture == "hh"
                else pulse_case(args.fixture, args.steps)
            )
            functional, parameters, prepared, state, samples, targets = case
            target = state["integrator"]["v"].new_full(
                (args.steps + 1, *functional.shape), -62.0
            )
            if args.target_pattern == "ramp":
                # Distinct frames expose wrong indexing that a constant target hides.
                offsets = torch.arange(
                    args.steps + 1, dtype=target.dtype, device=target.device
                )
                target = target + 0.7 * offsets.reshape(
                    -1, *([1] * len(functional.shape))
                )
            configured = {}
            if args.mode in ("mse", "both"):
                configured["mse"] = TraceMSE(target, getattr(torch, args.counter_dtype))
            if args.mode in ("recorder", "both"):
                configured["trace"] = dn.func.Recorder(
                    ["v", "hh.m", "t"] if args.fixture == "hh" else ["v", "t"]
                )
            if args.mode == "energy":
                configured["energy"] = VoltageEnergy()
            callbacks = functional.make_callbacks(configured)
            callback_state, _initial_emissions = callbacks._initialize(state)
            carries = callback_state.carries
            first_sample = pytree.tree_map(lambda x: x[0], samples)
            stage = "capture"
            body = CapturedCallbackStep(
                functional,
                callbacks,
                (parameters, prepared, state, carries, first_sample),
            )
            result["captured_body"] = graph_size(body.graph)
            graph_records = []
            if args.backend == "eager":
                fn = body.scanned
            else:
                from torch._dynamo.backends.registry import lookup_backend

                backend = lookup_backend(args.backend)

                def record_graph(graph, examples):
                    graph_records.append(graph_size(graph))
                    return backend(graph, examples)

                fn = torch.compile(
                    body.scanned, backend=record_graph, fullgraph=True, dynamic=False
                )

            stage = "oracle"
            expected = body.ordinary(parameters, prepared, state, carries, samples)
            expected_gradients = torch.autograd.grad(
                objective(expected),
                tuple(targets.values()),
                create_graph=args.hvp,
                retain_graph=True,
            )
            stage = "candidate_forward"
            if args.checkpoint_steps:
                parts = []
                current, current_carries = state, carries
                for start in range(0, args.steps, args.checkpoint_steps):
                    piece = pytree.tree_map(
                        lambda x: x[start : start + args.checkpoint_steps], samples
                    )

                    def execute(current, current_carries, piece):
                        return fn(parameters, prepared, current, current_carries, piece)

                    (current, current_carries), emitted = checkpoint(
                        execute, current, current_carries, piece, use_reentrant=False
                    )
                    parts.append(emitted)
                actual = (
                    (current, current_carries),
                    pytree.tree_map(lambda *xs: torch.cat(xs), *parts),
                )
            else:
                actual = fn(parameters, prepared, state, carries, samples)
            actual_leaves, spec = pytree.tree_flatten(actual)
            expected_leaves, expected_spec = pytree.tree_flatten(expected)
            assert spec == expected_spec
            value_rtol, value_atol = (
                (2e-10, 2e-11) if functional.dtype == torch.float64 else (2e-5, 2e-5)
            )
            for a, e in zip(actual_leaves, expected_leaves, strict=True):
                torch.testing.assert_close(a, e, rtol=value_rtol, atol=value_atol)
            result["forward"] = "pass"
            if "mse" in actual[0][1]:
                frames = actual[0][1]["mse"]["frames"]
                result["counter"] = {
                    "dtype": str(frames.dtype),
                    "value": frames.item(),
                    "requires_grad": frames.requires_grad,
                }
                if frames.requires_grad:
                    counter_gradients = torch.autograd.grad(
                        frames,
                        tuple(targets.values()),
                        retain_graph=True,
                        allow_unused=True,
                    )
                    zero_gradient = all(
                        gradient is None or bool((gradient == 0).all())
                        for gradient in counter_gradients
                    )
                    result["counter"]["zero_source_gradients"] = zero_gradient
                    assert zero_gradient, "bookkeeping counter depends on model sources"
            result["emissions"] = {
                name: list(value.shape)
                for name, value in actual[1].get("trace", {}).items()
            }
            stage = "candidate_backward"
            gradients = torch.autograd.grad(
                objective(actual),
                tuple(targets.values()),
                create_graph=args.hvp,
                retain_graph=True,
                allow_unused=True,
            )
            tolerance = 2e-8 if functional.dtype == torch.float64 else 2e-4
            result["gradients"] = compare_gradients(
                gradients, expected_gradients, targets, tolerance
            )
            result["accepted"] = all(
                value["pass"] for value in result["gradients"].values()
            )
            if args.hvp:
                stage = "candidate_hvp"
                direction_name = (
                    "initial_voltage" if "initial_voltage" in targets else "amplitude"
                )
                direction_index = list(targets).index(direction_name)
                expected_hvp = torch.autograd.grad(
                    expected_gradients[direction_index].sum(),
                    tuple(targets.values()),
                    retain_graph=True,
                )
                actual_hvp = torch.autograd.grad(
                    gradients[direction_index].sum(),
                    tuple(targets.values()),
                    allow_unused=True,
                )
                result["hvp"] = compare_gradients(
                    actual_hvp, expected_hvp, targets, tolerance
                )
                result["accepted"] &= all(
                    value["pass"] for value in result["hvp"].values()
                )
            elif args.checkpoint_steps:
                stage = "repeated_backward"
                repeated = torch.autograd.grad(
                    objective(actual), tuple(targets.values()), allow_unused=True
                )
                result["repeated_backward"] = compare_gradients(
                    repeated, expected_gradients, targets, tolerance
                )
                result["accepted"] &= all(
                    value["pass"] for value in result["repeated_backward"].values()
                )
            result["compiled_graphs"] = graph_records
    except Exception as error:
        result["accepted"] = False
        result["failure_stage"] = stage
        result["exception_type"] = type(error).__name__
        result["error"] = str(error)
        result["traceback"] = traceback.format_exc()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["eager", "aot_eager", "inductor"], default="eager"
    )
    parser.add_argument(
        "--mode", choices=["recorder", "energy", "mse", "both"], default="recorder"
    )
    parser.add_argument("--fixture", choices=["hh", "intra", "extra"], default="hh")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    parser.add_argument("--hvp", action="store_true")
    parser.add_argument(
        "--counter-dtype",
        choices=["int64", "float64"],
        default="int64",
        help="Experimental MSE carry representation; indexing still uses int64.",
    )
    parser.add_argument(
        "--target-pattern", choices=["constant", "ramp"], default="constant"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.checkpoint_steps < 0:
        parser.error("steps must be positive; checkpoint-steps must be nonnegative")
    result = run_probe(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("traceback", "error")
            },
            indent=2,
        ),
        flush=True,
    )
    return int(not result["accepted"])


if __name__ == "__main__":
    raise SystemExit(main())
