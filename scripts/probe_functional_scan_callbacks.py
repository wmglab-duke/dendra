"""Exercise real Dendra discrete callbacks with an optional local Torch patch.

This acceptance tool uses captured tensor kernels and callback lifecycle helpers;
it does not add a public scan runner. The model suite uses finite HH/PCR dynamics.
The callback suite uses prescribed observations (including nonfinite faults) and
an independent finite recurrence, so faults cannot contaminate gradient oracles.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import traceback
from pathlib import Path

import probe_functional_scan as base
import torch
from torch.utils import _pytree as pytree
from torch.utils.checkpoint import checkpoint

from dendra.func._callbacks import _update_callback_values


def load_patch(path):
    spec = importlib.util.spec_from_file_location("candidate_torch_patch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.install_patch()


class ObservationBody(base.CapturedCallbackStep):
    def authored(self, parameters, prepared, state, carries, sample):
        updated = {
            **state,
            "integrator": {**state["integrator"], "v": sample["observed_v"].clone()},
            "clock": {**state["clock"], "t": state["clock"]["t"] + base.DT},
            "latent": torch.tanh(
                0.8 * state["latent"] + parameters["gain"] * sample["drive"]
            ),
        }
        carries, emitted = _update_callback_values(
            self.callbacks._plans, carries, updated, {"v": updated["integrator"]["v"]}
        )
        return base.clone((updated, carries)), base.clone(emitted or {})


def fixture(args):
    steps = 10 if args.suite == "model" else 9
    functional, parameters, prepared, state, samples, targets = base.hh_case(steps)
    if args.suite == "model":
        signs = state["clock"]["t"].new_tensor([1, 1, -1, -1, 1, 1, -1, -1, 1, 1])
        samples["intra"] = (
            signs.reshape(steps, 1, 1).expand_as(samples["intra"]).clone() * 1e-8
        ).requires_grad_()
        targets["intra"] = samples["intra"]
        threshold, start, end = -60.0, 1, 9
        body_type = base.CapturedCallbackStep
    else:
        state = base.detached(state)
        state["integrator"]["v"].fill_(-1)
        state["latent"] = torch.tensor(0.13, dtype=torch.float64, requires_grad=True)
        parameters = {
            "gain": torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
        }
        prepared = {}
        observed = torch.full((steps, *functional.shape), -1.0, dtype=torch.float64)
        observed[:, :, 0] = observed.new_tensor([1, 1, 1, 1, -1, 0, 1, -1, 1])[:, None]
        observed[:, :, 1] = observed.new_tensor([-1, -1, -1, 0, 1, -1, 1, 1, -1])[
            :, None
        ]
        # Faults affect only detector/Recorder observations, outside the loss path.
        observed[4, 0, 4] = float("nan")
        observed[6, 1, 3] = float("inf")
        samples = {
            "observed_v": observed,
            "drive": torch.linspace(
                0.1, 0.9, steps, dtype=torch.float64
            ).requires_grad_(),
        }
        targets = {
            "gain": parameters["gain"],
            "initial": state["latent"],
            "drive": samples["drive"],
        }
        threshold, start, end = 0.0, 2, 8
        body_type = ObservationBody
    configured = {
        "raster": base.dn.func.Raster(
            threshold=threshold,
            node_check=[0, 1],
            t_start_check=start * base.DT,
            t_end_check=end * base.DT,
        ),
        "count": base.dn.func.APCount(
            threshold=threshold,
            node_check=[0, 1],
            t_start_check=start * base.DT,
            t_end_check=end * base.DT,
        ),
        "anomaly": base.dn.func.AnomalyDetector(),
    }
    if args.configuration == "combined":
        configured["trace"] = base.dn.func.Recorder(["v", "t"])
        if args.suite == "model":
            target = state["integrator"]["v"].new_full(
                (steps + 1, *functional.shape), -62.0
            )
            target += 0.7 * torch.arange(steps + 1, dtype=target.dtype).reshape(
                -1, 1, 1
            )
            configured["mse"] = base.TraceMSE(target)
    elif args.configuration == "reducers":
        del configured["raster"]
    if args.reverse:
        configured = dict(reversed(tuple(configured.items())))
    callbacks = functional.make_callbacks(configured)
    initial, emission = callbacks._initialize(state)
    if args.seed_anomaly:
        if args.suite != "model":
            raise ValueError("seeded anomaly carry is a model-suite option")
        fault = base.detached(state)
        fault["integrator"]["v"][0, 4] = float("nan")
        # Generate the resumed mask through the actual reducer, not a dtype cast.
        initial.carries["anomaly"], _ = configured["anomaly"].update(
            initial.carries["anomaly"], fault, None
        )
    body = body_type(
        functional,
        callbacks,
        (
            parameters,
            prepared,
            state,
            initial.carries,
            pytree.tree_map(lambda x: x[0], samples),
        ),
    )
    return (
        steps,
        parameters,
        prepared,
        state,
        samples,
        targets,
        callbacks,
        initial,
        emission,
        body,
    )


def objective(state, results):
    if "latent" in state:
        return state["latent"].square()
    value = state["integrator"]["v"].square().mean()
    if "mse" in results:
        value = value + results["mse"]
    if "trace" in results:
        value = value + results["trace"]["v"].square().mean()
    return value


def compare(actual, expected, *, exact=False):
    left, spec = pytree.tree_flatten(actual)
    right, expected_spec = pytree.tree_flatten(expected)
    assert spec == expected_spec
    for a, e in zip(left, right, strict=True):
        torch.testing.assert_close(
            a,
            e,
            rtol=0 if exact else 2e-9,
            atol=0 if exact else 2e-10,
            equal_nan=True,
        )
        if torch.is_tensor(e) and not (e.dtype.is_floating_point or e.dtype.is_complex):
            assert not a.requires_grad


def execute(args, structured):
    (
        steps,
        parameters,
        prepared,
        state,
        samples,
        targets,
        callbacks,
        carry,
        initial_emission,
        body,
    ) = fixture(args)
    spans = [int(x) for x in args.segments.split(",")] if args.segments else [steps]
    assert sum(spans) == steps and all(x >= 0 for x in spans)
    fn = body.scanned if structured else body.ordinary
    graphs = []
    if structured and args.backend != "eager":
        from torch._dynamo.backends.registry import lookup_backend

        backend = lookup_backend(args.backend)

        def record(graph, examples):
            graphs.append(base.graph_size(graph))
            return backend(graph, examples)

        fn = torch.compile(fn, backend=record, fullgraph=True, dynamic=False)
    original_inputs = base.clone((parameters, prepared, state, samples, carry.carries))
    live_inputs = (parameters, prepared, state, samples, carry.carries)
    parts = [callbacks._stack([initial_emission or {}])]
    post_parts, boundaries, consumed = [], [], 0
    for width in spans:
        old_carry = carry
        if not width:
            # Same host lifecycle as a zero-step continuation: no kernel invocation.
            emitted = callbacks._stack([])
            new_carries = carry.carries
        else:
            subparts = []
            new_carries = carry.carries
            interval = args.checkpoint_steps or width
            for offset in range(consumed, consumed + width, interval):
                stop = min(consumed + width, offset + interval)
                piece = pytree.tree_map(lambda x: x[offset:stop], samples)

                def block(current, carries, inputs):
                    return fn(parameters, prepared, current, carries, inputs)

                if args.checkpoint_steps:
                    (state, new_carries), output = checkpoint(
                        block, state, new_carries, piece, use_reentrant=False
                    )
                else:
                    (state, new_carries), output = block(state, new_carries, piece)
                subparts.append(
                    {plan.name: output.get(plan.name) for plan in callbacks._plans}
                )
            emitted = callbacks._concatenate(subparts)
        carry = callbacks._wrap_compiled_update(old_carry, new_carries, emitted)
        carry = callbacks._capture_emission_schemas(carry, emitted)
        consumed += width
        parts.append(emitted)
        post_parts.append(emitted)
        zero_results = callbacks._finalize(carry, emitted)
        boundaries.append(
            {
                "steps": consumed,
                "phase": carry["count"][2].item(),
                "count": carry["count"][1].tolist(),
                "anomaly": carry["anomaly"].tolist(),
                "raster_samples": zero_results["raster"].shape[0]
                if "raster" in zero_results
                else None,
                "raster_schema_known": any(
                    plan.name == "raster" and schema is not None
                    for plan, schema in zip(
                        callbacks._plans, carry.emission_schemas, strict=True
                    )
                ),
            }
        )
        assert carry["count"][2].item() == consumed
    outputs = callbacks._finalize(carry, callbacks._concatenate(parts))
    post = callbacks._concatenate(post_parts)
    if "raster" in outputs:
        assert outputs["raster"].dtype == torch.bool
        assert outputs["raster"].shape[0] == steps
        assert bool(outputs["raster"].any()) and bool((~outputs["raster"]).any())
        torch.testing.assert_close(outputs["count"], outputs["raster"].sum(0))
    assert outputs["count"].dtype == torch.int64 and bool((outputs["count"] > 0).all())
    if "trace" in outputs:
        assert outputs["trace"]["v"].shape[0] == steps + 1
    if "mse" in outputs:
        assert carry["mse"]["frames"].dtype == torch.int64
        assert carry["mse"]["frames"].item() == steps + 1
    if args.suite == "callbacks":
        expected_raster = torch.zeros((steps, 2, 2), dtype=torch.bool)
        expected_raster[[2, 5], :, 0] = True
        expected_raster[[3, 6], :, 1] = True
        if "raster" in outputs:
            torch.testing.assert_close(outputs["raster"], expected_raster)
        torch.testing.assert_close(
            outputs["count"], torch.full((2, 2), 2, dtype=torch.int64)
        )
        torch.testing.assert_close(
            carry["count"][0], torch.tensor([[True, False], [True, False]])
        )
        torch.testing.assert_close(outputs["anomaly"], torch.tensor([True, True]))
    else:
        torch.testing.assert_close(
            outputs["anomaly"], torch.tensor([args.seed_anomaly, False])
        )
    compare(live_inputs, original_inputs, exact=True)
    values = (state, carry.carries, outputs, post)
    before_backward = pytree.tree_map(
        lambda value: value.clone() if torch.is_tensor(value) else value, values
    )
    gradients = torch.autograd.grad(
        objective(state, outputs), tuple(targets.values()), retain_graph=True
    )
    repeated = torch.autograd.grad(
        objective(state, outputs), tuple(targets.values()), retain_graph=True
    )
    compare(values, before_backward, exact=True)
    compare(live_inputs, original_inputs, exact=True)
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and gradient.abs().amax() > 0
    return values, gradients, repeated, tuple(targets), boundaries, graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["model", "callbacks"], default="model")
    parser.add_argument(
        "--configuration",
        choices=["combined", "thresholds", "reducers"],
        default="combined",
    )
    parser.add_argument(
        "--backend", choices=["eager", "aot_eager", "inductor"], default="eager"
    )
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    parser.add_argument("--segments", default="")
    parser.add_argument("--seed-anomaly", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--patch-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint_steps < 0:
        parser.error("checkpoint steps must be nonnegative")
    torch.set_num_threads(1)
    result = {
        "torch": torch.__version__,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "accepted": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
    }
    stage = "patch"
    try:
        if args.patch_file:
            result["patch"] = load_patch(args.patch_file)
        with base.compiler_warnings():
            stage = "oracle"
            expected = execute(args, False)
            result["oracle_pass"] = True
            stage = "candidate"
            actual = execute(args, True)
            stage = "parity"
            compare(actual[0], expected[0])
            assert actual[3] == expected[3] and actual[4] == expected[4]
            result["gradients"] = base.compare_gradients(
                actual[1], expected[1], actual[3], 2e-8
            )
            result["repeated_backward"] = base.compare_gradients(
                actual[2], expected[1], actual[3], 2e-8
            )
            result["accepted"] = all(
                x["pass"]
                for collection in (result["gradients"], result["repeated_backward"])
                for x in collection.values()
            )
            result["boundaries"] = actual[4]
            result["compiled_graphs"] = actual[5]
    except Exception as error:
        result.update(
            failure_stage=stage, error=str(error), traceback=traceback.format_exc()
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("traceback", "error", "patch")
            },
            indent=2,
        ),
        flush=True,
    )
    raise SystemExit(int(not result["accepted"]))


if __name__ == "__main__":
    main()
