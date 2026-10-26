"""Recorded inference must not reuse a step graph from another assembly policy."""

import pytest
import torch

import dendra as dn

from .test_membrane_assembly_context import model


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "inductor"])
@pytest.mark.parametrize("execution", ["default", "unrolled"])
def test_existing_recorded_kernel_respects_policy_and_capture_failure_isolation(
    backend, execution
):
    source = model()
    functional, tensors = dn.func.make_functional(source, dt=0.01)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    callbacks = functional.make_callbacks({"trace": dn.func.Recorder(["v", "t"])})
    kernel = functional.compile_rollout_chunk(4, execution=execution, backend=backend)
    bound = kernel.bind(tensors.parameters, prepared)
    eager = functional.bind(tensors.parameters, prepared)

    def run(binding):
        final, auxiliary = dn.func.run(
            binding, tensors.state, steps=8, callbacks=callbacks
        )
        trace = auxiliary["callbacks"]["trace"]
        return final["integrator"]["v"], trace["v"], trace["t"]

    results = []
    for policy in ("auto", "off", "auto"):
        with dn.ctx(MEMBRANE_ASSEMBLY=policy), torch.no_grad():
            expected = run(eager)
            actual = run(bound)
        for value, reference in zip(actual, expected):
            torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)
        assert actual[1].shape == (9, *source.shape)
        assert actual[2].shape[0] == 9
        results.append(tuple(value.clone() for value in actual))
        assert kernel.execution_report().callbacks
        assert kernel.execution_report().strategy == (
            "compiled_while_loop" if execution == "default" else "compiled_unrolled"
        )

    for value, repeated in zip(results[0], results[2]):
        torch.testing.assert_close(value, repeated, rtol=0, atol=0)
    assert not torch.allclose(results[0][0], results[1][0])
    assert not torch.allclose(results[0][1], results[1][1])
    with dn.ctx(MEMBRANE_ASSEMBLY="validate"), torch.no_grad():
        with pytest.raises(RuntimeError, match="Invalid AFFINE"):
            run(bound)

    # A failed diagnostic capture must not poison the valid policy entries.
    for policy, reference in (("off", results[1]), ("auto", results[0])):
        with dn.ctx(MEMBRANE_ASSEMBLY=policy), torch.no_grad():
            actual = run(bound)
        for value, expected in zip(actual, reference):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_scan_graph_cache_is_partitioned_by_policy_without_executing_scan(monkeypatch):
    # Real scan execution has an exact supported-Torch gate. This unit test
    # tests only dispatch/cache ownership using a fake capture, not an
    # unsupported scan implementation or an altered numerical compatibility gate.
    from types import SimpleNamespace

    import dendra.func._scan as scan_module
    import dendra.func._scan_tree as scan_tree_module

    monkeypatch.setattr(scan_module, "install_scan_compatibility", lambda: None)
    monkeypatch.setattr(scan_tree_module, "install_scan_compatibility", lambda: None)
    captured = []

    def capture(
        self,
        parameters,
        prepared,
        state,
        carries,
        samples,
        sample_layout,
        *,
        grad_enabled,
    ):
        assert grad_enabled
        policy = dn.current_membrane_assembly()
        captured.append(policy)

        def execute(parameters, prepared, current, carries, samples):
            final = {
                "integrator": {
                    "v": current["integrator"]["v"].new_tensor(
                        ("auto", "off", "validate").index(policy)
                    )
                },
                "control": current["control"],
            }
            return (final, carries), {}

        layout = SimpleNamespace(unpack=lambda packed: packed)
        return execute, layout, lambda operands: operands

    monkeypatch.setattr(scan_tree_module._CompiledTensorTreeScan, "_capture", capture)
    functional = SimpleNamespace(
        device=torch.device("cpu"),
        intra=SimpleNamespace(enabled=False),
        extra=SimpleNamespace(enabled=False),
        _clone_state_tree=lambda value: value,
    )
    state = {
        "integrator": {"v": torch.tensor(0.0)},
        "control": {"duration_remainder": torch.tensor(0.0)},
    }
    kernel = scan_module._CompiledScanPopulationChunk(
        functional, steps=4, compile_options={}
    )
    for policy in ("auto", "off", "validate", "auto", "off", "validate"):
        with dn.ctx(MEMBRANE_ASSEMBLY=policy), torch.enable_grad():
            final, auxiliary = kernel({}, {}, state, None, None)
            assert final["integrator"]["v"].item() == (
                "auto",
                "off",
                "validate",
            ).index(policy)
            assert auxiliary["v"] is final["integrator"]["v"]
    assert captured == ["auto", "off", "validate"]
    assert len(kernel._training._kernels) == 3
