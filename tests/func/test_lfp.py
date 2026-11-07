"""Pure electrode projection, explicit batching, and differentiable lead fields."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._callback_carry import _CallbackCarryPartition
from dendra.func._callbacks import _prepare_callback
from dendra.func._carry_packing import _PackedCarryLayout
from dendra.models.mod import hh

DT = 0.01
STEPS = 5
pytestmark = pytest.mark.cpu


class _VoltageEnergy(dn.func.FunctionalCallback):
    """A genuinely changing custom carry, never eligible for invariant lifting."""

    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        del auxiliary
        return carry + state["integrator"]["v"].square().mean(), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


def _model(*, batch=(), imem=True, device="cpu", dtype=torch.float64):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=imem, DEVICE=device, DTYPE=dtype):
        model = dn.Population(
            N=2,
            C=3,
            celsius=6.3,
            v_init=-63.0,
            dtype=dtype,
            integrator=dn.bwd_euler_sc(imem=imem),
        )
        model.insert(hh)
        for size in batch:
            model.batch(size)
        model.initialize()
        model.train()
    return model


def _fields(model, *, trainable=False):
    field = torch.linspace(
        1.0, 2.0, 6, device=model.device(), dtype=model.dtype()
    ).reshape(2, 3)
    return [
        field.detach().clone().requires_grad_(trainable),
        field.flip(-1).detach().clone().requires_grad_(trainable),
    ]


def _inputs(model, steps=STEPS, *, trainable=False):
    drive = torch.linspace(
        -2.0e-7,
        3.0e-7,
        steps * model.v.numel(),
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(steps, *model.shape)
    return dn.func.RolloutInput(intra=drive.requires_grad_(trainable))


def _eager(functional, tensors, callbacks, inputs, *, state=None, callback_state=None):
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    return dn.func.run(
        functional.bind(tensors.parameters, prepared),
        tensors.state if state is None else state,
        inputs=inputs,
        steps=inputs.intra.shape[0],
        callbacks=callbacks,
        callback_state=callback_state,
    )


def _assert_projection(outputs, fields, batch):
    current = outputs["trace"]["i_membrane"]
    padded = torch.stack(fields).reshape(2, *((1,) * len(batch)), 2, 3)
    expected = (current.unsqueeze(1) * padded.unsqueeze(0)).sum((-1, -2))
    expected = expected.movedim(1, -1)
    assert outputs["lfp"].shape == (current.shape[0], *batch, 2)
    torch.testing.assert_close(outputs["lfp"], expected, rtol=2.0e-11, atol=1.0e-20)
    assert torch.count_nonzero(outputs["lfp"][0]) == 0


@pytest.mark.parametrize("batch", ((), (2,), (2, 2)))
def test_lfp_records_initial_and_accepted_projections_without_summing_batches(
    cpu_threads, batch
):
    model = _model(batch=batch)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    fields = _fields(model)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("v", "i_membrane")), "lfp": dn.func.LFP(fields)}
    )
    _, auxiliary = _eager(functional, tensors, callbacks, _inputs(model))
    outputs = auxiliary["callbacks"]
    _assert_projection(outputs, fields, batch)
    assert outputs["lfp"].shape[0] == STEPS + 1


def test_lfp_default_matches_imperative_and_explicit_rule(cpu_threads):
    model = _model()
    fields = _fields(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks(
        {"lfp": dn.func.LFP(fields), "custom": dn.func.LFP(fields, rule="nc,knc->k")}
    )
    inputs = _inputs(model)
    _, auxiliary = _eager(functional, tensors, callbacks, inputs)
    expected = dn.callbacks.LFP(fields)
    expected.pre_loop_hook(model)
    timestep = model.v.new_tensor(DT)
    model.integrator._initialize(
        model, timestep, force=True, compile_scope="population"
    )
    for index in range(STEPS):
        model.integrator.step(model, timestep, None, inputs.intra[index])
        model.t = model.t + timestep
        expected.post_step_hook(model)
    outputs = auxiliary["callbacks"]
    torch.testing.assert_close(outputs["lfp"], expected.lfp, rtol=2.0e-11, atol=1.0e-20)
    torch.testing.assert_close(
        outputs["custom"], expected.lfp, rtol=2.0e-11, atol=1.0e-20
    )


@pytest.mark.parametrize("shape", ((), (3,), (2, 1, 3)))
def test_lfp_broadcast_fields_and_dtype_conversion_preserve_gradients(
    cpu_threads, shape
):
    model = _model(batch=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    field = torch.full(shape, 1.25, dtype=torch.float32, requires_grad=True)
    callbacks = functional.make_callbacks(
        {"trace": dn.func.Recorder(("i_membrane",)), "lfp": dn.func.LFP([field])}
    )
    _, auxiliary = _eager(functional, tensors, callbacks, _inputs(model))
    outputs = auxiliary["callbacks"]
    expected = (outputs["trace"]["i_membrane"] * field.double()).sum((-1, -2))
    torch.testing.assert_close(outputs["lfp"].squeeze(-1), expected)
    gradient = torch.autograd.grad((outputs["lfp"] * 1.0e6).square().mean(), field)[0]
    assert gradient.dtype == torch.float32
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0


def test_lfp_field_and_stimulus_gradients_survive_fresh_runs_and_checkpointing(
    cpu_threads,
):
    model = _model(batch=(2,))
    functional, tensors = dn.func.make_functional(model, dt=DT)
    fields = _fields(model, trainable=True)
    callbacks = functional.make_callbacks({"lfp": dn.func.LFP(fields)})
    expected = None
    for runner in ("run", "run", "longrun_checkpointed"):
        inputs = _inputs(model, trainable=True)
        prepared = functional.prepare(tensors.parameters, tensors.constants)
        binding = functional.bind(tensors.parameters, prepared)
        if runner == "run":
            _, auxiliary = dn.func.run(
                binding, tensors.state, inputs=inputs, steps=STEPS, callbacks=callbacks
            )
        else:
            _, auxiliary = dn.func.longrun_checkpointed(
                functional,
                binding,
                tensors.state,
                tstop=STEPS * DT,
                chunklength=2,
                inputs=inputs,
                callbacks=callbacks,
            )
        outputs = auxiliary["callbacks"]
        assert tuple(outputs) == ("lfp",)  # No hidden full-current trajectory.
        result = outputs["lfp"]
        gradients = torch.autograd.grad(
            (result * 1.0e6).square().mean(), (*fields, inputs.intra)
        )
        for gradient in gradients:
            assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        if expected is None:
            expected = (result.detach(), tuple(value.detach() for value in gradients))
        else:
            torch.testing.assert_close(result, expected[0], rtol=2.0e-11, atol=1.0e-20)
            for actual, reference in zip(gradients, expected[1], strict=True):
                torch.testing.assert_close(
                    actual, reference, rtol=2.0e-10, atol=1.0e-15
                )


def test_lfp_explicit_continuation_has_no_duplicate_boundary(cpu_threads):
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks({"lfp": dn.func.LFP(_fields(model))})
    inputs = _inputs(model)
    _, all_auxiliary = _eager(functional, tensors, callbacks, inputs)
    first_state, first_auxiliary = _eager(
        functional, tensors, callbacks, dn.func.RolloutInput(intra=inputs.intra[:2])
    )
    _, second_auxiliary = _eager(
        functional,
        tensors,
        callbacks,
        dn.func.RolloutInput(intra=inputs.intra[2:]),
        state=first_state,
        callback_state=first_auxiliary["callbacks"].state,
    )
    joined = torch.cat(
        (first_auxiliary["callbacks"]["lfp"], second_auxiliary["callbacks"]["lfp"])
    )
    torch.testing.assert_close(joined, all_auxiliary["callbacks"]["lfp"])


@pytest.mark.parametrize(
    ("fields", "error", "message"),
    (
        (torch.ones(2, 3), TypeError, "list"),
        ([], ValueError, "at least one"),
        ([1.0], TypeError, "Tensor"),
        ([torch.ones(2), torch.ones(3)], ValueError, "same shape"),
    ),
)
def test_lfp_constructor_validates_fields(fields, error, message):
    with pytest.raises(error, match=message):
        dn.func.LFP(fields)


def test_lfp_requires_current_and_valid_default_broadcast():
    for imem, fields, message in (
        (False, [torch.ones(2, 3)], "membrane-current"),
        (True, [torch.ones(4)], "broadcast"),
    ):
        functional, _ = dn.func.make_functional(_model(imem=imem), dt=DT)
        with pytest.raises(dn.func.FunctionalizationError, match=message):
            functional.make_callbacks({"lfp": dn.func.LFP(fields)})


def test_only_exact_builtin_lfp_fields_are_excluded_from_recurrent_packing():
    class DerivedLFP(dn.func.LFP):
        pass

    options = dict(
        fields=(), rule=None, padding=0, device=torch.device("cpu"), dtype=torch.float64
    )
    fields = torch.zeros(64, 1024, 2).transpose(-1, -2)
    carries = {"lfp": fields, "custom": fields[:1], "reducer": torch.ones(3)}
    plans = (
        SimpleNamespace(name="lfp", callback=_prepare_callback(dn.func.LFP, options)),
        SimpleNamespace(name="custom", callback=_prepare_callback(DerivedLFP, options)),
        SimpleNamespace(name="reducer", callback=dn.func.AnomalyDetector()),
    )
    partition = _CallbackCarryPartition.from_plans(plans)
    recurrent, invariant = partition.split(carries)
    assert tuple(invariant) == ("lfp",) and invariant["lfp"] is fields
    assert tuple(recurrent) == ("custom", "reducer")
    layout = _PackedCarryLayout.from_tree((torch.ones(8), recurrent))
    assert sum(leaf.numel for leaf in layout.leaves) == 8 + fields[:1].numel() + 3
    restored = partition.merge(recurrent, invariant)
    assert tuple(restored) == tuple(carries)
    assert all(restored[name] is carry for name, carry in carries.items())
    empty = _CallbackCarryPartition.from_plans(plans[1:])
    assert empty.split(recurrent)[0] is recurrent
    assert empty.merge(recurrent, {}) is recurrent


@pytest.mark.slow
@pytest.mark.parametrize("execution", ("scan", "while", "scan_inference"))
def test_cpp_lfp_matches_eager_and_scan_preserves_field_gradients(
    cpu_threads, execution, monkeypatch
):
    packed_callback_keys = []
    original_from_tree = _PackedCarryLayout.from_tree.__func__

    def observe_layout(cls, tree):
        if isinstance(tree, tuple) and len(tree) == 2 and isinstance(tree[1], dict):
            packed_callback_keys.append(tuple(tree[1]))
        return original_from_tree(cls, tree)

    monkeypatch.setattr(_PackedCarryLayout, "from_tree", classmethod(observe_layout))
    model = _model(batch=(2,))
    # Non-contiguous user fields must remain explicit live runtime operands.
    fields = [
        field.T.contiguous().T.detach().requires_grad_(execution == "scan")
        for field in _fields(model)
    ]
    inputs = _inputs(model, trainable=execution == "scan")
    specifications = {"lfp": dn.func.LFP(fields)}
    functional, tensors = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks(specifications)
    _, expected_auxiliary = _eager(functional, tensors, callbacks, inputs)
    expected = expected_auxiliary["callbacks"]["lfp"]
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=STEPS,
        execution="while" if execution == "while" else "scan",
        inference=execution != "scan",
        callbacks=specifications,
        backend="inductor",
        options={"cpp_wrapper": True, "compile_threads": 1},
    )
    with torch_compiler_warning_context():
        _, auxiliary = runner(inputs=inputs)
    actual = auxiliary["callbacks"]["lfp"]
    torch.testing.assert_close(actual, expected, rtol=2.0e-10, atol=1.0e-18)
    assert runner.execution_report().strategy == (
        "compiled_while_loop" if execution == "while" else "compiled_scan"
    )
    # Blocked inference scan uses an unpacked tuple carry rather than the
    # training/while packed layout. Every layout that exists must omit fields.
    assert all("lfp" not in keys for keys in packed_callback_keys)
    if execution != "scan_inference":
        assert packed_callback_keys
    if execution == "scan":
        targets = (*fields, inputs.intra)
        reference = torch.autograd.grad((expected * 1.0e6).square().mean(), targets)
        gradients = torch.autograd.grad((actual * 1.0e6).square().mean(), targets)
        for gradient, expected_gradient in zip(gradients, reference, strict=True):
            torch.testing.assert_close(
                gradient, expected_gradient, rtol=2.0e-8, atol=1.0e-10
            )
    # Reuse the compiled callback after backward, with changed field values.
    # A cached stack/closed-over capture would either retain stale values or
    # attempt a second backward through the already-freed initialization graph.
    with torch.no_grad():
        fields[0].add_(0.125)
    _, repeated_reference = _eager(functional, tensors, callbacks, inputs)
    with torch_compiler_warning_context():
        _, repeated_auxiliary = runner(inputs=inputs)
    repeated = repeated_auxiliary["callbacks"]["lfp"]
    reference = repeated_reference["callbacks"]["lfp"]
    torch.testing.assert_close(repeated, reference, rtol=2.0e-10, atol=1.0e-18)
    if execution == "scan":
        expected_gradients = torch.autograd.grad(
            (reference * 1.0e6).square().mean(), targets
        )
        actual_gradients = torch.autograd.grad(
            (repeated * 1.0e6).square().mean(), targets
        )
        for gradient, reference_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            torch.testing.assert_close(
                gradient, reference_gradient, rtol=2.0e-8, atol=1.0e-10
            )


@pytest.mark.slow
def test_checkpointed_scan_resumes_strided_invariant_field_carry_with_full_gradients(
    cpu_threads,
):
    model = _model(batch=(2,))
    fields = _fields(model, trainable=True)
    inputs = _inputs(model, steps=8, trainable=True)
    specifications = {"lfp": dn.func.LFP(fields), "energy": _VoltageEnergy()}
    functional, tensors = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks(specifications)
    _, expected_auxiliary = _eager(functional, tensors, callbacks, inputs)
    reference = expected_auxiliary["callbacks"]["lfp"]
    runner = dn.func.compile(
        model,
        dt=DT,
        steps=4,
        execution="scan",
        inference=False,
        checkpoint_every=2,
        callbacks=specifications,
        backend="inductor",
        options={"cpp_wrapper": True, "compile_threads": 1},
    )
    with torch_compiler_warning_context():
        state, first = runner(inputs=dn.func.RolloutInput(intra=inputs.intra[:4]))
        callback_state = first["callbacks"].state
        value = callback_state.carries["lfp"]
        strided = value.unsqueeze(-1).expand(*value.shape, 2).clone()[..., 0]
        assert not strided.is_contiguous()
        resumed_carry = replace(
            callback_state, carries={**callback_state.carries, "lfp": strided}
        )
        _, second = runner(
            state=state,
            callback_state=resumed_carry,
            inputs=dn.func.RolloutInput(intra=inputs.intra[4:]),
        )
    actual = torch.cat((first["callbacks"]["lfp"], second["callbacks"]["lfp"]))
    torch.testing.assert_close(actual, reference, rtol=2.0e-10, atol=1.0e-18)
    actual_energy = second["callbacks"]["energy"]
    expected_energy = expected_auxiliary["callbacks"]["energy"]
    assert actual_energy > first["callbacks"]["energy"] > 0
    torch.testing.assert_close(
        actual_energy, expected_energy, rtol=2.0e-10, atol=1.0e-10
    )
    returned = second["callbacks"].state.carries["lfp"]
    torch.testing.assert_close(returned, strided)
    assert not torch._C._is_alias_of(returned, strided)
    targets = (*fields, inputs.intra)
    expected_gradients = torch.autograd.grad(
        (reference * 1.0e6).square().mean() + 1.0e-3 * expected_energy, targets
    )
    actual_gradients = torch.autograd.grad(
        (actual * 1.0e6).square().mean() + 1.0e-3 * actual_energy, targets
    )
    for gradient, reference_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            gradient, reference_gradient, rtol=2.0e-8, atol=1.0e-10
        )
