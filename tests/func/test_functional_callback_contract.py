from functools import partial

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import hh
from dendra.units import nA

DT = 0.005
GNABAR = "integrator.mech.mechanisms.hh.gnabar_param"


class NestedMoments(dn.func.FunctionalCallback):
    """Exercise nested carry, nested emissions, auxiliary reads, and finalize."""

    @staticmethod
    def _voltage(state):
        return state["integrator"]["v"]

    @staticmethod
    def _emission(voltage):
        return {
            "sample": voltage,
            "derived": (
                voltage.mean(dim=-1),
                {"energy": voltage.square().mean(dim=-1)},
            ),
        }

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        voltage = self._voltage(state)
        carry = {
            "count": voltage.new_ones((), dtype=torch.int64),
            "moments": (
                voltage,
                {"energy": voltage.square().mean(dim=-1)},
            ),
        }
        return carry, self._emission(voltage)

    def update(self, carry, state, auxiliary):
        del state
        voltage = auxiliary["v"]
        next_carry = {
            "count": carry["count"] + 1,
            "moments": (
                carry["moments"][0] + voltage,
                {
                    "energy": carry["moments"][1]["energy"]
                    + voltage.square().mean(dim=-1)
                },
            ),
        }
        return next_carry, self._emission(voltage)

    def finalize(self, carry, emissions):
        return {
            "series": emissions,
            "summary": {
                "count": carry["count"],
                "sum": carry["moments"][0],
                "energy": carry["moments"][1]["energy"],
            },
        }


class EnergyReducer(dn.func.FunctionalCallback):
    """A constant-memory callback whose public result is derived from carry."""

    @staticmethod
    def _voltage(state):
        return state["integrator"]["v"]

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        voltage = self._voltage(state)
        carry = {
            "steps": voltage.new_zeros((), dtype=torch.int64),
            "energy": voltage.new_zeros(voltage.shape[:-1]),
        }
        return carry, None

    def update(self, carry, state, auxiliary):
        del state
        voltage = auxiliary["v"]
        return {
            "steps": carry["steps"] + 1,
            "energy": carry["energy"] + voltage.square().mean(dim=-1),
        }, None

    def finalize(self, carry, emissions):
        assert emissions is None
        return {
            "steps": carry["steps"],
            "energy": carry["energy"],
        }


class AuxiliaryReducer(dn.func.FunctionalCallback):
    """Consume an auxiliary value supplied only by a user-authored step."""

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        voltage = state["integrator"]["v"]
        return voltage.new_zeros(voltage.shape[:-1]), None

    def update(self, carry, state, auxiliary):
        del state
        return carry + auxiliary["custom_energy"], None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


class PostStepRecorder(dn.func.FunctionalCallback):
    """Exercise a callback that emits after steps but not at initialization."""

    def initialize(self, state, auxiliary):
        assert auxiliary is None
        return {"template": torch.zeros_like(state["integrator"]["v"])}, None

    def update(self, carry, state, auxiliary):
        del state
        return carry, {"v": auxiliary["v"]}

    def finalize(self, carry, emissions):
        if emissions is None:
            template = carry["template"]
            return {"v": template.new_empty((0, *template.shape))}
        return emissions


class AuxiliaryEmitter(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        del state
        return carry, {"value": auxiliary["custom_value"]}

    def finalize(self, carry, emissions):
        if emissions is None:
            return carry
        return emissions["value"].sum()


class InferenceSeedReducer(dn.func.FunctionalCallback):
    def __init__(self, seed):
        self.seed = seed

    def initialize(self, state, auxiliary):
        del state, auxiliary
        return self.seed, None

    def update(self, carry, state, auxiliary):
        del state
        return carry + auxiliary["v"].square().mean(), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return carry


def _model():
    model = dn.Unmyelinated(
        [2.0],
        L=40.0,
        dx=10.0,
        celsius=6.3,
        v_init=-65.0,
        rhoa=100.0,
        integrator=dn.bwd_euler_ub(method="thomas", imem=False),
    ).double()
    model.insert(hh, gnabar=0.05, gkbar=0.05)
    model[0, 2].inject(
        dn.mono_rect(amp=2.0 * nA, delay=0.01, pw=0.03),
    )
    model.train()
    model.initialize()
    return model


def _case():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks(
        {
            "moments": NestedMoments(),
            "trace": dn.func.Recorder(["v"]),
            "energy": EnergyReducer(),
            "anomalous": dn.func.AnomalyDetector(),
        }
    )
    return functional, tensors, callbacks


def _host_run(
    runner,
    functional,
    tensors,
    callbacks,
    steps,
    *,
    parameters=None,
    state=None,
    callback_state=None,
):
    if parameters is None:
        parameters = tensors.parameters
    if state is None:
        state = tensors.state
    prepared = functional.prepare(parameters, tensors.constants)
    step = partial(functional.step, parameters, prepared)
    if runner == "run":
        return dn.func.run(
            functional,
            step,
            state,
            tstop=steps * DT,
            callbacks=callbacks,
            callback_state=callback_state,
        )
    return getattr(dn.func, runner)(
        functional,
        step,
        state,
        steps * DT,
        2,
        callbacks=callbacks,
        callback_state=callback_state,
    )


def _assert_tree_close(actual, expected):
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    assert actual_spec == expected_spec
    for actual_leaf, expected_leaf in zip(
        actual_leaves,
        expected_leaves,
        strict=True,
    ):
        torch.testing.assert_close(actual_leaf, expected_leaf)


def _assert_callback_results_close(actual, expected):
    assert tuple(actual) == tuple(expected)
    for name in actual:
        _assert_tree_close(actual[name], expected[name])
        _assert_tree_close(actual.state[name], expected.state[name])


def test_user_callbacks_support_nested_trees_no_emission_and_heterogeneous_plans():
    functional, tensors, callbacks = _case()
    _state, auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=3,
        callbacks=callbacks,
    )
    results = auxiliary["callbacks"]

    moments = results["moments"]
    trace = results["trace"]["v"]
    assert moments["series"]["sample"].shape == (4, 1, 5)
    assert moments["series"]["derived"][0].shape == (4, 1)
    torch.testing.assert_close(moments["series"]["sample"], trace)
    torch.testing.assert_close(
        moments["summary"]["sum"],
        trace.sum(dim=0),
    )
    torch.testing.assert_close(
        moments["summary"]["energy"],
        trace.square().mean(dim=-1).sum(dim=0),
    )
    assert moments["summary"]["count"].item() == 4

    energy = results["energy"]
    assert energy["steps"].item() == 3
    torch.testing.assert_close(
        energy["energy"],
        trace[1:].square().mean(dim=-1).sum(dim=0),
    )
    assert not results["anomalous"].any()


def test_zero_step_and_resume_have_explicit_nonduplicating_lifecycle():
    functional, tensors, callbacks = _case()
    zero_state, zero_auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=0,
        callbacks=callbacks,
    )
    zero = zero_auxiliary["callbacks"]
    assert zero["moments"]["series"]["sample"].shape == (1, 1, 5)
    assert zero["trace"]["v"].shape == (1, 1, 5)
    assert zero["energy"]["steps"].item() == 0

    same_state, empty_auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        zero_state,
        steps=0,
        callbacks=callbacks,
        callback_state=zero.state,
    )
    empty = empty_auxiliary["callbacks"]
    assert empty["moments"]["series"]["sample"].shape == (0, 1, 5)
    assert empty["trace"]["v"].shape == (0, 1, 5)
    _assert_tree_close(empty.state, zero.state)
    _assert_tree_close(same_state, zero_state)


def test_update_only_emissions_stack_across_chunks_and_define_empty_result():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks({"post": PostStepRecorder()})
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    step = partial(functional.step, tensors.parameters, prepared)

    _state, zero_auxiliary = dn.func.longrun(
        functional,
        step,
        tensors.state,
        0.0,
        2,
        callbacks=callbacks,
    )
    assert zero_auxiliary["callbacks"]["post"]["v"].shape == (0, 1, 5)

    _state, auxiliary = dn.func.longrun(
        functional,
        step,
        tensors.state,
        5 * DT,
        2,
        callbacks=callbacks,
    )
    assert auxiliary["callbacks"]["post"]["v"].shape == (5, 1, 5)


def test_callback_binding_does_not_probe_fabricated_step_auxiliary():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks(
        {
            "custom": AuxiliaryReducer(),
            "trace": dn.func.Recorder(["v"]),
        }
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def custom_step(state, inputs):
        next_state, auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            inputs,
        )
        voltage = auxiliary["v"]
        return next_state, {
            **auxiliary,
            "custom_energy": voltage.square().mean(dim=-1),
        }

    _state, auxiliary = dn.func.run(
        functional,
        custom_step,
        tensors.state,
        tstop=4 * DT,
        callbacks=callbacks,
    )
    results = auxiliary["callbacks"]
    torch.testing.assert_close(
        results["custom"],
        results["trace"]["v"][1:].square().mean(dim=-1).sum(dim=0),
    )


def test_update_only_emission_schema_is_preserved_across_resume_segments():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks({"emitter": AuxiliaryEmitter()})
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    def step_with_width(width):
        def step(state, inputs):
            next_state, auxiliary = functional.step(
                tensors.parameters,
                prepared,
                state,
                inputs,
            )
            return next_state, {
                **auxiliary,
                "custom_value": auxiliary["v"][..., :width],
            }

        return step

    first_state, first_auxiliary = dn.func.run(
        functional,
        step_with_width(1),
        tensors.state,
        tstop=DT,
        callbacks=callbacks,
    )
    with pytest.raises(dn.func.FunctionalizationError, match="changed shape"):
        dn.func.run(
            functional,
            step_with_width(2),
            first_state,
            tstop=DT,
            callbacks=callbacks,
            callback_state=first_auxiliary["callbacks"].state,
        )


@pytest.mark.parametrize("runner", ["run", "longrun", "longrun_checkpointed"])
def test_custom_callback_results_match_fixed_rollout_across_host_runners(runner):
    functional, tensors, callbacks = _case()
    expected_state, expected_auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=5,
        callbacks=callbacks,
    )
    actual_state, actual_auxiliary = _host_run(
        runner,
        functional,
        tensors,
        callbacks,
        5,
    )

    _assert_tree_close(actual_state, expected_state)
    _assert_callback_results_close(
        actual_auxiliary["callbacks"],
        expected_auxiliary["callbacks"],
    )


def test_callback_carry_can_resume_across_runner_families():
    functional, tensors, callbacks = _case()
    first_state, first_auxiliary = _host_run(
        "run",
        functional,
        tensors,
        callbacks,
        2,
    )
    final_state, second_auxiliary = _host_run(
        "longrun_checkpointed",
        functional,
        tensors,
        callbacks,
        3,
        state=first_state,
        callback_state=first_auxiliary["callbacks"].state,
    )
    expected_state, expected_auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=5,
        callbacks=callbacks,
    )

    first = first_auxiliary["callbacks"]
    second = second_auxiliary["callbacks"]
    resumed_moments = torch.cat(
        (
            first["moments"]["series"]["sample"],
            second["moments"]["series"]["sample"],
        )
    )
    resumed_trace = torch.cat((first["trace"]["v"], second["trace"]["v"]))
    torch.testing.assert_close(
        resumed_moments,
        expected_auxiliary["callbacks"]["moments"]["series"]["sample"],
    )
    torch.testing.assert_close(
        resumed_trace,
        expected_auxiliary["callbacks"]["trace"]["v"],
    )
    _assert_tree_close(
        second["energy"],
        expected_auxiliary["callbacks"]["energy"],
    )
    _assert_tree_close(final_state, expected_state)


def test_stateful_no_emission_result_supports_nested_torch_func_grad():
    functional, tensors, callbacks = _case()

    def loss(gnabar):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        _state, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=8,
            callbacks=callbacks,
        )
        return auxiliary["callbacks"]["energy"]["energy"].mean()

    gradient = torch.func.grad(loss)(tensors.parameters[GNABAR])
    nested_gradient = torch.func.grad(torch.func.grad(loss))(tensors.parameters[GNABAR])
    assert torch.isfinite(gradient)
    assert torch.isfinite(nested_gradient)
    assert gradient.abs() > 0
    assert nested_gradient.abs() > 0


def test_checkpointed_gradient_through_custom_carry_matches_longrun():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks({"energy": EnergyReducer()})

    def execute(runner):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = parameters[GNABAR].detach().clone().requires_grad_()
        prepared = functional.prepare(parameters, tensors.constants)
        step = partial(functional.step, parameters, prepared)
        _state, auxiliary = runner(
            functional,
            step,
            tensors.state,
            7 * DT,
            3,
            callbacks=callbacks,
        )
        loss = auxiliary["callbacks"]["energy"]["energy"].mean()
        gradient = torch.autograd.grad(loss, parameters[GNABAR])[0]
        return loss, gradient

    ordinary = execute(dn.func.longrun)
    checkpointed = execute(dn.func.longrun_checkpointed)
    for actual, expected in zip(checkpointed, ordinary, strict=True):
        torch.testing.assert_close(actual, expected)


def test_vmap_can_resume_nested_user_callback_carry():
    functional, tensors, callbacks = _case()
    base = tensors.parameters[GNABAR]
    lanes = torch.stack((0.8 * base, 1.2 * base))

    def parameters_for(gnabar):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        return parameters

    def first(gnabar):
        state, auxiliary = functional.prepare_and_rollout(
            parameters_for(gnabar),
            tensors.constants,
            tensors.state,
            steps=2,
            callbacks=callbacks,
        )
        results = auxiliary["callbacks"]
        return state, results.state, results["moments"]["series"]["sample"]

    first_states, callback_states, first_traces = torch.vmap(first)(lanes)

    def resume(gnabar, state, callback_state):
        final, auxiliary = functional.prepare_and_rollout(
            parameters_for(gnabar),
            tensors.constants,
            state,
            steps=2,
            callbacks=callbacks,
            callback_state=callback_state,
        )
        results = auxiliary["callbacks"]
        return (
            final["integrator"]["v"],
            results["moments"]["series"]["sample"],
            results["energy"]["energy"],
        )

    final_voltage, second_traces, final_energy = torch.vmap(resume)(
        lanes,
        first_states,
        callback_states,
    )

    def complete(gnabar):
        final, auxiliary = functional.prepare_and_rollout(
            parameters_for(gnabar),
            tensors.constants,
            tensors.state,
            steps=4,
            callbacks=callbacks,
        )
        results = auxiliary["callbacks"]
        return (
            final["integrator"]["v"],
            results["moments"]["series"]["sample"],
            results["energy"]["energy"],
        )

    expected_voltage, expected_traces, expected_energy = torch.vmap(complete)(lanes)
    torch.testing.assert_close(final_voltage, expected_voltage)
    torch.testing.assert_close(
        torch.cat((first_traces, second_traces), dim=1),
        expected_traces,
    )
    torch.testing.assert_close(final_energy, expected_energy)


def test_user_callback_composes_with_fullgraph_aot_eager_compile():
    functional, tensors, callbacks = _case()

    def execute(gnabar):
        parameters = dict(tensors.parameters)
        parameters[GNABAR] = gnabar
        final, auxiliary = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            steps=2,
            callbacks=callbacks,
        )
        results = auxiliary["callbacks"]
        return (
            final["integrator"]["v"],
            results["moments"]["series"]["derived"][1]["energy"],
            results["energy"]["energy"],
        )

    compiled = torch.compile(
        execute,
        backend="aot_eager",
        fullgraph=True,
        dynamic=False,
    )
    for scale in (0.9, 1.1):
        value = scale * tensors.parameters[GNABAR]
        expected = execute(value)
        with torch_compiler_warning_context():
            actual = compiled(value)
        _assert_tree_close(actual, expected)


def test_compiled_resume_reuses_graph_for_different_nested_callback_carry():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks({"energy": EnergyReducer()})

    first_state, first_auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        tensors.state,
        steps=1,
        callbacks=callbacks,
    )
    second_state, second_auxiliary = functional.prepare_and_rollout(
        tensors.parameters,
        tensors.constants,
        first_state,
        steps=1,
        callbacks=callbacks,
        callback_state=first_auxiliary["callbacks"].state,
    )
    compile_count = 0

    def backend(graph_module, _example_inputs):
        nonlocal compile_count
        compile_count += 1
        return graph_module.forward

    def resume(state, callback_state):
        final, auxiliary = functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            steps=1,
            callbacks=callbacks,
            callback_state=callback_state,
        )
        return final, auxiliary["callbacks"].state

    compiled = torch.compile(resume, backend=backend, fullgraph=True, dynamic=False)
    for state, callback_state in (
        (first_state, first_auxiliary["callbacks"].state),
        (second_state, second_auxiliary["callbacks"].state),
    ):
        expected = resume(state, callback_state)
        with torch_compiler_warning_context():
            actual = compiled(state, callback_state)
        _assert_tree_close(actual, expected)
    assert compile_count == 1


def test_inference_authored_callback_carry_is_safe_for_checkpointed_autograd():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    with torch.inference_mode():
        callbacks = functional.make_callbacks(
            {"energy": InferenceSeedReducer(torch.zeros((), dtype=torch.float64))}
        )

    parameters = dict(tensors.parameters)
    parameters[GNABAR] = parameters[GNABAR].detach().clone().requires_grad_()
    prepared = functional.prepare(parameters, tensors.constants)
    step = partial(functional.step, parameters, prepared)
    _state, auxiliary = dn.func.longrun_checkpointed(
        functional,
        step,
        tensors.state,
        4 * DT,
        2,
        callbacks=callbacks,
    )
    result = auxiliary["callbacks"]["energy"]
    gradient = torch.autograd.grad(result, parameters[GNABAR])[0]
    assert torch.isfinite(result)
    assert torch.isfinite(gradient)


class WrongArity(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"].new_zeros(())

    def update(self, carry, state, auxiliary):
        del state, auxiliary
        return carry, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


class NonTensorCarry(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del state, auxiliary
        return {"invalid": 1}, None

    def update(self, carry, state, auxiliary):
        del state, auxiliary
        return carry, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


class ChangingCarrySchema(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"].new_zeros(()), None

    def update(self, carry, state, auxiliary):
        del state, auxiliary
        return {"changed": carry}, None

    def finalize(self, carry, emissions):
        del emissions
        return carry


class ChangingEmissionSchema(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        value = state["integrator"]["v"]
        return value.new_zeros(()), {"value": value}

    def update(self, carry, state, auxiliary):
        del state
        return carry, (auxiliary["v"],)

    def finalize(self, carry, emissions):
        del carry
        return emissions


class ChangingReducerResultShape(dn.func.FunctionalCallback):
    def initialize(self, state, auxiliary):
        del auxiliary
        return state["integrator"]["v"].new_zeros(2, dtype=torch.bool), None

    def update(self, carry, state, auxiliary):
        del state, auxiliary
        return torch.ones_like(carry), None

    def finalize(self, carry, emissions):
        assert emissions is None
        return torch.nonzero(carry)


@pytest.mark.parametrize(
    ("callback", "match"),
    [
        (WrongArity(), "return exactly"),
        (NonTensorCarry(), "must be a Tensor"),
    ],
)
def test_user_callback_contract_fails_closed_during_binding(callback, match):
    functional, _tensors = dn.func.make_functional(_model(), dt=DT)
    with pytest.raises(dn.func.FunctionalizationError, match=match):
        functional.make_callbacks({"invalid": callback})


@pytest.mark.parametrize(
    "callback",
    [ChangingCarrySchema(), ChangingEmissionSchema()],
)
def test_user_callback_contract_fails_closed_on_runtime_schema_change(callback):
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks({"invalid": callback})
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="changed its tensor PyTree structure",
    ):
        functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=1,
            callbacks=callbacks,
        )


def test_reducer_finalized_result_must_keep_its_complete_shape():
    functional, tensors = dn.func.make_functional(_model(), dt=DT)
    callbacks = functional.make_callbacks({"invalid": ChangingReducerResultShape()})
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="changed a result leaf",
    ):
        functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            steps=1,
            callbacks=callbacks,
        )
