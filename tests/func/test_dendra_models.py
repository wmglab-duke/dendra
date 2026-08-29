from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.implicit import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas

peripheral = pytest.importorskip("dendra_models.models.cells.peripheral")
cadynamics = pytest.importorskip(
    "dendra_models.models.cells.cortical.mech.cadynamics"
).cadynamics
ca_hva = pytest.importorskip("dendra_models.models.cells.cortical.mech.ca_hva").ca_hva
ih = pytest.importorskip("dendra_models.models.cells.cortical.mech.ih").ih
rattay_aberham = pytest.importorskip(
    "dendra_models.models.cells.peripheral.mech.rattay_aberham"
).rattay_aberham
axnode_myel = pytest.importorskip(
    "dendra_models.models.cells.peripheral.mech.axnode_myel"
).axnode_myel

DT = 0.01
TIGERHOLM_DT = 0.001

pytestmark = [
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


@dataclass(frozen=True)
class _ModelCase:
    name: str
    kwargs: dict
    mechanism_states: dict[str, set[str]]


MODEL_CASES = (
    _ModelCase(
        "Rattay1993",
        {"diameters": [1.0], "L": 40.0, "dx": 10.0},
        {"rattay_aberham": {"m", "n", "h"}},
    ),
    _ModelCase(
        "Sundt2015",
        {"diameters": [1.0], "L": 40.0, "dx": 10.0},
        {"kdr": {"n", "l"}, "nahh": {"h", "m"}},
    ),
    _ModelCase(
        "FHUM",
        {"diameters": [2.0], "L": 100.0, "dx": 25.0},
        {"fh": {"m", "n", "h", "p"}},
    ),
)


def _case_id(case):
    return case.name


def _model(case: _ModelCase, *, method: str | None):
    kwargs = dict(case.kwargs)
    if method is not None:
        kwargs["integrator"] = dn.bwd_euler_ub(method=method, imem=False)
    constructor = getattr(peripheral, case.name)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = constructor(**kwargs).double()
        model.initialize()
        model.train()
    return model


def _single_compartment_rattay():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.SingleCompartment(
            N=1,
            C=3,
            dtype=torch.float64,
            integrator=dn.bwd_euler_sc(imem=False),
        )
        model.insert(rattay_aberham)
        model.initialize()
        model.train()
    return model


def _regional_cortical_calcium():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            v_init=-20.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas)
        target = model[:, 1:4]
        target.insert(ca_hva)
        target.insert(cadynamics)
        model.initialize()
        model.train()
    return model


def _regional_cortical_ih(*, v_init=-65.0):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.5, 2.0],
            L=4.0,
            dx=1.0,
            v_init=v_init,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model[:, torch.tensor([0, 2, 4])].insert(ih)
        model[:, torch.tensor([1, 3])].insert(pas)
        model.initialize()
        model.train()
    return model


def _unmyelinated_mrg_gates(*, v_init=-65.0, m_init=0.125):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [2.0],
            L=4.0,
            dx=1.0,
            v_init=v_init,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(axnode_myel, ic={"m": m_init})
        model.initialize()
        model.train()
    return model


def _mrg(name, diameter):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = getattr(peripheral, name)([diameter], n_node=2).double()
        model.initialize()
        model.train()
    return model


def _big_mrg():
    return _mrg("bigMRG", 8.0)


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


def _initialize_imperative_integrator(model, dt_value=DT):
    dt = torch.as_tensor(dt_value, device=model.device(), dtype=model.dtype())
    model.integrator._initialize(
        model,
        dt,
        force=True,
        compile_scope="population",
    )
    return dt


def _assert_every_leaf_close(actual, expected, *, rtol=2.0e-10, atol=2.0e-11):
    actual_with_paths, actual_spec = torch.utils._pytree.tree_flatten_with_path(actual)
    expected_with_paths, expected_spec = torch.utils._pytree.tree_flatten_with_path(
        expected
    )
    assert actual_spec == expected_spec
    for (actual_path, actual_leaf), (expected_path, expected_leaf) in zip(
        actual_with_paths,
        expected_with_paths,
        strict=True,
    ):
        assert actual_path == expected_path
        torch.testing.assert_close(
            actual_leaf,
            expected_leaf,
            rtol=rtol,
            atol=atol,
            msg=lambda message: f"state leaf {actual_path}: {message}",
        )


def _while_loop_nodes(graph_module):
    return [
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function"
        and node.target is torch.ops.higher_order.while_loop
    ]


def _assert_discovered_schema(state, case):
    assert set(state) == {"clock", "control", "integrator", "ions", "mechanisms"}
    assert set(state["integrator"]) == {"v"}
    assert set(state["mechanisms"]) == set(case.mechanism_states)
    for mechanism_name, expected_states in case.mechanism_states.items():
        assert set(state["mechanisms"][mechanism_name]) == expected_states
    assert set(state["ions"]) == {"k", "na"}
    assert set(state["ions"]["k"]) == {"ik", "ek", "ki", "ko"}
    assert set(state["ions"]["na"]) == {"ina", "ena", "nai", "nao"}


def test_regional_cortical_ih_fresh_initialization_matches_local_state_defaults_and_grad():
    source = _regional_cortical_ih()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    v_init = torch.linspace(
        -72.0,
        -54.0,
        source.v.numel(),
        dtype=source.dtype(),
        device=source.device(),
    ).reshape(source.shape)

    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        dn.func.InitializationInput(v_init=v_init),
    )
    reference = _regional_cortical_ih(v_init=v_init)
    expected = functional.extract(reference)
    _assert_every_leaf_close(initialized.state, expected.state, rtol=0.0, atol=0.0)

    local_v = reference.mech.ih.get(v_init)
    gate = reference.mech.ih.DE["m"]
    oracle = gate.state_defaults(
        local_v,
        {
            "celsius": gate.celsius,
            "diam": gate.diam,
        },
    )["m"]
    torch.testing.assert_close(
        initialized.state["mechanisms"]["ih"]["m"],
        oracle,
        rtol=0.0,
        atol=0.0,
    )

    gradient = torch.func.grad(
        lambda voltage: functional.initialize(
            tensors.parameters,
            tensors.constants,
            dn.func.InitializationInput(v_init=voltage),
        )
        .state["mechanisms"]["ih"]["m"]
        .sum()
    )(v_init)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_mrg_multi_state_defaults_and_explicit_ic_are_functional_inputs():
    source = _unmyelinated_mrg_gates()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    explicit_name = "mechanisms.axnode_myel.m"
    assert set(tensors.initialization.states) == {explicit_name}

    v_init = torch.linspace(
        -70.0,
        -58.0,
        source.v.numel(),
        dtype=source.dtype(),
        device=source.device(),
    ).reshape(source.shape)
    m_init = torch.linspace(
        0.1,
        0.2,
        source.v.numel(),
        dtype=source.dtype(),
        device=source.device(),
    ).reshape(source.shape)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        dn.func.InitializationInput(
            v_init=v_init,
            states={explicit_name: m_init},
        ),
    )
    reference = _unmyelinated_mrg_gates(v_init=v_init, m_init=m_init)
    expected = functional.extract(reference)
    _assert_every_leaf_close(initialized.state, expected.state, rtol=0.0, atol=0.0)
    assert set(initialized.state["mechanisms"]["axnode_myel"]) == {
        "m",
        "p",
        "h",
        "s",
    }
    torch.testing.assert_close(
        initialized.state["mechanisms"]["axnode_myel"]["m"],
        m_init,
        rtol=0.0,
        atol=0.0,
    )

    amp_name = tensors.parameter_name("ampA", within="model")

    def loss(amp, explicit_m):
        parameters = dict(tensors.parameters)
        parameters[amp_name] = amp
        state = functional.initialize(
            parameters,
            tensors.constants,
            dn.func.InitializationInput(
                v_init=v_init,
                states={explicit_name: explicit_m},
            ),
        ).state["mechanisms"]["axnode_myel"]
        return state["p"].mean() + state["m"].mean()

    amp_gradient, m_gradient = torch.func.grad(loss, argnums=(0, 1))(
        tensors.parameters[amp_name],
        m_init,
    )
    assert torch.isfinite(amp_gradient).all()
    assert torch.count_nonzero(amp_gradient) > 0
    assert torch.isfinite(m_gradient).all()
    assert torch.count_nonzero(m_gradient) > 0


def _assert_multistep_imperative_parity(functional_model, imperative_model, steps=4):
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(functional_model, steps)
    dt = _initialize_imperative_integrator(imperative_model)
    state = tensors.state

    for index in range(steps):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        imperative_model.integrator.step(
            imperative_model,
            dt,
            ve[index],
            intra[index],
        )
        imperative_model.t = imperative_model.t + dt
        expected = functional.extract(imperative_model).state
        _assert_every_leaf_close(state, expected)

    return functional, tensors


@pytest.mark.parametrize("case", MODEL_CASES, ids=_case_id)
def test_pcr_discovers_non_hh_state_and_matches_every_imperative_leaf(case):
    functional_model = _model(case, method="pcr")
    imperative_model = _model(case, method="pcr")

    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    _assert_discovered_schema(tensors.state, case)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(functional_model, 4)
    dt = _initialize_imperative_integrator(imperative_model)
    state = tensors.state

    for index in range(4):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        imperative_model.integrator.step(
            imperative_model,
            dt,
            ve[index],
            intra[index],
        )
        imperative_model.t = imperative_model.t + dt
        expected = functional.extract(imperative_model).state
        _assert_every_leaf_close(state, expected)


def test_sundt_native_callbacks_record_distinct_mechanism_state_checkpointed():
    case = next(case for case in MODEL_CASES if case.name == "Sundt2015")
    model = _model(case, method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    callbacks = functional.make_callbacks(
        {
            "trace": dn.func.Recorder(["v", "kdr.n", "nahh.m"]),
            "anomalous": dn.func.AnomalyDetector(),
        }
    )

    loss, results = model.longrun_checkpointed(
        tstop=2 * DT,
        chunklength=1,
        dt=DT,
        safe_checkpoint=True,
        functional_callbacks=callbacks,
    )

    assert loss is None
    assert not results["anomalous"].any()
    final_state = functional.extract(model).state
    expected_boundaries = {
        "v": (
            tensors.state["integrator"]["v"],
            final_state["integrator"]["v"],
        ),
        "kdr.n": (
            tensors.state["mechanisms"]["kdr"]["n"],
            final_state["mechanisms"]["kdr"]["n"],
        ),
        "nahh.m": (
            tensors.state["mechanisms"]["nahh"]["m"],
            final_state["mechanisms"]["nahh"]["m"],
        ),
    }
    for name, (initial, final) in expected_boundaries.items():
        trace = results["trace"][name]
        assert trace.shape == (3, *initial.shape)
        torch.testing.assert_close(trace[0], initial)
        torch.testing.assert_close(trace[-1], final)


@pytest.mark.parametrize("case", MODEL_CASES, ids=_case_id)
@pytest.mark.parametrize(
    "method",
    ["thomas", None],
    ids=["explicit-thomas", "dendra-default"],
)
def test_native_thomas_and_default_match_every_imperative_leaf(case, method):
    try:
        from dendra_solvers import thomas_solve_t
    except ImportError:
        pytest.skip("transform-compatible dendra-solvers facade is unavailable")

    functional_model = _model(case, method=method)
    imperative_model = _model(case, method=method)
    assert functional_model.integrator.method == "thomas"

    functional, _tensors = _assert_multistep_imperative_parity(
        functional_model,
        imperative_model,
        steps=3,
    )
    assert functional._transition.solver is thomas_solve_t


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="bigMRG CPU reference execution requires dendra-solvers",
)
def test_bigmrg_block_path_parametrizations_parity_and_gradient():
    source = _big_mrg()
    reference = _big_mrg()
    functional, tensors = dn.func.make_functional(source, dt=DT)

    expected_parametrization_constants = {
        "parametrizations._mrg_cm_transform.secd": source._mrg_cm_transform.secd,
        "parametrizations._mrg_cm_transform.fd": source._mrg_cm_transform.fd,
        "parametrizations._mrg_rhoa_transform.secd": (source._mrg_rhoa_transform.secd),
        "parametrizations._mrg_rhoa_transform.fd": source._mrg_rhoa_transform.fd,
    }
    actual_parametrization_names = {
        name for name in tensors.constants if name.startswith("parametrizations.")
    }
    assert actual_parametrization_names == set(expected_parametrization_constants)
    for name, expected in expected_parametrization_constants.items():
        torch.testing.assert_close(
            tensors.constants[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )

    assert {"cm_param.rho", "rhoa_param.rho"} <= set(tensors.parameters)
    assert set(tensors.state["integrator"]) == {"v", "vc"}
    assert set(tensors.state["mechanisms"]["axnode_myel"]) == {
        "h",
        "m",
        "p",
        "s",
    }

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    torch.testing.assert_close(
        prepared.values["population"]["cm"],
        source.cm,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        prepared.values["population"]["rhoa"],
        source.rhoa,
        rtol=0.0,
        atol=0.0,
    )

    counterfactual_constants = dict(tensors.constants)
    cm_secd_name = "parametrizations._mrg_cm_transform.secd"
    counterfactual_constants[cm_secd_name] = (
        1.01 * counterfactual_constants[cm_secd_name]
    )
    counterfactual = functional.prepare(
        tensors.parameters,
        counterfactual_constants,
    )
    torch.testing.assert_close(
        counterfactual.values["population"]["cm"],
        1.01 * source.cm,
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    ve, intra = _drives(source, 2)
    dt = _initialize_imperative_integrator(reference)
    state = tensors.state
    for index in range(2):
        state, _auxiliary = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        reference.integrator.step(reference, dt, ve[index], intra[index])
        reference.t = reference.t + dt

        _assert_every_leaf_close(state, functional.extract(reference).state)
        assert torch.equal(
            state["integrator"]["v"],
            state["integrator"]["vc"][..., 0] - state["integrator"]["vc"][..., 1],
        )

    step_input = dn.func.StepInput(ve=ve[0], intra=intra[0])

    for parameter_name in ("cm_param.rho", "rhoa_param.rho"):

        def next_voltage(raw_parameter):
            parameters = {**tensors.parameters, parameter_name: raw_parameter}
            next_state, _auxiliary = functional.prepare_and_step(
                parameters,
                tensors.constants,
                tensors.state,
                step_input,
            )
            return next_state["integrator"]["v"]

        reverse = torch.func.jacrev(next_voltage)(tensors.parameters[parameter_name])
        forward = torch.func.jacfwd(next_voltage)(tensors.parameters[parameter_name])
        torch.testing.assert_close(
            reverse,
            forward,
            rtol=2.0e-10,
            atol=3.0e-13,
        )
        assert torch.isfinite(reverse).all()
        assert torch.isfinite(forward).all()
        assert torch.count_nonzero(reverse) == reverse.numel()
        assert torch.count_nonzero(forward) == forward.numel()


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="MRG CPU reference execution requires dendra-solvers",
)
@pytest.mark.parametrize(
    ("model_name", "diameter"),
    (("smolMRG", 2.0), ("exactMRG", 5.7)),
)
def test_other_mrg_variants_match_every_imperative_leaf(model_name, diameter):
    functional_model = _mrg(model_name, diameter)
    imperative_model = _mrg(model_name, diameter)

    functional, _tensors = _assert_multistep_imperative_parity(
        functional_model,
        imperative_model,
        steps=1,
    )

    from dendra_solvers import solve_bt

    assert functional._transition.solver is solve_bt


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="bigMRG CPU lifecycle execution requires dendra-solvers",
)
def test_bigmrg_commit_then_imperative_resume_matches_uninterrupted_execution():
    source = _big_mrg()
    target = _big_mrg()
    reference = _big_mrg()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(source, 4)

    checkpoint_state, _auxiliary = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve[:2], intra=intra[:2]),
    )
    functional.commit_state_(target, checkpoint_state)
    _assert_every_leaf_close(functional.extract(target).state, checkpoint_state)

    target_dt = _initialize_imperative_integrator(target)
    assert target.integrator._solve is torch.ops.dendra_solvers.solve_bt
    for index in range(2, 4):
        target.integrator.step(target, target_dt, ve[index], intra[index])
        target.t = target.t + target_dt
    reference_dt = _initialize_imperative_integrator(reference)
    for index in range(4):
        reference.integrator.step(reference, reference_dt, ve[index], intra[index])
        reference.t = reference.t + reference_dt

    _assert_every_leaf_close(
        functional.extract(target).state,
        functional.extract(reference).state,
    )


def _bigmrg_runner_case(functional, tensors, ve, intra, *, checkpointed):
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    state = torch.utils._pytree.tree_map(
        lambda value: value.detach().clone(),
        tensors.state,
    )
    parameters["cm_param.rho"].requires_grad_()
    parameters["rhoa_param.rho"].requires_grad_()
    state["integrator"]["vc"].requires_grad_()
    ve = ve.detach().clone().requires_grad_()
    intra = intra.detach().clone().requires_grad_()
    targets = (
        parameters["cm_param.rho"],
        parameters["rhoa_param.rho"],
        state["integrator"]["vc"],
        ve,
        intra,
    )
    prepared = functional.prepare(parameters, constants)
    step = partial(functional.step, parameters, prepared)
    runner = dn.func.longrun_checkpointed if checkpointed else dn.func.longrun
    final, auxiliary = runner(
        functional,
        step,
        state,
        ve.shape[0] * DT,
        2,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    loss = (
        final["integrator"]["v"].square().mean()
        + 0.01 * final["integrator"]["vc"].square().mean()
        + 0.001 * auxiliary["v"].sin().mean()
    )
    return final, auxiliary, loss, torch.autograd.grad(loss, targets)


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="bigMRG checkpointed CPU BPTT requires dendra-solvers",
)
def test_bigmrg_checkpointed_runner_matches_ordinary_bptt():
    model = _big_mrg()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    from dendra_solvers import solve_bt

    assert functional._transition.solver is solve_bt
    ve, intra = _drives(model, 3)
    ordinary = _bigmrg_runner_case(
        functional,
        tensors,
        ve,
        intra,
        checkpointed=False,
    )
    checkpointed = _bigmrg_runner_case(
        functional,
        tensors,
        ve,
        intra,
        checkpointed=True,
    )

    _assert_every_leaf_close(checkpointed[0], ordinary[0])
    _assert_every_leaf_close(checkpointed[1], ordinary[1])
    torch.testing.assert_close(checkpointed[2], ordinary[2], rtol=2.0e-10, atol=2.0e-11)
    for actual, expected in zip(checkpointed[3], ordinary[3], strict=True):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


def _bigmrg_bptt_case(functional, tensors, ve, intra, *, compiled):
    parameters = {
        name: value.detach().clone() for name, value in tensors.parameters.items()
    }
    constants = {
        name: value.detach().clone() for name, value in tensors.constants.items()
    }
    state = torch.utils._pytree.tree_map(
        lambda value: value.detach().clone(),
        tensors.state,
    )
    parameters["cm_param.rho"].requires_grad_()
    constants["xg"].requires_grad_()
    state["integrator"]["v"].requires_grad_()
    state["integrator"]["vc"].requires_grad_()
    ve = ve.detach().clone().requires_grad_()
    intra = intra.detach().clone().requires_grad_()
    targets = (
        parameters["cm_param.rho"],
        constants["xg"],
        state["integrator"]["v"],
        state["integrator"]["vc"],
        ve,
        intra,
    )
    prepared = functional.prepare(parameters, constants)

    if compiled:
        chunk = functional.compile_rollout_chunk(2, backend="aot_eager")
        auxiliary = None
        with torch_compiler_warning_context():
            for start in range(0, ve.shape[0], chunk.steps):
                state, auxiliary = chunk(
                    parameters,
                    prepared,
                    state,
                    dn.func.RolloutInput(
                        ve=ve[start : start + chunk.steps],
                        intra=intra[start : start + chunk.steps],
                    ),
                )
    else:
        state, auxiliary = functional.rollout(
            parameters,
            prepared,
            state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )

    gates = state["mechanisms"]["axnode_myel"]
    loss = (
        state["integrator"]["v"].square().mean()
        + 0.01 * state["integrator"]["vc"].square().mean()
        + 0.001 * gates["m"].square().mean()
        + 0.0001 * auxiliary["v"].sin().mean()
    )
    gradients = torch.autograd.grad(loss, targets)
    return state, auxiliary, loss, gradients


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="bigMRG compiled CPU BPTT requires dendra-solvers",
)
def test_bigmrg_compiled_chunks_match_eager_bptt():
    model = _big_mrg()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    ve, intra = _drives(model, 4)

    eager = _bigmrg_bptt_case(
        functional,
        tensors,
        ve,
        intra,
        compiled=False,
    )
    compiled = _bigmrg_bptt_case(
        functional,
        tensors,
        ve,
        intra,
        compiled=True,
    )

    _assert_every_leaf_close(compiled[0], eager[0])
    _assert_every_leaf_close(compiled[1], eager[1])
    torch.testing.assert_close(compiled[2], eager[2], rtol=2.0e-10, atol=2.0e-11)
    for actual, expected in zip(compiled[3], eager[3], strict=True):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual)
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


@pytest.mark.skipif(
    not DENDRA_SOLVERS_AVAILABLE,
    reason="bigMRG compiled CPU inference requires dendra-solvers",
)
def test_bigmrg_no_grad_compiled_chunk_uses_bounded_native_block_loop():
    model = _big_mrg()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, intra = _drives(model, 5)
    expected = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    chunk = functional.compile_rollout_chunk(5, backend=backend)
    with torch.no_grad(), torch_compiler_warning_context():
        actual = chunk(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=ve, intra=intra),
        )

    _assert_every_leaf_close(actual[0], expected[0])
    _assert_every_leaf_close(actual[1], expected[1])
    assert len(graphs) == 1
    assert len(_while_loop_nodes(graphs[0])) == 1
    graph_modules = (
        module
        for module in graphs[0].modules()
        if isinstance(module, torch.fx.GraphModule)
    )
    assert any(
        node.op == "call_function"
        and node.target is torch.ops.dendra_solvers.solve_bt.default
        for graph_module in graph_modules
        for node in graph_module.graph.nodes
    )


def test_rattay_parameter_grad_vmap_and_fullgraph_compile():
    case = MODEL_CASES[0]
    model = _model(case, method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    ve, _intra = _drives(model, 3)

    def voltage(ve_values):
        state, _aux = functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=ve_values),
        )
        return state["integrator"]["v"]

    lanes = torch.stack((ve - 0.2, ve + 0.15))
    vmapped = torch.vmap(voltage)(lanes)
    explicit = torch.stack(tuple(voltage(lane) for lane in lanes))
    torch.testing.assert_close(vmapped, explicit, rtol=2.0e-10, atol=2.0e-11)

    parameter_name = "integrator.mech.mechanisms.rattay_aberham.gnabar_param.rho"

    def loss(raw_conductance):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = raw_conductance
        local_prepared = functional.prepare(parameters, tensors.constants)
        state, _aux = functional.rollout(
            parameters,
            local_prepared,
            tensors.state,
            steps=3,
        )
        return state["integrator"]["v"].square().mean()

    gradient = torch.func.grad(loss)(tensors.parameters[parameter_name])
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0.0

    def atomic_voltage(ve_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(ve=ve_values),
        )[0]["integrator"]["v"]

    compiled = torch.compile(atomic_voltage, backend="eager", fullgraph=True)
    with torch_compiler_warning_context():
        compiled_voltage = compiled(ve)
    torch.testing.assert_close(compiled_voltage, voltage(ve))


def test_rattay_mechanism_single_compartment_matches_every_leaf_and_differentiates():
    functional_model = _single_compartment_rattay()
    imperative_model = _single_compartment_rattay()
    functional, tensors = dn.func.make_functional(functional_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert set(tensors.state["mechanisms"]["rattay_aberham"]) == {"m", "n", "h"}
    assert set(tensors.state["ions"]) == {"k", "na"}

    ve, intra = _drives(functional_model, 3)
    dt = _initialize_imperative_integrator(imperative_model)
    state = tensors.state
    for index in range(3):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        imperative_model.integrator.step(
            imperative_model,
            dt,
            ve[index],
            intra[index],
        )
        imperative_model.t = imperative_model.t + dt
        _assert_every_leaf_close(
            state,
            functional.extract(imperative_model).state,
        )

    parameter_name = "integrator.mech.mechanisms.rattay_aberham.gnabar_param.rho"

    def final_voltage(raw_conductance):
        parameters = dict(tensors.parameters)
        parameters[parameter_name] = raw_conductance
        result, _aux = functional.prepare_and_rollout(
            parameters,
            tensors.constants,
            tensors.state,
            dn.func.RolloutInput(intra=intra[:2]),
        )
        return result["integrator"]["v"].square().mean()

    gradient = torch.func.grad(final_voltage)(tensors.parameters[parameter_name])
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0.0


def test_tigerholm_complex_state_matches_imperative_and_vmaps():
    case = _ModelCase(
        "Tigerholm2014",
        {"diameters": [1.0], "L": 40.0, "dx": 10.0},
        {
            "ks": {"f", "s"},
            "kf": {"m", "h"},
            "mh": {"f", "s"},
            "nattxs": {"m", "s", "h"},
            "nav1p8": {"h", "u", "m", "s"},
            "nav1p9": {"h", "m", "s"},
            "kdrTiger": {"n"},
            "naoiTiger": {"nai", "nao"},
            "koiTiger": {"ki", "ko"},
        },
    )
    functional_model = _model(case, method="pcr")
    imperative_model = _model(case, method="pcr")
    functional, tensors = dn.func.make_functional(
        functional_model,
        dt=TIGERHOLM_DT,
    )

    state = tensors.state
    assert set(state) == {
        "clock",
        "control",
        "integrator",
        "ions",
        "mechanisms",
    }
    assert {
        name: set(values) for name, values in state["mechanisms"].items()
    } == case.mechanism_states
    assert set(state["ions"]) == {"na", "k", "ca"}
    for ion_name, current_name, reversal_name, inner_name, outer_name in (
        ("na", "ina", "ena", "nai", "nao"),
        ("k", "ik", "ek", "ki", "ko"),
        ("ca", "ica", "eca", "cai", "cao"),
    ):
        assert set(state["ions"][ion_name]) == {
            current_name,
            reversal_name,
            inner_name,
            outer_name,
        }

    initial_nai = state["ions"]["na"]["nai"]
    initial_ki = state["ions"]["k"]["ki"]
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    leak_parameter = "integrator.mech.mechanisms.leak.gkleak_param"
    leak_buffer = "integrator.mech.mechanisms.leak.gkleak"
    torch.testing.assert_close(
        prepared.values["mechanisms"][leak_buffer],
        functional_model.mech.leak.gkleak,
    )

    def prepared_leak(raw_leak):
        parameters = dict(tensors.parameters)
        parameters[leak_parameter] = raw_leak
        return functional.prepare(parameters, tensors.constants).values["mechanisms"][
            leak_buffer
        ]

    leak_gradient = torch.func.grad(prepared_leak)(tensors.parameters[leak_parameter])
    torch.testing.assert_close(leak_gradient, torch.ones_like(leak_gradient))

    ve, intra = _drives(functional_model, 2)
    dt = _initialize_imperative_integrator(imperative_model, TIGERHOLM_DT)

    for index in range(2):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        imperative_model.integrator.step(
            imperative_model,
            dt,
            ve[index],
            intra[index],
        )
        imperative_model.t = imperative_model.t + dt
        expected = functional.extract(imperative_model).state
        _assert_every_leaf_close(state, expected)

    assert not torch.equal(state["ions"]["na"]["nai"], initial_nai)
    assert not torch.equal(state["ions"]["k"]["ki"], initial_ki)
    torch.testing.assert_close(
        state["ions"]["na"]["nai"],
        state["mechanisms"]["naoiTiger"]["nai"],
    )
    torch.testing.assert_close(
        state["ions"]["k"]["ki"],
        state["mechanisms"]["koiTiger"]["ki"],
    )

    def run_lane(lane_ve):
        return functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=lane_ve),
        )[0]

    lanes = torch.stack((ve - 0.1, ve + 0.2))
    vmapped = torch.vmap(run_lane)(lanes)
    explicit = torch.utils._pytree.tree_map(
        lambda *values: torch.stack(values),
        *(run_lane(lane) for lane in lanes),
    )
    _assert_every_leaf_close(vmapped, explicit)


def test_schild_reconstructs_derived_buffers_and_matches_imperative():
    case = _ModelCase(
        "Schild1997",
        {"diameters": [1.0], "L": 4.0, "dx": 1.0},
        {},
    )
    functional_model = _model(case, method="pcr")
    imperative_model = _model(case, method="pcr")
    functional, tensors = dn.func.make_functional(
        functional_model,
        dt=TIGERHOLM_DT,
    )

    # Initialization-static coefficients are reconstructed from parameters,
    # temperature, and local geometry. Ephemeral ASSIGNED observables are not
    # functional carry.
    assert "state_buffers" not in tensors.state
    assert "mechanism_buffers" not in tensors.state

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    derived_keys = {
        entry.key for entry in functional._prepared_buffers if entry.kind == "derived"
    }
    assert derived_keys == {
        "integrator.mech.mechanisms.caextscale.DE.cao.SA",
        "integrator.mech.mechanisms.caextscale.DE.cao.Vol_peri",
        "integrator.mech.mechanisms.caintscale.DE.oc_cai.SA",
        "integrator.mech.mechanisms.caintscale.DE.oc_cai.Vol",
        "integrator.mech.mechanisms.can.DE.d.q10",
        "integrator.mech.mechanisms.can.DE.f1.q10",
        "integrator.mech.mechanisms.can.DE.f2.q10",
        "integrator.mech.mechanisms.capump.ICaPmax",
        "integrator.mech.mechanisms.cat.DE.d.q10",
        "integrator.mech.mechanisms.cat.DE.f.q10",
        "integrator.mech.mechanisms.ka.DE.p.q10",
        "integrator.mech.mechanisms.ka.DE.q.q10",
        "integrator.mech.mechanisms.kca.DE.c.q10",
        "integrator.mech.mechanisms.kd.DE.n.q10",
        "integrator.mech.mechanisms.kds.DE.x.q10",
        "integrator.mech.mechanisms.kds.DE.y.q10",
        "integrator.mech.mechanisms.nacapump.DFin",
        "integrator.mech.mechanisms.nacapump.DFout",
        "integrator.mech.mechanisms.nacapump.KNaCa",
        "integrator.mech.mechanisms.naf97mean.DE.h.q10",
        "integrator.mech.mechanisms.naf97mean.DE.m.q10",
        "integrator.mech.mechanisms.nakpumpSchild.INaKmax",
        "integrator.mech.mechanisms.nas97mean.DE.h.q10",
        "integrator.mech.mechanisms.nas97mean.DE.m.q10",
    }
    for key in derived_keys:
        module_path, buffer_name = key.rsplit(".", 1)
        module = functional_model.get_submodule(module_path)
        torch.testing.assert_close(
            prepared.values["mechanisms"][key],
            module._buffers[buffer_name],
            rtol=0.0,
            atol=0.0,
        )

    ve, intra = _drives(functional_model, 2)
    dt = _initialize_imperative_integrator(imperative_model, TIGERHOLM_DT)
    state = tensors.state
    for index in range(2):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        imperative_model.integrator.step(
            imperative_model,
            dt,
            ve[index],
            intra[index],
        )
        imperative_model.t = imperative_model.t + dt
        _assert_every_leaf_close(
            state,
            functional.extract(imperative_model).state,
        )


def test_schild_explicit_batch_matches_imperative_fused_vmap_and_gradient():
    case = _ModelCase(
        "Schild1997",
        {"diameters": [1.0], "L": 4.0, "dx": 1.0},
        {},
    )

    def batched_model():
        model = _model(case, method="pcr")
        model.batch(2)
        model.initialize()
        model.train()
        return model

    functional_model = batched_model()
    imperative_model = batched_model()
    functional, tensors = dn.func.make_functional(
        functional_model,
        dt=TIGERHOLM_DT,
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert functional_model.shape == (2, *functional_model.core_shape())
    assert tensors.constants["diam"].shape == functional_model.core_shape()
    assert tensors.state["integrator"]["v"].shape == functional_model.shape
    assert "mechanism_buffers" not in tensors.state

    steps = 3
    ve, intra = _drives(functional_model, steps)
    dt = _initialize_imperative_integrator(imperative_model, TIGERHOLM_DT)
    state = tensors.state
    expected = None
    for index in range(steps):
        state, _aux = functional.step(
            tensors.parameters,
            prepared,
            state,
            dn.func.StepInput(ve=ve[index], intra=intra[index]),
        )
        imperative_model.integrator.step(
            imperative_model,
            dt,
            ve[index],
            intra[index],
        )
        imperative_model.t = imperative_model.t + dt
        expected = functional.extract(imperative_model).state
        _assert_every_leaf_close(state, expected)

    fused, _aux = functional.rollout(
        tensors.parameters,
        prepared,
        tensors.state,
        dn.func.RolloutInput(ve=ve, intra=intra),
    )
    _assert_every_leaf_close(fused, expected)

    def run_lane(lane_ve):
        return functional.rollout(
            tensors.parameters,
            prepared,
            tensors.state,
            dn.func.RolloutInput(ve=lane_ve, intra=intra),
        )[0]

    ve_lanes = torch.stack((ve - 0.05, ve + 0.075))
    vmapped = torch.vmap(run_lane)(ve_lanes)
    explicit = torch.utils._pytree.tree_map(
        lambda *values: torch.stack(values),
        *(run_lane(lane) for lane in ve_lanes),
    )
    _assert_every_leaf_close(vmapped, explicit)

    raw_knaca_name = "integrator.mech.mechanisms.nacapump.KNaCa22_param.rho"

    def voltage_loss(raw_knaca):
        parameters = dict(tensors.parameters)
        parameters[raw_knaca_name] = raw_knaca
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["integrator"]["v"].sum()

    gradient = torch.func.grad(voltage_loss)(tensors.parameters[raw_knaca_name])
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0.0


def test_schild_no_grad_compiled_rollout_uses_structured_loop_for_full_state():
    case = _ModelCase(
        "Schild1997",
        {"diameters": [1.0], "L": 4.0, "dx": 1.0},
        {},
    )
    model = _model(case, method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=TIGERHOLM_DT)
    assert not functional._structured_step_graphs
    assert functional.prewarm_structured_rollout(ve=True, intra=True)
    assert set(functional._structured_step_graphs) == {(True, True)}
    assert functional._structured_capture_error is None
    ve, intra = _drives(model, 5)
    graphs = []

    def backend(graph_module, _example_inputs):
        graphs.append(graph_module)
        return graph_module.forward

    def atomic(state, ve_values, intra_values):
        return functional.prepare_and_rollout(
            tensors.parameters,
            tensors.constants,
            state,
            dn.func.RolloutInput(ve=ve_values, intra=intra_values),
        )[0]

    expected = atomic(tensors.state, ve, intra)
    compiled = torch.compile(atomic, backend=backend, fullgraph=True)
    with torch.no_grad(), torch_compiler_warning_context():
        actual = compiled(tensors.state, ve, intra)

    _assert_every_leaf_close(actual, expected)
    assert len(graphs) == 1
    assert len(_while_loop_nodes(graphs[0])) == 1


def test_schild_compiled_chunks_match_eager_bptt_through_derived_workspaces():
    case = _ModelCase(
        "Schild1997",
        {"diameters": [1.0], "L": 4.0, "dx": 1.0},
        {},
    )
    model = _model(case, method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=TIGERHOLM_DT)
    ve, intra = _drives(model, 4)
    raw_knaca_name = "integrator.mech.mechanisms.nacapump.KNaCa22_param.rho"
    knaca_workspace = "integrator.mech.mechanisms.nacapump.KNaCa"
    assert knaca_workspace in {
        entry.key for entry in functional._prepared_buffers if entry.kind == "derived"
    }

    def differentiable_case():
        parameters = {
            name: value.detach().clone() for name, value in tensors.parameters.items()
        }
        parameters[raw_knaca_name].requires_grad_()
        constants = {
            name: value.detach().clone() for name, value in tensors.constants.items()
        }
        state = torch.utils._pytree.tree_map(
            lambda value: value.detach().clone(),
            tensors.state,
        )
        state["integrator"]["v"].requires_grad_()
        local_ve = ve.detach().clone().requires_grad_()
        local_intra = intra.detach().clone().requires_grad_()
        targets = (
            parameters[raw_knaca_name],
            state["integrator"]["v"],
            local_ve,
            local_intra,
        )
        return parameters, constants, state, local_ve, local_intra, targets

    def loss(state, aux):
        return state["integrator"]["v"].square().mean() + 0.01 * aux["v"].sin().mean()

    (
        eager_parameters,
        eager_constants,
        eager_state,
        eager_ve,
        eager_intra,
        eager_targets,
    ) = differentiable_case()
    eager_prepared = functional.prepare(eager_parameters, eager_constants)
    eager_final, eager_aux = functional.rollout(
        eager_parameters,
        eager_prepared,
        eager_state,
        dn.func.RolloutInput(ve=eager_ve, intra=eager_intra),
    )
    eager_loss = loss(eager_final, eager_aux)
    eager_gradients = torch.autograd.grad(eager_loss, eager_targets)

    (
        compiled_parameters,
        compiled_constants,
        compiled_state,
        compiled_ve,
        compiled_intra,
        compiled_targets,
    ) = differentiable_case()
    compiled_prepared = functional.prepare(compiled_parameters, compiled_constants)
    chunk = functional.compile_rollout_chunk(2, backend="aot_eager")
    with torch_compiler_warning_context():
        for start in range(0, 4, chunk.steps):
            compiled_state, compiled_aux = chunk(
                compiled_parameters,
                compiled_prepared,
                compiled_state,
                dn.func.RolloutInput(
                    ve=compiled_ve[start : start + chunk.steps],
                    intra=compiled_intra[start : start + chunk.steps],
                ),
            )
    compiled_loss = loss(compiled_state, compiled_aux)
    compiled_gradients = torch.autograd.grad(compiled_loss, compiled_targets)

    _assert_every_leaf_close(compiled_state, eager_final)
    _assert_every_leaf_close(compiled_aux, eager_aux)
    torch.testing.assert_close(
        compiled_loss,
        eager_loss,
        rtol=2.0e-10,
        atol=2.0e-11,
    )
    for actual, expected in zip(
        compiled_gradients,
        eager_gradients,
        strict=True,
    ):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, expected, rtol=2.0e-8, atol=2.0e-10)


def test_schild_derived_buffers_follow_temperature_geometry_and_raw_parameters():
    case = _ModelCase(
        "Schild1997",
        {"diameters": [1.0], "L": 4.0, "dx": 1.0},
        {},
    )
    model = _model(case, method="pcr")
    functional, tensors = dn.func.make_functional(model, dt=TIGERHOLM_DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    knaca_key = "integrator.mech.mechanisms.nacapump.KNaCa"
    dfin_key = "integrator.mech.mechanisms.nacapump.DFin"
    sa_key = "integrator.mech.mechanisms.caintscale.DE.oc_cai.SA"
    lseg_key = "integrator.mech.mechanisms.caintscale.DE.oc_cai.lseg"

    def temperature_workspaces(celsius):
        parameters = dict(tensors.parameters)
        parameters["celsius_param"] = celsius
        values = functional.prepare(parameters, tensors.constants).values["mechanisms"]
        return torch.stack((values[knaca_key], values[dfin_key]))

    temperature_jacobian = torch.func.jacrev(temperature_workspaces)(
        tensors.parameters["celsius_param"]
    )
    mechanism = model.mech.nacapump
    expected_temperature_jacobian = torch.stack(
        (
            -prepared.values["mechanisms"][knaca_key]
            * torch.log(mechanism.Q10NaCa)
            / mechanism.Q10TempB,
            -prepared.values["mechanisms"][dfin_key]
            / (tensors.parameters["celsius_param"] + 273.15),
        )
    )
    torch.testing.assert_close(
        temperature_jacobian,
        expected_temperature_jacobian,
        rtol=2.0e-12,
        atol=2.0e-14,
    )

    def intracellular_surface_area(diam):
        constants = dict(tensors.constants)
        constants["diam"] = diam
        return functional.prepare(tensors.parameters, constants).values["mechanisms"][
            sa_key
        ]

    diameter_jacobian = torch.func.jacrev(intracellular_surface_area)(
        tensors.constants["diam"]
    )
    coefficient = torch.pi * 1.0e-4 * prepared.values["mechanisms"][lseg_key]
    expected_diameter_jacobian = coefficient * torch.eye(
        tensors.constants["diam"].numel(),
        device=model.device(),
        dtype=model.dtype(),
    ).reshape(*model.shape, *model.shape)
    torch.testing.assert_close(
        diameter_jacobian,
        expected_diameter_jacobian,
        rtol=2.0e-12,
        atol=2.0e-14,
    )

    raw_knaca_name = "integrator.mech.mechanisms.nacapump.KNaCa22_param.rho"

    def next_voltage(raw_knaca):
        parameters = dict(tensors.parameters)
        parameters[raw_knaca_name] = raw_knaca
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["integrator"]["v"]

    voltage_jacobian = torch.func.jacrev(next_voltage)(
        tensors.parameters[raw_knaca_name]
    )
    assert torch.isfinite(voltage_jacobian).all()
    assert torch.count_nonzero(voltage_jacobian) == voltage_jacobian.numel()


def test_thio_nacx_derived_buffers_match_imperative_and_remain_differentiable():
    case = _ModelCase(
        "ThioCutaneous2024",
        {"diameters": [1.0], "L": 4.0, "dx": 1.0},
        {},
    )
    functional_model = _model(case, method="pcr")
    imperative_model = _model(case, method="pcr")
    functional, tensors = dn.func.make_functional(
        functional_model,
        dt=TIGERHOLM_DT,
    )
    prepared = functional.prepare(tensors.parameters, tensors.constants)

    assert "mechanism_buffers" not in tensors.state
    q10_key = "integrator.mech.mechanisms.nacx.q10"
    frt_key = "integrator.mech.mechanisms.nacx.FRT"
    q10 = prepared.values["mechanisms"][q10_key]
    assert q10.shape == functional_model.shape
    torch.testing.assert_close(q10, functional_model.mech.nacx.q10)
    torch.testing.assert_close(
        prepared.values["mechanisms"][frt_key],
        functional_model.mech.nacx.FRT,
    )

    state, _aux = functional.step(
        tensors.parameters,
        prepared,
        tensors.state,
    )
    dt = _initialize_imperative_integrator(imperative_model, TIGERHOLM_DT)
    imperative_model.integrator.step(imperative_model, dt)
    imperative_model.t = imperative_model.t + dt
    _assert_every_leaf_close(state, functional.extract(imperative_model).state)

    def temperature_workspaces(celsius):
        parameters = dict(tensors.parameters)
        parameters["celsius_param"] = celsius
        values = functional.prepare(parameters, tensors.constants).values["mechanisms"]
        return values[q10_key], values[frt_key]

    q10_gradient, frt_gradient = torch.func.jacrev(temperature_workspaces)(
        tensors.parameters["celsius_param"]
    )
    torch.testing.assert_close(
        q10_gradient,
        torch.full_like(q10, 1.2 / 14.0),
        rtol=2.0e-12,
        atol=2.0e-14,
    )
    temperature_kelvin = tensors.parameters["celsius_param"] + 273.0
    torch.testing.assert_close(
        frt_gradient,
        -prepared.values["mechanisms"][frt_key] / temperature_kelvin,
        rtol=2.0e-12,
        atol=2.0e-14,
    )

    raw_gbar_name = "integrator.mech.mechanisms.nacx.gbar_param.rho"

    def next_voltage(raw_gbar):
        parameters = dict(tensors.parameters)
        parameters[raw_gbar_name] = raw_gbar
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["integrator"]["v"]

    voltage_jacobian = torch.func.jacrev(next_voltage)(
        tensors.parameters[raw_gbar_name]
    )
    assert torch.isfinite(voltage_jacobian).all()
    assert torch.count_nonzero(voltage_jacobian) == voltage_jacobian.numel()


def test_cortical_calcium_workspace_is_prepared_and_affects_state_gradient():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            v_init=-20.0,
            dtype=torch.float64,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
        model.insert(pas)
        model.insert(ca_hva)
        model.insert(cadynamics)
        model.initialize()
        model.train()

    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    shell_key = "integrator.mech.mechanisms.cadynamics.DE.cai.shell"
    assert "state_buffers" not in tensors.state
    torch.testing.assert_close(
        prepared.values["mechanisms"][shell_key],
        model.mech.cadynamics.DE.cai.shell,
        rtol=0.0,
        atol=0.0,
    )

    raw_gamma_name = "integrator.mech.mechanisms.cadynamics.DE.cai.gamma_param.rho"

    def next_cai(raw_gamma):
        parameters = dict(tensors.parameters)
        parameters[raw_gamma_name] = raw_gamma
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["mechanisms"]["cadynamics"]["cai"]

    cai_jacobian = torch.func.jacrev(next_cai)(tensors.parameters[raw_gamma_name])
    assert torch.isfinite(cai_jacobian).all()
    assert torch.count_nonzero(cai_jacobian) == cai_jacobian.numel()


def test_regional_cortical_calcium_current_to_concentration_chain():
    functional_model = _regional_cortical_calcium()
    imperative_model = _regional_cortical_calcium()
    functional, tensors = _assert_multistep_imperative_parity(
        functional_model,
        imperative_model,
        steps=3,
    )

    assert "materials" not in tensors.state
    assert set(tensors.state["ions"]["ca"]) == {"ica", "eca", "cai", "cao"}
    assert tensors.state["mechanisms"]["cadynamics"]["cai"].shape == (1, 3)
    raw_gbar_name = next(
        name for name in tensors.parameters if name.endswith("ca_hva.gbar_param.rho")
    )
    raw_gbar = tensors.parameters[raw_gbar_name]

    def next_cai(value):
        parameters = dict(tensors.parameters)
        parameters[raw_gbar_name] = value
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["ions"]["ca"]["cai"]

    reverse = torch.func.jacrev(next_cai)(raw_gbar)
    forward = torch.func.jacfwd(next_cai)(raw_gbar)
    torch.testing.assert_close(reverse, forward, rtol=2.0e-10, atol=2.0e-12)
    assert torch.count_nonzero(reverse[:, 1:4]) == 3
    assert torch.count_nonzero(reverse[:, (0, 4)]) == 0

    transformed = torch.func.jacrev(next_cai)
    with torch_compiler_warning_context():
        compiled = torch.compile(
            transformed,
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(raw_gbar)
    torch.testing.assert_close(actual, reverse, rtol=2.0e-10, atol=2.0e-12)
