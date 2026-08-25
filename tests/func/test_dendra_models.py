from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import pas

peripheral = pytest.importorskip("dendra_models.models.cells.peripheral")
cadynamics = pytest.importorskip(
    "dendra_models.models.cells.cortical.mech.cadynamics"
).cadynamics
ca_hva = pytest.importorskip("dendra_models.models.cells.cortical.mech.ca_hva").ca_hva

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


def _assert_discovered_schema(state, case):
    assert set(state) == {"clock", "control", "integrator", "ions", "mechanisms"}
    assert set(state["integrator"]) == {"v"}
    assert set(state["mechanisms"]) == set(case.mechanism_states)
    for mechanism_name, expected_states in case.mechanism_states.items():
        assert set(state["mechanisms"][mechanism_name]) == expected_states
    assert set(state["ions"]) == {"k", "na"}
    assert set(state["ions"]["k"]) == {"ik", "ek", "ki", "ko"}
    assert set(state["ions"]["na"]) == {"ina", "ena", "nai", "nao"}


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
        "mechanism_buffers",
    }
    assert {
        name: set(values) for name, values in state["mechanisms"].items()
    } == case.mechanism_states
    assert {
        name: set(values) for name, values in state["mechanism_buffers"].items()
    } == {"mh": {"g"}, "nakpump": {"pump"}}
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
    # temperature, and local geometry. Breakpoint-written observables remain
    # ordinary carry.
    assert "state_buffers" not in tensors.state
    assert {
        name: set(values) for name, values in tensors.state["mechanism_buffers"].items()
    } == {
        "nacapump": {"inca"},
        "nakpumpSchild": {"ink"},
    }

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    derived_keys = {
        entry.key for entry in functional._prepared_buffers if entry.kind == "derived"
    }
    assert derived_keys == {
        "integrator.mech.mechanisms.caextscale.DE.cao.SA",
        "integrator.mech.mechanisms.caextscale.DE.cao.Vol_peri",
        "integrator.mech.mechanisms.caintscale.DE.oc_cai.SA",
        "integrator.mech.mechanisms.caintscale.DE.oc_cai.Vol",
        "integrator.mech.mechanisms.capump.ICaPmax",
        "integrator.mech.mechanisms.nacapump.DFin",
        "integrator.mech.mechanisms.nacapump.DFout",
        "integrator.mech.mechanisms.nacapump.KNaCa",
        "integrator.mech.mechanisms.nakpumpSchild.INaKmax",
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

    def next_inca(raw_knaca):
        parameters = dict(tensors.parameters)
        parameters[raw_knaca_name] = raw_knaca
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["mechanism_buffers"]["nacapump"]["inca"]

    inca_jacobian = torch.func.jacrev(next_inca)(tensors.parameters[raw_knaca_name])
    assert torch.isfinite(inca_jacobian).all()
    assert torch.count_nonzero(inca_jacobian) == inca_jacobian.numel()


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

    assert {
        name: set(values) for name, values in tensors.state["mechanism_buffers"].items()
    } == {"nacx": {"inaca"}, "nakpumpSchild": {"ink"}}
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

    def next_inaca(raw_gbar):
        parameters = dict(tensors.parameters)
        parameters[raw_gbar_name] = raw_gbar
        next_state, _aux = functional.prepare_and_step(
            parameters,
            tensors.constants,
            tensors.state,
        )
        return next_state["mechanism_buffers"]["nacx"]["inaca"]

    inaca_jacobian = torch.func.jacrev(next_inaca)(tensors.parameters[raw_gbar_name])
    assert torch.isfinite(inaca_jacobian).all()
    assert torch.count_nonzero(inaca_jacobian) == inaca_jacobian.numel()


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
