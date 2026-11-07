"""Ordinary PyTorch parametrizations use the shared parameter materializer."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.func import functional_call
from torch.nn.utils.parametrize import cached, register_parametrization

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mod import hh

DT = 0.0125
DTYPE = torch.float64
HH = "integrator.mech.mechanisms.hh"

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _context():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=DTYPE, DEVICE="cpu", CHECK_PURITY=False):
        yield


class _Affine(torch.nn.Module):
    def __init__(self, scale=1.1, offset=0.002):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale, dtype=DTYPE))
        self.register_buffer("offset", torch.tensor(offset, dtype=DTYPE))

    def forward(self, value):
        return self.scale * value + self.offset


class _Smooth(torch.nn.Module):
    def forward(self, value):
        return value + 0.02 * value.square()


class _InvertibleAffine(_Affine):
    def right_inverse(self, value):
        return (value - self.offset) / self.scale


class _ConfiguredAffine(_Affine):
    def __init__(self):
        super().__init__()
        self.gain = 1.2
        self.factors = (0.9, 1.1)

    def forward(self, value):
        value = super().forward(value)
        offset = 0.2 if self.training else 0.0
        return self.gain * self.factors[0] * value + self.factors[1] + offset


class _Product(torch.nn.Module):
    """A right inverse exposes two trainable original tensors."""

    def forward(self, left, right):
        return left * right

    def right_inverse(self, value):
        return value.clone(), torch.ones_like(value)


class _CountingIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value


class _VariableShape(torch.nn.Module):
    def forward(self, value):
        if (value < 0).any():
            return value.unsqueeze(-1)
        return value


def _population(kind="sc"):
    if kind == "sc":
        model = dn.SingleCompartment(
            N=1,
            C=2,
            v_init=torch.tensor([-63.0, -58.0], dtype=DTYPE),
            dtype=DTYPE,
            integrator=dn.bwd_euler_sc(imem=False),
        )
    else:
        model = dn.Unmyelinated(
            [2.0],
            L=3.0,
            dx=1.0,
            dtype=DTYPE,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        )
    model.insert(hh)
    model.initialize()
    model.train()
    return model


def _owner(model, path):
    return model.get_submodule(path) if path else model


def _refresh(model):
    model.populate_parameter_buffers()
    model.integrator.initialized = False
    model.initialize()
    model.train()


def _public_name(tree, path):
    matches = [name for name in tree if name == path or name.endswith("." + path)]
    assert len(matches) == 1, (path, tuple(tree))
    return matches[0]


def _effective(prepared, path, field):
    if not path:
        return prepared.values["population"][field]
    return prepared.values["mechanisms"][f"{path}.{field}"]


class _NativeMaterializer(torch.nn.Module):
    def __init__(self, model, owner_path, field):
        super().__init__()
        self.population = model
        self.owner_path = owner_path
        self.field = field

    def forward(self):
        return _owner(self.population, self.owner_path)._derive_parameter_buffers()[
            self.field
        ]


class _NativeDefaults(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.state = state

    def forward(self, voltage):
        return self.state.state_defaults(voltage, {})["m"]


def _native_value(model, owner_path, field, replacements):
    native = _NativeMaterializer(copy.deepcopy(model), owner_path, field)
    return functional_call(
        native,
        {"population." + name: value for name, value in replacements.items()},
        (),
        tie_weights=False,
    )


def _assert_native_rollout_and_gradients(model, paths, *, reference=None, steps=3):
    if reference is None:
        reference = copy.deepcopy(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    names = tuple(_public_name(tensors.parameters, path) for path in paths)
    leaves = tuple(
        tensors.parameters[name].detach().clone().requires_grad_() for name in names
    )
    actual, _ = functional.prepare_and_rollout(
        {**tensors.parameters, **dict(zip(names, leaves, strict=True))},
        tensors.constants,
        tensors.state,
        steps=steps,
    )
    dt = torch.tensor(DT, dtype=DTYPE)
    reference.integrator._initialize(
        reference, dt, force=True, compile_scope="population"
    )
    for _ in range(steps):
        reference.integrator.step(reference, dt)
        reference.t = reference.t + dt
    voltage = actual["integrator"]["v"]
    torch.testing.assert_close(voltage, reference.v, rtol=2e-11, atol=2e-11)
    actual_gradients = torch.autograd.grad(voltage.square().sum(), leaves)
    expected_gradients = torch.autograd.grad(
        reference.v.square().sum(),
        tuple(reference.get_parameter(path) for path in paths),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-10, atol=2e-10
        )
        assert torch.count_nonzero(actual_gradient)


@pytest.mark.parametrize(
    ("kind", "owner_path", "target", "field"),
    [
        ("sc", "", "cm", "cm"),
        ("ub", "", "diam", "diam"),
        ("sc", "", "celsius_param", "celsius"),
        ("sc", "cm_param", "rho", "cm"),
        ("sc", HH, "gnabar", "gnabar"),
        ("sc", HH + ".DE.mhn", "am1", "am1"),
        ("sc", HH, "gnabar_param", "gnabar"),
    ],
)
def test_torch_parametrizations_share_native_materialization_and_explicit_sources(
    kind, owner_path, target, field
):
    model = _population(kind)
    owner = _owner(model, owner_path)
    materializer_path = "" if owner_path == "cm_param" else owner_path
    register_parametrization(owner, target, _Affine())
    _refresh(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    prefix = f"{owner_path + '.' if owner_path else ''}parametrizations.{target}"
    scale_name = _public_name(tensors.parameters, prefix + ".0.scale")
    offset_name = _public_name(tensors.constants, prefix + ".0.offset")
    original = owner.parametrizations[target].original
    original_tree = (
        tensors.parameters
        if isinstance(original, torch.nn.Parameter)
        else tensors.constants
    )
    original_name = _public_name(original_tree, prefix + ".original")
    original_value = original_tree[original_name].detach().clone().requires_grad_()
    scale = tensors.parameters[scale_name].detach().new_tensor(1.25).requires_grad_()
    offset = tensors.constants[offset_name].detach().new_tensor(0.004).requires_grad_()
    parameters = {**tensors.parameters, scale_name: scale}
    constants = {**tensors.constants, offset_name: offset}
    (parameters if original_tree is tensors.parameters else constants)[
        original_name
    ] = original_value

    prepared = functional.prepare(parameters, constants)
    actual = _effective(prepared, materializer_path, field)
    expected = _native_value(
        model,
        materializer_path,
        field,
        {
            prefix + ".original": original_value,
            prefix + ".0.scale": scale,
            prefix + ".0.offset": offset,
        },
    )
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    leaves = (original_value, scale, offset)
    actual_grad = torch.autograd.grad(actual.square().sum(), leaves)
    expected_grad = torch.autograd.grad(expected.square().sum(), leaves)
    for actual_leaf, expected_leaf in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_leaf, expected_leaf, rtol=2e-12, atol=2e-12)
        assert torch.isfinite(actual_leaf).all()
        assert torch.count_nonzero(actual_leaf)


def test_chained_torch_parametrizations_support_func_transforms_and_fullgraph():
    model = _population()
    register_parametrization(model, "cm", _Affine())
    register_parametrization(model, "cm", _Smooth())
    _refresh(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    scale_name = _public_name(tensors.parameters, "parametrizations.cm.0.scale")
    original_name = _public_name(tensors.constants, "parametrizations.cm.original")
    offset_name = _public_name(tensors.constants, "parametrizations.cm.0.offset")
    original = tensors.constants[original_name]

    def materialized(scale):
        prepared = functional.prepare(
            {**tensors.parameters, scale_name: scale}, tensors.constants
        )
        return prepared.values["population"]["cm"]

    def native(scale):
        return _Smooth()(scale * original + tensors.constants[offset_name])

    scale = tensors.parameters[scale_name]
    for transform in (torch.func.jacrev, torch.func.jacfwd):
        torch.testing.assert_close(
            transform(materialized)(scale), transform(native)(scale)
        )
    lanes = scale * torch.tensor([0.9, 1.0, 1.2], dtype=DTYPE)
    torch.testing.assert_close(
        torch.vmap(materialized)(lanes), torch.vmap(native)(lanes)
    )

    def advance(local_scale):
        state, _ = functional.prepare_and_step(
            {**tensors.parameters, scale_name: local_scale},
            tensors.constants,
            tensors.state,
        )
        return state["integrator"]["v"]

    expected_voltage = advance(scale)
    with torch_compiler_warning_context():
        compiled = torch.compile(advance, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(scale), expected_voltage)
        torch.testing.assert_close(
            torch.autograd.grad(compiled(scale).sum(), scale)[0],
            torch.autograd.grad(expected_voltage.sum(), scale)[0],
        )


@pytest.mark.parametrize(
    ("kind", "owner_path", "target", "field"),
    [("sc", HH, "gnabar_param", "gnabar"), ("ub", "", "diam", "diam")],
)
def test_multiple_originals_are_explicit_and_preserve_native_gradients(
    kind, owner_path, target, field
):
    model = _population(kind)
    owner = _owner(model, owner_path)
    register_parametrization(owner, target, _Product())
    _refresh(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prefix = f"{owner_path + '.' if owner_path else ''}parametrizations.{target}"
    tree = tensors.parameters if target.endswith("_param") else tensors.constants
    names = tuple(_public_name(tree, prefix + f".original{i}") for i in range(2))
    leaves = tuple(tree[name].detach().clone().requires_grad_() for name in names)
    parameters, constants = dict(tensors.parameters), dict(tensors.constants)
    (parameters if tree is tensors.parameters else constants).update(
        dict(zip(names, leaves, strict=True))
    )
    prepared = functional.prepare(parameters, constants)
    actual = _effective(prepared, owner_path, field)
    expected = _native_value(
        model,
        owner_path,
        field,
        {prefix + f".original{i}": value for i, value in enumerate(leaves)},
    )
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), leaves)
    expected_grad = torch.autograd.grad(expected.sum(), leaves)
    for actual_leaf, expected_leaf in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_leaf, expected_leaf)


def test_shared_originals_and_transform_parameters_have_one_public_leaf():
    model = _population()
    mechanism = model.mech.mechanisms["hh"]
    shared = _Affine()
    for target in ("gnabar_param", "gkbar_param"):
        register_parametrization(mechanism, target, shared)
    first = mechanism.parametrizations.gnabar_param
    mechanism.parametrizations.gkbar_param.original = first.original
    _refresh(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    raw_names = [name for name in tensors.parameters if name.endswith(".original")]
    scale_names = [name for name in tensors.parameters if name.endswith(".0.scale")]
    assert len(raw_names) == len(scale_names) == 1
    assert len([name for name in tensors.constants if name.endswith(".0.offset")]) == 1
    raw_name, scale_name = raw_names[0], scale_names[0]
    raw = tensors.parameters[raw_name].detach().new_tensor(0.15).requires_grad_()
    scale = tensors.parameters[scale_name].detach().new_tensor(1.4).requires_grad_()
    prepared = functional.prepare(
        {**tensors.parameters, raw_name: raw, scale_name: scale}, tensors.constants
    )
    sodium = _effective(prepared, HH, "gnabar")
    potassium = _effective(prepared, HH, "gkbar")
    torch.testing.assert_close(sodium, potassium)
    gradient = torch.autograd.grad((sodium + potassium).sum(), (raw, scale))
    torch.testing.assert_close(gradient[0], 2 * scale)
    torch.testing.assert_close(gradient[1], 2 * raw)


@pytest.mark.parametrize("checked", [False, True])
def test_fresh_initialization_retains_parametrized_state_defaults_and_gradients(
    checked,
):
    model = _population()
    state = model.mech.mechanisms["hh"].DE.mhn
    register_parametrization(state, "am1", _Affine(scale=1.2, offset=0.003))
    _refresh(model)
    with dn.ctx(CHECK_PURITY=checked):
        functional, tensors = dn.func.make_functional(model, dt=DT)
    name = _public_name(tensors.parameters, HH + ".DE.mhn.parametrizations.am1.0.scale")
    scale = tensors.parameters[name].detach().new_tensor(1.3).requires_grad_()
    initialized = functional.initialize(
        {**tensors.parameters, name: scale}, tensors.constants, tensors.initialization
    )
    actual = initialized.state["mechanisms"]["hh"]["m"]
    reference = copy.deepcopy(model)
    reference_state = reference.mech.mechanisms["hh"].DE.mhn
    reference_state.parametrizations.am1[0].scale = torch.nn.Parameter(
        scale.detach().clone()
    )
    reference.initialize()
    torch.testing.assert_close(actual, reference.mech.mechanisms["hh"].m)
    native_defaults = _NativeDefaults(reference_state)
    expected = functional_call(
        native_defaults,
        {"state.parametrizations.am1.0.scale": scale},
        (initialized.state["integrator"]["v"],),
        tie_weights=False,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), scale)[0],
        torch.autograd.grad(expected.sum(), scale)[0],
    )


def test_parametrization_mutation_diagnostics_are_globally_opt_in():
    trusted_model = _population()
    register_parametrization(trusted_model, "cm", _CountingIdentity())
    _refresh(trusted_model)
    functional, tensors = dn.func.make_functional(trusted_model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    torch.testing.assert_close(prepared.values["population"]["cm"], trusted_model.cm)
    with dn.ctx(CHECK_PURITY=True):
        with pytest.raises(
            dn.func.FunctionalizationError, match="mutat|purity|state|write"
        ):
            dn.func.make_functional(trusted_model, dt=DT)


def test_parametrized_voltage_rollout_and_parameter_gradient_match_native():
    model = _population()
    register_parametrization(model, "cm", _Affine(scale=1.2, offset=0.01))
    register_parametrization(model.mech.mechanisms["hh"], "gnabar", _Affine())
    _refresh(model)
    reference = copy.deepcopy(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    scale_name = _public_name(tensors.parameters, "parametrizations.cm.0.scale")
    scale = tensors.parameters[scale_name].detach().clone().requires_grad_()
    actual, _ = functional.prepare_and_rollout(
        {**tensors.parameters, scale_name: scale},
        tensors.constants,
        tensors.state,
        steps=3,
    )

    native_scale = reference.parametrizations.cm[0].scale
    dt = torch.tensor(DT, dtype=DTYPE)
    reference.integrator._initialize(
        reference, dt, force=True, compile_scope="population"
    )
    for _ in range(3):
        reference.integrator.step(reference, dt)
        reference.t = reference.t + dt
    voltage = actual["integrator"]["v"]
    torch.testing.assert_close(voltage, reference.v, rtol=2e-11, atol=2e-11)
    torch.testing.assert_close(
        torch.autograd.grad(voltage.square().sum(), scale)[0],
        torch.autograd.grad(reference.v.square().sum(), native_scale)[0],
        rtol=2e-10,
        atol=2e-10,
    )


def test_parametrization_runtime_output_shape_remains_a_required_contract():
    model = _population()
    register_parametrization(model, "celsius_param", _VariableShape())
    _refresh(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    original_name = _public_name(
        tensors.parameters, "parametrizations.celsius_param.original"
    )
    parameters = {
        **tensors.parameters,
        original_name: -tensors.parameters[original_name],
    }
    with pytest.raises(
        (dn.func.FunctionalizationError, ValueError), match="shape|schema"
    ):
        functional.prepare(parameters, tensors.constants)


def test_external_torch_parametrization_cache_cannot_hide_explicit_tensor_updates():
    model = _population()
    register_parametrization(model, "cm", _Affine())
    _refresh(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    name = _public_name(tensors.parameters, "parametrizations.cm.0.scale")
    scales = tuple(tensors.parameters[name] * factor for factor in (0.8, 1.3))

    def advance(scale):
        state, _ = functional.prepare_and_step(
            {**tensors.parameters, name: scale}, tensors.constants, tensors.state
        )
        return state["integrator"]["v"]

    expected = tuple(advance(scale) for scale in scales)
    native_values = tuple(
        _native_value(model, "", "cm", {"parametrizations.cm.0.scale": scale})
        for scale in scales
    )
    assert torch.count_nonzero(expected[0] - expected[1])
    with cached():
        for scale, voltage, native in zip(scales, expected, native_values, strict=True):
            prepared = functional.prepare(
                {**tensors.parameters, name: scale}, tensors.constants
            )
            torch.testing.assert_close(prepared.values["population"]["cm"], native)
            torch.testing.assert_close(advance(scale), voltage)
        with torch_compiler_warning_context():
            compiled = torch.compile(advance, backend="aot_eager", fullgraph=True)
            for scale, voltage in zip(scales, expected, strict=True):
                torch.testing.assert_close(compiled(scale), voltage)


def test_canonical_cable_parametrized_coefficient_matches_native_rollout_and_gradient():
    morphology = dn.Morphology()
    morphology.section("axon", L=15.0, diam=2.0, nseg=3)
    model = dn.Cable.from_morphology(
        morphology,
        N=1,
        dtype=DTYPE,
        v_init=torch.tensor([-64.0, -59.0, -55.0], dtype=DTYPE),
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    model.insert(hh)
    model.initialize()
    register_parametrization(model, "cm_scale", _Affine(scale=1.2, offset=0.01))
    _refresh(model)
    _assert_native_rollout_and_gradients(model, ("parametrizations.cm_scale.0.scale",))


def _mixed_parametrized_population(geometry_field):
    from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE

    if not DENDRA_SOLVERS_AVAILABLE:
        pytest.skip("mixed scalar MultiPopulation execution requires dendra-solvers")

    point = _population()
    cable = _population("ub")
    cable.set_v_init(torch.tensor([-65.0, -59.0, -55.0], dtype=DTYPE))
    register_parametrization(point, "cm", _Affine(scale=1.2, offset=0.01))
    register_parametrization(cable, geometry_field, _Affine(scale=1.1, offset=0.02))
    _refresh(point)
    _refresh(cable)
    model = dn.concat_models({"point": point, "cable": cable}, threads=2)
    model.initialize()
    model.train()
    return model


@pytest.mark.parametrize("geometry_field", ["diam", "dx"])
def test_packed_component_parametrizations_match_native_rollout_and_gradients(
    geometry_field,
):
    model = _mixed_parametrized_population(geometry_field)
    _assert_native_rollout_and_gradients(
        model,
        (
            "populations.point.parametrizations.cm.0.scale",
            f"populations.cable.parametrizations.{geometry_field}.0.scale",
        ),
        reference=_mixed_parametrized_population(geometry_field),
    )


@pytest.mark.parametrize("target", ["cm", "celsius_param"])
def test_packed_parent_parametrizations_fail_closed_after_valid_native_initialization(
    target,
):
    model = dn.concat_models({"left": _population(), "right": _population()}, threads=2)
    model.initialize()
    register_parametrization(model, target, _InvertibleAffine())
    # Native initialization writes assembled fields through the right inverse.
    # Functional getter lowering must not silently discard that source/phase.
    model.initialize()
    with pytest.raises(dn.func.FunctionalizationError, match="packed parent"):
        dn.func.make_functional(model, dt=DT)


@pytest.mark.parametrize("target", ["cm", "diam"])
def test_packed_owned_cache_freezes_assembled_parametrized_physical_fields(target):
    def population():
        left, right = _population(), _population()
        register_parametrization(left, target, _Affine())
        _refresh(left)
        model = dn.concat_models({"left": left, "right": right}, threads=2)
        model.initialize()
        model.train()
        model.cache("_steady_state")
        return model

    source = population()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    key = _public_name(
        initialized.parameters, f"populations.left.parametrizations.{target}.0.scale"
    )
    # Explicit injected current depends on physical membrane area, making the
    # parametrized diameter case sensitive to cached assembled geometry too.
    intra = torch.full((2, *source.shape), 3e-10, dtype=DTYPE)
    for multiplier in (1.0, 1.2):
        native = population()
        native.initialize()
        scale = (initialized.parameters[key] * multiplier).detach().requires_grad_()
        with torch.no_grad():
            native.populations.left.parametrizations[target][0].scale.copy_(scale)
        actual, _ = functional.prepare_and_rollout(
            {**initialized.parameters, key: scale},
            initialized.constants,
            initialized.state,
            dn.func.RolloutInput(intra=intra),
            steps=2,
        )
        dt = torch.tensor(DT, dtype=DTYPE)
        native.integrator._initialize(
            native, dt, force=True, compile_scope="population"
        )
        for step in range(2):
            native.integrator.step(native, dt, None, intra[step])
        voltage = actual["integrator"]["v"]
        torch.testing.assert_close(voltage, native.v, rtol=2e-11, atol=2e-11)
        # Restoring a packed point cache detaches the assembled parent cm and
        # area. Changing a child getter does not rebuild those parent fields.
        assert native.v.grad_fn is None
        gradient = (
            torch.autograd.grad(voltage.square().sum(), scale, allow_unused=True)[0]
            if voltage.requires_grad
            else None
        )
        assert gradient is None or torch.count_nonzero(gradient) == 0


def test_packed_scalar_owned_cache_retains_native_child_solver_gradients():
    source, native = (
        _mixed_parametrized_population("diam"),
        _mixed_parametrized_population("diam"),
    )
    for model in (source, native):
        model.cache("_steady_state")
    functional, tensors = dn.func.make_functional(source, dt=DT)
    initialized = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    actual, _ = functional.prepare_and_rollout(
        initialized.parameters, initialized.constants, initialized.state, steps=3
    )
    dt = torch.tensor(DT, dtype=DTYPE)
    native.integrator._initialize(native, dt, force=True, compile_scope="population")
    for _ in range(3):
        native.integrator.step(native, dt)
    voltage = actual["integrator"]["v"]
    torch.testing.assert_close(voltage, native.v, rtol=2e-11, atol=2e-11)
    paths = (
        "populations.point.parametrizations.cm.0.scale",
        "populations.cable.parametrizations.diam.0.scale",
    )
    leaves = tuple(
        initialized.parameters[_public_name(initialized.parameters, path)]
        for path in paths
    )
    observed = torch.autograd.grad(voltage.square().sum(), leaves)
    oracle = torch.autograd.grad(
        native.v.square().sum(), tuple(native.get_parameter(path) for path in paths)
    )
    # The scalar packed DHS solver reads these child properties directly even
    # though the packed parent's saved fields belong to a detached cache.
    for actual_gradient, expected_gradient in zip(observed, oracle, strict=True):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-10, atol=2e-10
        )
        assert actual_gradient.abs().sum() > 1e-6


@pytest.mark.parametrize("change", ["float", "tuple", "training"])
def test_warmed_fullgraph_rejects_changed_transform_config_and_accepts_explicit_tensors(
    change,
):
    source = _population()
    transform = _ConfiguredAffine()
    register_parametrization(source, "cm", transform)
    _refresh(source)
    functional, tensors = dn.func.make_functional(source, dt=DT)

    def evaluate(parameters, constants, state):
        return functional.prepare_and_step(parameters, constants, state)[0][
            "integrator"
        ]["v"]

    compiled = torch.compile(evaluate, backend="eager", fullgraph=True)
    baseline = compiled(tensors.parameters, tensors.constants, tensors.state)
    scale_key = _public_name(tensors.parameters, "parametrizations.cm.0.scale")
    offset_key = _public_name(tensors.constants, "parametrizations.cm.0.offset")
    replacements = (
        (
            {**tensors.parameters, scale_key: tensors.parameters[scale_key] * 1.15},
            tensors.constants,
        ),
        (
            tensors.parameters,
            {**tensors.constants, offset_key: tensors.constants[offset_key] + 0.04},
        ),
    )
    for parameters, constants in replacements:
        actual = compiled(parameters, constants, tensors.state)
        expected = evaluate(parameters, constants, tensors.state)
        torch.testing.assert_close(actual, expected)
        assert torch.count_nonzero(actual - baseline)

    if change == "float":
        transform.gain = 1.4
    elif change == "tuple":
        transform.factors = (0.8, 1.2)
    else:
        transform.train(False)
    with pytest.raises(dn.func.FunctionalizationError, match="structure changed"):
        evaluate(tensors.parameters, tensors.constants, tensors.state)
    with pytest.raises(
        (dn.func.FunctionalizationError, RuntimeError, torch._dynamo.exc.Unsupported)
    ):
        compiled(tensors.parameters, tensors.constants, tensors.state)
