"""Native cache buffers select explicit, detached preparation operands."""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.func._population_cache import _cache_storage_frame, cache_storage_replays
from dendra.models.networks._population_cache import replay_population_cache_storage
from tests.func.test_membrane_current import DT, _model
from tests.func.test_network import _clone, _close

pytestmark = pytest.mark.cpu


class _SqrtParameter(torch.nn.Module):
    def forward(self, value):
        return value.sqrt()


class _SqrtConstant(torch.nn.Module):
    def forward(self, value, factor):
        return value * factor.sqrt()


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


@pytest.mark.parametrize(
    "kind",
    (
        "point",
        "path",
        "cable",
        "tree",
        "block_path",
        "block_tree",
        "multi_point",
        "multi_scalar",
        "multi_block",
    ),
)
def test_cached_effective_capacitance_is_explicit_and_snapshot_is_inactive(kind):
    source = _model(kind, batch=None)
    functional, original = dn.func.make_functional(source, dt=DT)
    expected = functional.prepare(original.parameters, original.constants).values
    functional._enable_cache_materializations()
    tensors = functional.extract()
    assert not tensors.constants["cache.active"].item()
    inactive = functional.prepare(tensors.parameters, tensors.constants).values
    torch.testing.assert_close(
        inactive["population"]["cm"], expected["population"]["cm"], rtol=0, atol=0
    )

    constants = dict(tensors.constants)
    constants["cache.active"] = torch.ones((), dtype=torch.bool)
    for name, path, *_ in functional._cache_value_specs:
        if path.endswith("cm"):
            constants[name] = constants[name] * 2
    prepared = functional.prepare(tensors.parameters, constants).values
    torch.testing.assert_close(
        prepared["population"]["cm"], 2 * inactive["population"]["cm"], rtol=0, atol=0
    )

    alternative = _model(kind, batch=None)
    refreshed = functional.extract(alternative)
    torch.testing.assert_close(
        refreshed.constants["cache.values.cm"], alternative.cm, rtol=0, atol=0
    )


def test_cache_materialization_defaults_follow_a_compatible_source():
    source = _model("point", batch=None)
    functional, _ = dn.func.make_functional(source, dt=DT)
    functional._enable_cache_materializations()
    alternative = _model("point", batch=None)
    alternative.cm.mul_(3)
    refreshed = functional.extract(alternative)
    torch.testing.assert_close(
        refreshed.constants["cache.values.cm"], alternative.cm, rtol=0, atol=0
    )


def test_cached_effective_and_derived_buffers_sever_raw_and_cache_gradients():
    source = _model("path", batch=None)
    functional, _ = dn.func.make_functional(source, dt=DT)
    functional._enable_cache_materializations()
    tensors = functional.extract()
    constants = {
        name: value.detach().clone().requires_grad_(value.is_floating_point())
        for name, value in tensors.constants.items()
    }
    constants["cache.active"] = torch.ones((), dtype=torch.bool)
    selected = [
        next(entry for entry in functional._prepared_buffers if entry.kind == kind)
        for kind in ("effective_parameter", "derived")
    ]
    for entry in selected:
        path = f"{entry.module_path}.{entry.buffer_name}"
        constants[f"cache.values.{path}"] = constants[f"cache.values.{path}"] + 0.375
    prepared = functional.prepare(tensors.parameters, constants).values
    result = prepared["population"]["cm"].sum()
    for entry in selected:
        path = f"{entry.module_path}.{entry.buffer_name}"
        torch.testing.assert_close(
            prepared["mechanisms"][entry.key],
            constants[f"cache.values.{path}"],
            rtol=0,
            atol=0,
        )
        result = result + prepared["mechanisms"][entry.key].sum()
    operands = tuple(tensors.parameters.values()) + tuple(
        value for value in constants.values() if value.requires_grad
    )
    gradients = torch.autograd.grad(result, operands, allow_unused=True)
    assert all(value is None or torch.count_nonzero(value) == 0 for value in gradients)


def test_cached_materialization_selection_compiles_and_has_zero_jvp():
    source = _model("path", batch=None)
    functional, _ = dn.func.make_functional(source, dt=DT)
    functional._enable_cache_materializations()
    tensors = functional.extract()
    constants = dict(tensors.constants)
    constants["cache.active"] = torch.ones((), dtype=torch.bool)

    def materialize(parameters, cache_cm):
        local = {**constants, "cache.values.cm": cache_cm}
        return functional._prepare_values(parameters, local)["population"]["cm"]

    expected = materialize(tensors.parameters, constants["cache.values.cm"])
    actual = torch.compile(materialize, backend="aot_eager", fullgraph=True)(
        tensors.parameters, constants["cache.values.cm"]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _, tangent = torch.func.jvp(
        materialize,
        (tensors.parameters, constants["cache.values.cm"]),
        (
            {
                name: torch.ones_like(value)
                for name, value in tensors.parameters.items()
            },
            torch.ones_like(constants["cache.values.cm"]),
        ),
    )
    torch.testing.assert_close(tangent, torch.zeros_like(tangent), rtol=0, atol=0)


def _cached_population_network(kind="path"):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=torch.float64, DEVICE="cpu"):
        network = dn.Network({"cell": _model(kind, batch=None)})
        network.train()
        network.initialize(DT)
        with torch.no_grad():
            for _ in range(3):
                network.step()
            network.cache_state()
        network.initialize(DT)
    return network


def test_network_cached_raw_and_effective_capacitance_match_native_and_cut_dynamics_gradient():
    source, native = _cached_population_network(), _cached_population_network()
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    cache = _clone(defaults.cache, detached=True)
    raw_name = "cm_param"
    cache.populations["cell"][raw_name].fill_(3)
    cache.populations["cell"]["cm"].fill_(2)
    native._state_cache["cell"][raw_name].fill_(3)
    native._state_cache["cell"]["cm"].fill_(2)
    initialized = functional.initialize(*tensors, defaults._replace(cache=cache))
    native.initialize(DT)
    torch.testing.assert_close(
        initialized.parameters["populations"]["cell"][raw_name], native.cell.cm_param
    )
    prepared = functional.populations["cell"].prepare(
        initialized.parameters["populations"]["cell"],
        initialized.constants["populations"]["cell"],
    )
    torch.testing.assert_close(prepared.values["population"]["cm"], native.cell.cm)
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)
    raw = tensors.parameters["populations"]["cell"][raw_name]
    voltage = actual["populations"]["cell"]["integrator"]["v"]
    gradient = torch.autograd.grad(voltage.sum(), raw, allow_unused=True)[0]
    assert gradient is None or torch.count_nonzero(gradient) == 0
    assert not native.cell.v.requires_grad


@pytest.mark.parametrize("coherent", (False, True))
def test_cached_temperature_uses_native_last_write_through_detached_storage_views(
    coherent,
):
    source, native = _cached_population_network(), _cached_population_network()
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    cache = _clone(defaults.cache, detached=True)
    for name, value in cache.populations["cell"].items():
        if name.endswith("celsius") if coherent else name == "celsius":
            value.fill_(0.9)
            native._state_cache["cell"][name].fill_(0.9)
    initialized = functional.initialize(*tensors, defaults._replace(cache=cache))
    native.initialize(DT)
    constants = initialized.constants["populations"]["cell"]
    expected = 0.9 if coherent else 6.3
    torch.testing.assert_close(
        native.cell.celsius, native.cell.celsius.new_tensor(expected)
    )
    torch.testing.assert_close(constants["cache.values.celsius"], native.cell.celsius)
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


def test_network_cached_materializations_compile_and_preserve_parameter_jvp_identity():
    functional, tensors = dn.func.make_functional_network(_cached_population_network())
    defaults = functional.initialization_inputs
    raw = tensors.parameters["populations"]["cell"]["cm_param"]
    effective = defaults.cache.populations["cell"]["cm"]

    def run(raw_cm, cached_cm):
        parameters = {
            **tensors.parameters,
            "populations": {
                "cell": {
                    **tensors.parameters["populations"]["cell"],
                    "cm_param": raw_cm,
                }
            },
        }
        cache = defaults.cache._replace(
            populations={
                "cell": {**defaults.cache.populations["cell"], "cm": cached_cm}
            }
        )
        initialized = functional.initialize(
            parameters, tensors.constants, tensors.state, defaults._replace(cache=cache)
        )
        state, _ = functional.prepare_and_step(*initialized)
        return (
            initialized.parameters["populations"]["cell"]["cm_param"],
            state["populations"]["cell"]["integrator"]["v"],
        )

    expected = run(raw, effective * 2)
    actual = torch.compile(run, backend="aot_eager", fullgraph=True)(raw, effective * 2)
    _close(actual, expected)
    primal, tangent = torch.func.jvp(
        run, (raw, effective * 2), (torch.ones_like(raw), torch.ones_like(effective))
    )
    _close(primal, expected)
    torch.testing.assert_close(tangent[0], torch.ones_like(raw), rtol=0, atol=0)
    torch.testing.assert_close(tangent[1], torch.zeros_like(tangent[1]), rtol=0, atol=0)
    for lanes in (0, 2):
        mapped = torch.vmap(run)(
            raw.expand(lanes, *raw.shape),
            (effective * 2).expand(lanes, *effective.shape),
        )
        _close(mapped, tuple(value.expand(lanes, *value.shape) for value in expected))


@pytest.mark.parametrize("source_kind", ("parameter", "constant"))
def test_cached_materialization_masks_nonfinite_inactive_authored_derivatives(
    source_kind,
):
    source = _model("path", batch=None)
    if source_kind == "parameter":
        source.register_parametrization_in_graph("cm", _SqrtParameter())
        name = "cm_param"
    else:
        source.register_buffer("factor", torch.ones((), dtype=source.dtype()))
        source.register_parametrization_in_graph(
            "cm", _SqrtConstant(), args=("factor",)
        )
        name = "parametrizations.factor"
    source.initialize()
    functional, _ = dn.func.make_functional(source, dt=DT)
    functional._enable_cache_materializations()
    tensors = functional.extract()
    operand = torch.zeros((), dtype=source.dtype(), requires_grad=True)
    parameters = dict(tensors.parameters)
    constants = dict(tensors.constants)
    (parameters if source_kind == "parameter" else constants)[name] = operand
    constants["cache.active"] = torch.ones((), dtype=torch.bool)
    result = functional.prepare(parameters, constants).values["population"]["cm"]
    torch.testing.assert_close(
        result, tensors.constants["cache.values.cm"], rtol=0, atol=0
    )
    gradient = torch.autograd.grad(result.sum(), operand)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)

    def materialize(value, active):
        local_parameters = dict(parameters)
        local_constants = {**constants, "cache.active": active}
        (local_parameters if source_kind == "parameter" else local_constants)[name] = (
            value
        )
        return functional._prepare_values(local_parameters, local_constants)[
            "population"
        ]["cm"]

    compiled = torch.compile(materialize, backend="aot_eager", fullgraph=True)
    actual = compiled(operand, constants["cache.active"])
    gradient = torch.autograd.grad(actual.sum(), operand)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)
    finite = torch.full_like(operand, 4, requires_grad=True)
    inactive = materialize(finite, torch.zeros((), dtype=torch.bool))
    gradient = torch.autograd.grad(inactive.sum(), finite)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("kind", ("multi_point", "multi_scalar"))
def test_partial_child_voltage_cache_edits_match_native_packed_restore(kind):
    source, native = _cached_population_network(kind), _cached_population_network(kind)
    functional, tensors = dn.func.make_functional_network(source)
    defaults = functional.initialization_inputs
    cache = _clone(defaults.cache, detached=True)
    cache.populations["cell"]["populations.point.v"].add_(5)
    native._state_cache["cell"]["populations.point.v"].add_(5)
    initialized = functional.initialize(*tensors, defaults._replace(cache=cache))
    native.initialize(DT)
    _close(initialized.state, functional.extract(native).state)
    actual, _ = functional.prepare_and_step(*initialized)
    native.step()
    _close(actual, functional.extract(native).state)


def test_partial_storage_replay_compiles_maps_and_preserves_last_copy_order():
    original = torch.arange(12, dtype=torch.float64)
    parent, child = original.reshape(1, 12), original[:6].reshape(2, 3)
    frames = tuple(_cache_storage_frame(value) for value in (parent, child))
    recipes = cache_storage_replays(
        [("parent", parent, frames[0]), ("child", child, frames[1])],
        [("parent", parent, frames[0]), ("child", child, frames[1])],
    )
    assert len(recipes) == 1
    recipe = recipes[0]

    def replay(parent_value, child_value):
        return replay_population_cache_storage(
            (parent_value, child_value),
            recipe.write_indices,
            recipe.output_indices,
            recipe.output_shapes,
            size=recipe.size,
        )

    parent_value = torch.arange(12, dtype=torch.float64).reshape_as(parent)
    child_value = torch.full_like(child, 30)
    expected_parent = parent_value.clone()
    expected_parent.reshape(-1)[:6].copy_(child_value.reshape(-1))
    expected = expected_parent, child_value
    _close(replay(parent_value, child_value), expected)
    _close(
        torch.compile(replay, backend="aot_eager", fullgraph=True)(
            parent_value, child_value
        ),
        expected,
    )
    for lanes in (0, 2):
        actual = torch.vmap(replay)(
            parent_value.expand(lanes, *parent.shape),
            child_value.expand(lanes, *child.shape),
        )
        _close(actual, tuple(value.expand(lanes, *value.shape) for value in expected))
    _, tangent = torch.func.jvp(
        replay,
        (parent_value, child_value),
        (torch.ones_like(parent_value), torch.ones_like(child_value)),
    )
    _close(tangent, tuple(torch.zeros_like(value) for value in expected))
    repeated = cache_storage_replays(
        [
            ("old_parent", parent, frames[0]),
            ("old_child", child, frames[1]),
            ("parent", parent, frames[0]),
            ("child", child, frames[1]),
        ],
        [("parent", parent, frames[0])],
    )
    assert repeated[0].write_paths == ("parent", "child")
    scalar = original[0]
    expanded = scalar.expand(2, 3)
    repeated_read = cache_storage_replays(
        [("scalar", scalar, _cache_storage_frame(scalar))],
        [("expanded", expanded, _cache_storage_frame(expanded))],
    )[0]
    actual = replay_population_cache_storage(
        (scalar.new_tensor(4),),
        repeated_read.write_indices,
        repeated_read.output_indices,
        repeated_read.output_shapes,
        size=repeated_read.size,
    )
    _close(actual, (torch.full_like(expanded, 4),))
    sparse = torch.empty(12, dtype=torch.float64)
    columns = sparse[::10], sparse[1::10]
    entries = [
        (name, value, _cache_storage_frame(value))
        for name, value in zip(("left", "right"), columns, strict=True)
    ]
    compact = cache_storage_replays(entries, entries)[0]
    assert compact.size == 4
    actual = replay_population_cache_storage(
        (
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            torch.tensor([4.0, 5.0], dtype=torch.float64),
        ),
        compact.write_indices,
        compact.output_indices,
        compact.output_shapes,
        size=compact.size,
    )
    _close(
        actual,
        (
            torch.tensor([2.0, 3.0], dtype=torch.float64),
            torch.tensor([4.0, 5.0], dtype=torch.float64),
        ),
    )


@pytest.mark.parametrize("kind", ("negative", "conjugate"))
def test_cache_storage_metadata_rejects_value_transforming_views(kind):
    value = (
        torch.ones(2, dtype=torch.float64)._neg_view()
        if kind == "negative"
        else torch.ones(2, dtype=torch.complex128).conj()
    )
    with pytest.raises(dn.func.FunctionalizationError, match="currently unsupported"):
        _cache_storage_frame(value)
