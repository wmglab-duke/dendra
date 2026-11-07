"""Construction-only subclasses retain each canonical Population contract."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators import dhs_bt
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import pas
from dendra.models.multi import MultiPopulation
from dendra.models.rng import RNGModule
from tests.func.test_extcell_tree import _branched_nonzero_root_graph
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_native_cable import _morphology
from tests.func.test_network_runtime_noise import _NoisyLeak
from tests.func.test_scalar_tree import _reordered_root_graph

DT = 0.01
DTYPE = torch.float64
PAS_G = "integrator.mech.mechanisms.pas.g_param"

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script` is deprecated.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        r"ignore:`torch\.jit\.script_method` is deprecated.*:DeprecationWarning"
    ),
]


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(previous)


class _Configured:
    """Authored constructors and additional helpers need no numerical adapter."""

    description = "a configured passive model"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.insert(pas, g=4.0e-4, e=-71.0)

    def model_description(self):
        return self.description


class _Point(_Configured, dn.Population):
    pass


class _Cell(_Configured, dn.SingleCompartment):
    pass


class _Cable(_Configured, dn.Cable):
    pass


class _Tree(_Configured, dn.Tree):
    pass


class _Unmyelinated(_Configured, dn.Unmyelinated):
    pass


class _Myelinated(_Configured, dn.Myelinated):
    pass


class _ExtCellAxon(_Configured, dn.ExtCellAxon):
    pass


class _ExtCellTree(_Configured, dn.ExtCellTree):
    pass


class _Packed(MultiPopulation):
    def __init__(self, **kwargs):
        if not kwargs:
            kwargs = {
                "point": _Point(N=1, C=1, dtype=DTYPE),
                "cell": _Cell(N=1, C=2, dtype=DTYPE),
            }
        super().__init__(**kwargs)


class _StochasticPoint(_Point):
    dn.Population.GLOBALRAND(
        "trial_value", mu=0.2, sigma=0.01, seed=17, rng_name="shared_rng"
    )
    dn.Population.GLOBALNOISE(
        "runtime_value", mu=0.0, sigma=1.0, seed=17, rng_name="shared_rng"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.insert(_NoisyLeak)


class _StochasticCell(_Cell):
    dn.SingleCompartment.GLOBALRAND(
        "trial_value", mu=0.2, sigma=0.01, seed=17, rng_name="shared_rng"
    )
    dn.SingleCompartment.GLOBALNOISE(
        "runtime_value", mu=0.0, sigma=1.0, seed=17, rng_name="shared_rng"
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.insert(_NoisyLeak)


FAMILIES = {
    "population": _Point,
    "single_compartment": _Cell,
    "cable": _Cable,
    "tree": _Tree,
    "unmyelinated": _Unmyelinated,
    "myelinated": _Myelinated,
    "extcell_axon": _ExtCellAxon,
    "extcell_tree": _ExtCellTree,
    "multi_population": _Packed,
}


def _model(family, *, model_type=None):
    if family in {"tree", "extcell_axon", "extcell_tree"}:
        if not DENDRA_SOLVERS_AVAILABLE:
            pytest.skip("this canonical solver requires dendra-solvers")
    constructor = FAMILIES[family] if model_type is None else model_type
    common = {"dtype": DTYPE, "v_init": -64.0}
    if family in {"population", "single_compartment"}:
        model = constructor(N=2, C=3, **common)
    elif family == "cable":
        model = constructor.from_morphology(
            _morphology(), N=2, integrator=dn.bwd_euler_ub(method="pcr"), **common
        )
    elif family == "tree":
        model = constructor.from_graph(
            _reordered_root_graph(), N=2, integrator=dn.dhs(threads=1), **common
        )
    elif family == "unmyelinated":
        model = constructor(
            [2.0, 2.5],
            L=4.0,
            dx=1.0,
            integrator=dn.bwd_euler_ub(method="pcr"),
            **common,
        )
    elif family == "myelinated":
        model = constructor(
            [8.0, 10.0],
            n_node=3,
            integrator=dn.bwd_euler_ub(method="pcr"),
            **common,
        )
    elif family == "extcell_axon":
        model = constructor(
            [5.0, 8.0],
            n_comp=3,
            integrator=dn.bwd_euler_bt(method="thomas"),
            **common,
        )
        model.xraxial.fill_(2.0)
        model.xc.fill_(0.1)
        model.xg.fill_(2.0e-4)
    elif family == "extcell_tree":
        model = constructor.from_graph(
            _branched_nonzero_root_graph(),
            N=2,
            integrator=dhs_bt(threads=1),
            **common,
        )
    else:
        model = constructor()
    model.initialize()
    model.train()
    return model


def _drive(model):
    return torch.linspace(-1.0e-9, 2.0e-9, model.v.numel(), dtype=DTYPE).reshape(
        model.shape
    )


def _native_step(model, drive):
    dt = model.v.new_tensor(DT)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    model.integrator.step(model, dt, None, drive)
    model.t = model.t + dt


def _clone(value):
    return pytree.tree_map(
        lambda leaf: leaf.detach().clone()
        if torch.is_tensor(leaf)
        else copy.deepcopy(leaf),
        value,
    )


@pytest.mark.parametrize("family", FAMILIES)
def test_subclasses_reuse_native_transition_gradients_and_fresh_initialization(family):
    source, native = _model(family), _model(family)
    before = _module_snapshot(source)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    conductance = (
        "integrator.mech.mechanisms.pas.g_point"
        if family == "multi_population"
        else PAS_G
    )
    parameters = tensors.independent_parameters((conductance,))
    prepared = functional.prepare(parameters, tensors.constants)
    state = tensors.state
    drive = _drive(source)
    native_g = dict(native.named_parameters())[conductance].requires_grad_(True)
    native.populate_parameter_buffers()
    for _ in range(3):
        state, _ = functional.step(
            parameters, prepared, state, dn.func.StepInput(intra=drive)
        )
        _native_step(native, drive)
        _assert_tree_close(
            state, functional.extract(native).state, rtol=2.0e-11, atol=2.0e-10
        )
    actual_grad = torch.autograd.grad(
        state["integrator"]["v"].square().mean(), parameters[conductance]
    )[0]
    expected_grad = torch.autograd.grad(native.v.square().mean(), native_g)[0]
    assert torch.isfinite(actual_grad).all() and actual_grad.abs().sum() > 0
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2.0e-9, atol=2.0e-7)

    initial_voltage = torch.linspace(
        -69.0, -59.0, source.v.numel(), dtype=DTYPE
    ).reshape(source.shape)
    initialized = functional.initialize(
        parameters,
        tensors.constants,
        tensors.initialization._replace(v_init=initial_voltage),
    )
    native.set_v_init(initial_voltage)
    native.initialize()
    _assert_tree_close(
        initialized.state, functional.extract(native).state, rtol=2.0e-11, atol=2.0e-10
    )
    _assert_module_snapshot(source, before)


@pytest.mark.parametrize("family", FAMILIES)
def test_subclass_checkpoint_commit_and_native_continuation_keep_concrete_contract(
    family,
):
    source, target = _model(family), _model(family)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    drive = _drive(source)
    checkpoint, _ = functional.step(
        tensors.parameters, prepared, tensors.state, dn.func.StepInput(intra=drive)
    )
    functional.commit_state_(target, checkpoint)
    _assert_tree_close(functional.extract(target).state, checkpoint)
    restored = _model(family)
    restored.restore_dict_from_checkpoint(_clone(target.state_dict_for_checkpoint()))
    _assert_tree_close(functional.extract(restored).state, checkpoint)
    expected, _ = functional.step(
        tensors.parameters, prepared, checkpoint, dn.func.StepInput(intra=drive)
    )
    _native_step(restored, drive)
    _assert_tree_close(
        functional.extract(restored).state, expected, rtol=2.0e-11, atol=2.0e-10
    )

    sibling = type("Sibling", (FAMILIES[family],), {})
    incompatible = _model(family, model_type=sibling)
    before = _module_snapshot(incompatible)
    with pytest.raises(dn.func.FunctionalizationError, match="structure"):
        functional.extract(incompatible)
    with pytest.raises(dn.func.FunctionalizationError, match="structure"):
        functional.commit_state_(incompatible, checkpoint)
    _assert_module_snapshot(incompatible, before)
    # A second authored subclass remains independently lowerable. Its class
    # identity only prevents exchanging state through a different class's plan.
    sibling_functional, sibling_tensors = dn.func.make_functional(incompatible, dt=DT)
    sibling_initialized = sibling_functional.initialize(
        sibling_tensors.parameters,
        sibling_tensors.constants,
        sibling_tensors.initialization,
    )
    _assert_tree_close(sibling_initialized.state, sibling_tensors.state)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "hook", ("step", "initialize", "_refresh_parameter_views_for_initialization")
)
def test_subclasses_reject_overridden_lifecycle_hooks_before_they_can_run(family, hook):
    source = _model(family)
    calls = []

    def unsafe(self, *args, **kwargs):
        calls.append(True)
        raise AssertionError("overridden native hook ran during lowering")

    changed = type("ChangedLifecycle", (FAMILIES[family],), {hook: unsafe})
    source.__class__ = changed
    with pytest.raises(
        dn.func.FunctionalizationError,
        match="step|initialize|physical|canonical|override|replaces",
    ):
        dn.func.make_functional(source, dt=DT)
    assert calls == []


@pytest.mark.parametrize("family", FAMILIES)
def test_subclasses_reject_overridden_geometry_even_when_the_body_delegates(family):
    class ChangedArea(FAMILIES[family]):
        @property
        def area(self):
            return super().area

    source = _model(family, model_type=ChangedArea)
    with pytest.raises(
        dn.func.FunctionalizationError, match="area|geometry|canonical|exact"
    ):
        dn.func.make_functional(source, dt=DT)


@pytest.mark.parametrize("model_type", (_StochasticPoint, _StochasticCell))
def test_constructor_and_stochastic_declarations_keep_explicit_rng_and_checkpoints(
    model_type,
):
    def model():
        result = model_type(N=2, C=2, dtype=DTYPE)
        result.initialize()
        return result

    source, native, restored = model(), model(), model()
    before = _module_snapshot(source)
    source_rng = {
        name: _clone(module.get_extra_state())
        for name, module in source.named_modules()
        if isinstance(module, RNGModule)
    }
    global_rng = torch.get_rng_state().clone()
    functional, tensors = dn.func.make_functional(source, dt=DT)
    draw = functional.sample_initialization_draw(tensors.parameters, tensors.state)
    initialized = functional.initialize(
        tensors.parameters,
        tensors.constants,
        tensors.initialization,
        state=tensors.state,
        random_draw=draw,
    )
    native.initialize()
    _assert_tree_close(initialized.state, functional.extract(native).state)
    torch.testing.assert_close(
        initialized.constants["random.values.trial_value"], native.trial_value
    )

    tape = functional.sample_noise_draw_tape(initialized.state, 3)
    checkpoint, _ = functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        initialized.state,
        dn.func.RolloutInput(noise=tape),
    )

    def native_steps(population, steps):
        dt = population.v.new_tensor(DT)
        population.integrator._initialize(
            population, dt, force=True, compile_scope="population"
        )
        for _ in range(steps):
            population._step(population.integrator, population, dt, None, None)
            population.t = population.t + dt

    native_steps(native, 3)
    _assert_tree_close(checkpoint, functional.extract(native).state)

    # Native checkpoint restoration and functional commit preserve the same
    # stream, including its position shared by RAND and NOISE declarations.
    restored.restore_dict_from_checkpoint(_clone(native.state_dict_for_checkpoint()))
    functional.commit_state_(restored, checkpoint)
    _assert_tree_close(functional.extract(restored).state, checkpoint)
    suffix, _ = functional.prepare_and_rollout(
        initialized.parameters,
        initialized.constants,
        checkpoint,
        dn.func.RolloutInput(noise=functional.sample_noise_draw_tape(checkpoint, 2)),
    )
    native_steps(restored, 2)
    _assert_tree_close(suffix, functional.extract(restored).state)
    continuation = functional.sample_initialization_draw(initialized.parameters, suffix)
    expected = functional.initialize(
        initialized.parameters,
        initialized.constants,
        tensors.initialization,
        state=suffix,
        random_draw=continuation,
    )
    restored.initialize()
    _assert_tree_close(expected.state, functional.extract(restored).state)
    _assert_module_snapshot(source, before)
    torch.testing.assert_close(
        {
            name: module.get_extra_state()
            for name, module in source.named_modules()
            if isinstance(module, RNGModule)
        },
        source_rng,
    )
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
