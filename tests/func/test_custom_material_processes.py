"""Custom pure full-field operators retain native lifecycle and transforms."""

from __future__ import annotations

import copy

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import MaterialProcess, Mechanism
from dendra.models.mod import pas
from tests.func.test_functional_initialization import _assert_tree_close

pytestmark = pytest.mark.cpu
DT = 0.125
STEPS = 4
DTYPE = torch.float64


class _Transfer(MaterialProcess):
    """Conservative, noncommuting operators with all parameter layouts."""

    MaterialProcess.GLOBAL(fraction=0.25)
    MaterialProcess.RANGE(rate=0.5)
    MaterialProcess.BATCH(gain=1.0)

    def __init__(self, *args, **kwargs):
        MaterialProcess.__init__(self, *args, **kwargs)
        self.scale = torch.nn.Parameter(self.celsius.new_tensor(0.8))
        self.register_buffer("offset", self.celsius.new_tensor(0.75))

    @property
    def material_fields(self):
        return (("pool", "amount"), ("pool", "reserve"))

    def prepare_materials(self, fields, dt):
        like = fields[("pool", "amount")]
        rate = self._resolve_quantity("rate", like)
        gain = self._resolve_quantity("gain", like)
        fraction = self._resolve_quantity("fraction", like)
        coefficient = dt * rate * gain * self.scale * self.offset
        return {"coefficient": coefficient, "fraction": (fraction,)}

    def material_updates(self, fields, prepared):
        amount, reserve = fields[("pool", "amount")], fields[("pool", "reserve")]
        coefficient, (fraction,) = prepared["coefficient"], prepared["fraction"]
        transfer = coefficient * ((1.0 - fraction) * amount - fraction * reserve)
        return self._mask_material_updates(
            fields,
            {
                ("pool", "amount"): amount - transfer,
                ("pool", "reserve"): reserve + transfer,
            },
        )


class _TransportTransfer(_Transfer):
    MaterialProcess.PHASE("transport")
    MaterialProcess.GLOBAL(fraction=0.75)


class _FinalTransfer(_Transfer):
    MaterialProcess.PHASE("post_transport")
    MaterialProcess.GLOBAL(fraction=0.5)


class _SodiumTransfer(MaterialProcess):
    MaterialProcess.PHASE("post_transport")
    MaterialProcess.GLOBAL(rate=20.0)

    @property
    def material_fields(self):
        return (("na", "nai"), ("na", "nao"))

    def prepare_materials(self, fields, dt):
        return (dt * self._resolve_quantity("rate", fields[("na", "nai")]),)

    def material_updates(self, fields, prepared):
        (transfer,) = prepared
        return self._mask_material_updates(
            fields,
            {
                ("na", "nai"): fields[("na", "nai")] - transfer,
                ("na", "nao"): fields[("na", "nao")] + transfer,
            },
        )


class _ReductionWorkspace(MaterialProcess):
    material_fields = (("pool", "amount"),)

    def __init__(self, *args, **kwargs):
        MaterialProcess.__init__(self, *args, **kwargs)
        self.scale = torch.nn.Parameter(self.celsius.new_tensor(0.5))
        self.register_buffer("coefficients", self.celsius.new_tensor([1.0, 2.0, 3.0]))

    def prepare_materials(self, fields, dt):
        del fields, dt
        return self.scale, self.coefficients * self.scale

    def material_updates(self, fields, prepared):
        scalar, vector = prepared
        key = ("pool", "amount")
        return {key: fields[key] + scalar.square() + vector.sum(dim=0)}


class _SourceCurrent(Mechanism):
    Mechanism.USEMATERIAL("pool", read=["amount"], source={"amount": "delta"})
    Mechanism.USEION("na", read=["nai", "ena"], write=["ina"])
    Mechanism.GLOBAL(production=0.2, g=0.0001)
    Mechanism.AFFINE("ina")

    def advance(self, v, dt, values):
        del v
        return {"delta": dt * self.production * (1.0 + 0.01 * values["nai"])}

    def ina_affine_terms(self, v):
        del v
        conductance = self.g * (1.0 + 0.1 * self.amount)
        return conductance, conductance * self.ena


@pytest.fixture(autouse=True)
def _environment():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=DTYPE, DEVICE="cpu"),
        ):
            yield
    finally:
        torch.set_num_threads(threads)


def _population(*, compartments=3, batch=None, regional=False, physiology=True):
    model = dn.SingleCompartment(
        N=1,
        C=compartments,
        v_init=-60.0,
        integrator=dn.bwd_euler_sc(imem=False),
    )
    model.material(
        "pool",
        fields={
            "amount": torch.nn.Parameter(model.v.new_tensor(1.7)),
            "reserve": torch.nn.Parameter(model.v.new_tensor(0.4)),
        },
        min_values={"amount": 0.0, "reserve": 0.0},
    )
    model.insert(pas, g=0.001, e=-65.0)
    if physiology:
        model.concentrations(nai0=5.0, nao0=140.0)
        model.ion_style("na", 1, 1)
        model.insert(_SourceCurrent.rename("source"))
        model.insert(_SodiumTransfer.rename("sodium"))
    # Reverse insertion catches phase-order regressions. The unequal fractions
    # make the conservative operators noncommuting.
    model.insert(_FinalTransfer.rename("final"))
    model.insert(_TransportTransfer.rename("transport"))
    owner = model[:, [0, compartments - 1]] if regional else model
    owner.insert(_Transfer.rename("local"))
    if batch is not None:
        model.batch(batch)
    model.train()
    with torch.no_grad():
        model.initialize()
    model.detach()
    return model


def _network():
    network = dn.Network(
        {"left": _population(compartments=2), "right": _population(compartments=1)}
    )
    # Register one shared raw Parameter on both ordinary custom operators.
    left = network.left.mech.material_processes["local"]
    network.right.mech.material_processes["local"].scale = left.scale
    network.train()
    with torch.no_grad():
        network.initialize(DT)
    for population in network.populations.values():
        population.detach()
    return network


def _fields(state):
    return state["materials"]["pool"]


def _source_key(parameters, suffix):
    names = [name for name in parameters if name.endswith(suffix)]
    assert len(names) == 1, names
    return names[0]


def _assert_local_fields(state, native):
    # Read-only locals are regenerated from accepted canonical fields rather
    # than stored as redundant functional carry. Check Nernst explicitly and
    # the corresponding imperative workspace after every phase transaction.
    source = native.mech.mechanisms["source"]
    torch.testing.assert_close(source.amount, _fields(state)["amount"])
    ions = state["ions"]["na"]
    for field in ("nai", "ena"):
        torch.testing.assert_close(getattr(source, field), ions[field])
    assert ions["nai"].min() > 0
    expected = (
        torch.log(ions["nao"] / ions["nai"])
        * native.mech.ions["na"].rzf
        * (273.15 + native.celsius)
    )
    torch.testing.assert_close(ions["ena"], expected)
    assert torch.isfinite(ions["ena"]).all()


def _oracle(amount, reserve, nai, coefficients):
    amount = amount + DT * 0.2 * (1.0 + 0.01 * nai)
    for coefficient, fraction in coefficients:
        transfer = coefficient * ((1.0 - fraction) * amount - fraction * reserve)
        amount, reserve = amount - transfer, reserve + transfer
    return amount, reserve


@pytest.mark.parametrize("batch", (None, 2))
def test_custom_processes_preserve_native_phases_source_guard_nernst_and_fresh_init(
    batch,
):
    model = _population(batch=batch)
    native = copy.deepcopy(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    frozen = pytree.tree_map_only(torch.Tensor, lambda x: x.detach().clone(), tensors)
    initial = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    native.initialize()
    _assert_tree_close(initial.state, functional.extract(native).state)
    torch.testing.assert_close(
        _fields(initial.state)["amount"], model.v.new_full(model.shape, 1.7)
    )
    torch.testing.assert_close(
        _fields(initial.state)["reserve"], model.v.new_full(model.shape, 0.4)
    )
    prepared = functional.prepare(initial.parameters, initial.constants)
    state = initial.state
    for _ in range(STEPS):
        before = _fields(state)
        nai = state["ions"]["na"]["nai"]
        expected = _oracle(
            before["amount"],
            before["reserve"],
            nai,
            ((DT * 0.5 * 0.8 * 0.75, fraction) for fraction in (0.25, 0.75, 0.5)),
        )
        state, _ = functional.step(initial.parameters, prepared, state)
        native.step(DT)
        _assert_tree_close(
            state, functional.extract(native).state, rtol=2e-10, atol=2e-11
        )
        for field, value in zip(("amount", "reserve"), expected, strict=True):
            torch.testing.assert_close(_fields(state)[field], value)
        _assert_local_fields(state, native)
    # The third sodium update reaches its lower guard; Nernst and the local
    # reads must use that accepted guarded value before the next local step.
    assert state["ions"]["na"]["nai"].max() < 1e-6
    _assert_tree_close(tensors, frozen)
    _assert_tree_close(functional.extract().state, frozen.state)


def test_custom_process_preparation_keeps_raw_parameter_and_buffer_inputs_differentiable():
    model = _population(physiology=False)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    scale_key = _source_key(tensors.parameters, "material_processes.local.scale")
    offset_key = _source_key(tensors.constants, "material_processes.local.offset")
    scale = tensors.parameters[scale_key].detach()
    offset = tensors.constants[offset_key].detach()

    def evaluate(raw, buffer, dt):
        parameters = {**tensors.parameters, scale_key: raw}
        constants = {**tensors.constants, offset_key: buffer, "dt": dt}
        initial = functional.initialize(parameters, constants, tensors.initialization)
        state, _ = functional.prepare_and_rollout(
            initial.parameters, initial.constants, initial.state, steps=STEPS
        )
        fields = _fields(state)
        return torch.cat(
            (fields["amount"].flatten(), fields["reserve"].square().flatten())
        )

    dt = tensors.constants["dt"]
    arguments = (scale, offset, dt)
    reverse = torch.func.jacrev(evaluate, argnums=(0, 1, 2))(*arguments)
    forward = torch.func.jacfwd(evaluate, argnums=(0, 1, 2))(*arguments)
    _assert_tree_close(reverse, forward, rtol=2e-10, atol=2e-11)
    for gradient in reverse:
        assert gradient.abs().sum() > 0 and gradient.isfinite().all()

    def loss(raw):
        return evaluate(raw, offset, dt).square().sum()

    hessian = torch.func.hessian(loss)(scale)
    assert hessian.isfinite().all() and hessian.abs().sum() > 0
    lanes = torch.stack((scale * 0.7, scale * 1.2))
    torch.testing.assert_close(
        torch.vmap(lambda raw: evaluate(raw, offset, dt))(lanes),
        torch.stack(tuple(evaluate(raw, offset, dt) for raw in lanes)),
    )
    compiled = torch.compile(
        torch.func.jacrev(evaluate, argnums=(0, 1, 2)),
        backend="aot_eager",
        fullgraph=True,
    )
    _assert_tree_close(compiled(*arguments), reverse, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("check_purity", (False, True))
def test_custom_workspace_ranks_preserve_native_reductions_and_transforms(check_purity):
    model = dn.SingleCompartment(N=1, C=3, integrator=dn.bwd_euler_sc(imem=False))
    model.material("pool", fields={"amount": 1.0})
    model.insert(pas)
    model.insert(_ReductionWorkspace.rename("reduction"))
    model.train()
    with torch.no_grad():
        model.initialize()
    model.detach()
    native = copy.deepcopy(model)
    with dn.ctx(CHECK_PURITY=check_purity):
        functional, tensors = dn.func.make_functional(model, dt=DT)
    scale_key = _source_key(tensors.parameters, "material_processes.reduction.scale")

    def evaluate(scale):
        parameters = {**tensors.parameters, scale_key: scale}
        state, _ = functional.prepare_and_step(
            parameters, tensors.constants, tensors.state
        )
        return state["materials"]["pool"]["amount"]

    scale = tensors.parameters[scale_key]
    actual = evaluate(scale)
    native.step(DT)
    torch.testing.assert_close(actual, native.mech.materials["pool"].amount)
    torch.testing.assert_close(actual, torch.full_like(actual, 4.25))
    reverse = torch.func.jacrev(evaluate)(scale)
    torch.testing.assert_close(reverse, torch.full_like(reverse, 7.0))
    torch.testing.assert_close(torch.func.jacfwd(evaluate)(scale), reverse)
    compiled = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(scale), actual)
    lanes = scale.detach().new_full((2,), 0.75)
    batched = torch.vmap(evaluate)(lanes)
    assert batched.shape == (2, *actual.shape)
    torch.testing.assert_close(batched, torch.full_like(batched, 6.0625))

    def evaluate_field(amount):
        state = {
            **tensors.state,
            "materials": {"pool": {"amount": amount}},
        }
        result, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state
        )
        return result["materials"]["pool"]["amount"]

    # Empty scalar-parameter broadcasting is unsupported by PyTorch itself;
    # an empty field batch exercises the authored workspace ranks unchanged.
    empty_fields = actual.detach().new_empty((0, *actual.shape))
    assert torch.vmap(evaluate_field)(empty_fields).shape == empty_fields.shape


@pytest.mark.parametrize("regional", (False, True))
def test_custom_process_packed_components_keep_region_support_and_native_gradients(
    regional,
):
    left = _population(compartments=3, regional=regional, physiology=False)
    right = _population(compartments=2, physiology=False)
    packed = dn.concat_models({"left": left, "right": right})
    packed.train()
    with torch.no_grad():
        packed.initialize()
    packed.detach()
    native = copy.deepcopy(packed)
    functional, tensors = dn.func.make_functional(packed, dt=DT)
    initial = functional.initialize(
        tensors.parameters, tensors.constants, tensors.initialization
    )
    state, _ = functional.prepare_and_rollout(
        initial.parameters, initial.constants, initial.state, steps=STEPS
    )
    native.initialize()
    for _ in range(STEPS):
        native.step(DT)
    _assert_tree_close(state, functional.extract(native).state, rtol=2e-10, atol=2e-11)
    actual_parameters = tuple(
        value
        for name, value in initial.parameters.items()
        if name.endswith("local.scale")
    )
    native_parameters = tuple(
        value
        for name, value in native.mech.material_processes.named_parameters()
        if name.endswith("local.scale")
    )
    assert len(actual_parameters) == len(native_parameters) == 2
    actual = torch.autograd.grad(
        _fields(state)["amount"].square().sum(), actual_parameters
    )
    expected = torch.autograd.grad(
        native.mech.materials["pool"].amount.square().sum(), native_parameters
    )
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
    total = _fields(state)["amount"] + _fields(state)["reserve"]
    torch.testing.assert_close(total, torch.full_like(total, 2.1))


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_compiled_network_custom_processes_share_parameters_and_resume_checkpoints(
    execution,
):
    network = _network()
    native = copy.deepcopy(network)
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=execution == "while",
        execution=execution,
        checkpoint_every=2 if execution == "scan" else None,
        backend="aot_eager",
        trainable=("*material_processes.local.scale",) if execution == "scan" else (),
        callbacks={"trace": dn.func.Recorder(("left.v", "right.v", "t"))},
    )
    if execution == "scan":
        assert runner.get_parameter("*left*local.scale") is runner.get_parameter(
            "*right*local.scale"
        )
    prefix, first = runner()
    repeated, _ = runner()
    _assert_tree_close(prefix, repeated)
    final, second = runner(
        prefix,
        parameters=first["parameters"],
        constants=first["constants"],
        callback_state=first["callbacks"].state,
    )
    native.initialize(DT)
    frames = [native.left.v.detach().clone()]
    for _ in range(2 * STEPS):
        native.step()
        frames.append(native.left.v.detach().clone())
    _assert_tree_close(
        final, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )
    torch.testing.assert_close(
        torch.cat(
            (
                first["callbacks"]["trace"]["left.v"],
                second["callbacks"]["trace"]["left.v"],
            )
        ),
        torch.stack(frames),
    )
    if execution == "scan":
        leaf = runner.get_parameter("*left*local.scale")
        actual = torch.autograd.grad(
            sum(
                _fields(state)["amount"].square().sum()
                for state in final["populations"].values()
            ),
            leaf,
        )[0]
        expected = torch.autograd.grad(
            sum(
                pop.mech.materials["pool"].amount.square().sum()
                for pop in native.populations.values()
            ),
            native.left.mech.material_processes["local"].scale,
        )[0]
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)
    runner.functional.commit_state_(network, final)
    checkpoint = network.state_dict_for_checkpoint()
    resumed = _network()
    resumed.restore_dict_from_checkpoint(checkpoint)
    resumed.step()
    continued, _ = runner.functional.prepare_and_step(
        first["parameters"], first["constants"], final
    )
    _assert_tree_close(
        continued, runner.functional.extract(resumed).state, rtol=2e-10, atol=2e-11
    )


@pytest.mark.parametrize("cache_kind", ("population", "network"))
def test_custom_process_caches_restore_without_advancing_and_continue(cache_kind):
    network = _network()
    with torch.no_grad():
        for _ in range(2):
            network.step()
        if cache_kind == "population":
            for population in network.populations.values():
                population.cache("_steady_state")
        else:
            network.cache_state()
        network.left.mech.material_processes["local"].scale.fill_(1.3)
    native = copy.deepcopy(network)
    runner = dn.compile(
        network,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=False,
        backend="aot_eager",
    )
    prefix, first = runner()
    native.initialize(DT)
    for _ in range(STEPS):
        native.step()
    _assert_tree_close(
        prefix, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )
    continued, _ = runner(
        prefix, parameters=first["parameters"], constants=first["constants"]
    )
    for _ in range(STEPS):
        native.step()
    _assert_tree_close(
        continued, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )


@pytest.mark.parametrize("execution", ("scan", "while"))
def test_inductor_custom_material_process_rollouts_use_shared_engines(execution):
    model = _population(compartments=1, physiology=False)
    native = copy.deepcopy(model)
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        initialization="fresh",
        inference=execution == "while",
        execution=execution,
        backend="inductor",
        fullgraph=True,
        callbacks={"trace": dn.func.Recorder(("v", "pool_material.amount"))},
        trainable=("*material_processes.local.scale",) if execution == "scan" else (),
    )
    final, _ = runner()
    native.initialize()
    for _ in range(STEPS):
        native.step(DT)
    _assert_tree_close(
        final, runner.functional.extract(native).state, rtol=2e-10, atol=2e-11
    )
    assert runner.execution_report().strategy == (
        "compiled_scan" if execution == "scan" else "compiled_while_loop"
    )
    if execution == "scan":
        gradient = torch.autograd.grad(
            _fields(final)["amount"].square().sum(),
            runner.get_parameter("*local.scale"),
        )[0]
        assert gradient.isfinite().all() and gradient.abs().sum() > 0


def test_custom_material_diagnostics_are_optional_and_frozen_for_execution():
    model = _population(physiology=False)
    with dn.ctx(CHECK_PURITY=False):
        trusted, trusted_tensors = dn.func.make_functional(model, dt=DT)
    with dn.ctx(CHECK_PURITY=True):
        checked, checked_tensors = dn.func.make_functional(model, dt=DT)
        expected, _ = trusted.prepare_and_rollout(
            trusted_tensors.parameters,
            trusted_tensors.constants,
            trusted_tensors.state,
            steps=STEPS,
        )
    # Each plan retains its construction policy when the surrounding context
    # changes. Diagnostics do not alter the shared operator's arithmetic.
    actual, _ = checked.prepare_and_rollout(
        checked_tensors.parameters,
        checked_tensors.constants,
        checked_tensors.state,
        steps=STEPS,
    )
    _assert_tree_close(actual, expected, rtol=2e-10, atol=2e-11)
    compiled = torch.compile(
        checked.prepare_and_step, backend="aot_eager", fullgraph=True
    )
    arguments = (
        checked_tensors.parameters,
        checked_tensors.constants,
        checked_tensors.state,
    )
    _assert_tree_close(
        compiled(*arguments),
        checked.prepare_and_step(*arguments),
        rtol=2e-10,
        atol=2e-11,
    )
