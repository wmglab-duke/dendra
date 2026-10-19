"""Diffusion retains physical geometry, process order, and explicit state.

Full recordings use the actual imperative integrator as their value oracle.
An independently assembled finite-volume matrix checks source application and
conservation, while explicit functional recurrence and central differences
check gradients. The real compiler matrix is only one scan and one while-loop
configuration on each CPU thread count; no unrolled compilation is needed.
"""

from __future__ import annotations

from functools import lru_cache

import pytest
import torch
from torch.utils import _pytree as pytree

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import DiffusionProcess
from tests.func.test_clearance_process import (
    DT,
    NAI,
    POOL,
    PRODUCTION,
    STEPS,
    _clone,
    _close,
    _explicit_reference,
    _key,
    _loss,
    _MaterialSourceCurrent,
    _runner,
)
from tests.func.test_functional_initialization import (
    _assert_module_snapshot,
    _assert_tree_close,
    _module_snapshot,
)
from tests.func.test_initialization_parity import (
    _constant_target,
    _live_state,
    _read_path,
    _record_paths,
)
from tests.func.test_membrane_current import CHANNELS
from tests.func.test_membrane_current import _model as _membrane_model

pytestmark = pytest.mark.cpu


@pytest.fixture(params=(1, 4), ids=("threads1", "threads4"))
def cpu_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _compiler_warnings():
    with torch_compiler_warning_context():
        yield


def _is_tree_kind(kind):
    return kind in {"tree", "block_tree", "cable"}


@lru_cache
def _process(
    method, solver, geometry, relaxation, edge_diffusivity, interface, regional=False
):
    class _FunctionalDiffusion(DiffusionProcess):
        DiffusionProcess.GLOBAL(rate=0.13, target=1.4)
        if regional:
            DiffusionProcess.RANGE(D=0.65)
        else:
            DiffusionProcess.GLOBAL(D=0.65)
        DiffusionProcess.METHOD(method, solver=solver, interface_scheme=interface)
        DiffusionProcess.DIFFUSE(
            "pool",
            field="amount",
            D="edge_diffusivity" if edge_diffusivity else "D",
            geometry="chemical" if geometry is not None else None,
            D_location="edge" if edge_diffusivity else "node",
        )
        DiffusionProcess.DIFFUSE(
            "na",
            field="nai",
            D="D",
            geometry="chemical" if geometry is not None else None,
        )
        if relaxation:
            DiffusionProcess.RELAX("pool", field="amount", rate="rate", target="target")

    return _FunctionalDiffusion


def _model(
    kind="path",
    *,
    method="implicit",
    solver="auto",
    geometry=None,
    relaxation=True,
    edge_diffusivity=False,
    interface="arithmetic",
    regional=False,
    device="cpu",
    dtype=torch.float64,
    shell_resistance_scale=1.0,
):
    model = _membrane_model(
        kind, device=device, dtype=dtype, batch=None, method="thomas"
    )
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DEVICE=device, DTYPE=dtype):
        if hasattr(model, "xraxial"):
            model.xraxial.mul_(shell_resistance_scale)
        if kind in {"path", "block_path"}:
            # Small positive lengths make transport sensitivity meaningful in
            # float32 too, rather than measuring a virtually static field.
            model.dx.copy_(
                torch.linspace(
                    0.7, 1.3, model.dx.numel(), device=device, dtype=dtype
                ).reshape_as(model.dx)
            )
        initial = torch.linspace(
            1.1, 2.1, model.v.numel(), device=device, dtype=dtype
        ).reshape(model.shape)
        model.material(
            "pool",
            fields={"amount": torch.nn.Parameter(initial)},
            min_values={"amount": 0.0},
            domain="intracellular",
        )
        initial_nai = torch.linspace(
            8.7, 9.7, model.v.numel(), device=device, dtype=dtype
        ).reshape(model.shape)
        model.concentrations(nai0=torch.nn.Parameter(initial_nai))
        model.ion_style("na", 1, 1)
        if geometry is not None:
            n_edges = model.shape[-1] - 1
            volume = torch.linspace(
                0.9, 1.4, model.v.numel(), device=device, dtype=dtype
            ).reshape(model.shape)
            edge = torch.linspace(
                0.6, 1.0, model.shape[-2] * n_edges, device=device, dtype=dtype
            ).reshape(model.shape[-2], n_edges)
            quantities = {"volume": torch.nn.Parameter(volume)}
            if geometry == "factor":
                quantities["edge_factor"] = torch.nn.Parameter(edge)
            else:
                assert geometry == "area_distance"
                quantities["edge_area"] = torch.nn.Parameter(edge * 1.3)
                quantities["edge_distance"] = torch.nn.Parameter(
                    torch.full_like(edge, 1.3)
                )
            model.register_material_geometry(
                "chemical", domain="intracellular", **quantities
            )
        if edge_diffusivity:
            model.register_parameter(
                "edge_diffusivity",
                torch.nn.Parameter(
                    model.v.new_full((model.shape[-2], model.shape[-1] - 1), 0.65)
                ),
            )
        process = _process(
            method, solver, geometry, relaxation, edge_diffusivity, interface, regional
        )
        model.insert(_MaterialSourceCurrent)
        (model[:, :2] if regional else model).insert(process)
        model.batch(2)
        model.train()
        model.initialize()
        model.t.fill_(1.25)
    model._diffusion_test_options = dict(
        kind=kind,
        method=method,
        solver=solver,
        geometry=geometry,
        relaxation=relaxation,
        edge_diffusivity=edge_diffusivity,
        interface=interface,
        regional=regional,
        device=device,
        dtype=dtype,
        shell_resistance_scale=shell_resistance_scale,
    )
    return model


def _constant_name(constants, *candidates):
    for candidate in candidates:
        if candidate in constants:
            return candidate
        names = [name for name in constants if name.endswith("." + candidate)]
        if len(names) == 1:
            return names[0]
    raise AssertionError((candidates, tuple(constants)))


def _arguments(runner, model):
    parameters = _clone(dict(runner.parameters))
    targets = {
        channel: parameters[
            _key(parameters, f"mechanisms.hh.{channel}_param")
        ].requires_grad_()
        for channel in CHANNELS
    }
    for name, value in parameters.items():
        if ".material_processes." in name:
            label = name.split(".material_processes.", 1)[1].removesuffix("_param")
            # Rate/target are deliberately unused when RELAX is not declared.
            if not model._diffusion_test_options["relaxation"] and label.endswith(
                (".rate", ".target")
            ):
                continue
            targets[label] = value.requires_grad_()
    for label, suffix in (
        ("initial_pool", "materials.pool._initial_sources.amount"),
        ("initial_nai", "ions.na.i_init"),
    ):
        targets[label] = parameters[_key(parameters, suffix)].requires_grad_()
    if "noded2" in parameters:
        targets["geometry"] = parameters["noded2"].requires_grad_()
    else:
        name = _constant_name(runner.constants, "canonical_area_cm2", "diam")
        targets["geometry"] = runner.constants[name].requires_grad_()
    geometry = model._diffusion_test_options["geometry"]
    if geometry is not None:
        for component, source in model.material_geometry("chemical").items():
            if component not in {"name", "domain"}:
                targets["chemical_" + component] = parameters[
                    _key(parameters, source)
                ].requires_grad_()
    elif _is_tree_kind(model._diffusion_test_options["kind"]):
        for label, candidates in (
            ("chemical_volume", ("volume_i", "_mp_volume_i")),
            ("chemical_edge_factor", ("diff_geom_um", "_mp_diff_geom_um")),
        ):
            targets[label] = runner.constants[
                _constant_name(runner.constants, *candidates)
            ].requires_grad_()
    if (
        model._diffusion_test_options["kind"] in {"path", "block_path"}
        and geometry is None
    ):
        targets["length"] = runner.constants[
            _constant_name(runner.constants, "dx", "_mp_dx")
        ].requires_grad_()
    if model._diffusion_test_options["edge_diffusivity"]:
        targets["edge_diffusivity"] = parameters[
            _key(parameters, "edge_diffusivity")
        ].requires_grad_()
    targets["dt"] = runner.constants["dt"].requires_grad_()
    voltage = torch.linspace(
        -72.0, -52.0, model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(model.shape)
    targets["initial_voltage"] = voltage.requires_grad_()
    targets["stimulus"] = model.v.new_ones((), requires_grad=True)
    drive = torch.linspace(
        -2e-9, 3e-9, STEPS * model.v.numel(), device=model.device(), dtype=model.dtype()
    ).reshape(STEPS, *model.shape)
    ve = torch.linspace(
        -0.3, 0.4, drive.numel(), device=model.device(), dtype=model.dtype()
    ).reshape_as(drive)
    initialization = _clone(runner.initialization_inputs)._replace(v_init=voltage)
    return (
        parameters,
        initialization,
        dn.func.RolloutInput(ve, drive * targets["stimulus"]),
        targets,
    )


def _gradients(recorded, targets):
    gradients = torch.autograd.grad(_loss(recorded), tuple(targets.values()))
    result = dict(zip(targets, gradients, strict=True))
    for name, value in result.items():
        assert torch.isfinite(value).all(), name
        assert torch.count_nonzero(value), name
    return result


def _imperative_reference(
    model, runner, parameters, initialization, inputs, *, steps=STEPS
):
    reference = _model(**model._diffusion_test_options)
    with torch.no_grad():
        raw = dict(reference.named_parameters())
        for name, value in parameters.items():
            raw[name].copy_(value)
        for name, value in runner.constants.items():
            if name != "dt" and not name.startswith("parametrizations."):
                target = (
                    _constant_target(reference.mech, name)
                    if name.startswith("material_processes.")
                    else _constant_target(reference, name)
                )
                if not torch.equal(target, value):
                    target.copy_(value)
        reference.set_v_init(initialization.v_init)
        reference.initialize()
        paths = _record_paths(runner.functional)
        state = _live_state(runner.functional, reference)
        history = [{name: _read_path(state, path) for name, path in paths.items()}]
        dt = runner.constants["dt"].detach()
        reference.integrator._initialize(
            reference, dt, force=True, compile_scope="population"
        )
        for index in range(steps):
            reference.integrator.step(
                reference, dt, inputs.ve[index], inputs.intra[index]
            )
            reference.t = reference.t + dt
            state = _live_state(runner.functional, reference)
            history.append(
                {name: _read_path(state, path) for name, path in paths.items()}
            )
    return {name: torch.stack([frame[name] for frame in history]) for name in paths}


@torch.no_grad()
def _matrix_oracle(model, recorded, parameters, dt):
    """Independently solve in storage order, including zero-volume junctions."""
    options = model._diffusion_test_options
    process = model.mech.material_processes["_FunctionalDiffusion"]
    pool, sodium = recorded[POOL][0], recorded[NAI][0]
    count = pool.shape[-1]
    if _is_tree_kind(options["kind"]):
        if hasattr(model, "material_edge_index"):
            parents, children = model.material_edge_index
            parents, children = parents.long(), children.long()
        else:
            # Native Cable retains the same authoritative storage-order
            # parent vector, without Tree's compact-edge convenience property.
            parent_index = model.diff_parent_index.long()
            children = torch.nonzero(parent_index >= 0, as_tuple=False).flatten()
            parents = parent_index.index_select(0, children)
    else:
        parents = torch.arange(count - 1, device=pool.device)
        children = parents + 1
    active = torch.ones_like(pool, dtype=torch.bool)
    if options["regional"]:
        active[..., 2:] = False
    edge_active = active.index_select(-1, parents) & active.index_select(-1, children)
    geometry = options["geometry"]
    if geometry is not None:
        config = model.material_geometry("chemical")
        volume = parameters[_key(parameters, config["volume"])].expand_as(pool)
        if geometry == "factor":
            factor = parameters[_key(parameters, config["edge_factor"])]
        else:
            factor = (
                parameters[_key(parameters, config["edge_area"])]
                / parameters[_key(parameters, config["edge_distance"])]
            )
    elif _is_tree_kind(options["kind"]):
        volume = model.volume_i.expand_as(pool)
        factor = model.diff_geom_um.index_select(-1, children)
    else:
        diameter, length = model.diam.expand_as(pool), model.dx.expand_as(pool)
        area = torch.pi * (diameter * 0.5).square()
        volume = area * length
        factor = (
            0.5
            * (area[..., :-1] + area[..., 1:])
            / (0.5 * (length[..., :-1] + length[..., 1:]))
        )
    diffusion = parameters[
        _key(parameters, "material_processes._FunctionalDiffusion.D_param")
    ]
    if options["regional"]:
        local = diffusion.expand_as(pool[..., :2])
        diffusion = torch.cat((local, torch.zeros_like(pool[..., 2:])), dim=-1)
    else:
        diffusion = diffusion.expand_as(pool)
    node_g = (
        0.5
        * (diffusion.index_select(-1, parents) + diffusion.index_select(-1, children))
        * factor
    )
    if options["interface"] == "series":
        half = 0.5 * length / (diffusion * area)
        node_g = (half[..., :-1] + half[..., 1:]).reciprocal()
    pool_g = node_g
    if options["edge_diffusivity"]:
        pool_g = parameters[_key(parameters, "edge_diffusivity")] * factor
    pool_g = torch.where(edge_active, pool_g, 0.0)
    node_g = torch.where(edge_active, node_g, 0.0)
    incidence = pool.new_zeros((count - 1, count))
    edge_index = torch.arange(count - 1, device=pool.device)
    incidence[edge_index, parents] = 1.0
    incidence[edge_index, children] = -1.0

    def laplace(g):
        return torch.einsum("...e,ei,ej->...ij", g, incidence, incidence)

    pool_laplace, sodium_laplace = laplace(pool_g), laplace(node_g)
    solve_volume = torch.where(active, volume, torch.ones_like(volume))
    mass = torch.diag_embed(solve_volume / dt)
    rate = pool.new_zeros(())
    target = pool.new_zeros(())
    if options["relaxation"]:
        rate = parameters[
            _key(parameters, "material_processes._FunctionalDiffusion.rate_param")
        ]
        target = parameters[
            _key(parameters, "material_processes._FunctionalDiffusion.target_param")
        ]
    reaction = torch.where(active, volume * rate, 0.0)
    pools, ions = [pool], [sodium]
    for _ in range(recorded[POOL].shape[0] - 1):
        source = pool + dt * PRODUCTION * (1.0 + 0.01 * sodium)
        if options["method"] == "implicit":
            rhs = solve_volume / dt * source + reaction * target
            pool = torch.linalg.solve(
                mass + pool_laplace + torch.diag_embed(reaction), rhs.unsqueeze(-1)
            ).squeeze(-1)
            sodium = torch.linalg.solve(
                mass + sodium_laplace, (solve_volume / dt * sodium).unsqueeze(-1)
            ).squeeze(-1)
        else:
            pool = (
                source
                - dt * (pool_laplace @ source.unsqueeze(-1)).squeeze(-1) / solve_volume
            )
            sodium = (
                sodium
                - dt
                * (sodium_laplace @ sodium.unsqueeze(-1)).squeeze(-1)
                / solve_volume
            )
        # This balance catches duplicate SOURCE application and split rather
        # than fused reservoir integration, independent of storage ordering.
        expected_mass = (volume * source).sum(-1) + dt * (
            reaction * (target - pool)
        ).sum(-1)
        torch.testing.assert_close(
            (volume * pool).sum(-1),
            expected_mass,
            rtol=2e-5 if pool.dtype == torch.float32 else 3e-10,
            atol=2e-5 if pool.dtype == torch.float32 else 3e-10,
        )
        pools.append(pool)
        ions.append(sodium)
    _close(
        {POOL: recorded[POOL], NAI: recorded[NAI]},
        {POOL: torch.stack(pools), NAI: torch.stack(ions)},
        dtype=pool.dtype,
    )
    assert process._diffusion_geometry_kind == (
        "tree" if _is_tree_kind(options["kind"]) else "1d"
    )


def _exercise(
    kind="path", *, backend="eager", execution="scan", checkpoint_every=None, **options
):
    model = _model(kind, **options)
    runner = _runner(
        model, backend=backend, execution=execution, checkpoint_every=checkpoint_every
    )
    source = _module_snapshot(model)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    result = runner(parameters=parameters, initialization=initialization, inputs=inputs)
    recorded = dict(result[1]["callbacks"]["trace"])
    gradients = _gradients(recorded, targets) if execution == "scan" else {}
    other_parameters, other_initialization, other_inputs, other_targets = _arguments(
        runner, model
    )
    expected = _explicit_reference(
        runner, other_parameters, other_initialization, other_inputs
    )
    tolerance = (
        dict(rtol=2e-4, atol=2e-5)
        if model.dtype() == torch.float32
        else dict(rtol=3e-9, atol=3e-10)
    )
    _assert_tree_close(result[0], expected[0], **tolerance)
    expected_recorded = dict(expected[1]["callbacks"]["trace"])
    _close(recorded, expected_recorded, dtype=model.dtype())
    if gradients:
        _close(
            gradients, _gradients(expected_recorded, other_targets), dtype=model.dtype()
        )
        assert {
            "stimulus",
            "initial_voltage",
            *CHANNELS,
            "initial_pool",
            "initial_nai",
            "geometry",
            "dt",
        } <= gradients.keys()
    _close(
        recorded,
        _imperative_reference(model, runner, parameters, initialization, inputs),
        dtype=model.dtype(),
    )
    _matrix_oracle(model, recorded, parameters, runner.constants["dt"])
    assert recorded["v"].shape == (STEPS + 1, *model.shape)
    if hasattr(model, "vc"):
        assert recorded["vc"].shape == (*recorded["v"].shape, 3)
        torch.testing.assert_close(
            recorded["vc"][0, ..., 0], recorded["v"][0], rtol=0, atol=0
        )
        assert torch.count_nonzero(recorded["vc"][0, ..., 1:]) == 0
    assert recorded["t"][0].item() == 0
    assert torch.count_nonzero(recorded["i_membrane"][0]) == 0
    torch.testing.assert_close(
        recorded[POOL][0],
        parameters[
            _key(parameters, "materials.pool._initial_sources.amount")
        ].expand_as(recorded[POOL][0]),
    )
    _assert_module_snapshot(model, source)
    return recorded, gradients


@pytest.mark.parametrize(
    "options",
    (
        {"kind": "path", "solver": "thomas"},
        {"kind": "path", "solver": "pcr", "geometry": "factor"},
        {
            "kind": "path",
            "solver": "dense",
            "method": "explicit",
            "relaxation": False,
            "interface": "series",
            "regional": True,
        },
        {"kind": "tree", "solver": "auto"},
        {"kind": "block_path", "solver": "thomas"},
        {"kind": "block_tree", "solver": "auto"},
        {"kind": "cable", "solver": "auto"},
        {
            "kind": "tree",
            "solver": "dense",
            "geometry": "area_distance",
            "edge_diffusivity": True,
        },
        {
            "kind": "tree",
            "solver": "auto",
            "geometry": "factor",
            "method": "explicit",
            "relaxation": False,
        },
    ),
)
def test_diffusion_full_traces_live_geometry_and_all_gradients(options, cpu_threads):
    _exercise(**options)


def test_diffusion_float32_zero_volume_tree_full_preparation_gradients(cpu_threads):
    _exercise("tree", dtype=torch.float32)


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_diffusion_checkpointed_continuation_full_traces_and_gradients(
    kind, cpu_threads
):
    model = _model(kind, geometry="factor")
    runner = _runner(model, checkpoint_every=2)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    state, auxiliary = runner(
        parameters=parameters, initialization=initialization, inputs=inputs
    )
    final, continued = runner(
        state,
        parameters=auxiliary["parameters"],
        inputs=inputs,
        callback_state=auxiliary["callbacks"].state,
    )
    recorded = {
        name: torch.cat((value, continued["callbacks"]["trace"][name]))
        for name, value in auxiliary["callbacks"]["trace"].items()
    }
    gradients = _gradients(recorded, targets)
    rp, ri, rx, rt = _arguments(runner, model)
    expected = _explicit_reference(
        runner,
        rp,
        ri,
        dn.func.RolloutInput(*(torch.cat((value, value)) for value in rx)),
        steps=2 * STEPS,
    )
    _assert_tree_close(final, expected[0], rtol=3e-9, atol=3e-10)
    expected_recorded = dict(expected[1]["callbacks"]["trace"])
    _close(recorded, expected_recorded)
    _close(gradients, _gradients(expected_recorded, rt))
    _close(
        recorded,
        _imperative_reference(
            model,
            runner,
            parameters,
            initialization,
            dn.func.RolloutInput(*(torch.cat((value, value)) for value in inputs)),
            steps=2 * STEPS,
        ),
    )
    _matrix_oracle(model, recorded, parameters, runner.constants["dt"])


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_diffusion_snapshot_restarts_do_not_advance_material_at_t0(kind, cpu_threads):
    model = _model(kind, geometry="factor")
    source = _module_snapshot(model)
    runner = _runner(model, initialization="snapshot")
    first, second = runner(), runner()
    _assert_tree_close(first[0], second[0])
    _close(dict(first[1]["callbacks"]["trace"]), dict(second[1]["callbacks"]["trace"]))
    trace = first[1]["callbacks"]["trace"]
    assert trace["t"][0].item() == 1.25
    torch.testing.assert_close(trace[POOL][0], model.mech.materials["pool"].amount)
    _assert_module_snapshot(model, source)


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_diffusion_commit_then_imperative_continuation(kind, cpu_threads):
    model = _model(kind, geometry="factor")
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    state, _ = functional.step(tensors.parameters, prepared, tensors.state)
    expected, _ = functional.step(tensors.parameters, prepared, state)
    target = _model(kind, geometry="factor")
    functional.commit_state_(target, state)
    dt = tensors.constants["dt"]
    target.integrator._initialize(target, dt, force=True, compile_scope="population")
    target.integrator.step(target, dt, None, None)
    target.t = target.t + dt
    _assert_tree_close(_live_state(functional, target), expected, rtol=3e-9, atol=3e-10)


def test_diffusion_parameter_geometry_and_dt_gradients_match_central_differences(
    cpu_threads,
):
    model = _model("path", geometry="area_distance", edge_diffusivity=True)
    runner = _runner(model)
    parameters, initialization, inputs, targets = _arguments(runner, model)
    recorded = dict(
        runner(parameters=parameters, initialization=initialization, inputs=inputs)[1][
            "callbacks"
        ]["trace"]
    )
    gradients = _gradients(recorded, targets)
    for label in (
        "edge_diffusivity",
        "chemical_volume",
        "chemical_edge_area",
        "chemical_edge_distance",
        "initial_pool",
        "initial_voltage",
        "dt",
    ):
        leaf, step = targets[label], 1e-6 if label == "dt" else 1e-5
        losses = []
        for sign in (1, -1):
            with torch.no_grad():
                leaf.add_(sign * step)
            losses.append(
                _loss(
                    _imperative_reference(
                        model, runner, parameters, initialization, inputs
                    )
                )
            )
            with torch.no_grad():
                leaf.sub_(sign * step)
        torch.testing.assert_close(
            gradients[label].sum(),
            (losses[0] - losses[1]) / (2 * step),
            rtol=4e-5,
            atol=3e-7,
        )


@pytest.mark.parametrize("kind", ("path", "tree"))
def test_diffusion_vmap_empty_vmap_jvp_full_recording(kind, cpu_threads):
    _exercise_transforms(kind)


def _exercise_transforms(kind, *, device="cpu", dtype=torch.float64):
    model = _model(kind, geometry="factor", device=device, dtype=dtype)
    runner = _runner(model)
    parameters, initialization, inputs, _targets = _arguments(runner, model)
    functional = runner.functional
    key = _key(parameters, "material_processes._FunctionalDiffusion.D_param")

    def rollout(diffusion, voltage, drive):
        initialized = functional.initialize(
            {**parameters, key: diffusion},
            runner.constants,
            initialization._replace(v_init=voltage),
        )
        _, auxiliary = functional.prepare_and_rollout(
            initialized.parameters,
            initialized.constants,
            initialized.state,
            steps=STEPS,
            inputs=inputs._replace(intra=drive),
            callbacks=runner._callbacks,
        )
        return dict(auxiliary["callbacks"]["trace"])

    def objective(diffusion, voltage, drive):
        return _loss(rollout(diffusion, voltage, drive))

    values = parameters[key], initialization.v_init, inputs.intra
    lanes = tuple(torch.stack((value * 0.97, value * 1.03)) for value in values)
    expected = pytree.tree_map(
        lambda *leaves: torch.stack(leaves),
        *[rollout(*(value[i] for value in lanes)) for i in range(2)],
    )
    _assert_tree_close(
        torch.func.vmap(rollout)(*lanes), expected, rtol=3e-9, atol=3e-10
    )
    derivative = torch.func.grad(objective, argnums=(0, 1, 2))
    expected_gradients = pytree.tree_map(
        lambda *leaves: torch.stack(leaves),
        *[derivative(*(value[i] for value in lanes)) for i in range(2)],
    )
    _assert_tree_close(
        torch.func.vmap(derivative)(*lanes), expected_gradients, rtol=3e-9, atol=3e-10
    )
    empty = torch.func.vmap(rollout)(*(value[:0] for value in lanes))
    assert empty[POOL].shape == (0, STEPS + 1, *model.shape)

    def material_sum(diffusion, voltage, drive):
        return rollout(diffusion, voltage, drive)[POOL].sum()

    # The compound scalar loss hits PyTorch's independent zero-lane scalar
    # arithmetic limitation. Full-state recording and full-loss nonempty AD
    # remain checked above; a direct material reduction checks empty AD.
    empty_gradients = torch.func.vmap(torch.func.grad(material_sum, argnums=(0, 1, 2)))(
        *(value[:0] for value in lanes)
    )
    assert all(value.shape[0] == 0 for value in empty_gradients)
    directions = tuple(
        torch.full_like(value, 0.01 if index < 2 else 1e-10)
        for index, value in enumerate(values)
    )
    _, tangent = torch.func.jvp(objective, values, directions)
    expected_tangent = sum(
        (gradient * direction).sum()
        for gradient, direction in zip(derivative(*values), directions, strict=True)
    )
    torch.testing.assert_close(tangent, expected_tangent, rtol=3e-9, atol=3e-10)


def test_explicit_native_tree_zero_volume_junction_fails_closed():
    model = _model("tree", method="explicit", relaxation=False)
    with pytest.raises(
        (dn.func.FunctionalizationError, RuntimeError), match="zero.volume|branchpoint"
    ):
        dn.func.make_functional(model, dt=DT)


def test_packed_diffusion_has_independent_chemical_topology():
    model = _membrane_model("multi_scalar", batch=None)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=torch.float64):
        model.material("pool", fields={"amount": 1.7}, domain="intracellular")
        model.concentrations(nai0=9.25)
        model.ion_style("na", 1, 1)
        model.insert(_MaterialSourceCurrent)
        model.insert(_process("implicit", "auto", None, False, False, "arithmetic"))
        model.initialize()
    dn.func.make_functional(model, dt=DT)
    assert torch.count_nonzero(model.diff_parent_index < 0) >= len(model.populations)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("kind", "execution", "checkpoint_every"),
    (("path", "scan", 2), ("tree", "while", None)),
)
def test_inductor_diffusion_scan_gradients_and_while_full_traces(
    kind, execution, checkpoint_every, cpu_threads
):
    _exercise(
        kind,
        geometry="factor",
        backend="inductor",
        execution=execution,
        checkpoint_every=checkpoint_every,
    )
