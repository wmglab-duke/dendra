"""Holistic simulation contracts for heterogeneous scalar MultiPopulation.

These tests deliberately exercise the public simulation lifecycle rather than
calling the packed integrator directly.  A packed component must reproduce the
trajectory of the same independently initialized Dendra model, including
mechanism evolution, intracellular stimulation, batching, and state writeback.

``concat_models`` is an execution packing operation: it does not create axial
edges between its component populations.  Physically connected soma/dendrite/
axon models belong in one Tree morphology instead.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

import dendra as dn
from dendra.models.integrators.tree import DENDRA_SOLVERS_AVAILABLE
from dendra.models.mod import hh, pas
from dendra.units import nA

pytest.importorskip("neuron")

pytestmark = [
    pytest.mark.neuron,
    pytest.mark.skipif(
        not DENDRA_SOLVERS_AVAILABLE,
        reason="Holistic scalar packing requires the CPU dendra_solvers package",
    ),
]

DTYPE = torch.float64
CELSIUS = 6.3
V_INIT = -65.0


def _native_tree(*, variant: int, active: bool, injected: bool = True):
    """Build a small branched native Tree with deliberately distinct geometry."""
    morphology = dn.Morphology(rhoa=35.4 + 8.0 * variant, cm=0.9 + 0.1 * variant)
    soma = morphology.section(
        "soma",
        L=18.0 + 2.0 * variant,
        diam=16.0 - variant,
        nseg=1,
        labels="injection_site",
    )
    trunk = morphology.section(
        "trunk",
        points=[
            (0.0, 0.0, 0.0, 4.0 + 0.2 * variant),
            (0.0, 55.0 + 5.0 * variant, 0.0, 2.0 + 0.1 * variant),
        ],
        nseg=3,
    )
    left = morphology.section(
        "left",
        points=[
            (0.0, 55.0 + 5.0 * variant, 0.0, 2.0),
            (-35.0, 90.0 + 5.0 * variant, 0.0, 1.1),
        ],
        nseg=2,
        labels="recording_branch",
    )
    right = morphology.section(
        "right",
        points=[
            (0.0, 55.0 + 5.0 * variant, 0.0, 1.8),
            (42.0, 94.0 + 5.0 * variant, 4.0, 0.9),
        ],
        nseg=2,
    )
    trunk.connect(soma.at(1.0), child_end=0)
    left.connect(trunk.at(1.0), child_end=0)
    right.connect(trunk.at(1.0), child_end=0)

    model = dn.Tree.from_morphology(
        morphology,
        N=1,
        celsius=CELSIUS,
        v_init=V_INIT,
        dtype=DTYPE,
    )
    if active:
        model.insert(hh)
        amplitude = (0.55 + 0.08 * variant) * nA
    else:
        model.insert(
            pas,
            g=2.0e-4 + 0.6e-4 * variant,
            e=-72.0 + 2.0 * variant,
        )
        amplitude = (0.12 + 0.03 * variant) * nA
    if injected:
        model.injection_site.inject(dn.mono_rect(amp=amplitude, delay=0.35, pw=0.55))
    return model


def _native_cable(*, active: bool, injected: bool = True, amplitude=None):
    """Build a tapered native path that cannot be reduced to fiber diameter."""
    morphology = dn.Morphology(rhoa=90.0, cm=1.0)
    source = morphology.section(
        "source",
        points=[(-12.0, 0.0, 0.0, 16.0), (12.0, 0.0, 0.0, 10.0)],
        nseg=1,
        rhoa=75.0,
        cm=0.85,
        labels=("membrane", "cable_stimulus"),
    )
    middle = morphology.section(
        "middle",
        points=[
            (12.0, 0.0, 0.0, 5.0),
            (18.0, 65.0, 7.0, 3.2),
            (42.0, 108.0, -3.0, 1.8),
        ],
        nseg=4,
        rhoa=125.0,
        cm=1.2,
        labels=("membrane", "cable_middle"),
    )
    # Authored distal-to-proximal. Native child_end=1 keeps authored x stable
    # while ordering the tridiagonal solve from the physical source to the tip.
    tip = morphology.section(
        "tip",
        points=[(88.0, 154.0, 12.0, 0.6), (42.0, 108.0, -3.0, 2.0)],
        nseg=4,
        rhoa=170.0,
        cm=1.05,
        labels=("membrane", "cable_tip"),
    )
    middle.connect(source.at(1.0), child_end=0)
    tip.connect(middle.at(1.0), child_end=1)

    model = dn.Cable.from_morphology(
        morphology,
        N=1,
        celsius=CELSIUS,
        v_init=V_INIT,
        dtype=DTYPE,
    )
    if active:
        model.membrane.insert(hh)
        default_amplitude = 8.0 * nA
    else:
        model.membrane.insert(pas, g=2.4e-4, e=-71.0)
        default_amplitude = 0.16 * nA
    if injected:
        model.cable_stimulus.inject(
            dn.mono_rect(
                amp=default_amplitude if amplitude is None else amplitude,
                delay=0.30,
                pw=0.50,
            )
        )
    return model


def _unmyelinated(*, active: bool, injected: bool = True, amplitude=None):
    model = dn.Unmyelinated(
        diameters=[2.2],
        L=160.0,
        dx=20.0,
        celsius=CELSIUS,
        v_init=V_INIT,
        rhoa=35.4,
        cm=1.0,
        dtype=DTYPE,
    )
    if active:
        model.insert(hh)
        default_amplitude = 0.72 * nA
    else:
        model.insert(pas, g=2.8e-4, e=-70.5)
        default_amplitude = 0.10 * nA
    if injected:
        model[..., 1:2].inject(
            dn.mono_rect(
                amp=default_amplitude if amplitude is None else amplitude,
                delay=0.30,
                pw=0.50,
            )
        )
    return model


def _myelinated(*, active: bool, injected: bool = True, amplitude=None):
    model = dn.Myelinated(
        diameters=[10.0],
        n_node=7,
        node_length=2.0,
        celsius=CELSIUS,
        v_init=V_INIT,
        rhoa=35.4,
        cm=1.0,
        dtype=DTYPE,
    )
    if active:
        model.insert(hh)
        default_amplitude = 0.72 * nA
    else:
        model.insert(pas, g=3.4e-4, e=-68.5)
        default_amplitude = 0.09 * nA
    if injected:
        model[..., 1:2].inject(
            dn.mono_rect(
                amp=default_amplitude if amplitude is None else amplitude,
                delay=0.40,
                pw=0.35,
            )
        )
    return model


_Builder = Callable[[], dn.Population]


def _factories(*names: str, active: bool) -> dict[str, _Builder]:
    available: dict[str, _Builder] = {
        "tree_a": lambda: _native_tree(variant=0, active=active),
        "tree_b": lambda: _native_tree(variant=1, active=active),
        "native_cable": lambda: _native_cable(active=active),
        "unmyelinated": lambda: _unmyelinated(active=active),
        "myelinated": lambda: _myelinated(active=active),
    }
    return {name: available[name] for name in names}


def _run_standalone(
    factories: dict[str, _Builder], *, dt: float, tstop: float
) -> dict[str, torch.Tensor]:
    traces = {}
    for name, factory in factories.items():
        model = factory()
        recorder = dn.callbacks.Recorder(states=["v"])
        model.eval()
        model.initialize()
        model.run(tstop=tstop, dt=dt, callbacks=[recorder])
        traces[name] = torch.from_numpy(recorder.numpy("v"))
    return traces


def _run_packed(
    factories: dict[str, _Builder],
    *,
    dt: float,
    tstop: float,
    write_back: bool = True,
):
    components = {name: factory() for name, factory in factories.items()}
    model = dn.concat_models(
        components,
        celsius=CELSIUS,
        write_back=write_back,
        threads=2,
    )
    recorder = dn.callbacks.RecorderLambda(
        {
            name: (lambda _, component=component: component.v.clone())
            for name, component in components.items()
        }
    )
    model.eval()
    model.initialize()
    model.run(tstop=tstop, dt=dt, callbacks=[recorder])
    traces = {name: torch.from_numpy(recorder.numpy(name)) for name in components}
    return model, components, traces


@pytest.mark.parametrize(
    "names",
    [
        ("tree_a", "tree_b"),
        ("unmyelinated", "myelinated"),
        ("tree_a", "unmyelinated"),
        ("tree_a", "myelinated"),
        ("tree_a", "unmyelinated", "myelinated"),
        ("native_cable", "tree_a"),
        ("native_cable", "unmyelinated", "myelinated"),
        ("native_cable", "tree_a", "unmyelinated", "myelinated"),
    ],
    ids=(
        "trees",
        "axons",
        "tree-unmyelinated",
        "tree-myelinated",
        "tree-and-axons",
        "native-cable-tree",
        "native-cable-axons",
        "all-scalar-paths",
    ),
)
def test_passive_scalar_combinations_match_independent_public_runs(names):
    """Packing changes launch layout, never a component trajectory."""
    dt = 0.025
    tstop = 2.0
    factories = _factories(*names, active=False)

    expected = _run_standalone(factories, dt=dt, tstop=tstop)
    model, components, actual = _run_packed(factories, dt=dt, tstop=tstop)

    for name in names:
        assert actual[name].shape == expected[name].shape
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-10, atol=2e-10)
        excursion = (actual[name] - actual[name][0]).abs().amax()
        assert float(excursion) > 0.02
        # write_back=True makes each public component state the final packed view.
        torch.testing.assert_close(components[name].v, actual[name][-1])
    assert model.t.item() == pytest.approx(tstop)


def test_active_native_cable_tree_and_both_axon_types_match_independent_trajectories():
    """HH state evolution and propagated spikes remain component-local."""
    dt = 0.0125
    tstop = 4.0
    factories = _factories(
        "native_cable",
        "tree_a",
        "unmyelinated",
        "myelinated",
        active=True,
    )

    expected = _run_standalone(factories, dt=dt, tstop=tstop)
    _, _, actual = _run_packed(factories, dt=dt, tstop=tstop)

    for name in factories:
        torch.testing.assert_close(actual[name], expected[name], rtol=3e-9, atol=3e-8)
        assert float(actual[name].amax()) > -20.0


def test_packed_native_cable_uses_exact_canonical_edge_resistance():
    """Packed DHS consumes compiled edges, not a center-cylinder surrogate."""
    dt = 0.025
    tstop = 2.0
    cable = _native_cable(active=False)

    # The tapered pt3d path is intentionally one for which reconstructing each
    # edge from center diameter and length is detectably wrong. The standalone
    # Tree below therefore serves as an independent exact-edge solver oracle.
    radius_cm = cable.diam * 1.0e-4 / 2.0
    segment_resistance = (
        cable.rhoa * cable.dx * 1.0e-4 / (torch.pi * radius_cm.square())
    )
    cylindrical_conductance = 2.0 / (
        segment_resistance[..., :-1] + segment_resistance[..., 1:]
    )
    canonical_conductance = cable.edge_resistance_ohm[..., 1:].reciprocal()
    relative_difference = (
        (cylindrical_conductance - canonical_conductance) / canonical_conductance
    ).abs()
    assert float(relative_difference.amax()) > 0.05

    tree = dn.Tree.from_compartment_graph(
        cable.compartment_graph,
        N=1,
        celsius=CELSIUS,
        v_init=V_INIT,
        dtype=DTYPE,
    )
    tree.membrane.insert(pas, g=2.4e-4, e=-71.0)
    tree.cable_stimulus.inject(dn.mono_rect(amp=0.16 * nA, delay=0.30, pw=0.50))
    tree_recorder = dn.callbacks.Recorder(states=["v"])
    tree.eval()
    tree.initialize()
    tree.run(tstop=tstop, dt=dt, callbacks=[tree_recorder])
    expected = torch.from_numpy(tree_recorder.numpy("v"))

    # Include a legacy Axon so this exercises heterogeneous packing and the
    # native Cable and legacy cylinder paths in the same dhs_multi launch.
    legacy_axon = _unmyelinated(active=False, injected=False)
    packed = dn.concat_models(
        {"native_cable": cable, "legacy_axon": legacy_axon},
        celsius=CELSIUS,
        threads=2,
    )
    packed_recorder = dn.callbacks.RecorderLambda(
        {"native_cable": lambda _: cable.v.clone()}
    )
    packed.eval()
    packed.initialize()
    packed.run(tstop=tstop, dt=dt, callbacks=[packed_recorder])
    actual = torch.from_numpy(packed_recorder.numpy("native_cable"))

    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)


def test_packing_does_not_electrically_connect_components():
    """A driven Tree cannot perturb an independently packed Axon."""
    dt = 0.025
    tstop = 2.0
    factories = {
        "driven_tree": lambda: _native_tree(variant=0, active=False, injected=True),
        "quiet_axon": lambda: _unmyelinated(active=False, injected=False),
    }
    expected = _run_standalone(factories, dt=dt, tstop=tstop)
    _, _, actual = _run_packed(factories, dt=dt, tstop=tstop)

    torch.testing.assert_close(
        actual["quiet_axon"], expected["quiet_axon"], rtol=0, atol=2e-12
    )
    assert float((actual["driven_tree"] - actual["driven_tree"][0]).abs().amax()) > 0.02


def test_native_cable_obeys_disabled_writeback_contract():
    """write_back=False updates the packed Slice, not the source Cable state."""
    cable = _native_cable(active=False)
    tree = _native_tree(variant=0, active=False)
    packed = dn.concat_models(
        {"native_cable": cable, "tree": tree},
        celsius=CELSIUS,
        write_back=False,
        threads=2,
    )
    packed.eval()
    packed.initialize()
    component_initial = cable.v.clone()
    packed_initial = packed.native_cable.v.clone()
    packed.run(tstop=1.0, dt=0.025)

    torch.testing.assert_close(cable.v, component_initial, rtol=0, atol=0)
    assert not torch.equal(packed.native_cable.v, packed_initial)


def test_packed_native_cable_rejects_geometry_mutation_before_state_advances():
    """Packed public execution revalidates each component's frozen geometry."""
    dt = 0.025
    cable = _native_cable(active=False)
    tree = _native_tree(variant=0, active=False)
    packed = dn.concat_models(
        {"native_cable": cable, "tree": tree},
        celsius=CELSIUS,
        threads=2,
    )
    packed.eval()
    packed.initialize()
    packed.run(tstop=dt, dt=dt)

    # Mutation happens only after a successful packed step, exercising the
    # validation boundary on a continuation with an already-built integrator.
    cable._canonical_edge_resistance_ohm[..., 1].mul_(1.01)
    packed_time = packed.t.clone()
    packed_voltage = packed.v.clone()
    cable_voltage = cable.v.clone()
    tree_voltage = tree.v.clone()

    with pytest.raises(RuntimeError, match="edge_resistance.*modified"):
        packed.run(tstop=dt, dt=dt)

    # Contract validation precedes duration accounting, mechanisms, callbacks,
    # or a solver launch; both packed and writeback component state are atomic.
    torch.testing.assert_close(packed.t, packed_time, rtol=0, atol=0)
    torch.testing.assert_close(packed.v, packed_voltage, rtol=0, atol=0)
    torch.testing.assert_close(cable.v, cable_voltage, rtol=0, atol=0)
    torch.testing.assert_close(tree.v, tree_voltage, rtol=0, atol=0)


def test_packed_native_cable_dependency_change_requires_reinitialization():
    """Component scale changes cannot use stale concatenated DHS coefficients."""
    dt = 0.025
    cable = _native_cable(active=False)
    tree = _native_tree(variant=0, active=False)
    packed = dn.concat_models(
        {"native_cable": cable, "tree": tree},
        celsius=CELSIUS,
        threads=2,
    )
    packed.eval()
    packed.initialize()
    packed.run(tstop=dt, dt=dt)

    cable.rhoa_scale.mul_(1.125)
    packed_time = packed.t.clone()
    packed_voltage = packed.v.clone()
    cable_voltage = cable.v.clone()
    tree_voltage = tree.v.clone()

    with pytest.raises(RuntimeError, match="initializ"):
        packed.run(tstop=dt, dt=dt)

    torch.testing.assert_close(packed.t, packed_time, rtol=0, atol=0)
    torch.testing.assert_close(packed.v, packed_voltage, rtol=0, atol=0)
    torch.testing.assert_close(cable.v, cable_voltage, rtol=0, atol=0)
    torch.testing.assert_close(tree.v, tree_voltage, rtol=0, atol=0)

    packed.initialize()
    packed.run(tstop=dt, dt=dt)
    assert float(packed.t) == pytest.approx(dt)


@pytest.mark.parametrize("name", ("rhoa_scale", "cm"))
def test_training_packed_dependency_change_requires_owner_reinitialization(name):
    """A training rebuild cannot silently bless stale concatenated fields."""
    dt = 0.025
    cable = _native_cable(active=False)
    tree = _native_tree(variant=0, active=False)
    packed = dn.concat_models(
        {"native_cable": cable, "tree": tree},
        celsius=CELSIUS,
        threads=2,
    )
    packed.train()
    packed.initialize()
    packed.step(dt=dt)

    dependency = getattr(cable, name)
    changed = dependency.detach().clone() * 1.125
    dependency.mul_(1.125)
    packed_field_before = getattr(packed, name).clone()
    packed_time = packed.t.clone()
    packed_voltage = packed.v.clone()
    cable_voltage = cable.v.clone()
    tree_voltage = tree.v.clone()

    with pytest.raises(RuntimeError, match="initializ"):
        packed.step(dt=dt)

    torch.testing.assert_close(packed.t, packed_time, rtol=0.0, atol=0.0)
    torch.testing.assert_close(packed.v, packed_voltage, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cable.v, cable_voltage, rtol=0.0, atol=0.0)
    torch.testing.assert_close(tree.v, tree_voltage, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        getattr(packed, name), packed_field_before, rtol=0.0, atol=0.0
    )

    # Skipping parameter repopulation retains the intentional child override,
    # while owner initialization refreshes the concatenated solver source.
    packed.initialize(populate_parameter_buffers=False)
    torch.testing.assert_close(getattr(cable, name), changed, rtol=0.0, atol=0.0)
    expected = torch.broadcast_to(changed, cable.shape).reshape(-1)
    actual = getattr(packed, name).reshape(-1)[: expected.numel()]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    packed.step(dt=dt)
    assert float(packed.t) == pytest.approx(dt)


def test_packed_inference_cable_dependency_change_is_rejected_atomically():
    """Packed execution also snapshots unversioned component dependencies."""
    dt = 0.025
    with torch.inference_mode():
        cable = _native_cable(active=False)
        tree = _native_tree(variant=0, active=False)
        packed = dn.concat_models(
            {"native_cable": cable, "tree": tree},
            celsius=CELSIUS,
            threads=2,
        )
        packed.eval()
        packed.initialize()
        packed.run(tstop=dt, dt=dt)

        assert torch.is_inference(cable.rhoa_scale)
        packed_time = packed.t.clone()
        packed_voltage = packed.v.clone()
        cable_voltage = cable.v.clone()
        tree_voltage = tree.v.clone()

        cable.rhoa_scale.mul_(1.125)
        with dn.ctx(RUNTIME_CONTRACT_VALIDATION="versioned"):
            with pytest.raises(RuntimeError, match="initializ"):
                packed.run(tstop=dt, dt=dt)

        torch.testing.assert_close(packed.t, packed_time, rtol=0, atol=0)
        torch.testing.assert_close(packed.v, packed_voltage, rtol=0, atol=0)
        torch.testing.assert_close(cable.v, cable_voltage, rtol=0, atol=0)
        torch.testing.assert_close(tree.v, tree_voltage, rtol=0, atol=0)


def _run_batched(*, packed: bool, amplitudes: torch.Tensor, dt: float, tstop: float):
    factories = {
        "tree": lambda: _native_tree(variant=0, active=False, injected=False),
        "native_cable": lambda: _native_cable(active=False, injected=False),
        "axon": lambda: _unmyelinated(
            active=False,
            amplitude=amplitudes,
        ),
        "myelinated": lambda: _myelinated(
            active=False,
            amplitude=amplitudes,
        ),
    }
    if not packed:
        traces = {}
        for name, factory in factories.items():
            model = factory().batch(amplitudes.shape[0])
            if name == "tree":
                stimulus_site = model.injection_site
            elif name == "native_cable":
                stimulus_site = model.cable_stimulus
            else:
                stimulus_site = None
            if stimulus_site is not None:
                stimulus_site.inject(dn.mono_rect(amp=amplitudes, delay=0.25, pw=0.65))
            recorder = dn.callbacks.Recorder(states=["v"])
            model.eval()
            model.initialize()
            model.run(tstop=tstop, dt=dt, callbacks=[recorder])
            traces[name] = torch.from_numpy(recorder.numpy("v"))
        return traces

    components = {name: factory() for name, factory in factories.items()}
    model = dn.concat_models(components, celsius=CELSIUS, threads=2).batch(
        amplitudes.shape[0]
    )
    model.tree.injection_site.inject(dn.mono_rect(amp=amplitudes, delay=0.25, pw=0.65))
    model.native_cable.cable_stimulus.inject(
        dn.mono_rect(amp=amplitudes, delay=0.25, pw=0.65)
    )
    recorder = dn.callbacks.RecorderLambda(
        {
            name: (lambda _, component=component: component.v.clone())
            for name, component in components.items()
        }
    )
    model.eval()
    model.initialize()
    model.run(tstop=tstop, dt=dt, callbacks=[recorder])
    return {name: torch.from_numpy(recorder.numpy(name)) for name in components}


def test_explicit_per_batch_stimuli_match_batched_standalone_models():
    """Explicit per-step ``[batch, 1, 1]`` currents survive scalar packing."""
    dt = 0.025
    tstop = 2.0
    # Waveforms are time-last.  The parameter therefore carries a final time
    # singleton so mono_rect evaluates to [batch, 1, 1, time] and unbinding
    # time presents Intra with the explicit [batch, 1, 1] spatial sample.
    amplitudes = torch.tensor([0.0, 0.08, 0.17], dtype=DTYPE).reshape(3, 1, 1, 1) * nA

    expected = _run_batched(
        packed=False,
        amplitudes=amplitudes,
        dt=dt,
        tstop=tstop,
    )
    actual = _run_batched(
        packed=True,
        amplitudes=amplitudes,
        dt=dt,
        tstop=tstop,
    )

    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=3e-10, atol=3e-10)
        assert actual[name].shape[1] == amplitudes.shape[0]
        # All replicas share the same passive relaxation.  Measure only the
        # stimulus-dependent departure from the zero-current first replica.
        response = (actual[name] - actual[name][:, :1]).abs().amax(dim=(0, -2, -1))
        assert response[0] < response[1] < response[2]
