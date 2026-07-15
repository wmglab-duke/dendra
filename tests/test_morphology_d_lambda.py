"""Contracts for native Morphology d-lambda discretization."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

import dendra as dn

pytestmark = pytest.mark.cpu


def _uniform_lambda(*, diameter, rhoa, cm, freq_hz):
    return 1.0e5 * math.sqrt(diameter / (4.0 * math.pi * freq_hz * rhoa * cm))


def _pt3d_lambda(points, *, rhoa, cm, freq_hz):
    length = 0.0
    electrotonic_integral = 0.0
    for first, second in zip(points, points[1:]):
        span = math.dist(first[:3], second[:3])
        length += span
        electrotonic_integral += span / math.sqrt(first[3] + second[3])
    electrotonic_length = (
        electrotonic_integral
        * math.sqrt(2.0)
        * 1.0e-5
        * math.sqrt(4.0 * math.pi * freq_hz * rhoa * cm)
    )
    return length / electrotonic_length


def _neuron_nseg(*, length, lambda_um, d_lambda):
    return max(1, int((length / (d_lambda * lambda_um) + 0.9) / 2.0) * 2 + 1)


def test_stylized_lambda_f_matches_the_uniform_cylinder_formula():
    morphology = dn.Morphology(rhoa=100.0, cm=1.0)
    section = morphology.section("cable", L=100.0, diam=2.0, nseg=4)
    expected = _uniform_lambda(
        diameter=2.0,
        rhoa=100.0,
        cm=1.0,
        freq_hz=100.0,
    )

    assert section.lambda_f() == pytest.approx(expected, rel=1e-14)
    assert section.lambda_f(freq_hz=400.0) == pytest.approx(expected / 2.0)
    assert section.nseg == 4

    override = morphology.section(
        "override",
        L=80.0,
        diam=4.0,
        rhoa=75.0,
        cm=1.5,
    )
    assert override.lambda_f(freq_hz=250.0) == pytest.approx(
        _uniform_lambda(
            diameter=4.0,
            rhoa=75.0,
            cm=1.5,
            freq_hz=250.0,
        ),
        rel=1e-14,
    )


def test_pt3d_lambda_f_integrates_bent_tapered_arclength():
    points = (
        (0.0, 0.0, 0.0, 4.0),
        (3.0, 4.0, 0.0, 2.0),  # 5 µm from the first control
        (3.0, 4.0, 12.0, 8.0),  # another 12 µm, around a bend
    )
    morphology = dn.Morphology(rhoa=120.0, cm=1.25)
    section = morphology.section("taper", points=points)

    expected = _pt3d_lambda(
        points,
        rhoa=120.0,
        cm=1.25,
        freq_hz=175.0,
    )
    assert section.L == pytest.approx(17.0)
    assert section.lambda_f(freq_hz=175.0) == pytest.approx(expected, rel=1e-14)


def test_zero_length_diameter_steps_add_no_electrotonic_length():
    points = (
        (0.0, 0.0, 0.0, 2.0),
        (10.0, 0.0, 0.0, 2.0),
        (10.0, 0.0, 0.0, 8.0),
        (20.0, 0.0, 0.0, 8.0),
    )
    morphology = dn.Morphology(rhoa=100.0, cm=1.0)
    section = morphology.section("step", points=points)

    # The coincident 2→8 µm controls contribute annular membrane area to
    # compilation, but their zero span contributes exactly zero here.
    expected_integral = 10.0 / math.sqrt(4.0) + 10.0 / math.sqrt(16.0)
    expected_electrotonic = (
        expected_integral
        * math.sqrt(2.0)
        * 1.0e-5
        * math.sqrt(4.0 * math.pi * 100.0 * 100.0 * 1.0)
    )
    expected = 20.0 / expected_electrotonic

    assert section.lambda_f() == pytest.approx(expected, rel=1e-14)
    assert section.lambda_f() == pytest.approx(
        _pt3d_lambda(points, rhoa=100.0, cm=1.0, freq_hz=100.0),
        rel=1e-14,
    )


def test_colocated_round_trip_diameter_steps_change_area_but_not_lambda_f():
    baseline_morphology = dn.Morphology()
    baseline = baseline_morphology.section(
        "cable",
        points=((0.0, 0.0, 0.0, 2.0), (20.0, 0.0, 0.0, 2.0)),
    )
    stepped_morphology = dn.Morphology()
    stepped = stepped_morphology.section(
        "cable",
        points=(
            (0.0, 0.0, 0.0, 2.0),
            (10.0, 0.0, 0.0, 2.0),
            (10.0, 0.0, 0.0, 4.0),
            (10.0, 0.0, 0.0, 2.0),
            (20.0, 0.0, 0.0, 2.0),
        ),
    )

    assert stepped.lambda_f() == baseline.lambda_f()
    assert (
        stepped_morphology.compile().geometry.area_um2[0]
        > baseline_morphology.compile().geometry.area_um2[0]
    )
    assert stepped_morphology.apply_d_lambda() == baseline_morphology.apply_d_lambda()


def test_lambda_f_uses_the_same_binary64_representable_spans_as_compilation():
    morphology = dn.Morphology()
    reference = morphology.section(
        "reference",
        points=((0.0, 0.0, 0.0, 1.0e32), (1.0e16, 0.0, 0.0, 1.0e32)),
    )
    collapsed = morphology.section(
        "collapsed",
        points=(
            (0.0, 0.0, 0.0, 1.0e32),
            (1.0e16, 0.0, 0.0, 1.0e32),
            (1.0e16, 1.0, 0.0, 1.0e-32),
            (1.0e16, 2.0, 0.0, 1.0e-32),
        ),
    )

    # Adding one micrometer after a 1e16 µm accumulated path is below that
    # arclength's binary64 resolution. Compilation omits both trailing spans;
    # d-lambda analysis must use the same representable cable rather than raw
    # coordinate chords (the final tiny-diameter chord would otherwise dominate).
    assert collapsed.L == reference.L == 1.0e16
    assert collapsed.lambda_f() == reference.lambda_f()

    selected = morphology.apply_d_lambda(d_lambda=0.1, freq_hz=100.0)
    assert selected["collapsed"] == selected["reference"]


def test_apply_d_lambda_uses_neuron_odd_grid_rounding_thresholds():
    morphology = dn.Morphology(rhoa=100.0, cm=1.0)
    lambda_um = _uniform_lambda(
        diameter=2.0,
        rhoa=100.0,
        cm=1.0,
        freq_hz=100.0,
    )
    d_lambda = 0.1
    targets = (1.05, 1.11, 3.05, 3.11)
    expected_nseg = (1, 3, 3, 5)
    sections = [
        morphology.section(
            f"q_{index}",
            L=target * d_lambda * lambda_um,
            diam=2.0,
            nseg=8,
        )
        for index, target in enumerate(targets)
    ]

    result = morphology.apply_d_lambda(d_lambda=d_lambda, freq_hz=100.0)

    assert result == {
        section.name: expected for section, expected in zip(sections, expected_nseg)
    }
    assert tuple(section.nseg for section in sections) == expected_nseg
    assert all(section.nseg % 2 == 1 for section in sections)
    assert tuple(result) == tuple(section.name for section in sections)


def test_apply_d_lambda_returns_declaration_order_and_preserves_topology_identity():
    morphology = dn.Morphology()
    root = morphology.section("root", L=500.0, diam=8.0, nseg=2)
    left = morphology.section("left", L=800.0, diam=2.0, nseg=4)
    right = morphology.section(
        "right",
        points=((0.0, 0.0, 0.0, 3.0), (0.0, 600.0, 0.0, 1.0)),
        nseg=6,
    )
    left.connect(root.at(0.25), child_end=1)
    right.connect(root.at(0.75), child_end=0)
    saved_location = root.at(0.5)
    declarations = morphology.sections

    result = morphology.apply_d_lambda(d_lambda=0.1, freq_hz=100.0)

    assert tuple(result) == ("root", "left", "right")
    assert morphology.sections == declarations
    assert all(
        current is original
        for current, original in zip(morphology.sections, declarations)
    )
    assert saved_location.section is root
    assert result == {section.name: section.nseg for section in declarations}
    graph = morphology.compile()
    assert set(graph.metadata.section_name).issuperset({"root", "left", "right"})


def test_d_lambda_is_one_shot_idempotent_and_monotone_when_reapplied():
    morphology = dn.Morphology()
    section = morphology.section("cable", L=400.0, diam=2.0, nseg=2)

    initial = morphology.apply_d_lambda(d_lambda=0.2, freq_hz=100.0)
    section.update(L=800.0)
    assert section.nseg == initial["cable"]

    after_geometry = morphology.apply_d_lambda(d_lambda=0.2, freq_hz=100.0)
    assert after_geometry["cable"] > initial["cable"]
    assert morphology.apply_d_lambda(d_lambda=0.2, freq_hz=100.0) == after_geometry

    finer = morphology.apply_d_lambda(d_lambda=0.1, freq_hz=100.0)
    higher_frequency = morphology.apply_d_lambda(d_lambda=0.1, freq_hz=400.0)
    assert finer["cable"] >= after_geometry["cable"]
    assert higher_frequency["cable"] >= finer["cable"]


@pytest.mark.parametrize("invalid", [0.0, -1.0, math.nan, math.inf])
def test_apply_d_lambda_validates_all_scalars_before_mutating(invalid):
    morphology = dn.Morphology()
    first = morphology.section("first", L=100.0, diam=2.0, nseg=2)
    second = morphology.section("second", L=200.0, diam=1.0, nseg=4)
    before = tuple(section.nseg for section in morphology.sections)

    with pytest.raises(ValueError, match=r"(?i)d.lambda|positive|finite"):
        morphology.apply_d_lambda(d_lambda=invalid)
    assert tuple(section.nseg for section in morphology.sections) == before

    with pytest.raises(ValueError, match=r"(?i)freq|positive|finite"):
        morphology.apply_d_lambda(freq_hz=invalid)
    assert tuple(section.nseg for section in morphology.sections) == before

    with pytest.raises(ValueError, match=r"(?i)freq|positive|finite"):
        first.lambda_f(freq_hz=invalid)
    assert (first.nseg, second.nseg) == before


def test_d_lambda_boolean_inputs_are_rejected_without_mutation():
    morphology = dn.Morphology()
    section = morphology.section("cable", L=100.0, diam=2.0, nseg=2)

    with pytest.raises(TypeError, match=r"(?i)d.lambda|boolean"):
        morphology.apply_d_lambda(d_lambda=True)
    with pytest.raises(TypeError, match=r"(?i)freq|boolean"):
        morphology.apply_d_lambda(freq_hz=False)
    with pytest.raises(TypeError, match=r"(?i)freq|boolean"):
        section.lambda_f(freq_hz=True)
    assert section.nseg == 2


def test_late_section_numeric_failure_rolls_back_every_nseg():
    morphology = dn.Morphology()
    first = morphology.section("first", L=100.0, diam=2.0, nseg=2)
    second = morphology.section(
        "overflow",
        L=100.0,
        diam=2.0,
        nseg=4,
        rhoa=1.0e308,
        cm=1.0e308,
    )

    with pytest.raises(ValueError, match=r"(?i)finite|frequency factor"):
        morphology.apply_d_lambda()

    assert (first.nseg, second.nseg) == (2, 4)


def test_empty_morphology_returns_an_empty_mapping_after_validation():
    morphology = dn.Morphology()

    assert morphology.apply_d_lambda() == {}
    with pytest.raises(ValueError, match=r"(?i)d.lambda|positive"):
        morphology.apply_d_lambda(d_lambda=0.0)
    with pytest.raises(ValueError, match=r"(?i)freq|positive"):
        morphology.apply_d_lambda(freq_hz=0.0)


def test_stale_and_forged_section_handles_cannot_compute_lambda():
    morphology = dn.Morphology()
    old = morphology.section("cable", L=100.0, diam=2.0)
    forged = replace(old)

    with pytest.raises(ValueError, match=r"(?i)canonical|registered"):
        forged.lambda_f()

    assert old.delete() == ("cable",)
    replacement = morphology.section("cable", L=200.0, diam=4.0)
    with pytest.raises(ValueError, match=r"(?i)canonical|registered|deleted"):
        old.lambda_f()

    assert replacement.lambda_f() == pytest.approx(
        _uniform_lambda(
            diameter=4.0,
            rhoa=replacement.rhoa,
            cm=replacement.cm,
            freq_hz=100.0,
        )
    )


def test_existing_graph_and_model_are_independent_of_later_d_lambda_changes():
    morphology = dn.Morphology()
    root = morphology.section("root", L=500.0, diam=6.0, nseg=1)
    child = morphology.section("child", L=1000.0, diam=1.0, nseg=1)
    child.connect(root.at(1.0), child_end=0)
    graph_before = morphology.compile()
    tree = dn.Tree.from_morphology(morphology)
    area_before = tree.area.clone()
    dx_before = tree.dx.clone()

    result = morphology.apply_d_lambda(d_lambda=0.05, freq_hz=200.0)
    graph_after = morphology.compile()

    assert any(nseg > 1 for nseg in result.values())
    assert graph_after != graph_before
    assert tree.compartment_graph == graph_before
    assert tree.compartment_graph.n_compartments == 2
    torch.testing.assert_close(tree.area, area_before)
    torch.testing.assert_close(tree.dx, dx_before)
    assert graph_after.n_compartments == sum(result.values())


def test_native_swc_can_apply_d_lambda_without_losing_type_provenance(tmp_path):
    swc = tmp_path / "long_soma_dendrite.swc"
    swc.write_text(
        "1 1 0 0 0 5 -1\n2 1 10 0 0 5 1\n3 3 20 0 0 1 2\n4 3 1020 0 0 1 3\n",
        encoding="utf-8",
    )
    morphology = dn.Morphology.from_swc(swc, nseg=2, rhoa=90.0, cm=1.2)
    declarations = morphology.sections
    provenance = morphology.swc_section_types
    expected_provenance = dict(provenance)

    result = morphology.apply_d_lambda(d_lambda=0.1, freq_hz=100.0)

    assert tuple(result) == tuple(section.name for section in declarations)
    assert all(value >= 1 and value % 2 == 1 for value in result.values())
    assert any(value > 1 for value in result.values())
    assert morphology.sections == declarations
    assert dict(provenance) == expected_provenance
    graph = morphology.compile()
    assert graph.n_compartments == sum(result.values())
    assert dn.Tree.from_morphology(morphology).compartment_graph == graph


def test_reported_nseg_matches_the_independent_analytic_rule():
    morphology = dn.Morphology(rhoa=110.0, cm=0.9)
    uniform = morphology.section("uniform", L=700.0, diam=3.0, nseg=2)
    points = (
        (0.0, 0.0, 0.0, 5.0),
        (200.0, 0.0, 0.0, 2.0),
        (500.0, 0.0, 0.0, 1.0),
    )
    taper = morphology.section("taper", points=points, nseg=2)
    d_lambda = 0.075
    freq_hz = 250.0

    expected = {
        "uniform": _neuron_nseg(
            length=uniform.L,
            lambda_um=_uniform_lambda(
                diameter=3.0,
                rhoa=110.0,
                cm=0.9,
                freq_hz=freq_hz,
            ),
            d_lambda=d_lambda,
        ),
        "taper": _neuron_nseg(
            length=taper.L,
            lambda_um=_pt3d_lambda(
                points,
                rhoa=110.0,
                cm=0.9,
                freq_hz=freq_hz,
            ),
            d_lambda=d_lambda,
        ),
    }

    assert (
        morphology.apply_d_lambda(
            d_lambda=d_lambda,
            freq_hz=freq_hz,
        )
        == expected
    )


@pytest.mark.neuron
@pytest.mark.parametrize(
    ("case", "points", "length", "diameter"),
    [
        ("uniform", None, 300.0, 2.0),
        (
            "bent_taper",
            (
                (0.0, 0.0, 0.0, 5.0),
                (30.0, 40.0, 0.0, 3.0),
                (30.0, 40.0, 120.0, 1.0),
            ),
            None,
            None,
        ),
        (
            "diameter_step",
            (
                (0.0, 0.0, 0.0, 2.0),
                (100.0, 0.0, 0.0, 2.0),
                (100.0, 0.0, 0.0, 8.0),
                (300.0, 0.0, 0.0, 8.0),
            ),
            None,
            None,
        ),
    ],
)
def test_native_d_lambda_matches_neuron_oracle(case, points, length, diameter):
    from neuron import h

    from dendra.models.io import apply_d_lambda as apply_neuron_d_lambda
    from dendra.models.io import lambda_f as neuron_lambda_f

    rhoa = 87.0
    cm = 1.3
    freq_hz = 173.0
    d_lambda = 0.08
    morphology = dn.Morphology(rhoa=rhoa, cm=cm)
    if points is None:
        native = morphology.section("cable", L=length, diam=diameter, nseg=2)
    else:
        native = morphology.section("cable", points=points, nseg=2)

    neuron_section = h.Section(name=f"native_dlambda_{case}")
    try:
        neuron_section.Ra = rhoa
        neuron_section.cm = cm
        neuron_section.nseg = 2
        if points is None:
            neuron_section.L = length
            neuron_section.diam = diameter
        else:
            h.pt3dclear(sec=neuron_section)
            for x, y, z, diam in points:
                h.pt3dadd(x, y, z, diam, sec=neuron_section)

        expected_lambda = neuron_lambda_f(neuron_section, freq_hz)
        assert native.lambda_f(freq_hz=freq_hz) == pytest.approx(
            expected_lambda,
            rel=1e-12,
        )

        native_result = morphology.apply_d_lambda(
            d_lambda=d_lambda,
            freq_hz=freq_hz,
        )
        apply_neuron_d_lambda(
            [neuron_section],
            d_lambda=d_lambda,
            freq=freq_hz,
        )
        assert native_result == {"cable": int(neuron_section.nseg)}
        assert native.nseg == int(neuron_section.nseg)
    finally:
        h.delete_section(sec=neuron_section)
