"""Isolation fixtures for simulator-backed tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_neuron_process_state():
    """Give every simtest a clean NEURON namespace and restore globals."""
    neuron = pytest.importorskip("neuron")
    h = neuron.h
    existing_sections = list(h.allsec())
    assert not existing_sections, (
        "NEURON simtests require an empty section namespace at setup; "
        f"found {[section.name() for section in existing_sections]!r}."
    )

    cvode = h.CVode()
    assert hasattr(h, "usetable_hh"), (
        "NEURON's built-in hh mechanism must expose usetable_hh so the "
        "simulator oracles can disable rate-table interpolation explicitly."
    )
    settings = {
        "celsius": float(h.celsius),
        "dt": float(h.dt),
        "secondorder": int(h.secondorder),
        "t": float(h.t),
        "cvode_active": bool(cvode.active()),
        "usetable_hh": float(h.usetable_hh),
    }
    # Dendra evaluates the canonical HH rate expressions directly.  NEURON's
    # built-in hh mechanism otherwise interpolates those rates from a voltage
    # table, which is a different model approximation rather than a solver
    # discrepancy.
    h.usetable_hh = 0
    assert h.usetable_hh == 0
    try:
        yield
    finally:
        for section in list(h.allsec()):
            h.delete_section(sec=section)
        h.celsius = settings["celsius"]
        h.secondorder = settings["secondorder"]
        h.dt = settings["dt"]
        h.t = settings["t"]
        h.usetable_hh = settings["usetable_hh"]
        cvode.active(settings["cvode_active"])
