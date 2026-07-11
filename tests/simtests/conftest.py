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
    settings = {
        "celsius": float(h.celsius),
        "dt": float(h.dt),
        "secondorder": int(h.secondorder),
        "t": float(h.t),
        "cvode_active": bool(cvode.active()),
    }
    try:
        yield
    finally:
        for section in list(h.allsec()):
            h.delete_section(sec=section)
        h.celsius = settings["celsius"]
        h.secondorder = settings["secondorder"]
        h.dt = settings["dt"]
        h.t = settings["t"]
        cvode.active(settings["cvode_active"])
