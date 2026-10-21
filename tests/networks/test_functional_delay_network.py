"""Connected population gradients and saved-state replay with physical delays."""

import runpy
from pathlib import Path

import pytest

from dendra._bootstrap import reset_torch_compiler

DEMO = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "examples"
        / "functional_delay_interpolation.py"
    )
)


@pytest.fixture(autouse=True)
def _compiler_cache():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


@pytest.mark.parametrize("taps", (2, 3))
@pytest.mark.parametrize("synapse,batch", (("expsyn", None), ("exp2syn", 2)))
def test_interpolated_connections_match_native_voltages_gradients_and_replay(
    taps, synapse, batch
):
    DEMO["run_demo"](taps=taps, synapse=synapse, batch=batch)


@pytest.mark.parametrize("taps", (2, 3))
def test_fullgraph_interpolated_network_composition_keeps_checkpoint_gradients(taps):
    DEMO["run_demo"](taps=taps, synapse="exp2syn", batch=2, compiled=True)
