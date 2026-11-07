"""Connected continuous delivery shares native values, gradients, and replay."""

import runpy
from pathlib import Path

import pytest

DEMO = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "examples/networks/functional_continuous_connectivity.py"
    )
)


@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("compiled", (False, True))
def test_connected_continuous_network_gradients_and_checkpoint_replay(batch, compiled):
    DEMO["run_demo"](batch=batch, compiled=compiled)
