"""Native cache loading owns broadcast buffers and preserves strict restore."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from tests.func.test_membrane_current import _model

pytestmark = pytest.mark.cpu
DT = 0.01


@pytest.fixture(autouse=True)
def _execution_environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch_compiler_warning_context():
            yield
    finally:
        torch.set_num_threads(previous)


def _network(kind):
    population = _model(kind, batch=None)
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=True, DTYPE=torch.float64, DEVICE="cpu"):
        network = dn.Network({"cell": population})
        network.batch(2)
        network.train()
        network.initialize(DT)
        with torch.no_grad():
            for _ in range(3):
                network.step()
            network.cache_state()
    return network


def _broadcast_buffers(population):
    return [
        (module, name, value)
        for module in population.modules()
        for name, value in module._buffers.items()
        if torch.is_tensor(value)
        and value.layout is torch.strided
        and any(
            size > 1 and stride == 0
            for size, stride in zip(value.shape, value.stride())
        )
    ]


def _snapshot(population):
    return {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in population.state_dict().items()
    }


@pytest.mark.parametrize("kind", ("tree", "block_tree", "multi_scalar", "multi_block"))
def test_batched_tree_cache_restores_expanded_geometry_and_matches_uninterrupted_step(
    kind,
):
    network, reference = _network(kind), _network(kind)
    assert _broadcast_buffers(network.cell)
    saved = network.cell.v.detach().clone()
    parameters = dict(network.cell.named_parameters())
    with torch.no_grad():
        network.cell.v.add_(7)
    network.initialize(DT)
    torch.testing.assert_close(network.cell.v, saved)
    assert all(
        dict(network.cell.named_parameters())[name] is value
        for name, value in parameters.items()
    )
    # The reference continues the same saved numerical state without loading
    # any cache, supplying an independent oracle for workspace reconstruction.
    reference._reset_runtime_clock(0)
    network.step()
    reference.step()
    torch.testing.assert_close(network.cell.v, reference.cell.v, rtol=2e-9, atol=2e-10)
    torch.testing.assert_close(
        network.cell.i_membrane, reference.cell.i_membrane, rtol=2e-9, atol=2e-12
    )
    if hasattr(network.cell, "vc"):
        torch.testing.assert_close(
            network.cell.vc, reference.cell.vc, rtol=2e-9, atol=2e-10
        )


def test_owned_cache_geometry_preserves_registered_buffer_aliases():
    network = _network("tree")
    module, name, original = next(
        (module, name, value)
        for module, name, value in _broadcast_buffers(network.cell)
        if name == "area"
    )
    module.register_buffer("_cache_area_alias", original)
    network.cache_state()
    network.initialize_pops_from_state_cache(dt=DT)
    restored = module._buffers[name]
    assert restored is module._buffers["_cache_area_alias"]
    assert restored is not original
    assert restored.is_contiguous()
    torch.testing.assert_close(restored, original)


def test_failed_canonical_cache_restore_rolls_back_values_and_broadcast_buffer_objects():
    network = _network("multi_scalar")
    module, name, original = next(
        (module, name, value)
        for module, name, value in _broadcast_buffers(network.cell)
        if name == "area"
    )
    module.register_buffer("_cache_area_alias", original)
    network.cache_state()
    before = _snapshot(network.cell)
    parameters = dict(network.cell.named_parameters())
    cache = network._state_cache["cell"]
    topology = next(
        key
        for key in cache
        if key.startswith("populations.tree.") and key.endswith("diff_parent_index")
    )
    cache[topology].reshape(-1)[0].bitwise_xor_(1)
    # Change an earlier parameter as well: a failing strict load must not
    # publish a partial copy before the later canonical component rejects it.
    parameter = next(iter(parameters))
    cache[parameter].add_(1)
    with pytest.raises(
        RuntimeError, match="canonical|morphology|topology|geometry|graph"
    ):
        network.initialize_pops_from_state_cache(dt=DT)
    assert module._buffers[name] is original
    assert module._buffers["_cache_area_alias"] is original
    assert all(
        dict(network.cell.named_parameters())[name] is value
        for name, value in parameters.items()
    )
    torch.testing.assert_close(_snapshot(network.cell), before)
