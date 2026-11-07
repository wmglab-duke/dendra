"""Authored waveform call graphs are observed on disposable explicit operands."""

import copy

import pytest
import torch

import dendra as dn
from dendra.func._poisson_invocations import profile_waveform_invocations
from dendra.func._poisson_stimulation import configure_execution, randomized_nodes
from dendra.models.stim.waveform.core import Waveform

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        yield


class _Leaf(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self, gain):
        super().__init__()
        self.register_parameter("gain", gain)

    def fn(self, times):
        return self.gain * torch.exp(-times.square())


class _Reordered(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self, first, second, unused, *, keyword=False):
        super().__init__()
        self.first, self.second, self.unused = first, second, unused
        self.keyword = keyword

    def fn(self, times):
        assert self.first.waveform.gain is self.second.waveform.gain
        if self.keyword:
            second = self.second.fn(t=times)
            first = self.first(t=times)
        else:
            second = self.second(times)
            first = self.first.fn(times)
        return second + first + 0.25 * self.first(times + 0.1)


class _Effectful(_Reordered):
    FUNCTIONAL_PURE = True
    observations = []

    def fn(self, times):
        type(self).observations.append("probe")
        torch.rand(2)
        times.add_(0.25)
        self.first.waveform.gain.add_(1.0)
        self.register_buffer("temporary", torch.ones(1))
        result = super().fn(times)
        if self.keyword:
            raise RuntimeError("Authored probe failure")
        return result


def _case(container=_Reordered, *, keyword=False, nested=False):
    gain = torch.nn.Parameter(torch.tensor(0.3))
    generator = torch.Generator().manual_seed(127)
    children = [
        _Leaf(gain).poisson(
            interval=0.2,
            n=2,
            start=-0.05,
            randomize_every_call=True,
            generator=generator,
        )
        for _ in range(3)
    ]
    source = container(*children, keyword=keyword)
    if nested:
        source = source.poisson(
            interval=0.3,
            n=2,
            noise=0.0,
            randomize_every_call=True,
            generator=generator,
        )
    nodes = tuple(path for path, _node in randomized_nodes(source))
    execution = copy.deepcopy(source)
    configure_execution(execution)
    return source, execution, nodes, generator


def _registered(module):
    return {
        **dict(module.named_parameters(remove_duplicate=False)),
        **dict(module.named_buffers(remove_duplicate=False)),
    }


@pytest.mark.parametrize("keyword", (False, True))
@pytest.mark.parametrize("nested", (False, True))
def test_profile_preserves_authored_order_repetitions_aliases_and_unused_children(
    keyword, nested
):
    source, execution, nodes, generator = _case(keyword=keyword, nested=nested)
    source_tensors, execution_tensors = _registered(source), _registered(execution)
    values = {
        name: tensor.detach().clone() for name, tensor in execution_tensors.items()
    }
    source_rng = generator.get_state().clone()
    global_rng = torch.get_rng_state().clone()
    times = torch.tensor([0.0], requires_grad=True)
    expected = ("second", "first", "first")
    if nested:
        expected = ("",) + tuple(f"waveform.{path}" for path in expected) * 2
    for _ in range(2):
        assert profile_waveform_invocations(execution, nodes, times) == expected
    assert all(
        _registered(source)[name] is value for name, value in source_tensors.items()
    )
    current = _registered(execution)
    assert all(current[name] is value for name, value in execution_tensors.items())
    for name, value in current.items():
        torch.testing.assert_close(value, values[name], rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), source_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    torch.testing.assert_close(times, torch.tensor([0.0]), rtol=0, atol=0)


@pytest.mark.parametrize("raises", (False, True))
def test_profile_disposable_operands_restore_global_and_class_state_on_failure(raises):
    source, execution, nodes, generator = _case(_Effectful, keyword=raises)
    source_tensors, execution_tensors = _registered(source), _registered(execution)
    values = {name: value.detach().clone() for name, value in execution_tensors.items()}
    source_rng, global_rng = (
        generator.get_state().clone(),
        torch.get_rng_state().clone(),
    )
    times = torch.tensor([0.0])
    if raises:
        with pytest.raises(RuntimeError, match="Authored probe failure"):
            profile_waveform_invocations(execution, nodes, times, check_purity=True)
    else:
        assert profile_waveform_invocations(
            execution, nodes, times, check_purity=True
        ) == (
            "second",
            "first",
            "first",
        )
    assert _Effectful.observations == []
    current = _registered(execution)
    assert all(current[name] is value for name, value in execution_tensors.items())
    assert all(
        _registered(source)[name] is value for name, value in source_tensors.items()
    )
    assert "temporary" not in execution._buffers
    for name, value in current.items():
        torch.testing.assert_close(value, values[name], rtol=0, atol=0)
    torch.testing.assert_close(generator.get_state(), source_rng, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
    torch.testing.assert_close(times, torch.tensor([0.0]), rtol=0, atol=0)
