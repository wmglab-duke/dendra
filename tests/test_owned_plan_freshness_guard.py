"""Plan metadata forwarding preserves native freshness and authored execution."""

from __future__ import annotations

import sys
import weakref

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._device import DeviceContext

import dendra as dn
from dendra.models.integrators import tree as tree_module
from dendra.models.integrators.core import Integrator

pytestmark = [
    pytest.mark.cpu,
    pytest.mark.skipif(
        not callable(getattr(tree_module.dendra_solvers, "DHSPlan", None)),
        reason="requires owning native topology plans",
    ),
]


class _Owner(Integrator):
    """Exercise shared freshness bookkeeping without running a solver."""

    def __init__(self, sources):
        torch.nn.Module.__init__(self)
        for name, value in zip(("parent", "order", "layers"), sources, strict=True):
            self.register_buffer(name, value)
        self._cpu_topology_plan = None
        self._cpu_plan_sources = ()
        self._cpu_plan_versions = ()
        self._compiled_kernels = {}
        self.events = []

    def sources(self):
        return tuple(self._buffers[name] for name in ("parent", "order", "layers"))

    def clear_jit_cache(self):
        self.events.append(("clear", self._cpu_topology_plan is None))
        return super().clear_jit_cache()


@pytest.fixture(autouse=True)
def _environment():
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DEVICE="cpu", DTYPE=torch.float64):
        yield


def _sources():
    return tuple(
        torch.tensor(values, dtype=torch.int64)
        for values in ((-1, 0, 0), (2, 1, 0), (0, 2, 3))
    )


def _warm(sources=None, *, constructor=None):
    owner = _Owner(_sources() if sources is None else sources)
    native = tree_module.dendra_solvers.DHSPlan

    def construct(*values):
        owner.events.append(("construct", owner._cpu_topology_plan is None))
        return (native if constructor is None else constructor)(*values)

    tree_module._refresh_owned_cpu_plan(owner, owner.sources(), construct, True)
    assert owner._cpu_topology_plan is not None
    owner.events.clear()
    owner._compiled_kernels["sentinel"] = object()
    return owner, construct


def _metadata_forwarding(function, descriptors=None):
    """Observe stock DeviceContext forwarding without replacing any method."""
    if descriptors is None:
        descriptors = {"device": torch.Tensor.device, "version": torch.Tensor._version}
    counts = dict.fromkeys(descriptors, 0)
    code = DeviceContext.__torch_function__.__code__
    previous = sys.getprofile()

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is code:
            descriptor = getattr(frame.f_locals["func"], "__self__", None)
            for name, expected in descriptors.items():
                if descriptor is expected:
                    counts[name] += 1
        if callable(previous):
            previous(frame, event, arg)

    sys.setprofile(observe)
    try:
        function()
    finally:
        sys.setprofile(previous)
    return counts


def test_plain_native_metadata_reads_avoid_stock_device_context_forwarding():
    owner, constructor = _warm()
    previous_plan = owner._cpu_topology_plan
    counts = _metadata_forwarding(
        lambda: tree_module._refresh_owned_cpu_plan(
            owner, owner.sources(), constructor, True
        )
    )
    assert counts == {"device": 0, "version": 0}
    assert owner._cpu_topology_plan is previous_plan
    assert owner.events == [] and owner._compiled_kernels


def test_public_identical_topology_swap_keeps_identity_and_legacy_owner_reuse():
    owner, constructor = _warm()
    previous_plan = owner._cpu_topology_plan
    source = owner.parent
    identity, version, uses = id(source), source._version, source._use_count()
    assert not weakref.getweakrefs(source)
    replacement = source.clone()
    assert replacement._version == version
    torch.utils.swap_tensors(source, replacement)
    assert id(source) == identity and source._version == version
    tree_module._refresh_owned_cpu_plan(owner, owner.sources(), constructor, True)
    assert source._use_count() == uses
    assert not weakref.getweakrefs(source)
    assert owner._cpu_topology_plan is previous_plan
    assert owner.events == [] and owner._compiled_kernels


@pytest.mark.parametrize("replacement", ("meta", "inference"))
def test_same_object_swap_keeps_device_and_versionless_raw_fallback(replacement):
    owner, constructor = _warm()
    source = owner.parent
    identity, version = id(source), source._version
    if replacement == "inference":
        with torch.inference_mode():
            other = torch.tensor([-1, 0, 0], dtype=torch.int64)
    else:
        other = torch.tensor([-1, 0, 0], dtype=torch.int64, device="meta")
        assert other._version == version
    torch.utils.swap_tensors(source, other)
    assert id(source) == identity
    if replacement == "inference":
        with pytest.raises(RuntimeError, match="Inference tensors"):
            _ = source._version
    else:
        assert source.device.type == "meta" and source._version == version
    tree_module._refresh_owned_cpu_plan(owner, owner.sources(), constructor, True)
    assert owner._cpu_topology_plan is None
    assert owner._cpu_plan_sources == owner._cpu_plan_versions == ()
    assert not owner._compiled_kernels
    assert owner.events == [("clear", True)]


def test_inplace_invalid_values_clear_owner_and_code_before_native_validation():
    owner, constructor = _warm()
    owner.parent[1] = 1
    with pytest.raises(RuntimeError, match="parent itself"):
        tree_module._refresh_owned_cpu_plan(owner, owner.sources(), constructor, True)
    assert owner._cpu_topology_plan is None
    assert owner._cpu_plan_sources == owner._cpu_plan_versions == ()
    assert not owner._compiled_kernels
    assert owner.events == [("clear", True), ("construct", True)]


def test_replacement_callable_keeps_legacy_reuse_when_sources_are_unchanged():
    owner, _ = _warm()
    previous_plan = owner._cpu_topology_plan

    def unexpected(*sources):
        raise AssertionError("unchanged sources must retain the existing owner")

    tree_module._refresh_owned_cpu_plan(owner, owner.sources(), unexpected, True)
    assert owner._cpu_topology_plan is previous_plan
    assert owner.events == [] and owner._compiled_kernels


@pytest.mark.parametrize("disabled", ("constructor", "eligibility"))
def test_disabled_plan_capability_preserves_short_circuit_and_clear_order(disabled):
    owner, constructor = _warm()
    observations = []

    class Eligibility:
        def __bool__(self):
            observations.append("eligibility")
            if disabled == "constructor":
                raise AssertionError("noncallable constructor must short-circuit")
            return False

    tree_module._refresh_owned_cpu_plan(
        owner,
        owner.sources(),
        None if disabled == "constructor" else constructor,
        Eligibility(),
    )
    assert observations == ([] if disabled == "constructor" else ["eligibility"])
    assert owner._cpu_topology_plan is None and not owner._compiled_kernels
    assert owner.events == [("clear", True)]


def test_eligibility_exception_precedes_clearing_existing_owner():
    owner, constructor = _warm()
    previous_plan = owner._cpu_topology_plan

    class Eligibility:
        def __bool__(self):
            raise ValueError("authored eligibility failure")

    with pytest.raises(ValueError, match="authored eligibility failure"):
        tree_module._refresh_owned_cpu_plan(
            owner, owner.sources(), constructor, Eligibility()
        )
    assert owner._cpu_topology_plan is previous_plan
    assert owner.events == [] and owner._compiled_kernels


def test_subclass_fallback_keeps_authored_device_and_version_read_order():
    observations = []

    class Authored(torch.Tensor):
        @property
        def device(self):
            observations.append("device")
            return super().device

        @property
        def _version(self):
            observations.append("version")
            return super()._version

        @property
        def _cdata(self):
            raise AssertionError("native implementation identity must not be read")

    sources = _sources()
    authored = torch.Tensor._make_subclass(Authored, sources[0], False)
    owner, constructor = _warm(
        (authored, *sources[1:]), constructor=lambda *args: object()
    )
    previous_plan = owner._cpu_topology_plan
    observations.clear()
    tree_module._refresh_owned_cpu_plan(owner, owner.sources(), constructor, True)
    assert observations == ["device", "version"]
    assert owner._cpu_topology_plan is previous_plan and owner.events == []


def test_authored_device_failure_precedes_clearing_existing_owner():
    owner, constructor = _warm()
    previous_plan = owner._cpu_topology_plan

    class Authored(torch.Tensor):
        @property
        def device(self):
            raise ValueError("authored device failure")

    owner.parent = torch.Tensor._make_subclass(Authored, owner.parent.clone(), False)
    with pytest.raises(ValueError, match="authored device failure"):
        tree_module._refresh_owned_cpu_plan(owner, owner.sources(), constructor, True)
    assert owner._cpu_topology_plan is previous_plan
    assert owner.events == [] and owner._compiled_kernels


def test_authored_cached_version_getter_runs_outside_metadata_suppression():
    owner, constructor = _warm()
    previous_plan = owner._cpu_topology_plan
    observations = []

    class AuthoredOwner(_Owner):
        @property
        def _cpu_plan_versions(self):
            observations.append(torch.empty(0).device.type)
            raise ValueError("authored cached-version failure")

    owner.__class__ = AuthoredOwner
    with dn.ctx(DEVICE="meta"):
        with pytest.raises(ValueError, match="authored cached-version failure"):
            tree_module._refresh_owned_cpu_plan(
                owner, owner.sources(), constructor, True
            )
    assert observations == ["meta"]
    assert owner._cpu_topology_plan is previous_plan
    assert owner.events == [] and owner._compiled_kernels


def test_custom_function_mode_retains_device_and_version_dispatch():
    owner, constructor = _warm()
    previous_plan = owner._cpu_topology_plan
    descriptors = []

    class Observe(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            descriptors.append(getattr(func, "__self__", None))
            return func(*args, **({} if kwargs is None else kwargs))

    with Observe():
        tree_module._refresh_owned_cpu_plan(owner, owner.sources(), constructor, True)
    assert any(item is torch.Tensor.device for item in descriptors)
    assert any(item is torch.Tensor._version for item in descriptors)
    assert not any(item is torch.Tensor._cdata for item in descriptors)
    assert owner._cpu_topology_plan is previous_plan and owner.events == []


@pytest.mark.parametrize("name", ("device", "_version"))
def test_authored_plain_tensor_getter_retains_metadata_forwarding(monkeypatch, name):
    owner, constructor = _warm()
    previous_plan = owner._cpu_topology_plan
    descriptors = {"device": torch.Tensor.device, "version": torch.Tensor._version}
    original = getattr(torch.Tensor, name)
    observations = []

    def authored(self):
        observations.append(name)
        return original.__get__(self, type(self))

    monkeypatch.setattr(torch.Tensor, name, property(authored))

    def original_read_order():
        sources = owner.sources()
        assert all(source.device.type in {"cpu", "mps"} for source in sources)
        return tuple(source._version for source in sources)

    expected_counts = _metadata_forwarding(original_read_order, descriptors)
    expected_observations = list(observations)
    observations.clear()
    counts = _metadata_forwarding(
        lambda: tree_module._refresh_owned_cpu_plan(
            owner, owner.sources(), constructor, True
        ),
        descriptors,
    )
    assert expected_observations
    assert observations == expected_observations
    assert counts == expected_counts
    assert owner._cpu_topology_plan is previous_plan and owner.events == []


def test_instance_torch_function_override_retains_native_metadata_dispatch():
    owner, constructor = _warm()
    previous_plan = owner._cpu_topology_plan

    def authored(func, types, args=(), kwargs=None):
        with torch._C.DisableTorchFunction():
            return func(*args, **({} if kwargs is None else kwargs))

    owner.parent.__torch_function__ = authored
    counts = _metadata_forwarding(
        lambda: tree_module._refresh_owned_cpu_plan(
            owner, owner.sources(), constructor, True
        )
    )
    assert counts == {"device": 3, "version": 3}
    assert owner._cpu_topology_plan is previous_plan and owner.events == []
