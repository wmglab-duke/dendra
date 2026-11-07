"""Source-backed Ion/Material derivations preserve values and graph ownership."""

from __future__ import annotations

import copy
from types import MethodType

import pytest
import torch

from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion, concentrations, equilibria
from dendra.models.mechanisms._materials import (
    Material,
    _configure_initial_source_execution,
    _resolve_initial_source,
)
from dendra.models.multi import _PackedMaterialInitial
from dendra.models.parametric import PositiveParam, cacheable


class _ModeSource(cacheable):
    def __init__(self, value):
        super().__init__()
        self.rho = torch.nn.Parameter(value)

    def __len__(self):
        return self.rho.numel()

    def _compute(self):
        return self.rho.square() + (2.0 if self.training else 3.0)


class _FieldValues(torch.nn.Module):
    def __init__(self, material):
        super().__init__()
        self.material = material

    def forward(self, celsius):
        if isinstance(self.material, Ion):
            values = self.material._derive_initial_field_values(celsius)
        else:
            values = self.material._derive_initial_field_values()
        return tuple(values[field] for field in self.material.fields)


class _NestedSource(torch.nn.Module):
    def __init__(self, source, *, fail=False):
        super().__init__()
        self.source = source
        self.alias = source
        self.fail = fail

    def forward(self):
        value = self.source() + 2 * self.alias()
        if self.fail:
            raise RuntimeError("source failure")
        return value


class _CustomCachedForward(PositiveParam):
    def forward(self):
        return super().forward() + 7.0


class _MetadataSource(_NestedSource):
    def modules(self):
        raise AssertionError("authored modules traversal")

    def named_modules(self, *args, **kwargs):
        raise AssertionError("authored named_modules traversal")


class _ReservedMarkerSource(_NestedSource):
    @property
    def _initialization_source_cache_free(self):
        raise AssertionError("authored execution-marker getter")

    @_initialization_source_cache_free.setter
    def _initialization_source_cache_free(self, value):
        raise AssertionError("authored execution-marker setter")


def _ion(source, *, nernst=False):
    with concentrations(nai0=source, nao0=source), equilibria(ena=source):
        return Ion("na", (2, 3), einit=int(nernst), eadvance=0).to(
            dtype=source.rho.dtype
        )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((), (3,), (2, 3)))
@pytest.mark.parametrize("kind", ("ion", "material"))
def test_shared_sources_preserve_native_derivation_and_gradient(kind, shape, dtype):
    raw = torch.arange(max(torch.Size(shape).numel(), 1), dtype=dtype).reshape(shape)
    source = PositiveParam(raw + 1.0, requires_grad=True)
    if kind == "ion":
        owner = _ion(source).train()
        fields = ("ena", "nai", "nao")
    else:
        owner = Material("pool", (2, 3), fields={"c": source, "d": source}).train()
        fields = ("c", "d")
    before = copy.deepcopy(owner.state_dict())
    identities = {name: id(value) for name, value in owner.named_buffers()}
    celsius = torch.tensor(34.0, dtype=dtype)
    values = (
        owner._derive_initial_field_values(celsius)
        if kind == "ion"
        else owner._derive_initial_field_values()
    )
    for field in fields:
        assert values[field].dtype == dtype
        torch.testing.assert_close(values[field], source._compute().expand(2, 3))
        assert values[field].grad_fn is not None
    assert {name: id(value) for name, value in owner.named_buffers()} == identities
    for name, value in owner.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    gradient = torch.autograd.grad(
        sum(values[field].sum() for field in fields), source.rho
    )[0]
    repetitions = 6 // max(source.rho.numel(), 1)
    expected = len(fields) * repetitions * torch.sigmoid(source.rho.detach())
    torch.testing.assert_close(gradient, expected)


@pytest.mark.parametrize("kind", ("ion", "material"))
@pytest.mark.parametrize("training", (False, True))
def test_source_training_flag_is_an_explicit_numerical_input(kind, training):
    source = _ModeSource(torch.tensor(2.0, dtype=torch.float64))
    owner = (
        _ion(source)
        if kind == "ion"
        else Material("pool", (2, 3), fields={"c": source})
    )
    owner.train()
    source.train(training)
    celsius = torch.tensor(34.0, dtype=torch.float64)
    values = (
        owner._derive_initial_field_values(celsius)
        if kind == "ion"
        else owner._derive_initial_field_values()
    )
    field = "nai" if kind == "ion" else "c"
    torch.testing.assert_close(
        values[field], torch.full((2, 3), 6.0 if training else 7.0, dtype=torch.float64)
    )
    assert source.training is training


@pytest.mark.parametrize("kind", ("ion", "material"))
def test_repeated_native_initialization_keeps_fresh_training_graphs(kind):
    source = PositiveParam(torch.tensor(1.5, dtype=torch.float64), requires_grad=True)
    owner = (
        _ion(source)
        if kind == "ion"
        else Material("pool", (2, 3), fields={"c": source})
    ).train()
    celsius = torch.tensor(34.0, dtype=torch.float64)
    field = "nai" if kind == "ion" else "c"
    losses = []
    for _ in range(2):
        owner.initialize(celsius)
        losses.append(getattr(owner, field).sum())
    gradient = torch.autograd.grad(sum(losses), source.rho)[0]
    torch.testing.assert_close(gradient, 12 * torch.sigmoid(source.rho.detach()))


@pytest.mark.parametrize("kind", ("ion", "material"))
def test_shared_initial_source_module_is_not_part_of_runtime_checkpoint(kind):
    source = PositiveParam(torch.tensor(1.5, dtype=torch.float64), requires_grad=True)
    owner = (
        _ion(source)
        if kind == "ion"
        else Material("pool", (2, 3), fields={"c": source, "d": source})
    )
    expected_names = (
        {"e_init.rho", "i_init.rho", "o_init.rho"}
        if kind == "ion"
        else {"_initial_sources.c.rho", "_initial_sources.d.rho"}
    )
    assert expected_names <= set(owner.state_dict())
    assert list(owner.parameters()) == [source.rho]
    assert len(tuple(owner.named_parameters(remove_duplicate=False))) == len(
        expected_names
    )
    owner.train().initialize(torch.tensor(34.0, dtype=torch.float64))
    handler = MechanismHandler(
        torch.tensor(34.0, dtype=torch.float64),
        torch.ones((2, 3), dtype=torch.float64),
        {},
        ions={"na": owner} if kind == "ion" else {},
        materials={"pool": owner} if kind == "material" else {},
    )
    runtime = handler.mutable_state_dict()
    assert not any("rho" in name or "initial" in name for name in runtime)
    field = "nai" if kind == "ion" else "c"
    expected = getattr(owner, field).detach().clone()
    with torch.no_grad():
        source.rho.add_(1.0)
        getattr(owner, field).add_(3.0)
    raw = source.rho.detach().clone()
    handler.restore_mutable_state_dict(runtime)
    torch.testing.assert_close(getattr(owner, field), expected)
    torch.testing.assert_close(source.rho, raw)


@pytest.mark.parametrize("kind", ("ion", "material"))
def test_fresh_initialization_uses_current_raw_leaves_without_changing_eval_cache(kind):
    source = PositiveParam(torch.tensor(1.5, dtype=torch.float64), requires_grad=True)
    owner = (
        _ion(source)
        if kind == "ion"
        else Material("pool", (2, 3), fields={"c": source})
    ).eval()
    cached = source()
    with torch.no_grad():
        source.rho.add_(1.0)
    current = source._compute()
    assert not torch.equal(current, cached)
    owner.initialize(torch.tensor(34.0, dtype=torch.float64))
    field = "nai" if kind == "ion" else "c"
    torch.testing.assert_close(getattr(owner, field), current.detach().expand(2, 3))
    assert getattr(owner, field).grad_fn is None
    assert source._cache is cached
    assert source() is cached
    assert "forward" not in vars(source)
    assert "_initialization_source_cache_free" not in vars(source)
    assert not source.training


@pytest.mark.parametrize("explicit_forward", (False, True))
@pytest.mark.parametrize("fail", (False, True))
def test_nested_cache_bypass_restores_method_identity_even_after_source_failure(
    explicit_forward, fail
):
    child = _ModeSource(torch.tensor(2.0, dtype=torch.float64)).eval()
    if explicit_forward:
        child.forward = MethodType(cacheable.forward, child)
    original_forward = vars(child).get("forward")
    cached = child()
    with torch.no_grad():
        child.rho.add_(1.0)
    source = _NestedSource(child, fail=fail)
    if fail:
        with pytest.raises(RuntimeError, match="source failure"):
            _resolve_initial_source(source)
    else:
        actual = _resolve_initial_source(source)
        torch.testing.assert_close(actual, 3 * child._compute())
        torch.testing.assert_close(
            torch.autograd.grad(actual, child.rho)[0], 6 * child.rho
        )
    assert vars(child).get("forward") is original_forward
    assert child._cache is cached
    assert child() is cached
    assert not child.training
    assert source.source is source.alias
    assert "_initialization_source_cache_free" not in vars(source)


def test_native_source_resolution_preserves_custom_forward_and_module_hooks():
    source = PositiveParam(torch.tensor(1.5, dtype=torch.float64)).eval()
    cached = source()
    with torch.no_grad():
        source.rho.add_(1.0)
    calls = []
    handle = source.register_forward_hook(lambda *_: calls.append("forward"))
    try:
        actual = _resolve_initial_source(source)
    finally:
        handle.remove()
    torch.testing.assert_close(actual, source._compute())
    assert calls == ["forward"]
    assert source._cache is cached

    custom = _CustomCachedForward(torch.tensor(1.5, dtype=torch.float64)).eval()
    expected = custom()
    cached = custom._cache
    with torch.no_grad():
        custom.rho.add_(1.0)
    torch.testing.assert_close(_resolve_initial_source(custom), expected)
    assert custom._cache is cached
    assert "forward" not in vars(custom)


def test_ion_source_order_keeps_discarded_reversal_call_and_nernst_gradients():
    dtype = torch.float64
    sources = {
        name: PositiveParam(torch.tensor(value, dtype=dtype), requires_grad=True)
        for name, value in (("e", 11.0), ("i", 3.0), ("o", 7.0))
    }
    with (
        concentrations(nai0=sources["i"], nao0=sources["o"]),
        equilibria(ena=sources["e"]),
    ):
        owner = Ion("na", (2, 3), einit=1, eadvance=1).to(dtype=dtype).train()
    calls = []
    handles = [
        source.register_forward_hook(lambda *_, name=name: calls.append(name))
        for name, source in sources.items()
    ]
    celsius = torch.tensor(34.0, dtype=dtype, requires_grad=True)
    try:
        owner.initialize(celsius)
    finally:
        for handle in handles:
            handle.remove()
    assert calls == ["e", "i", "o"]
    gradients = torch.autograd.grad(
        owner.ena.sum(),
        tuple(source.rho for source in sources.values()) + (celsius,),
        allow_unused=True,
    )
    assert gradients[0] is None
    coefficient = 6 * owner.rzf * (273.15 + celsius.detach())
    torch.testing.assert_close(
        gradients[1],
        -coefficient * torch.sigmoid(sources["i"].rho) / sources["i"]._compute(),
    )
    torch.testing.assert_close(
        gradients[2],
        coefficient * torch.sigmoid(sources["o"].rho) / sources["o"]._compute(),
    )
    torch.testing.assert_close(
        gradients[3],
        6 * owner.rzf * torch.log(sources["o"]._compute() / sources["i"]._compute()),
    )
    assert owner.ina.eq(0).all()


def test_material_source_order_precedes_field_commit_and_minimum_guards():
    sources = {
        name: _ModeSource(torch.tensor(value, dtype=torch.float64))
        for name, value in (("z", 1.0), ("a", 2.0))
    }
    owner = Material("pool", (2, 3), fields=sources, min_values={"z": 10.0}).train()
    owner.z = torch.full_like(owner.z, -4.0)
    owner.a = torch.full_like(owner.a, -5.0)
    calls = []

    def observe(name):
        def hook(*_):
            calls.append(name)
            assert owner.z.eq(-4.0).all()
            assert owner.a.eq(-5.0).all()

        return hook

    handles = [
        source.register_forward_hook(observe(name)) for name, source in sources.items()
    ]
    try:
        owner.initialize()
    finally:
        for handle in handles:
            handle.remove()
    assert calls == ["z", "a"]
    torch.testing.assert_close(owner.z, torch.full_like(owner.z, 10.0))
    torch.testing.assert_close(owner.a, torch.full_like(owner.a, 6.0))


@pytest.mark.parametrize("kind", ("ion", "material"))
def test_private_source_normalization_preserves_modes_and_allows_fullgraph_transforms(
    kind,
):
    source = PositiveParam(torch.tensor(1.5, dtype=torch.float64), requires_grad=True)
    owner = (
        _ion(source)
        if kind == "ion"
        else Material("pool", (2, 3), fields={"c": source})
    ).eval()
    celsius = torch.tensor(34.0, dtype=torch.float64)
    owner.initialize(celsius)
    cache = source()
    private = copy.deepcopy(owner)
    private_source = private.i_init if kind == "ion" else private.initial_source("c")
    _configure_initial_source_execution(private_source)
    assert not private_source.training
    assert source._cache is cache
    assert "forward" not in vars(source)
    module = _FieldValues(private)
    parameters = dict(module.named_parameters())
    parameter_name = next(iter(parameters))
    rho = parameters[parameter_name]

    def evaluate(raw):
        values = torch.func.functional_call(module, {parameter_name: raw}, (celsius,))
        return values[1] if kind == "ion" else values[0]

    expected = evaluate(rho)
    compiled = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(rho), expected)
    torch.testing.assert_close(
        torch.func.jacfwd(evaluate)(rho), torch.func.jacrev(evaluate)(rho)
    )
    lanes = rho.detach() + rho.new_tensor((-0.2, 0.0, 0.3))
    torch.testing.assert_close(
        torch.vmap(evaluate)(lanes), torch.stack([evaluate(raw) for raw in lanes])
    )
    assert torch.vmap(evaluate)(rho.new_empty((0,))).shape == (0, 2, 3)
    assert private_source._cache is None
    assert not private_source.training


def test_packed_initial_source_children_use_cache_free_fullgraph_path():
    packed = _PackedMaterialInitial(
        (
            PositiveParam(torch.tensor(1.5, dtype=torch.float64), requires_grad=True),
            _NestedSource(
                PositiveParam(
                    torch.tensor(2.5, dtype=torch.float64), requires_grad=True
                )
            ),
        ),
        ((1, 2), (1, 3)),
    )
    owner = Material("pool", (1, 5), fields={"c": packed}).eval()
    owner.initialize()
    private = copy.deepcopy(owner)
    source = private.initial_source("c")
    _configure_initial_source_execution(source)
    assert all(
        vars(child)["_initialization_source_cache_free"] is True
        for child in source.modules()
    )
    assert all(
        "_initialization_source_cache_free" not in vars(child)
        for child in packed.modules()
    )
    module = _FieldValues(private)
    celsius = torch.tensor(34.0, dtype=torch.float64)
    expected = module(celsius)
    actual = torch.compile(module, backend="aot_eager", fullgraph=True)(celsius)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        actual[0], torch.tensor([[1.5, 1.5, 7.5, 7.5, 7.5]], dtype=torch.float64)
    )
    assert all(
        child._cache is None
        for child in source.modules()
        if isinstance(child, cacheable)
    )


def test_private_source_normalization_never_calls_authored_metadata_traversal():
    child = PositiveParam(torch.tensor(1.5, dtype=torch.float64)).eval()
    source = _MetadataSource(child)
    _configure_initial_source_execution(source)
    assert vars(source)["_initialization_source_cache_free"] is True
    assert vars(child)["_initialization_source_cache_free"] is True
    torch.testing.assert_close(
        _resolve_initial_source(source), torch.tensor(4.5, dtype=torch.float64)
    )
    assert not child.training
    assert child._cache is None


def test_private_source_normalization_never_calls_reserved_marker_descriptors():
    child = PositiveParam(torch.tensor(1.5, dtype=torch.float64)).eval()
    source = _ReservedMarkerSource(child)
    _configure_initial_source_execution(source)
    assert vars(source)["_initialization_source_cache_free"] is True
    assert vars(child)["_initialization_source_cache_free"] is True
    assert not child.training
    assert child._cache is None
