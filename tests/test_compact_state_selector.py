"""Regression tests for compact selectors on authored State builders."""

import copy

import pytest
import torch

import dendra as dn
from dendra.models.mechanisms import Mechanism, State, SupportKind, SupportSpec

_CORE = (3, 5)
_REORDERED_SUBSET = [13, 11, 3, 1]
_SHARED = [0, 4, 5, 9, 10, 14]
_IRREGULAR = [13, 11, 2, 1]


class _CompactState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = -x")
    State.DERIVED_BUFFER("workspace")

    def derive_buffers(self):
        if isinstance(self.key, SupportSpec):
            indices = self.key.materialized_flat_indices(device=self.diam.device)
        else:
            indices = torch.as_tensor(
                self.key, dtype=torch.long, device=self.diam.device
            )
        return {"workspace": self.diam * (indices.to(self.diam.dtype) + 1)}

    def initial_values(self, v, values):
        return {"x": self.workspace}


class _LegacyState(State):
    supports_compact_selector = False

    State.STATE("y")
    State.DERIVATIVE("y' = -y")


class _ListState(_CompactState):
    supports_compact_selector = False


class _InheritedState(_ListState):
    pass


class _MutatingState(_CompactState):
    def derive_buffers(self):
        # Simulate authored code attempting to bypass the frozen dataclass.
        object.__setattr__(self.key, "column_indices", (0, 4))
        return super().derive_buffers()


class _CompactMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_CompactState, _LegacyState)


class _InheritedMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_InheritedState)


class _MutatingMechanism(Mechanism):
    Mechanism.STATE_BUNDLE(_MutatingState)


def _make(key, mechanism_class=_CompactMechanism, *, compiled=True):
    field = torch.ones(_CORE, dtype=torch.float64, requires_grad=True)
    local_shape = (len(key),)
    return (
        mechanism_class(
            "probe",
            torch.tensor(37.0, dtype=field.dtype),
            field,
            local_shape,
            local_shape,
            key=key,
            support_spec=SupportSpec.from_compiled(
                core_shape=_CORE,
                key=key,
                is_composable=False,
                local_shape=local_shape,
            )
            if compiled
            else None,
        ),
        field,
    )


def _initialize(mechanism, full_field):
    mechanism._init_buffers_s(mechanism.get(torch.zeros_like(full_field)))


@pytest.mark.parametrize("flat_key", (_REORDERED_SUBSET, _SHARED))
def test_state_builder_uses_ordered_compact_selector_by_default_with_gradients(
    flat_key,
):
    key = flat_key.copy()
    mechanism, field = _make(key)
    state = mechanism.DE._CompactState

    assert state.key is mechanism.support_spec
    assert state.key.kind is SupportKind.SHARED_COLUMNS
    assert state.key.compact_index_count <= len(key)
    assert mechanism.DE._LegacyState.key is key
    assert mechanism.key.tolist() == key

    _initialize(mechanism, field)
    expected = torch.tensor(key, dtype=field.dtype) + 1
    torch.testing.assert_close(state.workspace, expected, rtol=0, atol=0)
    (gradient,) = torch.autograd.grad(state.workspace.sum(), field)
    expected_gradient = torch.zeros_like(field)
    expected_gradient.reshape(-1)[key] = expected
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)


def test_legacy_selector_opt_out_is_inherited_by_custom_state_subclass():
    key = _REORDERED_SUBSET.copy()
    mechanism, field = _make(key, _InheritedMechanism)
    state = mechanism.DE._InheritedState
    assert state.key is key
    _initialize(mechanism, field)
    torch.testing.assert_close(
        state.workspace, torch.tensor(key, dtype=field.dtype) + 1
    )


def test_irregular_and_explicit_tensor_keys_keep_legacy_state_key():
    irregular = _IRREGULAR.copy()
    mechanism, field = _make(irregular)
    assert mechanism.support_spec.kind is SupportKind.PACKED_FLAT
    assert mechanism.DE._CompactState.key is irregular
    _initialize(mechanism, field)
    torch.testing.assert_close(
        mechanism.DE._CompactState.workspace,
        torch.tensor(irregular, dtype=field.dtype) + 1,
    )

    tensor = torch.tensor(_SHARED)
    mechanism, field = _make(tensor)
    assert mechanism.support_spec.kind is SupportKind.SHARED_COLUMNS
    assert mechanism.DE._CompactState.key is tensor
    _initialize(mechanism, field)
    torch.testing.assert_close(
        mechanism.DE._CompactState.workspace, tensor.to(field.dtype) + 1
    )


def test_compact_key_builder_mutation_is_rejected_and_isolated():
    mechanism, field = _make(_REORDERED_SUBSET.copy(), _MutatingMechanism)
    state = mechanism.DE._MutatingState
    original = copy.deepcopy(state.key)

    with pytest.raises(RuntimeError, match="must not mutate Python instance state"):
        _initialize(mechanism, field)

    assert state.key is mechanism.support_spec
    assert state.key == original
    assert state.key.materialized_flat_indices().tolist() == _REORDERED_SUBSET
    assert torch.count_nonzero(state.workspace) == 0


def test_standalone_state_dict_load_refreshes_or_decompacts_opted_key():
    target, field = _make(_REORDERED_SUBSET.copy())
    state = target.DE._CompactState
    other_shared = [2, 4, 7, 9]
    for key, kind in (
        (other_shared, SupportKind.SHARED_COLUMNS),
        (_IRREGULAR, SupportKind.PACKED_FLAT),
        (_REORDERED_SUBSET, SupportKind.SHARED_COLUMNS),
    ):
        source, _ = _make(key.copy())
        target.load_state_dict(source.state_dict())

        assert target.support_spec.kind is kind
        if kind is SupportKind.SHARED_COLUMNS:
            assert state.key is target.support_spec
            assert state.key.materialized_flat_indices().tolist() == key
        else:
            assert type(state.key) is list
            assert state.key == key
        _initialize(target, field)
        torch.testing.assert_close(
            state.workspace, torch.tensor(key, dtype=field.dtype) + 1
        )


def test_direct_mechanism_without_compiled_support_retains_original_key():
    key = _SHARED.copy()
    mechanism, field = _make(key, compiled=False)
    assert mechanism.DE._CompactState.key is key
    assert mechanism.support_spec.kind is SupportKind.SHARED_COLUMNS
    _initialize(mechanism, field)


def test_compiled_population_compact_selector_runs_native_and_functional():
    population = dn.Population(N=3, C=5, dtype=torch.float64).eval()
    population[:, torch.tensor([1, 3])].insert(_CompactMechanism)
    population.initialize()
    mechanism = population.mech._CompactMechanism
    assert mechanism.DE._CompactState.key is mechanism.support_spec

    program = dn.compile(population, dt=0.01, steps=2, backend="eager")
    final, _ = dn.func.run(program)
    population.initialize().run(steps=2, dt=0.01, progressbar=False)

    torch.testing.assert_close(final["integrator"]["v"], population.v)
    torch.testing.assert_close(final["mechanisms"][mechanism.name]["x"], mechanism.x)
    # Verify that a nonzero selector-dependent initial value reached the rollout.
    assert torch.count_nonzero(mechanism.x) == len(_SHARED)
