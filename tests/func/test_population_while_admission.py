"""Population admission reuses owned tapes and retains raw failure contracts."""

from __future__ import annotations

import gc
import weakref
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode

from dendra._bootstrap import torch_compiler_warning_context
from dendra.func import RuntimeDraws, RuntimeNoiseDraw, StimulationDraw
from dendra.func import _population_while as population_while
from dendra.func._inputs import _RuntimeInputs
from dendra.helpers import MEMBRANE_ASSEMBLY
from tests.func.test_while_block_ownership import (
    _assert_owned,
    _case,
    _clone,
    _close,
    _eager,
    _tensor_leaves,
)

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _warnings():
    with torch_compiler_warning_context():
        yield


def _executor(case, *, stepwise=True):
    return population_while._PopulationWhileExecutor(
        case.functional,
        steps=4,
        plans=case.plans,
        compile_options={"backend": "eager", "fullgraph": True},
        stepwise_stimulation=stepwise,
    )


def _arguments(case, ve=None, intra=None, *, noise=None, stimulation=None, draws=None):
    return (
        case.parameters,
        case.prepared,
        case.state,
        case.ve if ve is None else ve,
        case.intra if intra is None else intra,
        case.carries,
        noise,
        stimulation,
        draws,
    )


def _admit(executor, arguments):
    kernel, reason = executor.kernel_for(*arguments)
    assert kernel is not None and reason is None, reason
    return kernel


def _signature(arguments):
    return population_while._schema(arguments), MEMBRANE_ASSEMBLY.value


def _offset(value, offset):
    storage = torch.empty(value.numel() + offset, dtype=value.dtype)
    view = storage[offset:].reshape(value.shape)
    view.copy_(value)
    return view


def _tapes(case, kind, offset=1):
    if kind == "inference":
        with torch.inference_mode():
            return case.ve.clone(), case.intra.clone()
    if kind == "offset":
        return _offset(case.ve, offset), _offset(case.intra, offset + 1)
    if kind == "strided":
        return (
            case.ve.transpose(1, 2).contiguous().transpose(1, 2),
            case.intra.transpose(1, 2).contiguous().transpose(1, 2),
        )
    if kind == "same":
        return case.ve.clone(), case.ve.clone()
    if kind == "aliased":
        value = case.ve.clone()
        return value, value.view_as(value)
    if kind == "identical":
        value = case.ve.clone()
        return value, value
    return case.ve.clone(), case.intra.clone()


def test_population_success_cache_reuses_tape_layouts_and_reads_live_values():
    case = _case(4)
    executor = _executor(case)
    first_kernel = None
    with torch.no_grad():
        for index, kind in enumerate(
            (
                "fresh",
                *(["offset"] * 12),
                "strided",
                "inference",
                "same",
                "aliased",
                "identical",
            )
        ):
            ve, intra = _tapes(case, kind, index + 1)
            arguments = _arguments(case, ve, intra)
            frozen = _clone(arguments)
            versioned = [
                value for value in _tensor_leaves(arguments) if not value.is_inference()
            ]
            versions = [value._version for value in versioned]
            kernel = _admit(executor, arguments)
            if first_kernel is None:
                first_kernel = kernel
            assert kernel is first_kernel, kind
            assert len(executor._kernels) == 1
            assert len(executor._stepwise_engine._kernels) == 1
            if kind in ("fresh", "inference", "identical"):
                expected_case = SimpleNamespace(
                    **(vars(case) | {"ve": ve, "intra": intra})
                )
                actual = kernel(*arguments)
                _close(actual, _eager(expected_case, 4))
                _assert_owned(actual, arguments)
            _close(arguments, frozen, exact=True)
            assert [value._version for value in versioned] == versions
        ve, intra = _tapes(case, "offset", 31)
        arguments = _arguments(case, ve, intra)
        before = first_kernel(*arguments)
        ve.add_(0.125)
        intra.sub_(0.25)
        frozen = _clone(arguments)
        after = first_kernel(*arguments)
        expected_case = SimpleNamespace(**(vars(case) | {"ve": ve, "intra": intra}))
        _close(after, _eager(expected_case, 4))
        assert not torch.equal(
            before[0]["integrator"]["v"], after[0]["integrator"]["v"]
        )
        _close(arguments, frozen, exact=True)
    assert len(executor._kernels) == len(executor._stepwise_engine._kernels) == 1
    assert executor._capture_errors == {}


def test_successful_population_admission_retains_no_tape_or_autograd_examples():
    case = _case(4)
    executor = _executor(case)
    references = []

    def admit():
        with torch.enable_grad():
            root = torch.arange(
                case.ve.numel(), dtype=torch.float64, requires_grad=True
            )
            ve = (root * 0.01).reshape(case.ve.shape)
            intra = ve.view_as(ve)
        assert ve.grad_fn is not None
        references.extend(weakref.ref(value) for value in (root, ve, intra))
        return _admit(executor, _arguments(case, ve, intra))

    with torch.no_grad():
        kernel = admit()
    gc.collect()
    assert all(reference() is None for reference in references)
    with torch.no_grad():
        live_arguments = _arguments(case, *_tapes(case, "offset", 7))
        assert _admit(executor, live_arguments) is kernel
        _close(kernel(*live_arguments), _eager(case, 4))
    assert len(executor._kernels) == len(executor._stepwise_engine._kernels) == 1


def test_invalid_tape_axes_keep_raw_failures_even_after_a_normalized_success():
    case = _case(4)
    executor = _executor(case)
    with torch.no_grad():
        success = _admit(executor, _arguments(case))
        for value in (
            torch.zeros(3, 2, 6, dtype=torch.float64),
            torch.zeros(5, 2, 6, dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        ):
            arguments = _arguments(case, value)
            raw = _signature(arguments)
            assert executor._success_signature(arguments, raw) is raw
            kernel, reason = executor.kernel_for(*arguments)
            assert kernel is None and "matching leading time axis" in reason
            again, repeated = executor.kernel_for(*arguments)
            assert again is None and repeated is reason
        assert _admit(executor, _arguments(case, *_tapes(case, "offset", 9))) is success
    assert len(executor._kernels) == 1 and len(executor._capture_errors) == 3


def test_failed_raw_admission_is_not_hidden_by_a_later_equivalent_success():
    case = _case(4)

    class FailingFunctional(type(case.functional)):
        fail = True
        attempts = 0

        def _step_values(self, parameters, prepared, state, inputs):
            self.attempts += 1
            if self.fail:
                raise RuntimeError("the authored transition rejected this capture")
            return super()._step_values(parameters, prepared, state, inputs)

    case.functional = FailingFunctional()
    executor = _executor(case)
    failed = _arguments(case, *_tapes(case, "offset", 1))
    accepted = _arguments(case, *_tapes(case, "offset", 9))
    with torch.no_grad():
        kernel, reason = executor.kernel_for(*failed)
        assert kernel is None and "authored transition rejected" in reason
        case.functional.fail = False
        success = _admit(executor, accepted)
        attempts = case.functional.attempts
        retry, repeated = executor.kernel_for(*failed)
        assert retry is None and repeated is reason
        assert case.functional.attempts == attempts
        assert _admit(executor, accepted) is success
        _close(success(*accepted), _eager(case, 4))
    assert len(executor._kernels) == len(executor._capture_errors) == 1


@pytest.mark.parametrize("custom", ("capture", "tracer", "compiler"))
@pytest.mark.parametrize("warmed", (False, True), ids=("cold", "warm"))
def test_custom_population_capture_adapters_keep_their_raw_tape_keys(
    monkeypatch, custom, warmed
):
    case = _case(4)
    executor = _executor(case)
    calls = []
    name = {"capture": "_capture", "tracer": "make_fx", "compiler": "_compile_while"}[
        custom
    ]
    owner = executor if custom == "capture" else population_while
    original = getattr(owner, name)
    first = _arguments(case, *_tapes(case, "offset", 1))
    second = _arguments(case, *_tapes(case, "offset", 9))
    with torch.no_grad():
        previous = _admit(executor, first) if warmed else None

    def delegated(*args, **kwargs):
        calls.append(custom)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, delegated)
    with torch.no_grad():
        raw = _signature(first)
        assert executor._success_signature(first, raw) is raw
        left, right = _admit(executor, first), _admit(executor, second)
        assert left is not right
        assert left is not previous
        assert _admit(executor, first) is left
    assert bool(calls) == (not warmed or custom == "capture")
    assert len(executor._kernels) == 2 + int(warmed)
    assert len(executor._stepwise_engine._kernels) == 1


@pytest.mark.parametrize(
    "position", (0, 1, 2, 5), ids=("parameters", "prepared", "state", "carry")
)
def test_successful_tape_reuse_preserves_other_operand_view_signatures(position):
    case = _case(4)
    executor = _executor(case)
    first = _arguments(case)
    value = _tensor_leaves(first[position])[0]
    replacement = _offset(value, 3)
    second = list(first)
    second[position] = pytree.tree_map_only(
        torch.Tensor,
        lambda current: replacement if current is value else current,
        first[position],
    )
    with torch.no_grad():
        left = _admit(executor, first)
        right = _admit(executor, tuple(second))
        assert right is not left
        assert _admit(executor, first) is left
    assert len(executor._kernels) == 2


def test_named_runtime_draw_packets_normalize_views_but_preserve_none_positions():
    case = _case(4)
    for family in ("noise", "stimulation", "draws"):
        executor = _executor(case)

        def arguments(kind, *, optional=False):
            value = _tapes(case, kind, 7)[0]
            rng = torch.zeros(4, 16, dtype=torch.uint8)
            fields = {"value": value, "unused": value if optional else None}
            noise = RuntimeNoiseDraw(fields, {"shared": {"rng": rng}})
            stimulation = StimulationDraw(fields, {"shared": rng})
            draws = RuntimeDraws(
                {"noise": fields, "stimulation": None}, {"shared": rng}
            )
            packet = {"noise": noise, "stimulation": stimulation, "draws": draws}[
                family
            ]
            return _arguments(case, **{family: packet})

        with torch.no_grad():
            first = _admit(executor, arguments("fresh"))
            assert _admit(executor, arguments("offset")) is first
            assert _admit(executor, arguments("inference")) is first
            changed = _admit(executor, arguments("offset", optional=True))
            assert changed is not first
        assert len(executor._kernels) == len(executor._stepwise_engine._kernels) == 2


def test_observable_tensor_types_and_torch_modes_keep_raw_population_keys():
    class Subclass(torch.Tensor):
        pass

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            return function(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            return function(*args, **(kwargs or {}))

    case = _case(4)
    executor = _executor(case)
    for context, ve in (
        (nullcontext(), case.ve.as_subclass(Subclass)),
        (FunctionMode(), case.ve),
        (DispatchMode(), case.ve),
    ):
        arguments = _arguments(case, ve)
        raw = _signature(arguments)
        with context:
            assert executor._success_signature(arguments, raw) is raw


def test_observable_tensor_selection_descriptors_keep_raw_population_keys(monkeypatch):
    case = _case(4)
    executor = _executor(case)
    arguments = _arguments(case)
    raw = _signature(arguments)
    for name in ("index_select", "squeeze", "stride"):
        original = getattr(torch.Tensor, name)

        def delegated(self, *args, _original=original, **kwargs):
            return _original(self, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, name, delegated)
            assert executor._success_signature(arguments, raw) is raw
        with monkeypatch.context() as patch:
            patch.setattr(case.ve, name, MethodType(delegated, case.ve))
            assert executor._success_signature(arguments, raw) is raw


def test_population_admission_does_not_add_custom_key_or_metaclass_comparisons():
    comparisons = []

    class Key:
        def __hash__(self):
            return 0

        def __eq__(self, other):
            comparisons.append((self, other))
            return self is other

    class Meta(type(torch.Tensor)):
        def __eq__(cls, other):
            comparisons.append((cls, other))
            return cls is other

        __hash__ = type.__hash__

    class Subclass(torch.Tensor, metaclass=Meta):
        pass

    case = _case(4)
    executor = _executor(case)
    cases = (
        _arguments(case, noise=RuntimeNoiseDraw({Key(): case.ve}, {})),
        _arguments(case, case.ve.as_subclass(Subclass)),
    )
    for arguments in cases:
        raw = _signature(arguments)
        comparisons.clear()
        assert executor._success_signature(arguments, raw) is raw
        assert comparisons == []


def test_whole_tape_admission_stays_raw_until_runtime_stimulation_requires_steps():
    case = _case(4)

    class WholeTapeFunctional(type(case.functional)):
        intra = SimpleNamespace(enabled=False)
        extra = SimpleNamespace(enabled=False)
        transition_calls = 0

        def _transition_rollout_values(
            self,
            parameters,
            prepared,
            state,
            ve,
            intra,
            steps,
            noise=None,
            stimulation=None,
            draws=None,
        ):
            assert steps == 1
            self.transition_calls += 1
            return self._step_values(
                parameters,
                prepared,
                state,
                _RuntimeInputs(ve[0], intra[0], noise, stimulation, draws),
            )

    case.functional = WholeTapeFunctional()
    whole = _executor(case, stepwise=False)
    with torch.no_grad():
        first = _arguments(case, *_tapes(case, "offset", 1))
        second = _arguments(case, *_tapes(case, "offset", 9))
        left, right = _admit(whole, first), _admit(whole, second)
        assert left is not right and len(whole._kernels) == 2
        assert whole._stepwise_engine is None
        assert case.functional.transition_calls > 0
        before = case.functional.transition_calls
        runtime = _executor(case, stepwise=False)
        packet = StimulationDraw({}, {})
        first = _arguments(case, *_tapes(case, "offset", 1), stimulation=packet)
        second = _arguments(case, *_tapes(case, "offset", 9), stimulation=packet)
        left = _admit(runtime, first)
        assert _admit(runtime, second) is left
        assert case.functional.transition_calls == before
        assert len(runtime._kernels) == len(runtime._stepwise_engine._kernels) == 1
