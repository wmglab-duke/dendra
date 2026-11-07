"""Waveform occurrences retain tensor aliases without leaking explicit operands."""

import copy
import gc
import weakref

import pytest
import torch
from torch.func import functional_call

import dendra as dn
from dendra.func._poisson_execution import isolate_waveform_aliases
from dendra.func._stimuli import _FunctionalWaveformBank
from dendra.models.stim.waveform import Waveform
from dendra.models.stim.waveform._poisson import (
    POISSON_INVOCATION_POSITION,
    POISSON_INVOCATION_STACK,
    configure_poisson_invocations,
    evaluate_poisson_schedule,
)

pytestmark = pytest.mark.cpu


class _SlottedWaveform(Waveform):
    __slots__ = ("offset",)
    FUNCTIONAL_PURE = True
    Waveform.PARAMETER(amp=0.3)

    def __init__(self):
        super().__init__()
        self.offset = 0.2
        self.evaluate = self._evaluate

    def __copy__(self):
        raise AssertionError(
            "Execution occurrence isolation must not invoke copy hooks."
        )

    def __deepcopy__(self, memo):
        del memo
        raise AssertionError(
            "Execution occurrence isolation must not invoke copy hooks."
        )

    def _evaluate(self, times):
        return self.amp * (times + self.offset)

    def fn(self, times):
        return self.evaluate(times)


def test_strict_functional_call_restores_distinct_poisson_occurrences_and_aliases():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        source = dn.mono_rect(amp=0.4, pw=0.21).poisson(
            interval=0.3, n=3, generator=torch.Generator().manual_seed(83)
        )
        expression = copy.deepcopy(source + source)
        original_tensors = {
            **dict(expression.named_parameters(remove_duplicate=False)),
            **dict(expression.named_buffers(remove_duplicate=False)),
        }
        source_schedule = source._spike_times.clone()
        source_rng = source.generator.get_state().clone()
        assert isolate_waveform_aliases(expression) is expression
        first, second = expression.waveforms
        assert first is not second
        assert first.waveform is not second.waveform
        assert first.waveform.amp is second.waveform.amp
        retained = {
            **dict(expression.named_parameters(remove_duplicate=False)),
            **dict(expression.named_buffers(remove_duplicate=False)),
        }
        assert set(retained) == set(original_tensors)
        assert all(retained[name] is value for name, value in original_tensors.items())
        for waveform in (first, second):
            configure_poisson_invocations(waveform)
        template = {
            **dict(expression.named_parameters(remove_duplicate=False)),
            **dict(expression.named_buffers(remove_duplicate=False)),
        }
        amplitude = first.waveform.amp.detach().clone().requires_grad_()
        schedules = torch.tensor([[0.0, 0.3, torch.inf], [0.12, 0.37, 0.62]])
        times = torch.linspace(0.0, 0.9, 19)
        expected = sum(
            evaluate_poisson_schedule(source.waveform.fn, times, schedule)
            for schedule in schedules
        )
        expected_gradient = torch.autograd.grad(
            expected.square().sum(), source.waveform.amp
        )[0]
        for _ in range(2):
            mapping = dict(template)
            for index in range(2):
                prefix = f"waveforms.{index}."
                mapping[f"{prefix}waveform.amp"] = amplitude
                mapping[f"{prefix}{POISSON_INVOCATION_STACK}"] = schedules[
                    index : index + 1
                ]
                mapping[f"{prefix}{POISSON_INVOCATION_POSITION}"] = torch.zeros(
                    (1,), dtype=torch.int64
                )
            actual = functional_call(
                expression, mapping, (times,), tie_weights=False, strict=True
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                torch.autograd.grad(actual.square().sum(), amplitude)[0],
                expected_gradient,
                rtol=0,
                atol=0,
            )
            restored = {
                **dict(expression.named_parameters(remove_duplicate=False)),
                **dict(expression.named_buffers(remove_duplicate=False)),
            }
            assert all(restored[name] is value for name, value in template.items())
            assert all(
                getattr(waveform, POISSON_INVOCATION_STACK).shape[0] == 0
                and torch.count_nonzero(getattr(waveform, POISSON_INVOCATION_POSITION))
                == 0
                for waveform in (first, second)
            )
        assert torch.equal(source._spike_times, source_schedule)
        assert torch.equal(source.generator.get_state(), source_rng)
        assert not hasattr(source, POISSON_INVOCATION_POSITION)


def test_isolation_preserves_slotted_custom_waveforms_without_user_copy_hooks():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        source = _SlottedWaveform()
        owners = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
        owners[0].waveform = source
        owners[1].waveform = source
        assert isolate_waveform_aliases(owners) is owners
        first, second = owners[0].waveform, owners[1].waveform
        assert first is not second and first is not source and second is not source
        assert first.amp is second.amp is source.amp
        assert first.offset == second.offset == source.offset
        assert first.evaluate.__self__ is first
        assert second.evaluate.__self__ is second
        first.register_buffer("scratch", torch.zeros(()), persistent=False)
        assert "scratch" not in source._buffers
        assert "scratch" not in second._buffers
        assert "scratch" not in source._non_persistent_buffers_set
        assert "scratch" not in second._non_persistent_buffers_set
        handle = second.register_forward_hook(lambda module, args, output: output)
        handle.remove()
        value = torch.tensor(0.7, requires_grad=True)
        times = torch.linspace(0.0, 1.0, 7)
        actual = functional_call(first, {"amp": value}, (times,))
        expected = value * (times + source.offset)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), value)[0],
            (times + source.offset).sum(),
            rtol=0,
            atol=0,
        )
        assert first.amp is second.amp is source.amp


def test_waveform_registration_cycles_fail_before_execution():
    waveform = _SlottedWaveform()
    waveform.child = waveform
    with pytest.raises(dn.func.FunctionalizationError, match="cannot contain cycles"):
        isolate_waveform_aliases(waveform)


class _DeterministicLeaf(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.register_parameter("gain", torch.nn.Parameter(torch.tensor(0.3)))
        self.register_buffer("offset", torch.tensor(0.2))

    def fn(self, times):
        return self.gain * (times + self.offset)


class _DoubleGain(torch.nn.Module):
    FUNCTIONAL_PURE = True

    def __init__(self):
        super().__init__()
        self.register_buffer("factor", torch.tensor(2.0))

    def forward(self, gain):
        return gain * self.factor


def _registered_tensors(module):
    return {
        **{
            f"parameter:{name}": value
            for name, value in module.named_parameters(remove_duplicate=False)
        },
        **{
            f"buffer:{name}": value
            for name, value in module.named_buffers(remove_duplicate=False)
        },
    }


@pytest.mark.parametrize("operation", ("sum", "product"))
@pytest.mark.parametrize("parametrized", (False, True))
def test_deterministic_aliases_restore_every_registration_without_retaining_operands(
    operation, parametrized
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        source = _DeterministicLeaf()
        if parametrized:
            torch.nn.utils.parametrize.register_parametrization(
                source, "gain", _DoubleGain()
            )
        expression = source + source if operation == "sum" else source * source
        source_tensors = _registered_tensors(expression)
        source_values = {
            name: value.detach().clone() for name, value in source_tensors.items()
        }
        bank = _FunctionalWaveformBank(
            (expression,),
            namespace="test",
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        template = _registered_tensors(bank._bank)
        parameters, constants = bank.extract()
        assert len(parameters) == 1  # Authored parameter sharing remains canonical.
        parameter_name = next(iter(parameters))
        offset_name = next(name for name in constants if name.endswith(".offset"))
        times = torch.linspace(0.0, 0.9, 7)
        for initial in (0.6, 0.7):
            base = torch.tensor(initial, requires_grad=True)
            replacement = base.square()
            offset = torch.tensor(0.45)
            replacement_ref, offset_ref = weakref.ref(replacement), weakref.ref(offset)
            bound_parameters = {parameter_name: replacement}
            bound_constants = {**constants, offset_name: offset}
            actual = bank.evaluate(bound_parameters, bound_constants, times)[0]
            gain = replacement * 2.0 if parametrized else replacement
            first, second = gain * (times + offset), gain * (times + offset)
            expected = first + second if operation == "sum" else first * second
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual_gradient = torch.autograd.grad(actual.square().sum(), replacement)[0]
            expected_gradient = torch.autograd.grad(
                expected.square().sum(), replacement
            )[0]
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=0, atol=0
            )
            restored = _registered_tensors(bank._bank)
            assert all(restored[name] is value for name, value in template.items())
            del replacement, offset, actual, expected, first, second, gain
            del bound_parameters, bound_constants
            gc.collect()
            assert replacement_ref() is None
            assert offset_ref() is None
        current = _registered_tensors(expression)
        assert all(current[name] is value for name, value in source_tensors.items())
        for name, value in current.items():
            torch.testing.assert_close(value, source_values[name], rtol=0, atol=0)


def test_deterministic_aliases_do_not_retain_functorch_wrapped_parameters():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        source = _DeterministicLeaf()
        bank = _FunctionalWaveformBank(
            (source + source,),
            namespace="test",
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        template = _registered_tensors(bank._bank)
        parameters, constants = bank.extract()
        parameter_name = next(iter(parameters))
        times = torch.linspace(0.0, 0.9, 7)

        def response(gain):
            return bank.evaluate({parameter_name: gain}, constants, times)[0]

        actual = torch.func.jacrev(response)(torch.tensor(0.7))
        expected = 2.0 * (times + source.offset)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        restored = _registered_tensors(bank._bank)
        assert all(restored[name] is value for name, value in template.items())
        assert all(
            not torch._C._functorch.is_functorch_wrapped_tensor(value)
            for value in restored.values()
        )
