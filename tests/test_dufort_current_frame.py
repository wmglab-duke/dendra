"""Call-local and native Dufort current assembly share authored traversal."""

from __future__ import annotations

import copy
from types import MethodType

import pytest
import torch

import dendra as dn  # noqa: F401 - initialize mechanism compilation
from dendra.models.mechanisms import Mechanism, PointProcess
from dendra.models.mechanisms._handler import MechanismHandler
from dendra.models.mechanisms._ions import Ion

SHAPE = (2, 4)


class _AssignedAffineIon(Mechanism):
    Mechanism.RANGE(g=0.25)
    Mechanism.ASSIGNED("offset")
    Mechanism.USEION("na", write=["ina"])
    Mechanism.AFFINE("ina")
    Mechanism.SAVE_CURRENT("ina")

    def assigned_values(self, v, values):
        return {"offset": 0.1 * v.square()}

    def ina(self, v):
        return self.g * v + self.offset

    def ina_with_conductance(self, v):
        return self.ina(v), self.g


class _NonlinearSaved(Mechanism):
    Mechanism.RANGE(g=0.125)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.NUMERICAL("i")
    Mechanism.SAVE_CURRENT("i")

    def i(self, v):
        return self.g * v.square()

    def i_with_conductance(self, v):
        raise AssertionError("Dufort must not differentiate a non-affine current")


class _PointAffine(PointProcess):
    PointProcess.RANGE(g=0.5, bias=-0.75)
    PointProcess.NONSPECIFIC_CURRENT("i")
    PointProcess.AFFINE("i")

    def i(self, v):
        return self.g * v + self.bias

    def i_with_conductance(self, v):
        return self.i(v), self.g


class _MixedDtypeConstant(Mechanism):
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return torch.tensor(2048.0, dtype=torch.float16, device=v.device)

    def i_with_conductance(self, v):
        return self.i(v), torch.tensor(0.25, dtype=torch.float64, device=v.device)


def _mechanism(cls, name, dtype, key=None):
    field = torch.ones(SHAPE, dtype=dtype)
    local_shape = SHAPE if key is None else (key.numel(),)
    return cls(
        name,
        field * 34.0,
        field,
        local_shape,
        local_shape,
        key=key,
        dtype=dtype,
    )


def _handler(dtype=torch.float64, *, empty=False):
    field = torch.ones(SHAPE, dtype=dtype)
    sodium = Ion("na", SHAPE, einit=0, eadvance=0).to(dtype=dtype)
    potassium = Ion("k", SHAPE, einit=0, eadvance=0).to(dtype=dtype)
    mechanisms = {}
    currents = {}
    if not empty:
        mechanisms = {
            "ion": _mechanism(_AssignedAffineIon, "ion", dtype),
            "nonlinear": _mechanism(
                _NonlinearSaved, "nonlinear", dtype, torch.tensor([1, 1, 6])
            ),
            "point": _mechanism(_PointAffine, "point", dtype, torch.tensor([0, 4, 7])),
            "constant": _mechanism(_MixedDtypeConstant, "constant", dtype),
        }
        mechanisms["ion"].register_ion(sodium)
        currents = {
            "ina": {"ion": ["ina"]},
            "i": {"nonlinear": ["i"], "point": ["i"], "constant": ["i"]},
        }
    handler = MechanismHandler(
        field * 34.0,
        field * 1.0e-6,
        mechanisms,
        ions={"na": sodium, "k": potassium},
        currents=currents,
    ).to(dtype=dtype)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros(SHAPE, dtype=dtype))
    return handler


def _voltages(dtype=torch.float64):
    v = torch.linspace(-2.0, 3.0, 8, dtype=dtype).reshape(SHAPE)
    previous = torch.linspace(-4.0, 1.0, 8, dtype=dtype).reshape(SHAPE)
    return v, previous


def _manual_oracle(handler, v, previous):
    # Independent authored-current formulas, with category/destination ordering
    # matching the public current layout and duplicate indices retained.
    sodium = 0.25 * (0.5 * previous) + 0.1 * v.square()
    nonspecific = torch.zeros_like(v).flatten()
    nonlinear_keys = torch.tensor([1, 1, 6])
    nonlinear = 0.125 * v.flatten()[nonlinear_keys].square()
    nonspecific = nonspecific.index_add(0, nonlinear_keys, nonlinear)
    point_keys = torch.tensor([0, 4, 7])
    point = 0.5 * (0.5 * previous.flatten()[point_keys]) - 0.75
    nonspecific = nonspecific.index_add(0, point_keys, point).reshape_as(v)
    nonspecific = nonspecific + v.new_tensor(2048.0)
    nonspecific_g = torch.zeros_like(v).flatten()
    nonspecific_g = nonspecific_g.index_add(
        0, point_keys, v.new_full((3,), 0.5)
    ).reshape_as(v)
    nonspecific_g = nonspecific_g + 0.25
    sodium_g = torch.full_like(v, 0.25)
    return (
        torch.stack((sodium, nonspecific)).sum(0),
        torch.stack((sodium_g, nonspecific_g)).sum(0),
        (torch.zeros_like(v), sodium),
        (torch.zeros_like(v), sodium_g),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dufort_current_frame_preserves_centering_assigned_save_and_ionic_frames(dtype):
    handler = _handler(dtype)
    v, previous = _voltages(dtype)
    actual = handler._evaluate_dufort_current_frame(v, previous)
    expected = _manual_oracle(handler, v, previous)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(handler.mechanisms["ion"].offset, 0.1 * v.square())
    torch.testing.assert_close(handler.mechanisms["ion"].ina_, expected[2][1])
    keys = torch.tensor([1, 1, 6])
    torch.testing.assert_close(
        handler.mechanisms["nonlinear"].i_,
        0.125 * v.flatten()[keys].square(),
    )
    assert all(value.dtype == dtype for value in actual[:2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dufort_current_frame_matches_native_and_does_not_rebind_native_scratch(dtype):
    handler = _handler(dtype)
    native = copy.deepcopy(handler)
    v, previous = _voltages(dtype)
    scratch_before = tuple(
        (id(value), value._version, value.clone())
        for buffers in (handler._buf_i, handler._buf_g)
        for value in buffers
    )
    actual = handler._evaluate_dufort_current_frame(v, previous)
    for value, (identity, version, saved) in zip(
        (value for buffers in (handler._buf_i, handler._buf_g) for value in buffers),
        scratch_before,
        strict=True,
    ):
        assert id(value) == identity and value._version == version
        torch.testing.assert_close(value, saved, rtol=0.0, atol=0.0)
    expected = native.idf(v, previous)
    torch.testing.assert_close(actual[:2], expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual[2], native.capture_ion_current_frame(), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        actual[3], native.capture_ion_conductance_frame(), rtol=0.0, atol=0.0
    )


def test_dufort_current_frame_empty_handler_has_zero_ion_frames():
    handler = _handler(empty=True)
    v, previous = _voltages()
    actual = handler._evaluate_dufort_current_frame(v, previous)
    expected = (torch.zeros_like(v),) * 2
    torch.testing.assert_close(actual[:2], expected)
    assert len(actual[2]) == len(actual[3]) == 2
    for frame in actual[2:]:
        torch.testing.assert_close(frame, expected)
    torch.testing.assert_close(handler.idf(v, previous), expected)


@pytest.mark.parametrize("call_local", [False, True])
def test_dufort_current_frame_evaluates_each_authored_hook_once_in_native_order(
    call_local,
):
    handler = _handler()
    events = []
    for name, method_name in (
        ("ion", "assigned_values"),
        ("ion", "ina_with_conductance"),
        ("ion", "ina"),
        ("nonlinear", "i"),
        ("point", "i_with_g"),
        ("point", "i"),
        ("constant", "i_with_g"),
        ("constant", "i"),
    ):
        mechanism = handler.mechanisms[name]
        original = getattr(mechanism, method_name)

        def tracked(self, v, *args, original=original, label=(name, method_name)):
            del self
            events.append((label, v.detach().clone()))
            return original(v, *args)

        setattr(mechanism, method_name, MethodType(tracked, mechanism))
    v, previous = _voltages()
    if call_local:
        handler._evaluate_dufort_current_frame(v, previous)
    else:
        handler.idf(v, previous)
    assert [name for name, _ in events] == [
        ("ion", "assigned_values"),
        ("ion", "ina_with_conductance"),
        ("ion", "ina"),
        ("nonlinear", "i"),
        ("point", "i_with_g"),
        ("point", "i"),
        ("constant", "i_with_g"),
        ("constant", "i"),
    ]
    torch.testing.assert_close(events[0][1], v)
    torch.testing.assert_close(events[1][1], 0.5 * previous)
    torch.testing.assert_close(events[2][1], 0.5 * previous)
    torch.testing.assert_close(events[3][1], v.flatten()[torch.tensor([1, 1, 6])])


class _Frame(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.handler = _handler()

    def forward(self, v, previous):
        return self.handler._evaluate_dufort_current_frame(v, previous)


@pytest.mark.parametrize("lanes", [0, 3])
@pytest.mark.parametrize("field", ["ion.g", "nonlinear.g", "point.g"])
def test_dufort_current_frame_parameter_vmap_retains_empty_lanes_and_gradients(
    field, lanes
):
    module = _Frame()
    v, previous = _voltages()
    name = f"handler.mechanisms.{field}"
    value = dict(module.named_buffers())[name]
    values = value.expand((lanes,) + value.shape).clone().requires_grad_()

    def evaluate(candidate):
        return torch.func.functional_call(module, {name: candidate}, (v, previous))

    actual = torch.vmap(evaluate)(values)
    assert actual[0].shape == actual[1].shape == (lanes,) + SHAPE
    if lanes:
        references = [evaluate(candidate) for candidate in values]
        for slot in (0, 1):
            torch.testing.assert_close(
                actual[slot],
                torch.stack([reference[slot] for reference in references]),
                rtol=0.0,
                atol=0.0,
            )
    gradient = torch.autograd.grad(actual[0].sum() + actual[1].sum(), values)[0]
    assert gradient.shape == values.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("lanes", [0, 3])
def test_dufort_current_frame_voltage_vmap_retains_previous_voltage_gradients(lanes):
    module = _Frame()
    v, previous = _voltages()
    voltages = v.expand((lanes,) + SHAPE).clone().requires_grad_()
    histories = previous.expand((lanes,) + SHAPE).clone().requires_grad_()
    actual = torch.vmap(module)(voltages, histories)
    assert actual[0].shape == actual[1].shape == (lanes,) + SHAPE
    gradients = torch.autograd.grad(actual[0].square().sum(), (voltages, histories))
    for gradient, value in zip(gradients, (voltages, histories), strict=True):
        assert gradient.shape == value.shape
        assert torch.isfinite(gradient).all()


def test_dufort_current_frame_compiles_fullgraph_with_native_gradient_parity():
    module = _Frame()
    native = copy.deepcopy(module)
    inputs = tuple(value.requires_grad_() for value in _voltages())
    native_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    expected = native.handler.idf(*native_inputs)
    actual = torch.compile(module, backend="aot_eager", fullgraph=True)(*inputs)
    torch.testing.assert_close(actual[:2], expected, rtol=0.0, atol=0.0)
    actual_gradients = torch.autograd.grad(actual[0].square().sum(), inputs)
    expected_gradients = torch.autograd.grad(expected[0].square().sum(), native_inputs)
    torch.testing.assert_close(actual_gradients, expected_gradients)


@pytest.mark.parametrize("scratch", [([], []), ([None], [None]), ([],)])
def test_dufort_current_frame_rejects_incompatible_scratch(scratch):
    handler = _handler()
    with pytest.raises((TypeError, ValueError), match="Dufort current scratch"):
        handler._evaluate_dufort_current_frame(*_voltages(), scratch=scratch)


class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.5, dtype=torch.float64))

    def forward(self, value):
        return self.gain * value


class _ProgramFrame(torch.nn.Module):
    def __init__(self, *, empty=False, parametrized=False, dtype=torch.float64):
        super().__init__()
        self.handler = _handler(dtype=dtype, empty=empty)
        if parametrized:
            torch.nn.utils.parametrize.register_parametrization(
                self.handler.mechanisms["ion"], "g", _Scale()
            )
        from dendra.func._dufort import DufortCurrentPrograms

        object.__setattr__(
            self,
            "program",
            DufortCurrentPrograms(self.handler, _voltages(dtype)[0]),
        )

    def forward(self, voltage, previous, use_history):
        frame = self.program(voltage, previous, use_history)
        return frame, tuple(
            owner._buffers[name] for _slot, owner, name in self.program._output_slots
        )


def _program_mapping(module):
    return {
        f"{path + '.' if path else ''}{name}": value.clone()
        for path, owner in module.named_modules()
        for registry in (owner._parameters, owner._buffers)
        for name, value in registry.items()
        if torch.is_tensor(value)
    }


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_keep_registered_aliases_and_selected_products(
    use_history,
):
    module = _ProgramFrame()
    native = copy.deepcopy(module.handler)
    v, previous = _voltages()
    assert len(module.program._slots) == len(
        {(id(owner), name) for _slot, owner, name in module.program._slots}
    )
    # The ergonomic direct alias and ModuleDict alias share their live owner.
    assert module.handler.ion is module.handler.mechanisms["ion"]
    if use_history:
        expected = native._evaluate_dufort_current_frame(v, previous)
    else:
        current, frame = native._evaluate_explicit_current_frame(v)
        expected = (
            current,
            torch.zeros_like(current),
            frame,
            tuple(torch.zeros_like(value) for value in frame),
        )
    original = _program_mapping(module)
    actual, products = torch.func.functional_call(
        module,
        _program_mapping(module),
        (v, previous, torch.tensor(use_history)),
        tie_weights=False,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    for (slot, _owner, _name), value in zip(
        module.program._output_slots, products, strict=True
    ):
        path, _separator, name = slot.removeprefix("handler.").rpartition(".")
        torch.testing.assert_close(
            value,
            native.get_submodule(path)._buffers[name],
            rtol=0.0,
            atol=0.0,
        )
    torch.testing.assert_close(_program_mapping(module), original)


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_empty_handler_returns_independent_zero_frames(
    use_history,
):
    module = _ProgramFrame(empty=True)
    v, previous = _voltages()
    actual, products = module(v, previous, torch.tensor(use_history))
    assert products == ()
    assert len(actual[2]) == len(actual[3]) == 2
    for value in torch.utils._pytree.tree_leaves(actual):
        torch.testing.assert_close(value, torch.zeros_like(v))


@pytest.mark.parametrize("lanes", [0, 3])
def test_dufort_current_programs_vmap_selects_mixed_history_lanes_and_gradients(lanes):
    module = _ProgramFrame(parametrized=True)
    mapping = _program_mapping(module)
    name = next(key for key in mapping if key.endswith(".gain"))
    gains = (
        mapping[name].expand((lanes,) + mapping[name].shape).clone().requires_grad_()
    )
    flags = torch.arange(lanes) % 2 == 0
    v, previous = _voltages()

    def evaluate(gain, flag):
        values = dict(mapping)
        values[name] = gain
        return torch.func.functional_call(
            module, values, (v, previous, flag), tie_weights=False
        )

    frame, products = torch.vmap(evaluate)(gains, flags)
    assert frame[0].shape == frame[1].shape == (lanes,) + SHAPE
    if lanes:
        references = [evaluate(gain, flag) for gain, flag in zip(gains, flags)]
        expected = torch.utils._pytree.tree_map(
            lambda *values: torch.stack(values), *references
        )
        torch.testing.assert_close((frame, products), expected, rtol=0.0, atol=0.0)
    gradient = torch.autograd.grad(frame[0].square().sum(), gains)[0]
    assert gradient.shape == gains.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_jvp_matches_authored_frame_derivatives(use_history):
    module = _ProgramFrame()
    v, previous = _voltages()

    def evaluate(voltage, old_voltage):
        return torch.func.functional_call(
            module,
            _program_mapping(module),
            (voltage, old_voltage, torch.tensor(use_history)),
            tie_weights=False,
        )[0][0]

    actual, tangent = torch.func.jvp(
        evaluate,
        (v, previous),
        (torch.ones_like(v), torch.ones_like(previous)),
    )
    base_voltage, base_previous = (
        value.clone().requires_grad_() for value in (v, previous)
    )
    expected = evaluate(base_voltage, base_previous)
    gradients = torch.autograd.grad(
        expected.sum(), (base_voltage, base_previous), allow_unused=True
    )
    # Every authored current is pointwise; the all-ones tangent equals the
    # coordinate-wise derivative, including packed duplicates.
    expected_tangent = sum(
        torch.zeros_like(v) if value is None else value for value in gradients
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(tangent, expected_tangent)


def test_dufort_current_programs_initial_phase_does_not_consume_nan_history():
    module = _ProgramFrame()
    v, previous = _voltages()
    previous.fill_(torch.nan)
    # The shared voltage kernel provides its sanitized history operand to the
    # dispatcher; selected startup results and derivatives remain finite.
    valid = torch.tensor(False)
    safe_previous = torch.where(valid, previous, v)
    current = module(v, safe_previous, valid)[0][0]
    assert torch.isfinite(current).all()


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_compile_fullgraph_with_raw_gain_gradients(use_history):
    module = _ProgramFrame(parametrized=True)
    mapping = _program_mapping(module)
    name = next(key for key in mapping if key.endswith(".gain"))
    v, previous = _voltages()

    def evaluate(voltage, old_voltage, gain, valid):
        values = {key: value.clone() for key, value in mapping.items()}
        values[name] = gain
        return torch.func.functional_call(
            module, values, (voltage, old_voltage, valid), tie_weights=False
        )

    inputs = tuple(
        value.clone().requires_grad_() for value in (v, previous, mapping[name])
    )
    compiled_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    expected = evaluate(*inputs, torch.tensor(use_history))
    actual = torch.compile(evaluate, backend="aot_eager", fullgraph=True)(
        *compiled_inputs, torch.tensor(use_history)
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    gradients = torch.autograd.grad(actual[0][0].square().sum(), compiled_inputs)
    expected_gradients = torch.autograd.grad(
        expected[0][0].square().sum(), inputs, allow_unused=True
    )
    for gradient, reference in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(
            gradient,
            torch.zeros_like(gradient) if reference is None else reference,
        )


@pytest.mark.parametrize("lanes", [0, 2])
def test_dufort_current_programs_compiled_vmap_preserves_zero_lanes(lanes):
    module = _ProgramFrame(parametrized=True)
    mapping = _program_mapping(module)
    name = next(key for key in mapping if key.endswith(".gain"))
    v, previous = _voltages()

    def evaluate(gain, valid):
        values = {key: value.clone() for key, value in mapping.items()}
        values[name] = gain
        return torch.func.functional_call(
            module, values, (v, previous, valid), tie_weights=False
        )

    gains = (
        mapping[name].expand((lanes,) + mapping[name].shape).clone().requires_grad_()
    )
    flags = torch.arange(lanes) % 2 == 0
    vmapped = torch.vmap(evaluate)
    expected = vmapped(gains, flags)
    actual = torch.compile(vmapped, backend="aot_eager", fullgraph=True)(gains, flags)
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual[0][0].square().sum(), gains)[0]
    assert gradient.shape == gains.shape
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_program_diagnostic_exposes_raw_hooks_and_selected_products(
    use_history,
):
    module = _ProgramFrame()
    reference = copy.deepcopy(module)
    mechanism = module.handler.mechanisms["ion"]
    mechanism.current_calls = 0
    original = mechanism.ina

    def tracked(self, voltage):
        self.current_calls += 1
        return original(voltage)

    mechanism.ina = MethodType(tracked, mechanism)
    v, previous = _voltages()
    expected, products = reference(v, previous, torch.tensor(use_history))
    actual = module.program.diagnostic(v, previous, torch.tensor(use_history))
    assert mechanism.current_calls == 2
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        tuple(
            owner._buffers[name] for _slot, owner, name in module.program._output_slots
        ),
        products,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_rebind_live_effective_parameter_buffers(use_history):
    module = _ProgramFrame()
    mechanism = module.handler.mechanisms["ion"]
    mechanism.g = (mechanism.g.detach().clone() * 2.25).requires_grad_()
    native = copy.deepcopy(module.handler)
    v, previous = _voltages()
    actual = module(v, previous, torch.tensor(use_history))[0]
    expected = (
        native._evaluate_dufort_current_frame(v, previous)
        if use_history
        else native._evaluate_explicit_current_frame(v)
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    gradient = torch.autograd.grad(actual[0].sum(), mechanism.g)[0]
    torch.testing.assert_close(
        gradient, 0.5 * previous if use_history else v, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_preserve_scalar_gain_dtype_promotion(use_history):
    module = _ProgramFrame(parametrized=True, dtype=torch.float32)
    gain = module.handler.mechanisms["ion"].parametrizations.g[0].gain
    assert gain.dtype == torch.float64
    with torch.no_grad():
        gain.fill_(1.0000000596046448)
    native = copy.deepcopy(module.handler)
    v, previous = _voltages(torch.float32)
    actual = module(v, previous, torch.tensor(use_history))[0]
    expected = (
        native._evaluate_dufort_current_frame(v, previous)
        if use_history
        else native._evaluate_explicit_current_frame(v)
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    assert actual[0].dtype == actual[1].dtype == torch.float32
    gradient = torch.autograd.grad(actual[0].sum(), gain)[0]
    native_gain = native.mechanisms["ion"].parametrizations.g[0].gain
    expected_gradient = torch.autograd.grad(expected[0].sum(), native_gain)[0]
    torch.testing.assert_close(gradient, expected_gradient, rtol=0.0, atol=0.0)
    assert gradient.dtype == torch.float64


class _TimedAffineCurrent(Mechanism):
    Mechanism.RANGE(g=0.25, drive=0.125)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i(self, v):
        return self.g * v + self.drive * self.t


class _ClockedInjection(Mechanism):
    Mechanism.ASSIGNED("i_inj")
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def assigned_values(self, v, values):
        del values
        return {"i_inj": self.injection_values(v, t=self.t)}

    def i(self, v):
        return v * 0.0 + self.i_inj


def _clocked_program(cls):
    from dendra.func._dufort import DufortCurrentPrograms

    from .test_mechanism_injections_pure import _LinearWaveform

    field = torch.ones(SHAPE, dtype=torch.float64)
    owner = torch.nn.Module()
    owner.register_buffer("t", torch.tensor(0.0, dtype=torch.float64))
    mechanism = _mechanism(cls, "current", torch.float64)
    mechanism.setreference("t", lambda owner=owner: owner.t)
    if cls is _ClockedInjection:
        waveform = _LinearWaveform(2.0, slope=0.5).to(dtype=torch.float64)
        assert mechanism.register_waveform_injection(
            waveform, index=(slice(None), 1), model_shape=SHAPE
        )
    handler = MechanismHandler(
        field * 34.0,
        field,
        {"current": mechanism},
        currents={"i": {"current": ["i"]}},
    ).to(dtype=torch.float64)
    handler.make_maps()
    handler.init_i_g_bufs(torch.zeros_like(field))
    program = DufortCurrentPrograms(handler, _voltages()[0], clock_owner=owner)
    return program, handler, owner


@pytest.mark.parametrize("use_history", [False, True])
@pytest.mark.parametrize("cls", [_TimedAffineCurrent, _ClockedInjection])
def test_dufort_current_programs_keep_live_clock_and_injection_times(cls, use_history):
    program, handler, owner = _clocked_program(cls)
    v, previous = _voltages()
    for time in (0.25, 1.5):
        clock = torch.tensor(time, dtype=torch.float64, requires_grad=True)
        owner.t = clock
        actual = program(v, previous, torch.tensor(use_history))
        expected = (
            handler._evaluate_dufort_current_frame(v, previous)
            if use_history
            else handler._evaluate_explicit_current_frame(v)
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
        assert owner.t is clock
        gradient = torch.autograd.grad(actual[0].sum(), clock)[0]
        torch.testing.assert_close(
            gradient,
            clock.new_tensor(1.0),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.cpu
@pytest.mark.parametrize("cpp_wrapper", [False, True])
@pytest.mark.parametrize("execution", ["scan", "while"])
def test_dufort_current_conditional_preserves_inductor_convolution_operand_layout(
    cpp_wrapper, execution
):
    from tests.func.test_dufort_frankel import DT, _population

    # A smoothed multi-row cable makes Inductor fuse the sanitized numerical
    # history into a (N, 3*C) convolution stack. The current conditional keeps
    # its contiguous operand contract while sharing those numerical values.
    with dn.ctx(
        JIT=0,
        REQUIRE_GRAD=1,
        IMEM=False,
        DTYPE=torch.float64,
        DEVICE="cpu",
    ):
        source, native = (
            _population("dfh-conv", beta=0.8, imem=False) for _ in range(2)
        )
        network, native_network = (
            dn.Network({"pop": source}),
            dn.Network({"pop": native}),
        )
        for model in (network, native_network):
            model.train()
            model.initialize(DT)
        runner = dn.compile(
            network,
            dt=DT,
            steps=4,
            inference=True,
            execution=execution,
            backend="inductor",
            options={"cpp_wrapper": cpp_wrapper},
        )
        with torch.no_grad():
            state, _ = runner()
            for _ in range(4):
                native_network.step()
        actual = state["populations"]["pop"]
        for name in ("v", "v_prev"):
            torch.testing.assert_close(
                actual["integrator"][name],
                getattr(native, name),
                rtol=0.0,
                atol=2.0e-12,
            )
        for mechanism, field in (("curved", "gain"), ("probe", "seen")):
            torch.testing.assert_close(
                actual["mechanism_buffers"][mechanism][field],
                getattr(native.mech.mechanisms[mechanism], field),
                rtol=0.0,
                atol=2.0e-12,
            )


@pytest.mark.cpu
def test_dufort_current_conditional_preserves_checkpointed_inductor_gradients():
    from tests.func.test_dufort_frankel import DT, _close, _population

    # Functionalizing a checkpointed scan changes the storage-layout view into
    # as_strided_copy. Its local decomposition must preserve the conditional's
    # operand layout in both the forward and recomputed backward programs.
    with dn.ctx(
        JIT=0,
        REQUIRE_GRAD=1,
        IMEM=False,
        DTYPE=torch.float64,
        DEVICE="cpu",
    ):
        populations = tuple(
            _population("dfh-conv", beta=0.8, imem=False) for _ in range(2)
        )
        networks = tuple(dn.Network({"pop": population}) for population in populations)
        suffixes = ("ssd.weight", "filter.weight", "pas.g_param")
        for network in networks:
            network.train()
            for name, parameter in network.named_parameters():
                if name.endswith(suffixes):
                    parameter.requires_grad_()
            network.initialize(DT)
        source, native_network = networks
        _source_population, native = populations
        runner = dn.compile(
            source,
            dt=DT,
            steps=4,
            inference=False,
            checkpoint_every=2,
            trainable=tuple(f"*{suffix}" for suffix in suffixes),
            backend="inductor",
            options={"cpp_wrapper": False},
        )
        state, _ = runner()
        for _ in range(4):
            native_network.step()
        _close(state, runner.functional.extract(native_network).state)
        packet = state["populations"]["pop"]
        actual_loss = (
            packet["integrator"]["v"].square().mean()
            + packet["ions"]["na"]["ina"].square().mean()
        )
        expected_loss = (
            native.v.square().mean() + native.mech.ions["na"].ina.square().mean()
        )
        targets, expected_targets = [], []
        for name, parameter in native_network.named_parameters():
            if name.endswith(suffixes):
                targets.append(runner.get_parameter(f"populations.{name}"))
                expected_targets.append(parameter)
        actual_gradients = torch.autograd.grad(actual_loss, targets)
        expected_gradients = torch.autograd.grad(expected_loss, expected_targets)
        _close(actual_gradients, expected_gradients)
        assert all(torch.isfinite(gradient).all() for gradient in actual_gradients)
        assert all(gradient.abs().sum() > 0 for gradient in actual_gradients)


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_preserve_transposed_raw_parameters_and_voltage(
    use_history,
):
    from dendra.func._dufort import DufortCurrentPrograms

    handler = _handler()
    scale = _Scale()
    scale.gain = torch.nn.Parameter(
        torch.linspace(0.5, 1.5, 8, dtype=torch.float64).reshape(4, 2).T
    )
    torch.nn.utils.parametrize.register_parametrization(
        handler.mechanisms["ion"], "g", scale
    )
    native = copy.deepcopy(handler)
    voltage = torch.linspace(-2.0, 3.0, 8, dtype=torch.float64).reshape(4, 2).T
    previous = torch.linspace(-4.0, 1.0, 8, dtype=torch.float64).reshape(4, 2).T
    voltage.requires_grad_()
    previous.requires_grad_()
    assert not scale.gain.is_contiguous()
    assert not voltage.is_contiguous()
    program = DufortCurrentPrograms(handler, voltage)
    actual = program(voltage, previous, torch.tensor(use_history))
    expected = (
        native._evaluate_dufort_current_frame(voltage, previous)
        if use_history
        else native._evaluate_explicit_current_frame(voltage)
    )
    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    native_gain = native.mechanisms["ion"].parametrizations.g[0].gain
    actual_gradients = torch.autograd.grad(
        actual[0].square().sum(),
        (voltage, previous, scale.gain),
        allow_unused=True,
    )
    expected_gradients = torch.autograd.grad(
        expected[0].square().sum(),
        (voltage, previous, native_gain),
        allow_unused=True,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient,
            torch.zeros_like(actual_gradient)
            if expected_gradient is None
            else expected_gradient,
        )


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_manual_forward_ad_matches_jvp(use_history):
    v, previous = _voltages()
    module = _ProgramFrame()

    def evaluate(voltage, old_voltage):
        return torch.func.functional_call(
            module,
            _program_mapping(module),
            (voltage, old_voltage, torch.tensor(use_history)),
            tie_weights=False,
        )[0][0]

    expected, expected_tangent = torch.func.jvp(
        evaluate,
        (v, previous),
        (torch.ones_like(v), torch.ones_like(previous)),
    )
    with torch.autograd.forward_ad.dual_level():
        voltage = torch.autograd.forward_ad.make_dual(v, torch.ones_like(v))
        old_voltage = torch.autograd.forward_ad.make_dual(
            previous, torch.ones_like(previous)
        )
        primal, tangent = torch.autograd.forward_ad.unpack_dual(
            evaluate(voltage, old_voltage)
        )
        torch.testing.assert_close(primal, expected)
        torch.testing.assert_close(tangent, expected_tangent)


class _AuthoredCurrentAD(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(value):
        return 2.0 * value

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, gradient):
        return 3.0 * gradient

    @staticmethod
    def jvp(ctx, tangent):
        return 4.0 * tangent


class _AuthoredADCurrent(Mechanism):
    Mechanism.RANGE(g=0.25)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.EXPLICIT("i")
    Mechanism.SAVE_CURRENT("i")

    def i(self, v):
        return _AuthoredCurrentAD.apply(self.g * v)


def _authored_ad_program(implementation=_AuthoredADCurrent):
    from dendra.func._dufort import DufortCurrentPrograms

    field = torch.ones(SHAPE, dtype=torch.float64)
    mechanism = _mechanism(implementation, "current", torch.float64)
    handler = MechanismHandler(
        34.0 * field,
        field,
        {"current": mechanism},
        currents={"i": {"current": ["i"]}},
    ).to(dtype=field.dtype)
    handler.make_maps()
    handler.init_i_g_bufs(field)
    return DufortCurrentPrograms(handler, _voltages()[0]), handler


@pytest.mark.parametrize("use_history", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_dufort_current_programs_preserve_authored_backward(use_history, compiled):
    program, handler = _authored_ad_program()
    voltage, previous = _voltages()
    voltage.requires_grad_()
    gain = handler.mechanisms["current"].g.requires_grad_()

    def evaluate(voltage):
        return program(voltage, previous, torch.tensor(use_history))[0]

    if compiled:
        evaluate = torch.compile(evaluate, backend="aot_eager", fullgraph=True)
    actual = evaluate(voltage)
    native = handler._evaluate_explicit_current_frame(voltage)[0]
    torch.testing.assert_close(actual, native)
    gradients = torch.autograd.grad(actual.sum(), (voltage, gain))
    expected = torch.autograd.grad(native.sum(), (voltage, gain))
    torch.testing.assert_close(gradients, expected)
    torch.testing.assert_close(gradients[0], 3.0 * gain.expand_as(voltage))
    torch.testing.assert_close(gradients[1], 3.0 * voltage)


@pytest.mark.parametrize("use_history", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_dufort_current_programs_preserve_authored_forward_ad(
    use_history, grad_enabled
):
    program, handler = _authored_ad_program()
    voltage, previous = _voltages()
    direction = torch.ones_like(voltage)
    with torch.set_grad_enabled(grad_enabled), torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(voltage, direction)
        value, tangent = torch.autograd.forward_ad.unpack_dual(
            program(dual, previous, torch.tensor(use_history))[0]
        )
        expected, expected_tangent = torch.autograd.forward_ad.unpack_dual(
            handler._evaluate_explicit_current_frame(dual)[0]
        )
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(tangent, expected_tangent)
    torch.testing.assert_close(
        tangent, 4.0 * handler.mechanisms["current"].g.expand_as(voltage)
    )


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_preserve_authored_functorch_rules(use_history):
    program, handler = _authored_ad_program()
    voltage, previous = _voltages()

    def evaluate(value):
        return program(value, previous, torch.tensor(use_history))[0]

    value, tangent = torch.func.jvp(evaluate, (voltage,), (torch.ones_like(voltage),))
    torch.testing.assert_close(value, 2.0 * handler.mechanisms["current"].g * voltage)
    torch.testing.assert_close(
        tangent, 4.0 * handler.mechanisms["current"].g.expand_as(voltage)
    )
    gradient = torch.func.grad(lambda value: evaluate(value).sum())(voltage)
    torch.testing.assert_close(
        gradient, 3.0 * handler.mechanisms["current"].g.expand_as(voltage)
    )
    batch = torch.stack((voltage, voltage + 0.5, voltage - 0.25))
    actual = torch.vmap(evaluate)(batch)
    torch.testing.assert_close(actual, 2.0 * handler.mechanisms["current"].g * batch)


class _AuthoredNonDifferentiable(torch.autograd.Function):
    @staticmethod
    def forward(value):
        return 2.0 * value

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.mark_non_differentiable(output)

    @staticmethod
    def backward(ctx, gradient):
        return None


class _NonDifferentiableCurrent(_AuthoredADCurrent):
    def i(self, v):
        return _AuthoredNonDifferentiable.apply(self.g * v)


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_programs_preserve_authored_non_differentiable_outputs(
    use_history,
):
    program, handler = _authored_ad_program(_NonDifferentiableCurrent)
    assert program._authored_autograd
    voltage, previous = _voltages()
    voltage.requires_grad_()
    actual = program(voltage, previous, torch.tensor(use_history))[0]
    expected = handler._evaluate_explicit_current_frame(voltage)[0]
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad((actual + 0.0 * voltage).sum(), voltage)[0]
    expected_gradient = torch.autograd.grad((expected + 0.0 * voltage).sum(), voltage)[
        0
    ]
    torch.testing.assert_close(actual_gradient, expected_gradient)
    torch.testing.assert_close(actual_gradient, torch.zeros_like(voltage))


class _AuthoredNonDifferentiableIdentity(_AuthoredNonDifferentiable):
    @staticmethod
    def forward(value):
        return value


class _NonDifferentiableIdentityCurrent(_AuthoredADCurrent):
    def i(self, v):
        return _AuthoredNonDifferentiableIdentity.apply(self.g * v)


class _DetachedCurrent(_AuthoredADCurrent):
    def i(self, v):
        with torch.no_grad():
            return 2.0 * self.g * v


@pytest.mark.parametrize(
    "implementation", [_NonDifferentiableIdentityCurrent, _DetachedCurrent]
)
def test_dufort_current_programs_preserve_detached_authored_arithmetic(implementation):
    program, handler = _authored_ad_program(implementation)
    voltage, previous = _voltages()
    voltage.requires_grad_()
    actual = program(voltage, previous, torch.tensor(False))[0]
    expected = handler._evaluate_explicit_current_frame(voltage)[0]
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad((actual + 0.0 * voltage).sum(), voltage)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(voltage))


class _ForwardOnlyCurrentAD(torch.autograd.Function):
    @staticmethod
    def forward(value):
        return 2.0 * value

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass


class _ForwardOnlyCurrent(_AuthoredADCurrent):
    def i(self, v):
        return _ForwardOnlyCurrentAD.apply(self.g * v)


class _ConditionalCurrent(_AuthoredADCurrent):
    def i(self, v):
        return torch.cond(
            torch.all(v < 0.0),
            lambda voltage, gain: 2.0 * gain * voltage,
            lambda voltage, gain: 3.0 * gain * voltage,
            (v, self.g),
        )


@pytest.mark.parametrize("use_history", [False, True])
def test_dufort_current_capture_observer_preserves_authored_conditionals(use_history):
    program, handler = _authored_ad_program(_ConditionalCurrent)
    voltage, previous = _voltages()
    for value in (voltage, voltage - 10.0):
        value.requires_grad_()
        actual = program(value, previous, torch.tensor(use_history))[0]
        expected = handler._evaluate_explicit_current_frame(value)[0]
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(actual.sum(), value)[0]
        expected_gradient = torch.autograd.grad(expected.sum(), value)[0]
        torch.testing.assert_close(gradient, expected_gradient)


@pytest.mark.parametrize("execution", ["scan", "while"])
def test_dufort_network_inference_accepts_forward_only_authored_current(execution):
    from tests.func.test_dufort_frankel import DT, _close, _population

    with (
        torch.no_grad(),
        dn.ctx(JIT=0, REQUIRE_GRAD=0, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"),
    ):
        populations = tuple(_population("df", imem=False) for _ in range(2))
        for population in populations:
            population.insert(_ForwardOnlyCurrent.rename("forward_only"))
        networks = tuple(dn.Network({"pop": population}) for population in populations)
        for network in networks:
            network.initialize(DT)
        source, native = networks
        runner = dn.compile(
            source,
            dt=DT,
            steps=4,
            inference=True,
            execution=execution,
            backend="aot_eager",
        )
        assert runner.functional.populations[
            "pop"
        ]._transition._dufort_current_programs._grad_raw
        state, _recordings = runner()
        for _ in range(4):
            native.step()
        _close(state, runner.functional.extract(native).state)


@pytest.mark.parametrize("checkpoint_every", [None, 2])
def test_dufort_network_scan_preserves_authored_current_backward(checkpoint_every):
    from tests.func.test_dufort_frankel import DT, _close, _population

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, IMEM=False, DTYPE=torch.float64, DEVICE="cpu"):
        populations = tuple(_population("df", imem=False) for _ in range(2))
        for population in populations:
            population.insert(_AuthoredADCurrent.rename("authored"))
        networks = tuple(dn.Network({"pop": population}) for population in populations)
        for network in networks:
            network.train()
            for name, parameter in network.named_parameters():
                if name.endswith("authored.g_param"):
                    parameter.requires_grad_()
            network.initialize(DT)
        source, native = networks
        runner = dn.compile(
            source,
            dt=DT,
            steps=4,
            inference=False,
            checkpoint_every=checkpoint_every,
            trainable="*authored.g_param",
            backend="aot_eager",
        )
        state, _recordings = runner()
        for _ in range(4):
            native.step()
        _close(state, runner.functional.extract(native).state)
        name, target = next(
            (name, parameter)
            for name, parameter in native.named_parameters()
            if name.endswith("authored.g_param")
        )
        actual = torch.autograd.grad(
            state["populations"]["pop"]["integrator"]["v"].square().mean(),
            runner.get_parameter(f"populations.{name}"),
        )[0]
        expected = torch.autograd.grad(populations[1].v.square().mean(), target)[0]
        _close(actual, expected)
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
