"""Point containers preserve the selected native stencil and its AD contract."""

from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.integrators.explicit import SymmetricConv1D, _conv1d_forward
from dendra.models.mod import graded_syn, pas
from tests.func.test_euler import DT, STEPS, _clone, _close, _drives, _native_step

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _environment():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with dn.ctx(
            JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu", IMEM=True
        ):
            yield
    finally:
        torch.set_num_threads(previous)


def _point(method, *, training=True, population_type=dn.Population, compartments=3):
    options = {"imem": True}
    if method.startswith("df"):
        options["beta"] = 0.6
    if method == "dfh-conv":
        options["conv"] = True
        method = "dfh"
    population = population_type(
        N=2, C=compartments, integrator=getattr(dn, method)(**options)
    )
    population.diam.fill_(2.0)
    population.dx.fill_(10.0)
    population.insert(pas, g=0.0004, e=-71.0)
    population.batch(2)
    population.train(training)
    population.initialize()
    population.v = torch.linspace(-66.0, -59.0, population.v.numel()).reshape(
        population.shape
    )
    population.integrator._initialize(
        population, population.t.new_tensor(DT), force=True
    )
    return population


@pytest.mark.parametrize("method", ("euler", "rk2", "rk4", "df", "dfh-conv"))
@pytest.mark.parametrize("training", (False, True), ids=("eval", "train"))
def test_point_geometry_stencil_and_drive_gradients_match_native(method, training):
    source, native = (_point(method, training=training) for _ in range(2))
    names = ["cm_param.rho", "rhoa_param.rho"]
    if method.startswith("df"):
        names.append("integrator.filter.weight")
    if method != "df":
        names.append("integrator.ssd.weight")
        channels = source.integrator.ssd.in_channels
        weight = torch.linspace(0.4, 1.2, 3 * channels).reshape(1, channels, 3)
        weight[..., 1] *= -2.0
        for population in (source, native):
            with torch.no_grad():
                population.integrator.ssd.weight.copy_(weight)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    parameters, constants, state = map(
        _clone, (tensors.parameters, tensors.constants, tensors.state)
    )
    names.append(next(name for name in parameters if name.endswith(".pas.g_param")))
    targets = [parameters[name].requires_grad_() for name in names]
    targets.extend(constants[name].requires_grad_() for name in ("diam", "dx"))
    targets.append(state["integrator"]["v"].requires_grad_())
    inputs = _drives(source)
    inputs = inputs._replace(
        ve=inputs.ve.requires_grad_(), intra=inputs.intra.requires_grad_()
    )
    targets.extend((inputs.ve, inputs.intra))
    frozen = _clone((parameters, constants, state, inputs))
    final, _ = functional.prepare_and_rollout(
        parameters, constants, state, inputs, steps=STEPS
    )
    loss = (
        final["integrator"]["v"].square().mean()
        + (1e6 * final["integrator"]["i_membrane"]).square().mean()
    )
    gradients = torch.autograd.grad(loss, targets)
    native_parameters = dict(native.named_parameters())
    native_targets = [native_parameters[name].requires_grad_() for name in names]
    for name in ("diam", "dx"):
        value = getattr(native, name).detach().clone().requires_grad_()
        setattr(native, name, value)
        native_targets.append(value)
    native.v = native.v.detach().clone().requires_grad_()
    native_targets.append(native.v)
    native_inputs = _clone(inputs)
    native_inputs = native_inputs._replace(
        ve=native_inputs.ve.requires_grad_(), intra=native_inputs.intra.requires_grad_()
    )
    native_targets.extend((native_inputs.ve, native_inputs.intra))
    for index in range(STEPS):
        _native_step(
            native,
            dn.func.StepInput(
                ve=native_inputs.ve[index], intra=native_inputs.intra[index]
            ),
        )
    expected_loss = native.v.square().mean() + (1e6 * native.i_membrane).square().mean()
    expected_gradients = torch.autograd.grad(expected_loss, native_targets)
    _close(final, functional.extract(native).state)
    _close(gradients, expected_gradients)
    _close((parameters, constants, state, inputs), frozen)
    _close(functional.extract(source), tensors)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
    if method != "df":
        gradient = gradients[names.index("integrator.ssd.weight")]
        if training:
            _close(gradient, gradient.flip(-1))
        else:
            assert not torch.allclose(gradient, gradient.flip(-1))


@pytest.mark.parametrize("population_type", (dn.Population, dn.SingleCompartment))
@pytest.mark.parametrize(
    "method,change,message",
    (
        ("euler", "stencil", "reflection stencil"),
        ("rk4", "stencil", "reflection stencil"),
        ("df", "smoothing", "fixed smoothing configuration"),
        ("dfh-conv", "smoothing", "fixed smoothing configuration"),
        ("euler", "kernel", "canonical Euler"),
        ("df", "kernel", "canonical Euler"),
    ),
)
def test_point_lowering_preserves_integrator_configuration_guards(
    population_type, method, change, message
):
    source = _point(method, population_type=population_type)
    if change == "stencil":
        source.integrator.ssd.stride = (2,)
    elif change == "smoothing":
        source.integrator.smoothing = False
    else:
        source.integrator._step = source.integrator._step
    with pytest.raises(dn.func.FunctionalizationError, match=message):
        dn.func.make_functional(source, dt=DT)


@pytest.mark.parametrize("method", ("euler", "rk4", "df", "dfh-conv"))
def test_point_integrator_support_does_not_admit_custom_population_geometry(method):
    class CustomPoint(dn.Population):
        @property
        def area(self):
            return 2.0 * super().area

    source = _point(method, population_type=CustomPoint)
    with pytest.raises(dn.func.FunctionalizationError, match="supported topologies"):
        dn.func.make_functional(source, dt=DT)


@pytest.mark.parametrize("population_type", (dn.Population, dn.SingleCompartment))
@pytest.mark.parametrize(
    "method,map_filter", (("rk4", False), ("dfh-conv", False), ("dfh-conv", True))
)
@pytest.mark.parametrize("compartments", (1, 3))
def test_point_integrators_preserve_jvp_vjp_and_hidden_batch_axes(
    population_type, method, map_filter, compartments
):
    source = _point(method, population_type=population_type, compartments=compartments)
    functional, tensors = dn.func.make_functional(source, dt=DT)
    raw_cm = tensors.parameters["cm_param.rho"]
    voltage = tensors.state["integrator"]["v"]
    stencil = tensors.parameters["integrator.ssd.weight"]
    filter_weight = tensors.parameters.get("integrator.filter.weight", stencil)

    def solve(cm, v, weight, smoothing):
        parameters = {
            **tensors.parameters,
            "cm_param.rho": cm,
            "integrator.ssd.weight": weight,
        }
        if map_filter:
            parameters["integrator.filter.weight"] = smoothing
        state = {
            **tensors.state,
            "integrator": {**tensors.state["integrator"], "v": v},
        }
        final, _ = functional.prepare_and_rollout(
            parameters, tensors.constants, state, steps=2
        )
        return final["integrator"]["v"]

    variables = (raw_cm, voltage, stencil, filter_weight)
    directions = tuple(
        torch.linspace(0.02, 0.1, value.numel()).reshape_as(value)
        for value in variables
    )
    primal, tangent = torch.func.jvp(solve, variables, directions)
    weights = torch.linspace(0.1, 1.0, primal.numel()).reshape_as(primal)
    gradient = torch.func.grad(
        lambda *args: (solve(*args) * weights).sum(), argnums=(0, 1, 2, 3)
    )
    gradients = gradient(*variables)
    _close(
        (tangent * weights).sum(),
        sum(
            (value * direction).sum()
            for value, direction in zip(gradients, directions, strict=True)
        ),
    )
    lanes = tuple(
        torch.stack((value - 0.2 * direction, value + 0.5 * direction))
        for value, direction in zip(variables, directions, strict=True)
    )
    actual = torch.vmap(solve)(*lanes)
    expected = torch.stack(
        [solve(*(value[index] for value in lanes)) for index in range(2)]
    )
    _close(actual, expected)
    empty = torch.vmap(solve)(*(value[:0] for value in lanes))
    assert empty.shape == (0, *source.shape)
    empty_gradients = torch.vmap(gradient)(*(value[:0] for value in lanes))
    for actual_gradient, value in zip(empty_gradients, variables, strict=True):
        assert actual_gradient.shape == (0, *value.shape)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("training", (False, True), ids=("eval", "train"))
@pytest.mark.parametrize("rows,compartments", ((1, 1), (1, 3), (2, 1)))
@pytest.mark.parametrize(
    "conv_type,channels,width",
    ((SymmetricConv1D, 2, 3), (torch.nn.Conv1d, 1, 5)),
)
def test_small_stencil_outputs_and_vjp_match_explicit_padded_convolution(
    dtype, training, rows, compartments, conv_type, channels, width
):
    convolution = conv_type(
        channels, 1, width, padding="same", padding_mode="reflect", bias=False
    ).to(dtype=dtype)
    convolution.train(training)
    with torch.no_grad():
        convolution.weight.copy_(
            torch.linspace(-0.7, 1.2, channels * width, dtype=dtype).reshape_as(
                convolution.weight
            )
        )
    value = torch.linspace(
        -2.0, 0.8, rows * channels * compartments, dtype=dtype
    ).reshape(rows, channels, compartments)
    actual_input = value.clone().requires_grad_()
    expected_input = value.clone().requires_grad_()
    expected_weight = convolution.weight.detach().clone().requires_grad_()
    actual = (
        convolution(actual_input)
        if conv_type is SymmetricConv1D
        else _conv1d_forward(convolution, actual_input, convolution.weight)
    )
    effective = expected_weight
    if conv_type is SymmetricConv1D and training:
        effective = (expected_weight + expected_weight.flip(-1)) / 2.0
    expected = torch.nn.functional.conv1d(
        torch.nn.functional.pad(
            expected_input,
            (width // 2, width // 2),
            mode="replicate" if compartments == 1 else "reflect",
        ),
        effective,
    )
    cotangent = torch.linspace(0.3, -0.8, actual.numel(), dtype=dtype).reshape_as(
        actual
    )
    actual_gradients = torch.autograd.grad(
        actual, (actual_input, convolution.weight), cotangent
    )
    expected_gradients = torch.autograd.grad(
        expected, (expected_input, expected_weight), cotangent
    )
    _close(actual, expected, dtype=dtype)
    _close(actual_gradients, expected_gradients, dtype=dtype)


@pytest.mark.parametrize("cpp_wrapper", (False, True))
def test_inductor_single_row_rk4_preserves_stage_current_storage(cpp_wrapper):
    source = dn.SingleCompartment(
        N=1,
        C=3,
        v_init=torch.tensor([-66.0, -63.0, -60.0]),
        integrator=dn.rk4(imem=True),
    )
    source.insert(pas, g=0.0004, e=-71.0)
    source.insert(graded_syn.rename("syn"), e=0.0)
    source.eval()
    source.initialize()
    source.mech.syn.g_pre = torch.full_like(source.v, 9.85e-5)
    functional, tensors = dn.func.make_functional(source, dt=DT)

    def step(state):
        return functional.prepare_and_step(
            tensors.parameters, tensors.constants, state
        )[0]

    with torch.no_grad(), torch_compiler_warning_context():
        compiled = torch.compile(
            step,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"cpp_wrapper": cpp_wrapper},
        )
        actual, expected = _clone(tensors.state), _clone(tensors.state)
        for _ in range(3):
            actual = compiled(actual)
            expected = step(expected)
            _close(actual, expected)
