"""Cold stimulation correspondence preserves authored Tensor identities."""

from __future__ import annotations

import copy

import pytest
import torch

import dendra as dn
from dendra.func._network_bindings import AliasSnapshots
from dendra.func._stimuli import FunctionalExtra
from dendra.models.mod import pas
from dendra.models.stim.waveform import Waveform

pytestmark = pytest.mark.cpu
DT = 0.1


class _SourceWaveform(Waveform):
    FUNCTIONAL_PURE = True

    def __init__(self, *, dtype=torch.float64):
        super().__init__()
        self.register_parameter(
            "gain", torch.nn.Parameter(torch.tensor(0.2, dtype=dtype))
        )
        self.register_buffer("offset", torch.tensor(0.1, dtype=dtype))

    def fn(self, t):
        return torch.ones_like(t) * self.gain + self.offset


def _model(waveform=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64):
        model = dn.SingleCompartment(N=2, C=3, integrator=dn.bwd_euler_sc())
        model.insert(pas)
        if waveform is not None:
            model[:, 0].inject(waveform)
        model.train()
        model.initialize(DT)
    return model


def test_extra_sources_preserve_exact_identity_before_conversion_or_cloning():
    model = _model()
    waveform = _SourceWaveform(dtype=torch.float32)
    field = torch.arange(6.0, dtype=torch.float32).reshape(2, 3)
    view = field.view_as(field)
    python_field = [[1.0, 2.0, 3.0]]
    plan = FunctionalExtra(
        model,
        [
            (field, waveform),
            (field, waveform),
            (view, waveform),
            (python_field, waveform),
            (python_field, waveform),
        ],
        namespace="stimulation.extra",
    )
    sources = plan._tensor_sources()
    values = plan.extract()
    assert sources["stimulation.extra.contacts.0.field"] is field
    assert sources["stimulation.extra.contacts.1.field"] is field
    assert sources["stimulation.extra.contacts.2.field"] is view
    assert "stimulation.extra.contacts.3.field" not in sources
    assert "stimulation.extra.contacts.4.field" not in sources
    assert sources["stimulation.extra.waveforms.0.gain"] is waveform.gain
    assert sources["stimulation.extra.waveforms.0.offset"] is waveform.offset
    assert (
        values.parameters["stimulation.extra.waveforms.0.gain"].dtype == torch.float64
    )
    assert waveform.gain.dtype == torch.float32
    assert values.constants["stimulation.extra.contacts.0.field"] is not field

    # The aggregate memo preserves exact authored ties, without inferring an
    # alias from a shared storage allocation or repeated Python-valued fields.
    snapshot = AliasSnapshots().tree(sources, {**values.parameters, **values.constants})
    assert (
        snapshot["stimulation.extra.contacts.0.field"]
        is snapshot["stimulation.extra.contacts.1.field"]
    )
    assert (
        snapshot["stimulation.extra.contacts.0.field"]
        is not snapshot["stimulation.extra.contacts.2.field"]
    )
    assert (
        snapshot["stimulation.extra.contacts.3.field"]
        is not snapshot["stimulation.extra.contacts.0.field"]
    )
    assert (
        snapshot["stimulation.extra.contacts.3.field"]
        is not snapshot["stimulation.extra.contacts.4.field"]
    )


def test_population_sources_include_registered_intra_and_bound_extra():
    intra = _SourceWaveform()
    model = _model(intra)
    extra = _SourceWaveform()
    field = torch.ones_like(model.v)
    functional, tensors = dn.func.make_functional(model, dt=DT, extra=(field, extra))
    sources = functional._parameter_sources()
    assert set(sources) == set(tensors.parameters)
    for name, value in model.named_parameters():
        assert sources[name] is value
    assert sources["stimulation.intra.waveforms.0.gain"] is intra.gain
    assert sources["stimulation.intra.waveforms.0.offset"] is intra.offset
    assert sources["stimulation.extra.waveforms.0.gain"] is extra.gain
    assert sources["stimulation.extra.waveforms.0.offset"] is extra.offset
    assert sources["stimulation.extra.contacts.0.field"] is field


def test_compatible_population_sources_use_alternate_intra_and_original_extra():
    intra = _SourceWaveform()
    model = _model(intra)
    extra = _SourceWaveform()
    field = torch.ones_like(model.v)
    functional, _ = dn.func.make_functional(model, dt=DT, extra=(field, extra))
    alternate = copy.deepcopy(model)
    alternate_intra = alternate.injections[0][0]
    sources = functional._parameter_sources(alternate)
    packet = functional.extract(alternate)
    assert set(sources) == set(packet.parameters)
    assert sources["stimulation.intra.waveforms.0.gain"] is alternate_intra.gain
    assert sources["stimulation.intra.waveforms.0.offset"] is alternate_intra.offset
    assert sources["stimulation.intra.waveforms.0.gain"] is not intra.gain
    assert sources["stimulation.extra.waveforms.0.gain"] is extra.gain
    assert sources["stimulation.extra.contacts.0.field"] is field
    for name, value in alternate.named_parameters():
        assert sources[name] is value


def test_alternate_intra_sources_reject_incompatible_injection_layout():
    model = _model(_SourceWaveform())
    functional, _ = dn.func.make_functional(model, dt=DT)
    alternate = copy.deepcopy(model)
    alternate[:, 1].inject(_SourceWaveform())
    with pytest.raises(
        dn.func.FunctionalizationError, match="structure does not match"
    ):
        functional._parameter_sources(alternate)
    with pytest.raises(dn.func.FunctionalizationError, match="injection topology"):
        functional.intra._tensor_sources(alternate)


@pytest.mark.parametrize("role", ("intra", "population_parameter"))
def test_network_sources_preserve_aliases_across_stimulation_roles(role):
    waveform = _SourceWaveform()
    left = _model(waveform if role == "intra" else None)
    right = _model()
    if role == "population_parameter":
        waveform.gain = left.celsius_param
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64):
        network = dn.Network({"left": left, "right": right})
        network.train()
        network.initialize(DT)
    functional, tensors = dn.func.make_functional_network(
        network, extra={"right": (torch.ones_like(right.v), waveform)}
    )
    source_name = (
        "stimulation.intra.waveforms.0.gain" if role == "intra" else "celsius_param"
    )
    parameters = tensors.parameters["populations"]
    assert (
        parameters["left"][source_name]
        is parameters["right"]["stimulation.extra.waveforms.0.gain"]
    )
    if role == "intra":
        assert (
            parameters["left"]["stimulation.intra.waveforms.0.offset"]
            is parameters["right"]["stimulation.extra.waveforms.0.offset"]
        )
    independent = tensors.independent_parameters(
        "populations.right.stimulation.extra.waveforms.0.gain"
    )
    assert (
        independent["populations"]["left"][source_name]
        is independent["populations"]["right"]["stimulation.extra.waveforms.0.gain"]
    )
    assert independent["populations"]["left"][source_name].requires_grad
    # Alternate extraction retains the plan's external extra recipe. Copying
    # the native graph does not rebind that recipe to its copied native leaves.
    with pytest.raises(ValueError, match="aliases"):
        functional.extract(copy.deepcopy(network))


@pytest.mark.parametrize("leaf", ("gain", "offset"))
def test_network_extraction_rejects_split_canonical_waveform_source_aliases(leaf):
    first, second = _SourceWaveform(), _SourceWaveform()
    setattr(second, leaf, getattr(first, leaf))
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=torch.float64):
        network = dn.Network({"cell": _model()})
        network.train()
        network.initialize(DT)
    field = torch.ones_like(network.cell.v)
    functional, packet = dn.func.make_functional_network(
        network, extra={"cell": [(field, first), (field, second)]}
    )
    canonical = f"stimulation.extra.waveforms.0.{leaf}"
    assert [
        name
        for name in packet.parameters["populations"]["cell"]
        if name.endswith(f".{leaf}")
    ] == [canonical]
    replacement = torch.tensor(0.7, dtype=torch.float64)
    if leaf == "gain":
        replacement = torch.nn.Parameter(replacement)
    setattr(second, leaf, replacement)
    with pytest.raises(
        dn.func.FunctionalizationError, match="source aliases changed.*lower.*again"
    ):
        functional.extract()

    # Rebinding every participant to one new shared source retains the same
    # canonical alias layout and supplies its current value to extraction.
    setattr(first, leaf, replacement)
    refreshed = functional.extract()
    torch.testing.assert_close(
        refreshed.parameters["populations"]["cell"][canonical], replacement
    )
