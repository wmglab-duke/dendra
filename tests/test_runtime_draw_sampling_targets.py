"""Native sampler admission follows the existing canonical subclass contracts."""

import pytest
import torch

import dendra as dn
from dendra.models.integrators.implicit import _bwd_euler_sc
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import pas
from tests.func.test_callback_checkpoints import _assert_packet_equal
from tests.func.test_continuation import _snapshot
from tests.test_native_runtime_draws import _environment as _environment

pytestmark = pytest.mark.cpu
DT = 0.025


class _Cell(dn.SingleCompartment):
    def __init__(self):
        super().__init__(N=1, C=1, integrator=dn.bwd_euler_sc(imem=False))
        self.insert(pas)


class _Network(dn.Network):
    def __init__(self):
        super().__init__({"cell": _Cell()})


@pytest.mark.parametrize("network", (False, True))
def test_native_sampling_accepts_constructor_only_subclasses_and_keeps_empty_packets_pure(
    network,
):
    model = _Network() if network else _Cell()
    model.initialize(DT) if network else model.initialize()
    before = _snapshot(model.state_dict_for_checkpoint())
    packet = dn.func.sample_runtime_draws(model, dt=DT, steps=0)
    repeated = dn.func.sample_runtime_draws(model, dt=DT, steps=0)
    _assert_packet_equal(packet, repeated)
    _assert_packet_equal(model.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("network", (False, True))
def test_native_sampling_rejects_replaced_execution_hooks_before_calling_them(
    network, monkeypatch
):
    model = _Network() if network else _Cell()
    model.initialize(DT) if network else model.initialize()
    before = _snapshot(model.state_dict_for_checkpoint())

    def replaced(self, *args, **kwargs):
        raise AssertionError("Sampler invoked an authored execution replacement")

    monkeypatch.setattr(type(model), "step", replaced)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
        dn.func.sample_runtime_draws(model, dt=DT)
    _assert_packet_equal(model.state_dict_for_checkpoint(), before)


def test_network_runtime_sampling_rejects_unknown_extra_before_evaluation():
    model = _Network()
    model.initialize(DT)
    before = _snapshot(model.state_dict_for_checkpoint())
    with pytest.raises(ValueError, match="unknown Populations"):
        dn.func.sample_runtime_draws(
            model, extra={"absent": (torch.ones(1), dn.constant(value=1.0))}
        )
    _assert_packet_equal(model.state_dict_for_checkpoint(), before)


class _NoisyLeak(Mechanism):
    Mechanism.RANGE(g=0.001, e=-65.0)
    Mechanism.RANGENOISE("eta", mu=0.0, sigma=0.00001, seed=947)
    Mechanism.NONSPECIFIC_CURRENT("i")
    Mechanism.AFFINE("i")

    def i_affine_terms(self, v):
        del v
        return self.g, self.g * self.e + self.eta.clone()


class _ExtraNoiseStep(_bwd_euler_sc):
    def step(self, model, dt, ve=None, intra=None):
        self._sample_runtime_noises()
        return super().step(model, dt, ve, intra)


def test_native_sampling_rejects_custom_integrator_noise_schedule_without_probing(
    monkeypatch,
):
    model = dn.SingleCompartment(N=1, C=1, integrator=_ExtraNoiseStep)
    model.insert(_NoisyLeak.rename("leak"))
    model.train()
    model.initialize()
    before = _snapshot(model.state_dict_for_checkpoint())

    def forbidden(self):
        raise AssertionError("Sampler invoked an authored integrator spec")

    with monkeypatch.context() as guard:
        guard.setattr(_ExtraNoiseStep, "_functional_spec", forbidden)
        with pytest.raises(
            dn.func.FunctionalizationError, match="canonical integrator"
        ):
            dn.func.sample_runtime_draws(model, dt=DT)
    _assert_packet_equal(model.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize("member", ("step", "_functional_spec", "_call_kernel"))
def test_native_sampling_rejects_instance_integrator_hooks_without_calling_them(
    member, monkeypatch
):
    model = _Cell()
    model.initialize()
    before = _snapshot(model.state_dict_for_checkpoint())

    def forbidden(*args, **kwargs):
        raise AssertionError("Sampler invoked an authored integrator hook")

    monkeypatch.setattr(model.integrator, member, forbidden)
    with pytest.raises(dn.func.FunctionalizationError, match="canonical integrator"):
        dn.func.sample_runtime_draws(model, dt=DT)
    _assert_packet_equal(model.state_dict_for_checkpoint(), before)


@pytest.mark.parametrize(
    "network, member",
    (
        (False, "initialized"),
        (False, "integrator"),
        (True, "built"),
        (True, "populations"),
    ),
)
def test_native_sampling_rejects_masked_runtime_slots_before_executing_descriptors(
    network, member, monkeypatch
):
    model = _Network() if network else _Cell()
    model.initialize(DT) if network else model.initialize()
    before = _snapshot(model.state_dict_for_checkpoint())
    reads = []

    def forbidden(self):
        reads.append(member)
        raise AssertionError("Sampler resolved an authored runtime descriptor")

    with monkeypatch.context() as guard:
        guard.setattr(type(model), member, property(forbidden), raising=False)
        with pytest.raises(dn.func.FunctionalizationError, match="canonical"):
            dn.func.sample_runtime_draws(
                model,
                dt=DT,
                **(
                    {"extra": {"absent": (torch.ones(1), dn.constant(value=1.0))}}
                    if network
                    else {}
                ),
            )
    assert reads == []
    _assert_packet_equal(model.state_dict_for_checkpoint(), before)


def test_native_sampling_accepts_canonical_packed_integrator_without_lowering():
    model = dn.concat_models({"left": _Cell(), "right": _Cell()}, threads=1)
    model.initialize()
    before = _snapshot(model.state_dict_for_checkpoint())
    packet = dn.func.sample_runtime_draws(model, dt=DT, steps=2)
    assert packet.draws == {"noise": None, "stimulation": None}
    assert packet.rng == {"model": {}, "stimulation": {}}
    _assert_packet_equal(model.state_dict_for_checkpoint(), before)
