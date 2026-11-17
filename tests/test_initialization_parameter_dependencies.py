"""Exercise native initialization writes through parameter owner dependencies."""

import pytest
import torch

import dendra as dn
from dendra.models.core import Population
from dendra.models.mechanisms import Mechanism

PREFIX = "parameters.integrator.mech.mechanisms"


class _Conductance(Mechanism):
    Mechanism.GLOBAL(gain=1.0)


class _Workspace(Mechanism):
    Mechanism.GLOBAL(gain=1.0)
    Mechanism.DERIVED_BUFFER("double_gain")

    def derive_buffers(self):
        return {"double_gain": self.gain * 2}


class _Replace(torch.nn.Module):
    def forward(self, replacement):
        return (replacement,)


def _population(*, consumer=_Conductance):
    model = dn.SingleCompartment(N=1, C=2, dtype=torch.float64)
    model.insert(_Conductance.rename("writer"))
    model.insert(consumer.rename("consumer"))
    return model.eval()


def _register_write(model):
    model.register_post_initialize_transform(
        "replace_gain",
        _Replace(),
        writes=(f"{PREFIX}.writer.gain_param",),
        inputs={"replacement": torch.tensor(3.0, dtype=torch.float64)},
    )


@pytest.mark.cpu
def test_local_initialization_parameter_write_refreshes_only_its_owner(monkeypatch):
    with dn.ctx(JIT=0), torch.inference_mode():
        model = _population()
        model.initialize()
        _register_write(model)

        touched = []
        original = Population._rematerialize_parameter_owner

        def refresh(cls, owner, **kwargs):
            touched.append(owner)
            return original(owner, **kwargs)

        monkeypatch.setattr(
            Population, "_rematerialize_parameter_owner", classmethod(refresh)
        )
        model.initialize()

    assert touched == [model.mech.writer]
    assert model.mech.writer.gain.item() == 3.0
    assert model.mech.consumer.gain.item() == 1.0


@pytest.mark.cpu
@pytest.mark.parametrize("shared_storage_only", [False, True])
def test_initialization_write_refreshes_consumers_of_aliased_parameters(
    shared_storage_only,
):
    with dn.ctx(JIT=0), torch.inference_mode():
        model = _population(consumer=_Workspace)
        model.initialize()
        _register_write(model)
        model.initialize()
        assert model.mech.consumer.gain.item() == 1.0
        shared = model.mech.writer.gain_param
        shared.fill_(1.0)
        model.mech.consumer.gain_param = (
            torch.nn.Parameter(shared.detach(), requires_grad=False)
            if shared_storage_only
            else shared
        )
        model.initialize()

    assert model.mech.writer.gain.item() == 3.0
    assert model.mech.consumer.gain.item() == 3.0
    torch.testing.assert_close(
        model.mech.consumer.double_gain,
        torch.full_like(model.mech.consumer.double_gain, 6.0),
        rtol=0,
        atol=0,
    )


class _ReferencedWorkspace(Mechanism):
    Mechanism.DERIVED_BUFFER("double_gain")

    @property
    def shared_gain(self):
        return torch.ones_like(self.diam)

    def derive_buffers(self):
        return {"double_gain": self.shared_gain * 2}


@pytest.mark.cpu
def test_opaque_cross_mechanism_read_preserves_full_refresh():
    with dn.ctx(JIT=0), torch.inference_mode():
        model = _population(consumer=_ReferencedWorkspace)
        model.initialize()
        _register_write(model)
        model.initialize()
        model.mech.writer.gain_param.fill_(1.0)
        model.mech.consumer.setreference("shared_gain", lambda: model.mech.writer.gain)
        model.initialize()

    torch.testing.assert_close(
        model.mech.consumer.double_gain,
        torch.full_like(model.mech.consumer.double_gain, 6.0),
        rtol=0,
        atol=0,
    )


@pytest.mark.cpu
def test_initialization_refresh_failure_restores_raw_and_effective_values(monkeypatch):
    with dn.ctx(JIT=0), torch.inference_mode():
        model = _population()
        model.initialize()
        _register_write(model)

        original = Population._rematerialize_parameter_owner
        attempts = 0

        def fail_once(cls, owner, **kwargs):
            nonlocal attempts
            original(owner, **kwargs)
            if owner is model.mech.writer:
                attempts += 1
                if attempts == 1:
                    raise RuntimeError("injected parameter materialization failure")

        monkeypatch.setattr(
            Population, "_rematerialize_parameter_owner", classmethod(fail_once)
        )
        with pytest.raises(
            RuntimeError, match="injected parameter materialization failure"
        ):
            model.initialize()
        assert not model.initialized
        assert not model.integrator.initialized
        assert model.mech.writer.gain_param.item() == 1.0
        assert model.mech.writer.gain.item() == 1.0

        model.initialize()
        assert model.initialized
        assert model.mech.writer.gain.item() == 3.0


@pytest.mark.cpu
def test_initialization_write_refreshes_independently_wrapped_overlapping_storage():
    import numpy as np

    with dn.ctx(JIT=0), torch.inference_mode():
        model = _population(consumer=_Workspace)
        model.initialize()
        backing = np.ones(3, dtype=np.float64)
        writer = torch.from_numpy(backing).as_strided((), (), storage_offset=1)
        reader = torch.from_numpy(backing[1:2]).reshape(())
        model.mech.writer.gain_param = torch.nn.Parameter(writer, requires_grad=False)
        model.mech.consumer.gain_param = torch.nn.Parameter(reader, requires_grad=False)
        _register_write(model)
        model.initialize()

    assert model.mech.consumer.gain.item() == 3.0
    torch.testing.assert_close(
        model.mech.consumer.double_gain,
        torch.full_like(model.mech.consumer.double_gain, 6.0),
        rtol=0,
        atol=0,
    )


class _IonReader(_Conductance):
    Mechanism.USEION("na", read=["ena"])


class _ReversalSource(torch.nn.Module):
    def forward(self):
        return torch.tensor(50.0, dtype=torch.float64)


@pytest.mark.cpu
def test_ion_initialization_callback_retains_full_refresh():
    with dn.ctx(JIT=0), torch.inference_mode():
        model = dn.SingleCompartment(N=1, C=2, dtype=torch.float64).eval()
        model.insert(_IonReader.rename("writer"))
        model.insert(_Workspace.rename("consumer"))
        model.build()
        consumer = model.mech.consumer
        source = _ReversalSource()

        def update_parameter(_source, _arguments, _output):
            consumer.gain_param.fill_(7)

        source.register_forward_hook(update_parameter)
        model.equilibria(ena=source)
        _register_write(model)
        model.initialize()

    assert model.mech.writer.gain.item() == 3.0
    assert consumer.gain.item() == 7.0
    torch.testing.assert_close(
        consumer.double_gain,
        torch.full_like(consumer.double_gain, 14.0),
        rtol=0,
        atol=0,
    )


class _PopulationWithWorkspace(dn.SingleCompartment):
    def make_contiguous(self):
        super().make_contiguous()
        if getattr(self, "mech", None) is not None and "writer" in self.mech.mechanisms:
            self.celsius.copy_(self.mech.writer.gain_param)


@pytest.mark.cpu
def test_initialization_keeps_custom_population_workspace_refresh():
    with dn.ctx(JIT=0), torch.inference_mode():
        model = _PopulationWithWorkspace(N=1, C=2, dtype=torch.float64).eval()
        model.insert(_Conductance.rename("writer"))
        model.build()
        _register_write(model)
        model.initialize()

    assert model.mech.writer.gain.item() == 3.0
    assert model.celsius.item() == 3.0
