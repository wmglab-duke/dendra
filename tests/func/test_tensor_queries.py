from __future__ import annotations

import pytest
import torch

import dendra as dn
from dendra.models.mod import pas


def _references():
    return dn.func.PopulationTensors(
        parameters={
            "integrator.mech.mechanisms.hh.gnabar_param": torch.tensor(0.12),
            "integrator.mech.mechanisms.hh.gkbar_param": torch.tensor(0.036),
            "stimulation.intra.waveforms.0.amp": torch.tensor(2.0),
            "stimulation.extra.waveforms.0.amp": torch.tensor(3.0),
            "stimulation.extra.contacts.0.field": torch.tensor([1.0, 2.0]),
            "stimulation.extra.contacts.0.index": torch.tensor(1),
        },
        constants={"dt": torch.tensor(0.01)},
        state={},
        initialization=dn.func.InitializationInput(v_init=torch.tensor([[-65.0]])),
    )


def test_population_tensor_parameter_subsets_are_disjoint_ordered_views():
    tensors = _references()

    assert tuple(tensors.model_parameters) == (
        "integrator.mech.mechanisms.hh.gnabar_param",
        "integrator.mech.mechanisms.hh.gkbar_param",
    )
    assert tuple(tensors.intra_parameters) == ("stimulation.intra.waveforms.0.amp",)
    assert tuple(tensors.extra_parameters) == (
        "stimulation.extra.waveforms.0.amp",
        "stimulation.extra.contacts.0.field",
        "stimulation.extra.contacts.0.index",
    )
    assert {
        *tensors.model_parameters,
        *tensors.intra_parameters,
        *tensors.extra_parameters,
    } == set(tensors.parameters)
    for subset in (
        tensors.model_parameters,
        tensors.intra_parameters,
        tensors.extra_parameters,
    ):
        for name, value in subset.items():
            assert value is tensors.parameters[name]


def test_parameter_search_supports_scopes_and_strict_unique_resolution():
    tensors = _references()
    gnabar_name = "integrator.mech.mechanisms.hh.gnabar_param"

    assert tensors.find_parameters("gnabar") == {
        gnabar_name: tensors.parameters[gnabar_name]
    }
    assert tensors.parameter_name("gnabar", within="model") == gnabar_name
    assert tensors.parameter_name(gnabar_name, within="model") == gnabar_name
    assert (
        tensors.get_parameter("gnabar", within="model")
        is tensors.parameters[gnabar_name]
    )
    assert tuple(tensors.find_parameters("amp", within="intra")) == (
        "stimulation.intra.waveforms.0.amp",
    )
    assert tuple(tensors.find_parameters("amp", within="extra")) == (
        "stimulation.extra.waveforms.0.amp",
    )

    with pytest.raises(KeyError, match="ambiguous"):
        tensors.parameter_name("amp")
    with pytest.raises(KeyError, match="no parameter"):
        tensors.parameter_name("missing", within="model")
    with pytest.raises(ValueError, match="within must be"):
        tensors.find_parameters("amp", within="unknown")
    with pytest.raises(ValueError, match="non-empty string"):
        tensors.find_parameters("")
    with pytest.raises(TypeError, match="must be a string"):
        tensors.find_parameters(["gnabar"])


def test_independent_parameters_clones_every_leaf_and_resolves_partial_names():
    tensors = _references()
    independent = tensors.independent_parameters(
        ("gnabar", "gkbar"),
        within="model",
    )
    trainable_names = {
        tensors.parameter_name("gnabar", within="model"),
        tensors.parameter_name("gkbar", within="model"),
    }

    assert tuple(independent) == tuple(tensors.parameters)
    for name, value in independent.items():
        torch.testing.assert_close(value, tensors.parameters[name])
        assert (
            value.untyped_storage().data_ptr()
            != tensors.parameters[name].untyped_storage().data_ptr()
        )
        assert isinstance(value, torch.nn.Parameter) is (name in trainable_names)
        assert value.requires_grad is (name in trainable_names)

    mutated_name = next(iter(trainable_names))
    source_before = tensors.parameters[mutated_name].clone()
    with torch.no_grad():
        independent[mutated_name].add_(1.0)
    torch.testing.assert_close(tensors.parameters[mutated_name], source_before)

    with pytest.raises(ValueError, match="floating-point or complex"):
        tensors.independent_parameters("index", within="extra")


def test_independent_parameters_created_in_inference_mode_remain_ordinary():
    tensors = _references()
    with torch.inference_mode():
        independent = tensors.independent_parameters("gnabar", within="model")

    assert not torch.is_inference(independent[tensors.parameter_name("gkbar")])
    assert not torch.is_inference(independent[tensors.parameter_name("gnabar")])

    with pytest.raises(ValueError, match="within must be"):
        tensors.independent_parameters(within="unknown")


def test_population_tensor_helpers_preserve_namedtuple_pytree_structure():
    tensors = _references()
    leaves, spec = torch.utils._pytree.tree_flatten(tensors)
    restored = torch.utils._pytree.tree_unflatten(leaves, spec)

    assert isinstance(restored, dn.func.PopulationTensors)
    assert restored._fields == (
        "parameters",
        "constants",
        "state",
        "initialization",
    )
    assert tuple(restored.parameters) == tuple(tensors.parameters)
    for name in tensors.parameters:
        assert restored.parameters[name] is tensors.parameters[name]
    assert restored.initialization.v_init is tensors.initialization.v_init


def test_regional_overrides_are_returned_as_multiple_explicit_matches():
    with dn.ctx(JIT=0, REQUIRE_GRAD=1):
        model = dn.Unmyelinated(
            [1.0],
            L=4.0,
            dx=1.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        ).double()
        model[:, :2].insert(pas, alias="proximal", g=0.001)
        model[:, 2:].insert(pas, alias="distal", g=0.002)
        model.initialize()
        model.train()

    _functional, tensors = dn.func.make_functional(model, dt=0.01)
    matches = tensors.find_parameters("mechanisms.pas.g_", within="model")
    expected = (
        "integrator.mech.mechanisms.pas.g_param",
        "integrator.mech.mechanisms.pas.g_proximal",
        "integrator.mech.mechanisms.pas.g_distal",
    )

    assert tuple(matches) == expected
    with pytest.raises(KeyError, match="ambiguous"):
        tensors.parameter_name("mechanisms.pas.g_", within="model")
    assert (
        tensors.parameter_name("g_proximal", within="model")
        == "integrator.mech.mechanisms.pas.g_proximal"
    )

    independent = tensors.independent_parameters(
        trainable=matches,
        within="model",
    )
    assert (
        tuple(name for name, value in independent.items() if value.requires_grad)
        == expected
    )
