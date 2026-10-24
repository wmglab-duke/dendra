"""Prepared raw-materializer packets preserve ownership and graph lifetimes."""

from collections.abc import Mapping

import pytest
import torch

import dendra as dn
from dendra.func._connection_parameters import ConnectionParameterPlan
from dendra.func._netstim_parameters import NetStimParameterPlan
from tests.func.test_connection_parameter_sources import _connection
from tests.func.test_netstim_parameter_sources import _source

pytestmark = pytest.mark.cpu


@pytest.fixture(autouse=True)
def _context():
    with dn.ctx(
        JIT=0,
        REQUIRE_GRAD=1,
        DEVICE="cpu",
        DTYPE=torch.float64,
        CHECK_PURITY=False,
    ):
        yield


def _plan(kind):
    return (
        ConnectionParameterPlan(_connection())
        if kind == "connection"
        else NetStimParameterPlan(_source(parametrized=True))
    )


class _UnreadableParameters(Mapping):
    def __getitem__(self, key):
        raise AssertionError("A foreign packet accessed parameter values")

    def __iter__(self):
        raise AssertionError("A foreign packet accessed parameter names")

    def __len__(self):
        raise AssertionError("A foreign packet accessed the parameter schema")


@pytest.mark.parametrize("receiver_kind", ("connection", "netstim"))
@pytest.mark.parametrize("donor_kind", ("connection", "netstim"))
def test_foreign_prepared_packets_fail_before_parameter_access_or_materialization(
    receiver_kind, donor_kind, monkeypatch
):
    receiver, donor = _plan(receiver_kind), _plan(donor_kind)
    parameters, constants = donor.extract()
    prepared = donor.prepare(parameters, constants)

    def unexpected_materialization(*args, **kwargs):
        pytest.fail("A foreign packet reached authored materialization")

    monkeypatch.setattr(receiver, "materialize", unexpected_materialization)
    with pytest.raises(dn.func.FunctionalizationError, match="belong.*another plan"):
        receiver.validate_prepared(prepared, _UnreadableParameters())


@pytest.mark.parametrize("kind", ("connection", "netstim"))
def test_inference_preparation_keeps_versioned_inspection_values_and_later_gradients(
    kind,
):
    plan = _plan(kind)
    parameters, constants = plan.extract()
    with torch.inference_mode():
        prepared = plan.prepare(parameters, constants)
    assert all(
        not torch.is_inference(value) and value.grad_fn is None
        for value in prepared.values.values()
    )
    values = plan.validate_prepared(prepared, parameters)
    trainable = tuple(value for value in parameters.values() if value.requires_grad)
    assert trainable
    gradients = torch.autograd.grad(
        sum(value.square().sum() for value in values.values()), trainable
    )
    assert all(torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) for value in gradients)
