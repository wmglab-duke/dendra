"""Shared cold entry points for Population and Network lowering."""

from __future__ import annotations

from dendra.models.core import Population, _validate_time_scalar
from dendra.models.networks.net import Network

from ._network import FunctionalNetwork, NetworkTensors
from ._population import FunctionalPopulation
from ._population import make_functional as _make_functional_population
from ._types import PopulationTensors


def _lower_network(network, *, dt=None, extra=None, pending_capacity=None):
    """Lower a typed Network and check an optional normalized timestep."""
    plan = FunctionalNetwork(network, extra=extra, pending_capacity=pending_capacity)
    tensors = plan.extract()
    if dt is not None and dt != plan.dt:
        raise ValueError(
            "dt must match the initialized Network timestep; initialize and compile again to change it"
        )
    return plan, tensors


def make_functional(
    population: Population | Network,
    *,
    dt: float,
    extra=None,
    pending_capacity=None,
    _bind_owned_cache=True,
    _retain_initialization_materializations=False,
    _network_stimulation_rng_scope=False,
) -> (
    tuple[FunctionalPopulation, PopulationTensors]
    | tuple[FunctionalNetwork, NetworkTensors]
):
    """Lower an initialized Population or Network into explicit tensor execution.

    Parameters
    ----------
    population : Population or Network
        Initialized compatible source model. Unsupported hooks/configurations raise
        FunctionalizationError; lowering does not initialize or advance the source.
    dt : float
        Positive solver timestep in ms. A Network must already use that timestep.
    extra : stimulation specification or mapping, optional
        Extracellular field/waveform pair or contact sequence; Networks use
        Population-name mappings. Stimulation Tensor identities and aliases are
        retained in the returned source mappings.
    pending_capacity : int or mapping, optional
        Network event-calendar capacities. Overflow raises during host execution;
        retry from the input boundary with a larger-capacity plan.

    Returns
    -------
    functional : FunctionalPopulation or FunctionalNetwork
        Immutable model plan accepting explicit sources and state.
    tensors : PopulationTensors or NetworkTensors
        Raw parameters, constants, runtime state and model-specific initialization
        inputs. Population raw parameters retain native Tensor references; Network
        extraction owns detached snapshots preserving aliases.

    Notes
    -----
    Use ``tensors.independent_parameters(...)`` for independent optimization leaves.
    Bind raw sources with ``functional.bind(tensors)`` and execute with :func:`run`,
    or use combined preparation/transition methods inside tensor transforms.
    :func:`make_functional_network` is the Network-specific entry point using its
    initialized timestep without a separate dt argument.
    """
    if isinstance(population, Network):
        if (
            _bind_owned_cache is not True
            or _retain_initialization_materializations is not False
            or _network_stimulation_rng_scope is not False
        ):
            raise TypeError(
                "Population-only lowering options cannot be used with a Network"
            )
        dt_value = _validate_time_scalar(dt, name="dt", positive=True)
        return _lower_network(
            population, dt=dt_value, extra=extra, pending_capacity=pending_capacity
        )
    if pending_capacity is not None:
        raise ValueError("pending_capacity applies only to Network lowering")
    return _make_functional_population(
        population,
        dt=dt,
        extra=extra,
        _bind_owned_cache=_bind_owned_cache,
        _retain_initialization_materializations=(
            _retain_initialization_materializations
        ),
        _network_stimulation_rng_scope=_network_stimulation_rng_scope,
    )


__all__ = ["make_functional"]
