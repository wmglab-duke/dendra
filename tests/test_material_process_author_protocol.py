"""Authoring-contract tests for full-field MaterialProcess classes."""

from __future__ import annotations

import pytest
import torch

from dendra.models.mechanisms import MaterialProcess, Mechanism, State


def _process(cls=MaterialProcess):
    shape = (1, 2)
    return cls(
        name="process",
        celsius=torch.tensor(37.0),
        diameters=torch.ones(shape),
        shape=shape,
        shape_f=shape,
    )


class _PrivateState(State):
    State.STATE("x")
    State.DERIVATIVE("x' = 0.0 * x")


@pytest.mark.parametrize(
    ("declaration_name", "declaration"),
    [
        ("STATE_BUNDLE", lambda: MaterialProcess.STATE_BUNDLE(_PrivateState)),
        ("CARRY", lambda: MaterialProcess.CARRY("scratch")),
        ("ASSIGNED", lambda: MaterialProcess.ASSIGNED("scratch")),
        ("DERIVED_BUFFER", lambda: MaterialProcess.DERIVED_BUFFER("scratch")),
        ("TIMESTEP_BUFFER", lambda: MaterialProcess.TIMESTEP_BUFFER("scratch")),
        ("SAVE_CURRENT", lambda: MaterialProcess.SAVE_CURRENT("i")),
        ("USEION", lambda: MaterialProcess.USEION("na", read=["nai"])),
        (
            "USEMATERIAL",
            lambda: MaterialProcess.USEMATERIAL("pool", read=["amount"]),
        ),
        (
            "NONSPECIFIC_CURRENT",
            lambda: MaterialProcess.NONSPECIFIC_CURRENT("i"),
        ),
        ("EXPLICIT", lambda: MaterialProcess.EXPLICIT("i")),
        ("AFFINE", lambda: MaterialProcess.AFFINE("i")),
        ("NUMERICAL", lambda: MaterialProcess.NUMERICAL("i")),
    ],
)
def test_material_process_rejects_ordinary_mechanism_declarations(
    declaration_name,
    declaration,
):
    with pytest.raises(
        TypeError,
        match=rf"MaterialProcess .* does not support: .*{declaration_name}",
    ):

        class _InvalidProcess(MaterialProcess):
            declaration()

    # A rejected class body must not contaminate the next declaration.
    class _CleanProcess(MaterialProcess):
        pass

    assert _CleanProcess._state == ()
    assert _CleanProcess._carry == ()
    assert _CleanProcess._material == set()


def test_material_process_rejects_declarations_made_through_mechanism_base():
    with pytest.raises(TypeError, match=r"does not support: CARRY"):

        class _InvalidProcess(MaterialProcess):
            Mechanism.CARRY("scratch")


@pytest.mark.parametrize(
    "hook",
    [
        "initial_values",
        "assigned_values",
        "advance",
        "derive_buffers",
        "derive_timestep_buffers",
    ],
)
def test_material_process_rejects_ordinary_mechanism_hooks(hook):
    with pytest.raises(TypeError, match=rf"lifecycle hooks.*{hook}"):
        type(
            "_InvalidProcess",
            (MaterialProcess,),
            {hook: lambda *args, **kwargs: {}},
        )


@pytest.mark.parametrize(
    ("hook", "args", "replacement"),
    [
        ("initial_values", (torch.zeros(1, 2), {}), "configure_process"),
        ("assigned_values", (torch.zeros(1, 2), {}), "advance_materials"),
        ("advance", (torch.zeros(1, 2), 0.1, {}), "advance_materials"),
        ("derive_buffers", (), "configure_process"),
        ("derive_timestep_buffers", (0.1,), "set_dt"),
    ],
)
def test_material_process_direct_ordinary_lifecycle_calls_fail_actionably(
    hook,
    args,
    replacement,
):
    process = _process()
    with pytest.raises(
        RuntimeError,
        match=rf"does not participate in Mechanism\.{hook}.*{replacement}",
    ):
        getattr(process, hook)(*args)


def test_material_process_configuration_is_idempotent_and_helpers_are_private():
    class _ConfiguredProcess(MaterialProcess):
        def configure_process(self, population=None):
            self.configuration = tuple(population.shape)

        def advance_materials(self, dt):
            return dt

    process = _process(_ConfiguredProcess)
    population = type("PopulationShape", (), {"shape": (1, 2)})()
    resolver = {}.__getitem__

    process._bind_materials(resolver, population=population)
    first = process.configuration
    process._bind_materials(resolver, population=population)

    assert process.configuration == first == (1, 2)
    assert not hasattr(MaterialProcess, "material_field")
    assert not hasattr(MaterialProcess, "set_material_field")


def test_material_process_accepts_only_canonical_scheduler_phases():
    for phase in MaterialProcess._SUPPORTED_PHASES:

        class _ValidProcess(MaterialProcess):
            MaterialProcess.PHASE(phase)

        assert _ValidProcess._material_process_phase == phase

    with pytest.raises(ValueError, match="phase must be one of"):

        class _UnknownPhase(MaterialProcess):
            MaterialProcess.PHASE("after_everything")


def test_material_process_parameter_and_lifecycle_surface_remains_available():
    class _ConfiguredProcess(MaterialProcess):
        MaterialProcess.GLOBAL(rate=1.0)
        MaterialProcess.RANGE(scale=2.0)
        MaterialProcess.BATCH(offset=0.0)
        MaterialProcess.METHOD("custom", stages=2)
        MaterialProcess.PHASE("post_local")

        def configure_process(self, population=None):
            return population

        def set_dt(self, dt):
            self.dt = dt

        def advance_materials(self, dt):
            return dt

    assert _ConfiguredProcess._material_process_method == "custom"
    assert _ConfiguredProcess._material_process_method_kwargs == {"stages": 2}
    assert _ConfiguredProcess._material_process_phase == "post_local"
    assert not hasattr(MaterialProcess, "bind_materials")
    assert hasattr(MaterialProcess, "_bind_materials")
