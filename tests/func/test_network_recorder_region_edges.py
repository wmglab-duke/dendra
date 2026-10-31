"""Regional Network bindings retain physical coordinates and strict ownership."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch

import dendra as dn
from dendra.func._network_callbacks import make_network_callbacks
from dendra.models.slice import Slice

pytestmark = pytest.mark.cpu


def _schema(value):
    if isinstance(value, dict):
        return {name: _schema(child) for name, child in value.items()}
    return SimpleNamespace(shape=value.shape, dtype=value.dtype, device=value.device)


def _mock_network(*, packed=False, collision=False):
    with dn.ctx(JIT=0, DTYPE=torch.float64, DEVICE="cpu"):
        left = dn.Population(N=2, C=4, dtype=torch.float64)
        right = dn.Population(N=1, C=3, dtype=torch.float64)
        if packed:
            source = dn.concat_models({"child": left, "other": right})
            sources = {"packed": source}
        else:
            left.batch(2)
            sources = {"left": left, "right": right}
    states, plans = {}, {}
    for name, source in sources.items():
        voltage = torch.arange(source.v.numel(), dtype=torch.float64).reshape(
            source.shape
        )
        states[name] = {
            "integrator": {
                "v": voltage,
                "vc": torch.stack((voltage, voltage + 100, voltage + 200), dim=-1),
                "i_membrane": voltage * 1e-6,
            },
            "clock": {"t": torch.tensor(1.25, dtype=torch.float64)},
        }
        layout = [
            SimpleNamespace(public_path=("integrator", field), checkpoint_key=None)
            for field in ("v", "vc", "i_membrane")
        ]
        layout.append(SimpleNamespace(public_path=("clock", "t"), checkpoint_key=None))
        if collision:
            states[name]["mechanism_buffers"] = {
                "integrator": {"vc": voltage.clone()},
                "clock": {"t": torch.tensor(2.5, dtype=torch.float64)},
            }
            layout.extend(
                SimpleNamespace(
                    public_path=("mechanism_buffers", *path),
                    checkpoint_key=".".join(path),
                )
                for path in (("integrator", "vc"), ("clock", "t"))
            )
        plans[name] = SimpleNamespace(
            _population=source,
            shape=source.shape,
            device=source.device(),
            _state_layout=tuple(layout),
        )
    state = {
        "populations": states,
        "clock": {"t": torch.tensor(1.5, dtype=torch.float64)},
    }
    functional = SimpleNamespace(
        device=torch.device("cpu"),
        populations=plans,
        _state_schema=_schema(state),
        extract=lambda: SimpleNamespace(state=state),
    )
    return functional, sources, state


def _observe(functional, recorder, state):
    callbacks = make_network_callbacks(functional, {"trace": recorder})
    _, observed = callbacks._initialize(state)
    return callbacks, observed["trace"]


def test_qualified_vc_aliases_select_compartments_and_keep_every_payload_channel():
    functional, sources, state = _mock_network()
    region = sources["left"][..., [3, 1, 3]]
    fields = ("left.vc", "populations.left.integrator.vc", "left.v", "t", "left.t")
    _, observed = _observe(functional, dn.func.Recorder(fields, region=region), state)
    assert tuple(observed) == fields
    expected = state["populations"]["left"]["integrator"]["vc"][..., [3, 1, 3], :]
    for name in fields[:2]:
        torch.testing.assert_close(observed[name], expected)
        assert observed[name].shape == (*region.shape, 3)
    torch.testing.assert_close(observed["left.v"], expected[..., 0])
    assert observed["t"].ndim == observed["left.t"].ndim == 0
    assert observed["t"].item() == 1.5 and observed["left.t"].item() == 1.25


def test_clock_only_regions_accept_global_and_selected_population_clock_aliases():
    functional, sources, state = _mock_network()
    fields = ("t", "clock.t", "left.t", "populations.left.clock.t")
    _, observed = _observe(
        functional, dn.func.Recorder(fields, region=sources["left"][..., []]), state
    )
    assert tuple(observed) == fields
    assert all(value.ndim == 0 for value in observed.values())
    torch.testing.assert_close(observed["t"], observed["clock.t"])
    torch.testing.assert_close(observed["left.t"], observed["populations.left.clock.t"])


@pytest.mark.parametrize("field", ("right.t", "populations.right.clock.t", "right.v"))
def test_regional_recorder_rejects_foreign_population_even_for_clock(field):
    functional, sources, _ = _mock_network()
    with pytest.raises(
        dn.func.FunctionalizationError, match="same Population|belongs to"
    ):
        make_network_callbacks(
            functional,
            {"trace": dn.func.Recorder(("left.v", field), region=sources["left"][:])},
        )


@pytest.mark.parametrize(
    "field",
    (
        "populations.left.mechanism_buffers.integrator.vc",
        "populations.left.mechanism_buffers.clock.t",
    ),
)
def test_physical_suffixes_do_not_admit_mechanism_state(field):
    functional, sources, _ = _mock_network(collision=True)
    with pytest.raises(dn.func.FunctionalizationError, match="mechanism-local"):
        make_network_callbacks(
            functional, {"trace": dn.func.Recorder(field, region=sources["left"][:])}
        )


def test_regional_binding_retains_ambiguous_alias_rejection_and_explicit_escape():
    functional, sources, state = _mock_network(collision=True)
    region = sources["left"][..., [0, 3]]
    with pytest.raises(dn.func.FunctionalizationError, match="ambiguous"):
        make_network_callbacks(
            functional, {"trace": dn.func.Recorder("left.integrator.vc", region=region)}
        )
    _, observed = _observe(
        functional,
        dn.func.Recorder("populations.left.integrator.vc", region=region),
        state,
    )
    assert observed["populations.left.integrator.vc"].shape == (*region.shape, 3)


@pytest.mark.parametrize("field", ("v", "vc", "i_membrane"))
def test_region_does_not_implicitly_qualify_population_state_names(field):
    functional, sources, _ = _mock_network()
    with pytest.raises(dn.func.FunctionalizationError, match="unsupported selection"):
        make_network_callbacks(
            functional, {"trace": dn.func.Recorder(field, region=sources["left"][:])}
        )


def test_region_rejects_foreign_source_and_mechanism_slice_roots():
    functional, sources, _ = _mock_network()
    foreign = dn.Population(N=2, C=4, dtype=torch.float64)
    with pytest.raises(dn.func.FunctionalizationError, match="exact source Population"):
        make_network_callbacks(
            functional, {"trace": dn.func.Recorder("left.v", region=foreign[:])}
        )
    source = sources["left"]
    physical = source[:]
    mechanism = Slice(
        source, physical.index_spec, root_model=source, module_path=("mech",)
    )
    with pytest.raises(dn.func.FunctionalizationError, match="mechanism submodule"):
        make_network_callbacks(
            functional, {"trace": dn.func.Recorder("left.v", region=mechanism)}
        )


def test_packed_region_requires_parent_root_instead_of_original_child():
    functional, sources, state = _mock_network(packed=True)
    packed = sources["packed"]
    with pytest.raises(dn.func.FunctionalizationError, match="packed|exact source"):
        make_network_callbacks(
            functional,
            {
                "trace": dn.func.Recorder(
                    "packed.v", region=packed.populations["child"][:]
                )
            },
        )
    _, observed = _observe(
        functional, dn.func.Recorder("packed.v", region=packed.child), state
    )
    assert observed["packed.v"].numel() == 8


@pytest.mark.parametrize(
    "options",
    (
        {"node_indices": []},
        {"max_only": True},
        {"partition": [4]},
    ),
)
def test_network_regions_preserve_shared_recorder_option_restrictions(options):
    functional, sources, _ = _mock_network()
    with pytest.raises(dn.func.FunctionalizationError, match="combined"):
        make_network_callbacks(
            functional,
            {"trace": dn.func.Recorder("left.v", region=sources["left"][:], **options)},
        )


def test_bound_network_region_snapshots_indices_without_retaining_slice_or_recorder():
    functional, sources, state = _mock_network()

    def bind():
        with torch.inference_mode():
            region = sources["left"][..., torch.tensor([3, 0, 3])]
            recorder = dn.func.Recorder("left.v", region=region)
            callbacks = make_network_callbacks(functional, {"trace": recorder})
            region.index[-1].fill_(1)
        return callbacks, weakref.ref(region), weakref.ref(recorder)

    callbacks, region_ref, recorder_ref = bind()
    gc.collect()
    assert region_ref() is None and recorder_ref() is None
    bound_region = callbacks._plans[0].callback.region
    assert all(
        not torch.is_tensor(value) or not value.is_inference()
        for value in bound_region.index
    )
    _, observed = callbacks._initialize(state)
    expected = state["populations"]["left"]["integrator"]["v"][..., [3, 0, 3]]
    torch.testing.assert_close(observed["trace"]["left.v"], expected)
