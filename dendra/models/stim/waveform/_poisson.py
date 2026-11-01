"""Shared Poisson schedule recipes with explicit independent random draws.

Schedule construction and evaluation are pure tensor operations.  The host
adapter alone owns a Torch generator and preserves the native scalar draw order.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

POISSON_INVOCATION_STACK = "_poisson_invocation_stack"
POISSON_INVOCATION_POSITION = "_poisson_invocation_position"
POISSON_EVENT_SLOTS = "_poisson_event_slots"


def poisson_invocation_schedule(schedules, position):
    """Select one invocation and return its next explicit position.

    The position is a one-element integer tensor.  Copying that index keeps
    autograd's saved index independent of any caller-owned scratch update.
    Neither input is mutated, and the returned position has no persistent
    ownership or random-number-generator state.
    """
    selected = torch.index_select(schedules, 0, position.clone()).squeeze(0)
    return selected, position + 1


def configure_poisson_invocations(module, *, event_slots=None):
    """Install invocation operands on an isolated waveform execution copy.

    The schedule stack defaults to an empty tensor so cold profiling and snapshot
    evaluation retain the ordinary fixed-schedule path.  An execution adapter
    supplies the stack and a fresh position tensor for each evaluation.  These
    scratch buffers never become part of native checkpoints.

    ``event_slots`` can retain a statically known native schedule length when
    an execution adapter pads the schedule.  The empty-schedule infinity
    sentinel occupies one event slot.
    """
    if event_slots is not None and (type(event_slots) is not int or event_slots < 1):
        raise ValueError("Poisson event slots must be a positive integer or None.")
    if POISSON_INVOCATION_STACK not in module._buffers:
        module.register_buffer(
            POISSON_INVOCATION_STACK,
            module._spike_times.new_empty((0, *module._spike_times.shape)),
            persistent=False,
        )
    if POISSON_INVOCATION_POSITION not in module._buffers:
        module.register_buffer(
            POISSON_INVOCATION_POSITION,
            torch.zeros((1,), dtype=torch.int64, device=module._spike_times.device),
            persistent=False,
        )
    setattr(module, POISSON_EVENT_SLOTS, event_slots)


def poisson_interval_from_uniform(draw, *, interval, noise):
    """Return a detached float64 interval with native scalar arithmetic.

    Native Poisson sampling evaluates the exponential in the uniform draw's
    dtype, then uses its Python scalar value for the mixture and accumulated
    onset.  Promoting after the exponential preserves that precision boundary.
    """
    if noise == 0.0:
        return torch.full_like(draw, interval, dtype=torch.float64).detach()
    exponential = (-draw.detach().log() * interval).to(torch.float64)
    return ((1.0 - noise) * interval + noise * exponential).detach()


def poisson_schedule_from_draws(
    draws, *, interval, start, noise, off, dtype, device=None
):
    """Construct a fixed-size schedule from a one-dimensional uniform draw.

    The first onset is ``start``.  Each emitted onset owns an interval draw,
    including the last event.  Slots following ``off`` contain infinity, and
    an empty draw vector produces the historical single infinity sentinel.
    Uniforms and resulting onsets are detached from autograd; the wrapped
    waveform's own parameters retain their usual derivatives.
    """
    draws = draws.detach()
    if draws.numel() == 0:
        return torch.full((1,), torch.inf, dtype=dtype, device=device)
    current = draws.new_full((), start, dtype=torch.float64)
    active = draws.new_ones((), dtype=torch.bool)
    intervals = poisson_interval_from_uniform(draws, interval=interval, noise=noise)
    onsets = []
    for delta in intervals.unbind(0):
        active = active & (current < off)
        onsets.append(torch.where(active, current, torch.inf))
        current = current + delta
    return torch.stack(onsets).to(device=device, dtype=dtype).detach()


def poisson_invocation_event_count(draws, *, interval, start, noise, off):
    """Return the native child-call count for one invocation's uniform draws.

    Native schedule construction stops on its float64 clock before converting
    onsets to the waveform dtype. Counting in that clock's dtype distinguishes
    a genuinely terminated schedule from finite onsets which overflow during
    a float32 conversion. An empty schedule still invokes its child once for
    the infinity sentinel.
    """
    schedule = poisson_schedule_from_draws(
        draws,
        interval=interval,
        start=start,
        noise=noise,
        off=off,
        dtype=torch.float64,
        device=draws.device,
    )
    return torch.isfinite(schedule).sum().clamp_min(1)


def native_poisson_schedule_from_draws(waveform, draws, *, dtype=None, device=None):
    """Own the native variable-length schedule for one validated invocation.

    Count events on the float64 construction clock before converting to the
    waveform's buffer dtype. A finite native onset may overflow that dtype and
    must still retain its child invocation. Empty schedules retain the native
    single-infinity sentinel and registered intracellular singleton axes.
    Native banks may supply their receiving dtype/device before lazy waveform
    preparation moves the authored module there.
    """
    recipe = {
        "interval": waveform.interval,
        "start": waveform.start,
        "noise": waveform.noise,
        "off": waveform.off,
    }
    count = int(poisson_invocation_event_count(draws, **recipe).item())
    schedule = poisson_schedule_from_draws(
        draws,
        **recipe,
        dtype=waveform._spike_times.dtype if dtype is None else dtype,
        device=waveform._spike_times.device if device is None else device,
    )[:count]
    return schedule.reshape(schedule.shape + (1,) * waveform._spike_time_trailing_dims)


class _NativePoissonInvocationReplay:
    """Consume validated schedules in the native bank's physical call order.

    The eager caller temporarily binds this object to its waveform nodes.
    Schedules are already owned and shaped for their receiving nodes. No
    sampler, class conversion, or registered scratch buffer participates in
    consumption. Publication retains each physical waveform's latest schedule.
    """

    def __init__(self, invocations):
        self.invocations = tuple(invocations)
        self.position = 0

    def consume(self, waveform):
        if self.position >= len(self.invocations):
            raise ValueError(
                "Poisson replay has no remaining invocation for this waveform."
            )
        expected, schedule = self.invocations[self.position]
        if waveform is not expected:
            raise ValueError(
                "Poisson replay invocation order does not match the prepared "
                "waveform profile."
            )
        waveform._spike_times = schedule
        self.position += 1

    def finish(self):
        if self.position != len(self.invocations):
            raise ValueError(
                "Poisson replay did not consume its complete waveform "
                "invocation profile."
            )


def sample_poisson_draws(*, interval, n, start, noise, off, generator, dtype=None):
    """Sample native scalar uniforms, returning a padded draw and event count.

    ``n=None`` retains native host generation up to a finite ``off``.  Finite
    ``n`` yields a vector of that length, filling unused slots with 0.5 without
    consuming randomness.  ``noise=0`` never advances the supplied generator.
    """
    if dtype is None:
        dtype = torch.get_default_dtype()
    device = generator.device
    values = []
    current = start
    count = 0
    with torch.no_grad():
        while current < off and (n is None or count < n):
            if noise == 0.0:
                draw = torch.full((), 0.5, dtype=dtype, device=device)
                delta = interval
            else:
                draw = torch.rand((), generator=generator, device=device, dtype=dtype)
                delta = poisson_interval_from_uniform(
                    draw, interval=interval, noise=noise
                ).item()
            values.append(draw)
            current += delta
            count += 1
        if n is not None:
            values.extend(
                torch.full((), 0.5, dtype=dtype, device=device)
                for _ in range(n - count)
            )
        draws = (
            torch.stack(values)
            if values
            else torch.empty((0,), dtype=dtype, device=device)
        )
    return draws, count


def sample_poisson_draws_from_state(state, *, device="cpu", **kwargs):
    """Advance an explicit RNG frame without reading or changing global RNG."""
    generator = torch.Generator(device=device)
    generator.set_state(state.detach().clone())
    draws, count = sample_poisson_draws(generator=generator, **kwargs)
    return draws, count, generator.get_state().clone()


def evaluate_poisson_schedule(waveform_fn, t, schedule):
    """Evaluate each event independently, suppressing inactive padded slots."""
    responses = []
    for onset in schedule.unbind(dim=0):
        active = torch.isfinite(onset)
        safe_onset = torch.where(active, onset, 0.0)
        value = waveform_fn(t - safe_onset.unsqueeze(-1))
        responses.append(torch.where(active.unsqueeze(-1), value, 0.0))
    return torch.stack(responses, dim=0).sum(dim=0)


class _PendingPoissonRNG:
    """One deferred restoration shared by every waveform using a stream."""

    def __init__(self, generator, state):
        self.generator = generator
        self.state = state.detach().clone()


def defer_poisson_generator_restore(waveforms, state):
    """Carry a stream checkpoint until native evaluation first consumes it.

    Functional commits use this for global Torch streams: committing explicit
    state must not itself reset ambient randomness. All aliases receive one
    shared frame so subsequent waveform calls do not rewind one another.
    """
    waveforms = tuple(waveforms)
    if not waveforms:
        return
    generator = waveforms[0].generator
    if any(waveform.generator is not generator for waveform in waveforms):
        raise ValueError("A deferred Poisson RNG frame requires one shared generator.")
    _validate_poisson_rng_state(generator, state)
    frame = _PendingPoissonRNG(generator, state)
    for waveform in waveforms:
        waveform._functional_poisson_rng = frame


def restore_pending_poisson_rng(waveform):
    """Consume a committed stream frame once before native schedule sampling."""
    frame = vars(waveform).get("_functional_poisson_rng")
    if (
        frame is not None
        and frame.generator is waveform.generator
        and frame.state is not None
    ):
        waveform.generator.set_state(frame.state)
        frame.state = None


def _poisson_waveforms(owner):
    # Intra keeps its waveforms in a plain Python list; mechanism-owned
    # injections can also have been accepted before their runtime registration.
    # Visit the stable authored registries before the registered module graph.
    from .core import _randomized_poisson

    roots = []
    for owner_name, child_owner in owner.named_modules():
        for registry in ("injections", "mechanism_injections"):
            for index, spec in enumerate(getattr(child_owner, registry, ())):
                if (
                    isinstance(spec, (tuple, list))
                    and spec
                    and isinstance(spec[0], torch.nn.Module)
                ):
                    name = ".".join(
                        part for part in (owner_name, registry, str(index)) if part
                    )
                    roots.append((name, spec[0]))
    roots.append(("", owner))
    result = {}
    seen = set()
    for root_name, root in roots:
        if not isinstance(root, torch.nn.Module):
            continue
        for name, module in root.named_modules():
            if isinstance(module, _randomized_poisson) and id(module) not in seen:
                seen.add(id(module))
                result[".".join(part for part in (root_name, name) if part)] = module
    return result


def _poisson_recipe(waveform):
    return (
        waveform.interval,
        waveform.n,
        waveform.start,
        waveform.noise,
        waveform.off,
        waveform._spike_time_trailing_dims,
    )


def _validate_poisson_rng_state(generator, state):
    if (
        not torch.is_tensor(state)
        or state.dtype != torch.uint8
        or state.device.type != "cpu"
        or state.ndim != 1
    ):
        raise ValueError("Poisson RNG checkpoints require CPU byte vectors.")
    validator = torch.Generator(device=generator.device)
    validator.set_state(state.detach().clone())
    return validator.get_state().clone()


def _pending_poisson_state(waveform):
    frame = vars(waveform).get("_functional_poisson_rng")
    if frame is not None and frame.generator is waveform.generator:
        return frame.state
    return None


def poisson_generator_state(waveform):
    """Return the logical stream frame, including a deferred functional commit."""
    state = _pending_poisson_state(waveform)
    return (waveform.generator.get_state() if state is None else state).detach().clone()


def _is_default_poisson_generator(generator):
    if generator is torch.default_generator:
        return True
    return generator.device.type == "cuda" and any(
        generator is default for default in torch.cuda.default_generators
    )


def poisson_state_dict_for_checkpoint(owner):
    """Snapshot owned randomized schedules and one RNG frame per physical stream."""
    modules = {}
    streams = []
    stream_indices = {}
    for name, waveform in _poisson_waveforms(owner).items():
        generator = waveform.generator
        pending_state = _pending_poisson_state(waveform)
        state = generator.get_state() if pending_state is None else pending_state
        if id(generator) not in stream_indices:
            stream_indices[id(generator)] = len(streams)
            streams.append(
                {
                    "device": str(generator.device),
                    "state": state.detach().clone(),
                    "pending": pending_state is not None,
                }
            )
        stream = stream_indices[id(generator)]
        if not torch.equal(state, streams[stream]["state"]):
            raise ValueError(
                "Shared Poisson generators have inconsistent pending frames."
            )
        modules[name] = {
            "recipe": _poisson_recipe(waveform),
            "stream": stream,
            "schedule": waveform._spike_times.detach().clone(),
        }
    return {"version": 1, "modules": modules, "streams": tuple(streams)}


def prepare_poisson_state_from_checkpoint(owner, payload):
    """Validate every schedule, frame, and alias before changing runtime state."""
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or payload.get("version") != 1:
        raise ValueError("Unsupported Poisson stimulation checkpoint payload.")
    modules = payload.get("modules")
    streams = payload.get("streams")
    if not isinstance(modules, Mapping) or not isinstance(streams, (tuple, list)):
        raise TypeError(
            "Poisson checkpoint modules and streams must be mappings and sequences."
        )
    expected = poisson_state_dict_for_checkpoint(owner)
    waveforms = _poisson_waveforms(owner)
    if set(modules) != set(expected["modules"]) or len(streams) != len(
        expected["streams"]
    ):
        raise ValueError(
            "Poisson checkpoint waveform or RNG topology does not match this model."
        )
    prepared_modules = []
    stream_waveforms = [[] for _ in streams]
    for name, waveform in waveforms.items():
        candidate = modules[name]
        reference = expected["modules"][name]
        if not isinstance(candidate, Mapping):
            raise TypeError(f"Poisson checkpoint waveform {name!r} must be a mapping.")
        stream = candidate.get("stream")
        if (
            type(stream) is not int
            or stream != reference["stream"]
            or tuple(candidate.get("recipe", ())) != reference["recipe"]
        ):
            raise ValueError(
                f"Poisson checkpoint recipe or RNG aliases differ for {name!r}."
            )
        schedule = candidate.get("schedule")
        trailing_dims = waveform._spike_time_trailing_dims
        if (
            not torch.is_tensor(schedule)
            or not torch.is_floating_point(schedule)
            or schedule.ndim != trailing_dims + 1
            or schedule.shape[0] < 1
            or any(size != 1 for size in schedule.shape[1:])
            or (waveform.n is not None and schedule.shape[0] > max(waveform.n, 1))
        ):
            raise ValueError(
                f"Poisson checkpoint schedule shape or dtype is invalid for {name!r}."
            )
        flat = schedule.detach().reshape(-1)
        if (
            bool(torch.isnan(flat).any())
            or bool(torch.isneginf(flat).any())
            or bool((flat[1:] < flat[:-1]).any())
        ):
            raise ValueError(f"Poisson checkpoint schedule is invalid for {name!r}.")
        prepared_modules.append(
            (waveform, schedule.detach().to(waveform._spike_times).clone())
        )
        stream_waveforms[stream].append(waveform)
    prepared_streams = []
    for index, (candidate, aliases) in enumerate(zip(streams, stream_waveforms)):
        if not isinstance(candidate, Mapping):
            raise TypeError("Poisson checkpoint RNG entries must be mappings.")
        if (
            candidate.get("device") != expected["streams"][index]["device"]
            or type(candidate.get("pending", False)) is not bool
        ):
            raise ValueError(
                "Poisson checkpoint RNG device or pending policy is invalid."
            )
        state = _validate_poisson_rng_state(
            aliases[0].generator, candidate.get("state")
        )
        prepared_streams.append(
            (tuple(aliases), state, candidate.get("pending", False))
        )
    return tuple(prepared_modules), tuple(prepared_streams)


def apply_prepared_poisson_state(prepared, *, defer_global=False):
    """Restore a validated checkpoint while retaining physical generator aliases."""
    if prepared is None:
        return
    modules, streams = prepared
    for waveform, schedule in modules:
        waveform._spike_times = schedule
    for aliases, state, pending in streams:
        generator = aliases[0].generator
        if pending or (defer_global and _is_default_poisson_generator(generator)):
            defer_poisson_generator_restore(aliases, state)
        else:
            generator.set_state(state)
            for waveform in aliases:
                vars(waveform).pop("_functional_poisson_rng", None)


def snapshot_poisson_rng_lifecycle(owner):
    """Retain actual stream bytes and pending-frame bindings for rollback."""
    waveforms = tuple(_poisson_waveforms(owner).values())
    streams = {id(waveform.generator): waveform.generator for waveform in waveforms}
    return (
        tuple(
            (generator, generator.get_state().clone()) for generator in streams.values()
        ),
        tuple(
            (waveform, vars(waveform).get("_functional_poisson_rng"))
            for waveform in waveforms
        ),
    )


def restore_poisson_rng_lifecycle(snapshot):
    """Restore an enclosing transaction's streams without replacing handles."""
    streams, waveforms = snapshot
    for generator, state in streams:
        generator.set_state(state)
    for waveform, frame in waveforms:
        if frame is None:
            vars(waveform).pop("_functional_poisson_rng", None)
        else:
            waveform._functional_poisson_rng = frame


def coalesce_pending_poisson_rng(owner):
    """Restore one pending holder for aliases spanning Population boundaries."""
    groups = {}
    for waveform in _poisson_waveforms(owner).values():
        groups.setdefault(id(waveform.generator), []).append(waveform)
    prepared = []
    for waveforms in groups.values():
        states = [
            state
            for waveform in waveforms
            if (state := _pending_poisson_state(waveform)) is not None
        ]
        if states:
            if any(not torch.equal(state, states[0]) for state in states[1:]):
                raise ValueError(
                    "Shared Poisson generators have inconsistent pending frames."
                )
            prepared.append((waveforms, states[0]))
    for waveforms, state in prepared:
        defer_poisson_generator_restore(waveforms, state)


def validate_poisson_checkpoint_stream_aliases(populations, payloads):
    """Preflight shared stream bytes across all Population checkpoint sections."""
    seen = {}
    schedules = {}
    for name, population in populations.items():
        payload = payloads[name]
        if not isinstance(payload, Mapping):
            continue
        stochastic = payload.get("stochastic", {})
        if not isinstance(stochastic, Mapping):
            continue
        prepared = prepare_poisson_state_from_checkpoint(
            population, stochastic.get("poisson")
        )
        if prepared is None:
            continue
        for waveform, schedule in prepared[0]:
            previous = schedules.setdefault(id(waveform), schedule)
            if not torch.equal(schedule, previous):
                raise ValueError(
                    "Shared Poisson checkpoint waveform schedules differ across Populations."
                )
        for aliases, state, pending in prepared[1]:
            key = id(aliases[0].generator)
            if key in seen:
                previous_state, previous_pending = seen[key]
                if pending != previous_pending or not torch.equal(
                    state, previous_state
                ):
                    raise ValueError(
                        "Shared Poisson checkpoint RNG frames differ across Populations."
                    )
            else:
                seen[key] = state, pending
