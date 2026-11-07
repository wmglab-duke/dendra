"""Host-only Recorder persistence around completed numerical execution spans."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from ._callbacks import FunctionalCallbacks, _BoundCadencedRecorder, _CallbackCollector
from ._types import FunctionalizationError

_DISCARD_PUBLICATION = ContextVar("dendra_discard_recording_publication", default=False)


@contextmanager
def _discard_recording_publication():
    """Keep numerical Recorder plans and spans while discarding host publication."""
    token = _DISCARD_PUBLICATION.set(True)
    try:
        yield
    finally:
        _DISCARD_PUBLICATION.reset(token)


class _DiscardRecordingSession:
    """Keep bounded collection without opening or validating an I/O session.

    Warm-up validates the callback binding metadata, but does not qualify file
    writer availability or native writer state. Ordinary execution retains the
    real session's checks. No recorder, queue or history is retained or changed.
    """

    def append(self, values, *, steps, positions):
        pass

    def finish(self):
        pass

    def abort(self):
        pass


def _storage_bindings(callbacks):
    # Preserve validation of unsupported public objects without invoking their
    # attributes before the normal callback type/owner checks.
    return callbacks._storage if isinstance(callbacks, FunctionalCallbacks) else {}


@contextmanager
def _recording_sessions(callbacks):
    """Own one host invocation's storage outside numerical checkpoint replay."""
    storage = _storage_bindings(callbacks)
    if not storage:
        yield {}
        return
    from dendra.models._recorder_storage import _RecorderHDF5Session

    discard = _DISCARD_PUBLICATION.get()
    sessions = {}
    try:
        for name, (recorder, queue, path, interval, states) in storage.items():
            if (
                recorder.queue is not queue
                or recorder.hdf5_path != path
                or recorder.cache_every != interval
                or tuple(recorder.states) != states
                or not recorder.cache_with_hdf5
            ):
                raise FunctionalizationError(
                    "HDF5 Recorder storage changed after binding; bind a new callback collection"
                )
            sessions[name] = (
                _DiscardRecordingSession()
                if discard
                else _RecorderHDF5Session(recorder)
            )
        yield sessions
        for session in sessions.values():
            session.finish()
    except BaseException:
        for session in sessions.values():
            session.abort()
        raise


def _host_recording(function):
    """Keep file sessions outside numerical functions and checkpoint replay."""

    @wraps(function)
    def execute(*args, **kwargs):
        with _recording_sessions(kwargs.get("callbacks")) as sessions:
            if not sessions:
                return function(*args, **kwargs)
            return function(*args, **kwargs, _recording_sessions=sessions)

    return execute


def _require_recording_host(callbacks):
    if _storage_bindings(callbacks):
        raise FunctionalizationError(
            "HDF5 recording requires dn.func.run or a dn.compile runner; pure rollout kernels return tensors only"
        )


def _recording_span(callbacks, span):
    storage = _storage_bindings(callbacks)
    return min(span, *(binding[3] for binding in storage.values())) if storage else span


class _RecordingCollector(_CallbackCollector):
    def __init__(self, callbacks, callback_state, selections, sessions):
        super().__init__(callbacks, selections)
        self._sessions = sessions
        self.storage_names = frozenset(sessions)
        self._strides = {
            name: recorder.save_every
            for name, recorder in selections
            if type(recorder) is _BoundCadencedRecorder
        }
        self._phases = {
            name: int(callback_state[name].item()) if name in self._strides else 0
            for name in sessions
        }

    def append(self, state, emitted, *, retained=False, steps=None):
        if steps is None:
            raise FunctionalizationError(
                "HDF5 collection requires a completed host span"
            )
        super().append(state, emitted, retained=retained)
        emitted = dict(self.parts.pop())
        for name, session in self._sessions.items():
            values = emitted[name]
            size = next(iter(values.values())).shape[0]
            stride = self._strides.get(name)
            phase = self._phases[name]
            positions = (
                ((0,) if size else ())
                if steps == 0
                else tuple(
                    index
                    for index in range(1, steps + 1)
                    if stride is None or (phase + index) % stride == 0
                )
            )
            if len(positions) != size:
                raise FunctionalizationError(
                    "HDF5 Recorder samples do not match the completed host span"
                )
            session.append(values, steps=steps, positions=positions)
            self._phases[name] += steps
            emitted[name] = {
                field: value.new_empty((0, *value.shape[1:]))
                for field, value in values.items()
            }
        # Disk/reducer-only histories retain one schema part, not a growing
        # list. Real in-memory emissions retain their gradient-bearing parts.
        if not self.parts or any(
            name not in self._sessions and value is not None
            for name, value in emitted.items()
        ):
            self.parts.append(emitted)


def _recording_collector(callbacks, callback_state, selections, parts, sessions):
    collector = _RecordingCollector(callbacks, callback_state, selections, sessions)
    for part in parts:
        collector.append(callback_state, part, steps=0)
    return collector


__all__ = []
