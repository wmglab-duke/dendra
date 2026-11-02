"""Observe selected ordinary Python function entries without wrapping them."""

import sys
from contextlib import ExitStack, contextmanager
from threading import get_ident

_monitoring = getattr(sys, "monitoring", None)
_TOOL_IDS = (3, 4, 2, 5, 0, 1)


@contextmanager
def observe_calls(codes, callback):
    """Pass selected callee frames to ``callback`` in the calling thread.

    Local monitoring events keep unrelated compiler calls uninstrumented and
    preserve function identity. Python versions without monitoring, or with all
    tool IDs occupied, retain the previous profile callback through a fallback.
    The selected functions must be ordinary functions, not generator resumptions.
    """
    codes = frozenset(codes)
    monitoring = _monitoring
    thread_id = get_ident()
    tool_id = None
    with ExitStack() as cleanup:
        if monitoring is not None:
            for candidate in _TOOL_IDS:
                try:
                    monitoring.use_tool_id(candidate, "dendra-test-call-observer")
                except ValueError:
                    continue
                cleanup.callback(monitoring.free_tool_id, candidate)
                tool_id = candidate
                break

        if tool_id is not None:

            def started(code, instruction_offset):
                if code in codes and get_ident() == thread_id:
                    callback(sys._getframe(1))

            # free_tool_id does not remove event masks or callbacks. Restore them
            # explicitly, including any stale configuration left on a free ID.
            previous_events = monitoring.get_events(tool_id)
            cleanup.callback(monitoring.set_events, tool_id, previous_events)
            monitoring.set_events(tool_id, 0)
            event = monitoring.events.PY_START
            previous_callback = monitoring.register_callback(tool_id, event, started)
            cleanup.callback(
                monitoring.register_callback, tool_id, event, previous_callback
            )
            for code in codes:
                previous_local = monitoring.get_local_events(tool_id, code)
                cleanup.callback(
                    monitoring.set_local_events, tool_id, code, previous_local
                )
                monitoring.set_local_events(tool_id, code, event)
            yield
            return

    previous = sys.getprofile()
    if previous is not None and not callable(previous):
        raise RuntimeError(
            "call observation cannot replace a non-callable profile hook"
        )

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            callback(frame)
        if previous is not None:
            previous(frame, event, arg)

    try:
        sys.setprofile(profile)
        yield
    finally:
        sys.setprofile(previous)
