"""Call observation retains cold entries, ownership and other diagnostic tools."""

import sys
import unittest
from contextlib import ExitStack, contextmanager
from threading import Thread
from unittest.mock import patch

from tests import _call_observer as observer


def _selected(value, *, steps=1):
    return value + steps


def _unselected(value):
    return value * 2


class _Owner:
    def __init__(self, steps):
        self._steps = steps

    def selected(self, value):
        return value + self._steps


@contextmanager
def _previous_profile(callback):
    previous = sys.getprofile()
    sys.setprofile(callback)
    try:
        yield
    finally:
        sys.setprofile(previous)


class CallObserverTests(unittest.TestCase):
    def test_selected_cold_entries_keep_arguments_instances_and_identity(self):
        owner = _Owner(4)
        selected, method = _selected, _Owner.selected
        frames = []
        codes = {selected.__code__, method.__code__}
        with observer.observe_calls(
            codes, lambda frame: frames.append((frame.f_code, dict(frame.f_locals)))
        ):
            self.assertEqual(selected(2, steps=7), 9)
            self.assertEqual(owner.selected(3), 7)
            self.assertEqual(_unselected(5), 10)
        self.assertIs(_selected, selected)
        self.assertIs(_Owner.selected, method)
        self.assertEqual(frames[0], (selected.__code__, {"value": 2, "steps": 7}))
        self.assertEqual(frames[1], (method.__code__, {"self": owner, "value": 3}))

    def test_monitoring_leaves_an_existing_profile_callback_installed(self):
        if observer._monitoring is None:
            self.skipTest("local monitoring requires Python 3.12")
        previous_calls, observed = [], []

        def previous(frame, event, arg):
            if event == "call" and frame.f_code is _selected.__code__:
                previous_calls.append(frame.f_locals["value"])

        with _previous_profile(previous):
            with observer.observe_calls(
                {_selected.__code__},
                lambda frame: observed.append(frame.f_locals["value"]),
            ):
                self.assertIs(sys.getprofile(), previous)
                _selected(1)
            self.assertIs(sys.getprofile(), previous)
            _selected(2)
        self.assertEqual(observed, [1])
        self.assertEqual(previous_calls, [1, 2])

    def test_nested_observers_restore_outer_observation(self):
        outer, inner = [], []
        with observer.observe_calls(
            {_selected.__code__}, lambda frame: outer.append(frame.f_locals["value"])
        ):
            _selected(1)
            with observer.observe_calls(
                {_selected.__code__},
                lambda frame: inner.append(frame.f_locals["value"]),
            ):
                _selected(2)
            _selected(3)
        self.assertEqual(outer, [1, 2, 3])
        self.assertEqual(inner, [2])

    def test_monitoring_preserves_an_unrelated_tools_events_and_callback(self):
        monitoring = observer._monitoring
        if monitoring is None:
            self.skipTest("local monitoring requires Python 3.12")
        tool_id = next(
            (
                index
                for index in (3, 4, 2, 5, 0, 1)
                if monitoring.get_tool(index) is None
            ),
            None,
        )
        if tool_id is None:
            self.skipTest("all monitoring tool IDs are occupied")
        unrelated, selected = [], []

        def existing(code, offset):
            unrelated.append(sys._getframe(1).f_locals["value"])

        event = monitoring.events.PY_START
        monitoring.use_tool_id(tool_id, "unrelated-test-tool")
        with ExitStack() as cleanup:
            cleanup.callback(monitoring.free_tool_id, tool_id)
            previous = monitoring.register_callback(tool_id, event, existing)
            cleanup.callback(monitoring.register_callback, tool_id, event, previous)
            previous_local = monitoring.get_local_events(tool_id, _selected.__code__)
            cleanup.callback(
                monitoring.set_local_events, tool_id, _selected.__code__, previous_local
            )
            monitoring.set_local_events(tool_id, _selected.__code__, event)
            with observer.observe_calls(
                {_selected.__code__},
                lambda frame: selected.append(frame.f_locals["value"]),
            ):
                _selected(1)
            self.assertEqual(monitoring.get_tool(tool_id), "unrelated-test-tool")
            self.assertEqual(
                monitoring.get_local_events(tool_id, _selected.__code__), event
            )
            _selected(2)
        self.assertEqual(selected, [1])
        self.assertEqual(unrelated, [1, 2])

    def test_observation_is_limited_to_the_calling_thread(self):
        observed = []
        with observer.observe_calls(
            {_selected.__code__}, lambda frame: observed.append(frame.f_locals["value"])
        ):
            thread = Thread(target=_selected, args=(1,))
            thread.start()
            thread.join()
            _selected(2)
        self.assertEqual(observed, [2])

    def test_monitoring_releases_owned_events_after_user_or_callback_exception(self):
        monitoring = observer._monitoring
        if monitoring is None:
            self.skipTest("local monitoring requires Python 3.12")
        owners = [monitoring.get_tool(index) for index in range(6)]
        locals_before = [
            monitoring.get_local_events(index, _selected.__code__) for index in range(6)
        ]

        def failed(frame):
            raise LookupError("observer")

        for callback_error in (False, True):
            with self.subTest(callback_error=callback_error):
                with self.assertRaisesRegex(LookupError, "observer|body"):
                    with observer.observe_calls(
                        {_selected.__code__},
                        failed if callback_error else lambda frame: None,
                    ):
                        if callback_error:
                            _selected(1)
                        raise LookupError("body")
                self.assertEqual(
                    [monitoring.get_tool(index) for index in range(6)], owners
                )
                self.assertEqual(
                    [
                        monitoring.get_local_events(index, _selected.__code__)
                        for index in range(6)
                    ],
                    locals_before,
                )

    def test_monitoring_setup_failure_restores_partial_event_registration(self):
        monitoring = observer._monitoring
        if monitoring is None:
            self.skipTest("local monitoring requires Python 3.12")
        codes = (_selected.__code__, _unselected.__code__)
        owners = [monitoring.get_tool(index) for index in range(6)]
        masks = [
            [monitoring.get_local_events(index, code) for code in codes]
            for index in range(6)
        ]

        class PartialFailure:
            def __init__(self):
                self.enabled = 0

            def __getattr__(self, name):
                return getattr(monitoring, name)

            def set_local_events(self, tool_id, code, events):
                if events == monitoring.events.PY_START:
                    self.enabled += 1
                    if self.enabled == 2:
                        raise LookupError("registration")
                return monitoring.set_local_events(tool_id, code, events)

        with patch.object(observer, "_monitoring", PartialFailure()):
            with self.assertRaisesRegex(LookupError, "registration"):
                with observer.observe_calls(codes, lambda frame: None):
                    self.fail("setup failure must precede the observed body")
        self.assertEqual([monitoring.get_tool(index) for index in range(6)], owners)
        self.assertEqual(
            [
                [monitoring.get_local_events(index, code) for code in codes]
                for index in range(6)
            ],
            masks,
        )

    def test_setup_before_monitoring_reservation_cannot_leak_a_tool_id(self):
        monitoring = observer._monitoring
        if monitoring is None:
            self.skipTest("local monitoring requires Python 3.12")
        owners = [monitoring.get_tool(index) for index in range(6)]

        class BrokenStack:
            def __enter__(self):
                raise LookupError("stack entry")

            def __exit__(self, *args):
                raise AssertionError("failed entry must not call exit")

        faults = (
            patch.object(observer, "get_ident", side_effect=LookupError("thread")),
            patch.object(observer, "ExitStack", BrokenStack),
        )
        for fault in faults:
            with self.subTest(fault=fault):
                with fault:
                    with self.assertRaisesRegex(LookupError, "thread|stack entry"):
                        with observer.observe_calls(
                            {_selected.__code__}, lambda frame: None
                        ):
                            self.fail("setup failure must precede the observed body")
                self.assertEqual(
                    [monitoring.get_tool(index) for index in range(6)], owners
                )

    def test_profile_fallback_restores_previous_hook_if_installation_raises(self):
        events = []

        def previous(frame, event, arg):
            if sys.getprofile() is not previous:
                events.append((event, arg is sys.setprofile))
                raise LookupError("replacement installed")

        with patch.object(observer, "_monitoring", None):
            with _previous_profile(previous):
                with self.assertRaisesRegex(LookupError, "replacement installed"):
                    with observer.observe_calls(
                        {_selected.__code__}, lambda frame: None
                    ):
                        self.fail("failed hook installation must precede the body")
                self.assertIs(sys.getprofile(), previous)
        self.assertEqual(events, [("c_return", True)])

    def test_unavailable_monitoring_falls_back_and_preserves_prior_profile(self):
        class Unavailable:
            def use_tool_id(self, tool_id, name):
                raise ValueError("occupied")

        for monitoring in (None, Unavailable()):
            with self.subTest(monitoring=monitoring):
                prior, observed = [], []

                def previous(frame, event, arg):
                    if event == "call" and frame.f_code is _selected.__code__:
                        prior.append(frame.f_locals["value"])

                with patch.object(observer, "_monitoring", monitoring):
                    with _previous_profile(previous):
                        with observer.observe_calls(
                            {_selected.__code__},
                            lambda frame: observed.append(frame.f_locals["value"]),
                        ):
                            _selected(1)
                            _unselected(2)
                        self.assertIs(sys.getprofile(), previous)
                        _selected(3)
                self.assertEqual(observed, [1])
                self.assertEqual(prior, [1, 3])

    def test_profile_fallback_restores_previous_hook_after_exception(self):
        prior = []

        def previous(frame, event, arg):
            if event == "call" and frame.f_code is _selected.__code__:
                prior.append(frame.f_locals["value"])

        def failed(frame):
            raise LookupError("observer")

        with patch.object(observer, "_monitoring", None):
            with _previous_profile(previous):
                with self.assertRaisesRegex(LookupError, "observer"):
                    with observer.observe_calls({_selected.__code__}, failed):
                        _selected(1)
                self.assertIs(sys.getprofile(), previous)
                _selected(2)
        self.assertEqual(prior, [2])

    def test_profile_fallback_refuses_to_replace_a_noncallable_hook(self):
        with patch.object(observer, "_monitoring", None):
            with patch.object(sys, "getprofile", return_value=object()):
                with patch.object(sys, "setprofile") as setter:
                    with self.assertRaisesRegex(RuntimeError, "non-callable"):
                        with observer.observe_calls(
                            {_selected.__code__}, lambda frame: None
                        ):
                            pass
                    setter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
