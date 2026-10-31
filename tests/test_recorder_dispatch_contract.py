"""Generated Recorder dispatch admits shared cadence and rejects authored work."""

from types import FunctionType, MethodType

import pytest

import dendra as dn
from dendra.models.callbacks import CallbackList, _bind_callback_dt
from dendra.models.core import (
    _intra_fusion_callbacks_stable,
    _intra_fusion_recorder_buffers_safe,
)
from dendra.utils.dynamic_compilation import compile_generated_function
from tests.test_population_fused_intra import DT, _case

pytestmark = pytest.mark.cpu


def _recorder(*, dt=None):
    model, _ = _case(batch=1)
    callback = dn.Recorder(("v", "hh.m", "t"), dt=dt)
    _bind_callback_dt(callback, DT)
    return model, callback, callback._post_step_hook.__func__


def _stable(model, callback):
    return _intra_fusion_callbacks_stable(CallbackList([callback]), model)


@pytest.mark.parametrize("dt", (None, 3 * DT))
def test_framework_recorder_with_shared_sampling_cadence_is_statically_admitted(dt):
    model, callback, generated = _recorder(dt=dt)
    assert callback.save_every == (None if dt is None else 3)
    assert _intra_fusion_recorder_buffers_safe(callback, model, generated)
    assert _stable(model, callback)
    assert all(values == [] for values in callback.rec.values())


def test_rebound_generated_cadence_helper_rejects_without_running_authored_code(
    monkeypatch,
):
    model, callback, generated = _recorder(dt=3 * DT)
    calls = []

    def authored(counter, stride):
        calls.append((counter, stride))
        raise AssertionError("classification must not execute the cadence helper")

    assert _stable(model, callback)
    with monkeypatch.context() as changed:
        changed.setitem(generated.__globals__, "_recorder_should_save", authored)
        assert not _intra_fusion_recorder_buffers_safe(callback, model, generated)
        assert not _stable(model, callback)
        assert calls == []
    assert _stable(model, callback)
    assert callback.i == 0 and all(values == [] for values in callback.rec.values())


@pytest.mark.parametrize("change", ("source_only", "authored_statement"))
def test_generated_metadata_does_not_authorize_an_altered_recorder_body(
    change, monkeypatch
):
    model, callback, generated = _recorder()
    calls = []

    def authored(model):
        calls.append(model)
        raise AssertionError("classification must not execute authored statements")

    source = generated.__source__.replace(
        "def recorder(self, model):\n",
        "def recorder(self, model):\n  authored(model)\n",
        1,
    )
    assert source != generated.__source__
    if change == "source_only":
        monkeypatch.setattr(generated, "__source__", source)
    else:
        generated = compile_generated_function(
            source,
            func_name="recorder",
            filename_prefix="dendra.callbacks.recorder_func",
            global_ns=generated.__globals__,
            extra_ns={"authored": authored},
        )
        callback._post_step_hook = MethodType(generated, callback)
    assert not _intra_fusion_recorder_buffers_safe(callback, model, generated)
    assert not _stable(model, callback)
    assert calls == []


def test_rebound_append_helper_invalidates_an_existing_generated_recorder(monkeypatch):
    model, callback, generated = _recorder()
    calls = []

    def authored(*args):
        calls.append(args)
        raise AssertionError("classification must not execute the append helper")

    assert _stable(model, callback)
    with monkeypatch.context() as changed:
        changed.setitem(generated.__globals__, "_append_tensor", authored)
        assert not _intra_fusion_recorder_buffers_safe(callback, model, generated)
        assert not _stable(model, callback)
        assert calls == []
    assert _stable(model, callback)


def test_authored_generated_globals_are_rejected_without_running_mapping_accessors():
    model, callback, original = _recorder()
    calls = []

    class AuthoredGlobals(dict):
        def get(self, key, default=None):
            calls.append(key)
            raise AssertionError("classification must not call authored globals.get")

    generated = FunctionType(original.__code__, AuthoredGlobals(original.__globals__))
    generated.__source__ = original.__source__
    generated.__generated_filename__ = original.__generated_filename__
    callback._post_step_hook = MethodType(generated, callback)
    assert not _intra_fusion_recorder_buffers_safe(callback, model, generated)
    assert not _stable(model, callback)
    assert calls == []
