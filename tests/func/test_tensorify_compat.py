"""Owned compilation repairs only the recognized unused-scalar-memo pass."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu
_PREFIX = "DENDRA_TENSORIFY_COMPAT="
_REPOSITORY = Path(__file__).resolve().parents[2]


def _probe(tmp_path, script, *arguments):
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        TORCHINDUCTOR_CACHE_DIR=os.fspath(tmp_path / "inductor"),
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(script), *arguments],
        cwd=_REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    reports = [
        line.removeprefix(_PREFIX)
        for line in completed.stdout.splitlines()
        if line.startswith(_PREFIX)
    ]
    assert len(reports) == 1, completed.stdout + completed.stderr
    return json.loads(reports[0])


_IMPORT = """
import importlib
import json
import torch
from torch.compiler import config as compiler_config

compiler_config.cache_key_tag = 'caller-owned-cache-prefix'
original_tag = compiler_config.cache_key_tag
module = importlib.import_module('torch.fx.passes._tensorify_python_scalars')
original = module._tensorify_impl
original_code = original.__code__
cache_owner = importlib.import_module('torch._inductor.codecache').FxGraphHashDetails
original_constructor = cache_owner.__init__
import dendra
from dendra import _tensorify_compat as compat
from dendra._bootstrap import compile_owned

assert compat._compatibility_status() == 'uninitialized'
assert compiler_config.cache_key_tag == original_tag
assert module._tensorify_impl is original and original.__code__ is original_code
assert cache_owner.__init__ is original_constructor
calls = []
sentinel = object()
def compiler(function, **options):
    calls.append((function, options))
    return sentinel
torch.compile = compiler
function = lambda value: value
options = {'disable': True, 'backend': 'eager', 'options': {'nested': []}}
assert compile_owned(function, **options) is sentinel
assert calls == [(function, options)]
assert calls[0][1]['options'] is options['options']
assert compat._compatibility_status() == 'uninitialized'
assert compiler_config.cache_key_tag == original_tag
assert module._tensorify_impl is original and original.__code__ is original_code
assert cache_owner.__init__ is original_constructor
print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'qualified': True}))
"""


def test_import_and_disabled_owned_compilation_do_not_activate_repair(tmp_path):
    assert _probe(tmp_path, _IMPORT)["qualified"]


_INSTALL = """
from concurrent.futures import ThreadPoolExecutor
import ast
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
from threading import Barrier
import torch
from torch.compiler import config as compiler_config

compiler_config.cache_key_tag = 'caller-owned-cache-prefix'
original_tag = compiler_config.cache_key_tag
module = importlib.import_module('torch.fx.passes._tensorify_python_scalars')
original = module._tensorify_impl
cache_owner = importlib.import_module('torch._inductor.codecache').FxGraphHashDetails
original_constructor = cache_owner.__init__
constructor_code = original_constructor.__code__
import dendra
from dendra import _tensorify_compat as compat

if not (
    compat._known_function(original, vars(module))
    and compat._known_cache_function(cache_owner.__init__, cache_owner.__init__.__globals__)
):
    print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': False}))
else:
    source = Path(module.__file__)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    cache_source = Path(original_constructor.__globals__['__file__'])
    cache_before = hashlib.sha256(cache_source.read_bytes()).hexdigest()
    if sys.argv[1] == 'relocated_lines':
        original.__code__ = original.__code__.replace(
            co_firstlineno=original.__code__.co_firstlineno + 100
        )
        real_version = torch.__version__
        torch.__version__ = 'unrelated-version-label'
        assert compat._known_function(original, vars(module))
        torch.__version__ = real_version
    old_code = original.__code__
    definition = compat._known_definition(original, vars(module), old_code)
    assert definition is not None
    rewritten = deepcopy(definition)
    assert compat._rewrite_final_predicate(rewritten) is True
    original_ast = ast.dump(definition)
    assert ast.dump(rewritten) != original_ast
    definition.body[-3].body[0].body[1].test = rewritten.body[-3].body[0].body[1].test
    assert ast.dump(definition) == ast.dump(rewritten)
    assert compat._rewrite_final_predicate(rewritten) is False
    barrier = Barrier(8)
    def install(_index):
        barrier.wait(timeout=30)
        prefix = original_tag + '-' + str(_index)
        compiler_config.cache_key_tag = prefix
        active = compat.install_tensorify_compatibility()
        details = cache_owner(None, [], {}, [])
        tag = details.cache_key_tag
        assert tag == (compat._CACHE_TAG, prefix)
        assert compiler_config.cache_key_tag == prefix
        assert compat.install_tensorify_compatibility() is True
        assert cache_owner(None, [], {}, []).cache_key_tag == tag
        assert compiler_config.cache_key_tag == prefix
        current = module._tensorify_impl
        return active, id(current), id(current.__code__), id(cache_owner.__init__), tag
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(install, range(8)))
    assert all(row[0] for row in results)
    assert len({row[1:4] for row in results}) == 1
    assert compat._compatibility_status() == 'installed'
    expected_tag = (compat._CACHE_TAG, original_tag)
    assert compat._CACHE_TAG == 'dendra.tensorify-eq-v1'
    assert len({row[4] for row in results}) == 8
    repaired_constructor = cache_owner.__init__
    assert repaired_constructor is not original_constructor
    assert original_constructor.__code__ is constructor_code
    assert repaired_constructor.__defaults__ is original_constructor.__defaults__
    assert repaired_constructor.__kwdefaults__ is original_constructor.__kwdefaults__
    assert repaired_constructor.__annotations__ is original_constructor.__annotations__
    assert repaired_constructor.__doc__ == original_constructor.__doc__
    assert inspect.signature(repaired_constructor) == inspect.signature(original_constructor)
    repaired = module._tensorify_impl
    assert repaired is not original
    assert original.__code__ is old_code
    assert repaired.__code__ is not old_code
    assert repaired.__globals__ is original.__globals__
    assert repaired.__defaults__ is original.__defaults__
    assert repaired.__kwdefaults__ is original.__kwdefaults__
    assert repaired.__closure__ is original.__closure__
    assert repaired.__annotations__ is original.__annotations__
    assert repaired.__doc__ == original.__doc__
    assert inspect.signature(repaired) == inspect.signature(original)
    assert compat.install_tensorify_compatibility() is True
    assert compiler_config.cache_key_tag == original_tag
    assert cache_owner(None, [], {}, []).cache_key_tag == expected_tag
    assert compat.install_tensorify_compatibility() is True
    assert compiler_config.cache_key_tag == original_tag
    compiler_config.cache_key_tag = 'new-caller-prefix'
    assert compat.install_tensorify_compatibility() is True
    assert compiler_config.cache_key_tag == 'new-caller-prefix'
    assert cache_owner(None, [], {}, []).cache_key_tag == (compat._CACHE_TAG, 'new-caller-prefix')
    compiler_config.cache_key_tag = ''
    assert cache_owner(None, [], {}, []).cache_key_tag == (compat._CACHE_TAG, '')
    assert compiler_config.cache_key_tag == ''
    for prefix in ('', 'x'):
        stock_prefix = prefix + ('|' if prefix else '') + compat._CACHE_TAG
        with compiler_config.patch(cache_key_tag=stock_prefix):
            stock = object.__new__(cache_owner)
            original_constructor(stock, None, [], {}, [])
            assert stock.cache_key_tag == stock_prefix
            assert compiler_config.cache_key_tag == stock_prefix
        with compiler_config.patch(cache_key_tag=prefix):
            repaired_key = cache_owner(None, [], {}, []).cache_key_tag
            assert repaired_key == (compat._CACHE_TAG, prefix)
            assert repaired_key != stock.cache_key_tag
            assert compiler_config.cache_key_tag == prefix
    assert compiler_config.cache_key_tag == ''
    assert cache_owner.__init__ is repaired_constructor
    assert module._tensorify_impl is repaired
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert hashlib.sha256(cache_source.read_bytes()).hexdigest() == cache_before
    print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': True, 'qualified': True}))
"""


@pytest.mark.parametrize("location", ("original", "relocated_lines"))
def test_known_body_install_is_lazy_thread_safe_idempotent_and_location_independent(
    tmp_path, location
):
    report = _probe(tmp_path, _INSTALL, location)
    if not report["supported"]:
        pytest.skip("Installed Torch pass is outside the recognized repair contract")
    assert report["qualified"]


_UNKNOWN = """
import importlib
import json
import sys
from types import FunctionType
import torch
from torch.compiler import config as compiler_config

compiler_config.cache_key_tag = 'caller-owned-cache-prefix'
original_tag = compiler_config.cache_key_tag
module = importlib.import_module('torch.fx.passes._tensorify_python_scalars')
original = module._tensorify_impl
cache_owner = importlib.import_module('torch._inductor.codecache').FxGraphHashDetails
original_constructor = cache_owner.__init__
import dendra
from dendra import _tensorify_compat as compat

mode = sys.argv[1]
if mode.startswith('hash_'):
    if mode == 'hash_wrapper':
        def candidate(self, *args, **kwargs):
            return original_constructor(self, *args, **kwargs)
    else:
        code = original_constructor.__code__.replace(
            co_consts=(*original_constructor.__code__.co_consts, 'unknown-hash-body')
        )
        candidate = FunctionType(code, original_constructor.__globals__,
                                 original_constructor.__name__,
                                 original_constructor.__defaults__,
                                 original_constructor.__closure__)
        candidate.__module__ = original_constructor.__module__
        candidate.__qualname__ = original_constructor.__qualname__
        candidate.__annotations__ = original_constructor.__annotations__
        candidate.__kwdefaults__ = original_constructor.__kwdefaults__
    cache_owner.__init__ = candidate
elif mode == 'callable':
    class Caller:
        def __call__(self, *args, **kwargs):
            return original(*args, **kwargs)
    candidate = Caller()
elif mode == 'wrapper':
    def candidate(*args, **kwargs):
        return original(*args, **kwargs)
else:
    code = original.__code__
    if mode == 'fixed':
        if code.co_names.count('backed_var_to_val') != 1:
            print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': False}))
            raise SystemExit(0)
        names = tuple('replacements' if name == 'backed_var_to_val' else name
                      for name in code.co_names)
        code = code.replace(co_names=names)
    elif mode == 'code':
        code = code.replace(co_consts=(*code.co_consts, 'unrecognized-body-constant'))
    else:
        raise AssertionError(mode)
    candidate = FunctionType(code, original.__globals__, original.__name__,
                             original.__defaults__, original.__closure__)
    candidate.__module__ = original.__module__
    candidate.__qualname__ = original.__qualname__
    candidate.__annotations__ = original.__annotations__
    candidate.__kwdefaults__ = original.__kwdefaults__
if not mode.startswith('hash_'):
    module._tensorify_impl = candidate
expected_pass = module._tensorify_impl
expected_constructor = cache_owner.__init__
assert compat.install_tensorify_compatibility() is False
assert compat._compatibility_status() == 'unsupported'
assert compiler_config.cache_key_tag == original_tag
assert module._tensorify_impl is expected_pass
assert cache_owner.__init__ is expected_constructor
assert compat.install_tensorify_compatibility() is False
assert module._tensorify_impl is expected_pass
assert cache_owner.__init__ is expected_constructor
assert compiler_config.cache_key_tag == original_tag
print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': True, 'qualified': True}))
"""


@pytest.mark.parametrize(
    "mode", ("fixed", "code", "wrapper", "callable", "hash_code", "hash_wrapper")
)
def test_fixed_or_unrecognized_passes_remain_untouched(tmp_path, mode):
    report = _probe(tmp_path, _UNKNOWN, mode)
    if not report["supported"]:
        pytest.skip(
            "Installed pass has no legacy lookup to construct the fixed control"
        )
    assert report["qualified"]


_REPLACED = """
import importlib
import json
import sys
import torch
from torch.compiler import config as compiler_config

compiler_config.cache_key_tag = 'caller-owned-cache-prefix'
original_tag = compiler_config.cache_key_tag
module = importlib.import_module('torch.fx.passes._tensorify_python_scalars')
original = module._tensorify_impl
cache_owner = importlib.import_module('torch._inductor.codecache').FxGraphHashDetails
import dendra
from dendra import _tensorify_compat as compat

if not (
    compat._known_function(original, vars(module))
    and compat._known_cache_function(cache_owner.__init__, cache_owner.__init__.__globals__)
):
    print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': False}))
else:
    assert compat.install_tensorify_compatibility() is True
    active = module._tensorify_impl
    active_constructor = cache_owner.__init__
    if sys.argv[1] == 'hash_namespace':
        def replacement(self, *args, **kwargs):
            return active_constructor(self, *args, **kwargs)
        cache_owner.__init__ = replacement
    elif sys.argv[1] == 'hash_code':
        active_constructor.__code__ = active_constructor.__code__.replace(
            co_consts=(*active_constructor.__code__.co_consts, 'external-hash-change')
        )
    elif sys.argv[1] == 'namespace':
        def replacement(*args, **kwargs):
            return active(*args, **kwargs)
        module._tensorify_impl = replacement
    else:
        active.__code__ = active.__code__.replace(
            co_consts=(*active.__code__.co_consts, 'external-code-change')
        )
    current = module._tensorify_impl
    current_code = current.__code__
    current_constructor = cache_owner.__init__
    current_constructor_code = current_constructor.__code__
    compiler_config.cache_key_tag = 'externally-owned-namespace'
    for _ in range(2):
        assert compat.install_tensorify_compatibility() is False
        assert module._tensorify_impl is current and current.__code__ is current_code
        assert compat._compatibility_status() == 'replaced'
        assert cache_owner.__init__ is current_constructor
        assert current_constructor.__code__ is current_constructor_code
        assert compiler_config.cache_key_tag == 'externally-owned-namespace'
    print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': True, 'qualified': True}))
"""


@pytest.mark.parametrize("mode", ("namespace", "code", "hash_namespace", "hash_code"))
def test_external_replacements_after_installation_are_never_overwritten(tmp_path, mode):
    report = _probe(tmp_path, _REPLACED, mode)
    if not report["supported"]:
        pytest.skip("Installed Torch pass is outside the recognized repair contract")
    assert report["qualified"]


_UNAVAILABLE = """
import importlib
import json
from types import SimpleNamespace
import sys
import torch
from torch.compiler import config as compiler_config
compiler_config.cache_key_tag = 'caller-owned-cache-prefix'
original_tag = compiler_config.cache_key_tag
import dendra
from dendra import _tensorify_compat as compat
from dendra._bootstrap import compile_owned

real_import = importlib.import_module
mode = sys.argv[1]
def unavailable(name, *args, **kwargs):
    if name == compat._MODULE:
        if mode == 'invalid_module':
            return SimpleNamespace()
        raise ImportError('unavailable private compiler module')
    return real_import(name, *args, **kwargs)
compat.importlib.import_module = unavailable
try:
    assert compat.install_tensorify_compatibility() is False
finally:
    compat.importlib.import_module = real_import
assert compat._compatibility_status() == (
    'unsupported' if mode == 'invalid_module' else 'unavailable'
)
assert compiler_config.cache_key_tag == original_tag
calls = []
sentinel = object()
def compiler(function, **options):
    calls.append((function, options))
    return sentinel
torch.compile = compiler
function = lambda value: value
options = {'backend': 'eager', 'fullgraph': True, 'options': {'nested': []}}
assert compile_owned(function, **options) is sentinel
assert calls == [(function, options)]
assert calls[0][1]['options'] is options['options']
assert compiler_config.cache_key_tag == original_tag
print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'qualified': True}))
"""


@pytest.mark.parametrize("mode", ("missing_module", "invalid_module"))
def test_unavailable_private_api_falls_through_to_ordinary_compile(tmp_path, mode):
    assert _probe(tmp_path, _UNAVAILABLE, mode)["qualified"]


_NUMERICAL = """
from contextlib import nullcontext
import importlib
import json
import torch
from torch._dynamo.backends.registry import lookup_backend
import torch._dynamo as dynamo
import torch._dynamo.config as config
from torch._dynamo.utils import get_compilation_metrics

module = importlib.import_module('torch.fx.passes._tensorify_python_scalars')
original = module._tensorify_impl
cache_owner = importlib.import_module('torch._inductor.codecache').FxGraphHashDetails
import dendra as dn
from dendra import _tensorify_compat as compat
from dendra.models.mod import pas
from dendra._bootstrap import compile_owned, torch_compiler_warning_context

torch.set_num_threads(1)
if not (
    compat._known_function(original, vars(module))
    and compat._known_cache_function(cache_owner.__init__, cache_owner.__init__.__globals__)
):
    print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': False}))
    raise SystemExit(0)
assert compat._compatibility_status() == 'uninitialized'
source = {
    'sin': 'def fn(x):\\n    return x.sin() * x\\n',
    'python': 'def fn(x, a):\\n    return x + torch.full((2,), a, dtype=x.dtype)\\n',
    'item': 'def fn(x):\\n    return torch.full((2,), x.item(), dtype=x.dtype)\\n',
    'mixed': 'def fn(x):\\n    return x + torch.full((2,), x.item(), dtype=x.dtype)\\n',
}
with torch_compiler_warning_context(), dn.ctx(
    JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE='cpu', IMEM=False
):
    population = dn.Population(1, 1)
    population.insert(pas, g=1.0e-4, e=-65.0)
    population.initialize()
    runner = dn.compile(population, dt=0.01, steps=1, execution='scan',
                        inference=True, backend='aot_eager')
    with torch.no_grad():
        expected = runner.functional.prepare_and_rollout(
            runner.parameters, runner.constants, runner.initial_state, steps=1
        )[0]
    before = len(get_compilation_metrics())
    actual = runner()[0]
    assert compat._compatibility_status() == 'installed'
    actual_leaves, spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, wanted_spec = torch.utils._pytree.tree_flatten(expected)
    assert spec == wanted_spec
    for value, wanted in zip(actual_leaves, expected_leaves):
        torch.testing.assert_close(value, wanted, rtol=0, atol=0)
    assert not any(row.restart_reasons for row in get_compilation_metrics()[before:])

    baseline_config = (config.capture_scalar_outputs, config.specialize_float)
    cases = (
        ('scalar64', 'sin', torch.float64, (), False, 1, 0, 0),
        ('scalar64_grad', 'sin', torch.float64, (), True, 1, 0, 0),
        ('vector64_grad', 'sin', torch.float64, (1,), True, 1, 0, 0),
        ('scalar32', 'sin', torch.float32, (), False, 1, 0, 0),
        ('python', 'python', torch.float32, (2,), False, 2, 0, 1),
        ('item', 'item', torch.float64, (), False, 3, 1, 1),
        ('mixed_grad', 'mixed', torch.float64, (), True, 3, 1, 1),
    )
    reports = []
    for name, kind, dtype, shape, train, expected_entries, expected_restarts, expected_guards in cases:
        dynamo.reset()
        scope = {'torch': torch}
        exec(compile(source[kind].replace('\\\\n', '\\n'), '<tensorify-control-' + name + '>', 'exec'), scope)
        function = scope['fn']
        entries = []
        ordinary_backend = lookup_backend('aot_eager')
        def backend(graph, examples):
            entries.append(graph)
            return ordinary_backend(graph, examples)
        settings = config.patch(capture_scalar_outputs=True) if kind != 'sin' else nullcontext()
        with settings:
            compiled = compile_owned(function, backend=backend, fullgraph=True, dynamic=False)
            value = torch.full(shape, 1.3, dtype=dtype, requires_grad=train)
            metric_start = len(get_compilation_metrics())
            for selected in (1.3, 2.7, 1.3):
                with torch.no_grad():
                    value.fill_(selected)
                before_value = value.detach().clone()
                version = value._version
                eager_value = before_value.clone().requires_grad_(train)
                args = (value, selected) if kind == 'python' else (value,)
                eager_args = (eager_value, selected) if kind == 'python' else (eager_value,)
                context = torch.enable_grad() if train else torch.no_grad()
                with context:
                    expected = function(*eager_args)
                    actual = compiled(*args)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    if train:
                        actual_gradient = torch.autograd.grad(actual.sum(), value)[0]
                        expected_gradient = torch.autograd.grad(expected.sum(), eager_value)[0]
                        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
                torch.testing.assert_close(value, before_value, rtol=0, atol=0)
                assert value._version == version
            metrics = get_compilation_metrics()[metric_start:]
            restart_count = sum(bool(row.restart_reasons) for row in metrics)
            recompile_count = sum(row.recompile_reason is not None for row in metrics)
            assert len(entries) == expected_entries, (name, len(entries))
            assert restart_count == expected_restarts, (name, restart_count)
            assert recompile_count == expected_guards, (name, recompile_count)
            reports.append(name)
        assert (config.capture_scalar_outputs, config.specialize_float) == baseline_config
assert compat.install_tensorify_compatibility() is True
print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': True, 'qualified': True, 'cases': reports}))
"""


def test_owned_aot_compilation_avoids_unused_memo_restarts_and_keeps_scalar_guards(
    tmp_path,
):
    report = _probe(tmp_path, _NUMERICAL)
    if not report["supported"]:
        pytest.skip("Installed Torch pass is outside the recognized repair contract")
    assert report["qualified"] and len(report["cases"]) == 7


_DERIVED = """
import importlib
import json
import torch
import torch.nn.functional as F
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.utils import get_compilation_metrics
import dendra
from dendra import _tensorify_compat as compat
from dendra._bootstrap import compile_owned, torch_compiler_warning_context

module = importlib.import_module('torch.fx.passes._tensorify_python_scalars')
cache_owner = importlib.import_module('torch._inductor.codecache').FxGraphHashDetails
if not (
    compat._known_function(module._tensorify_impl, vars(module))
    and compat._known_cache_function(cache_owner.__init__, cache_owner.__init__.__globals__)
):
    print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': False}))
    raise SystemExit(0)

def norm_pure(x, s):
    return F.rms_norm(x, (4,), eps=s + .001)
def norm_mixed(x, s):
    return F.rms_norm(x * s, (4,), eps=s + .001)
def huber_pure(x, s):
    return F.huber_loss(x, torch.zeros_like(x), delta=s + .1)
def huber_mixed(x, s):
    return F.huber_loss(x * s, torch.zeros_like(x), delta=s + .1)
def resize_pure(x, s):
    return F.interpolate(x[None], scale_factor=1.0 + s)
def resize_mixed(x, s):
    return F.interpolate((x * s)[None], scale_factor=1.0 + s)

def signature():
    return [(str(row.compile_id), row.start_time_us, row.co_name)
            for row in get_compilation_metrics()]

torch.set_num_threads(1)
rng = torch.get_rng_state().clone()
reports = []
with torch_compiler_warning_context():
    for function in (norm_pure, norm_mixed, huber_pure, huber_mixed, resize_pure, resize_mixed):
        torch._dynamo.reset()
        entries = []
        ordinary_backend = lookup_backend('aot_eager')
        def backend(graph, examples):
            entries.append(graph)
            return ordinary_backend(graph, examples)
        compiled = compile_owned(function, backend=backend, fullgraph=True)
        def check(s):
            value = (torch.arange(8, dtype=torch.float64).reshape(2, 4) / 9).requires_grad_()
            source = value.detach().clone()
            version = value._version
            eager_value = source.clone().requires_grad_()
            expected = function(eager_value, s)
            actual = compiled(value, s)
            expected_gradient = torch.autograd.grad(expected.sum(), eager_value)[0]
            actual_gradient = torch.autograd.grad(actual.sum(), value)[0]
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-10, atol=1e-11)
            torch.testing.assert_close(value, source, rtol=0, atol=0)
            assert value._version == version
        values = (.9, .5, .25, 2.0)
        for selected in values:
            check(selected)
        before = signature()
        backend_count = len(entries)
        for selected in reversed(values):
            check(selected)
        assert signature() == before and len(entries) == backend_count, function.__name__
        reports.append(function.__name__)
assert compat._compatibility_status() == 'installed'
torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
print('DENDRA_TENSORIFY_COMPAT=' + json.dumps({'supported': True, 'qualified': True, 'cases': reports}))
"""


def test_derived_schema_scalars_and_mixed_tensor_uses_keep_values_and_gradients(
    tmp_path,
):
    report = _probe(tmp_path, _DERIVED)
    if not report["supported"]:
        pytest.skip("Installed Torch pass is outside the recognized repair contract")
    assert report["qualified"] and len(report["cases"]) == 6
