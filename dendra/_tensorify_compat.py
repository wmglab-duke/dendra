"""Lazy repair of a recognized unused-scalar-memo compiler defect.

The repair is confined to known tensorify and cache-key implementations.
Unfamiliar or externally replaced implementations retain Torch's behavior.
Compiler configuration, installed files, and numerical inputs are unchanged.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import sys
from contextlib import ExitStack
from functools import update_wrapper
from pathlib import Path
from threading import Lock
from types import CodeType, FunctionType, ModuleType

_MODULE = "torch.fx.passes._tensorify_python_scalars"
_CACHE_MODULE = "torch._inductor.codecache"
_AOT_MODULE = "torch._functorch._aot_autograd.autograd_cache"
_CACHE_TAG = "dendra.tensorify-eq-v1"
# The AST identifies the complete known defective body across Python versions.
_KNOWN_AST = "8c2cf550dc1555a46e069d1d36a6e2ecbfc6bed31abab926e1bf45e606767068"
_KNOWN_CACHE_AST = "fc3f7a145eb5a8f5a6b3db241d6a3e3400208ef400a30403546351f7da4f9ce7"
_lock = Lock()
_status = "uninitialized"
_installed = None


def _constant(value):
    kind = type(value)
    if kind is CodeType:
        return [
            "code",
            value.co_name,
            value.co_qualname,
            value.co_argcount,
            value.co_posonlyargcount,
            value.co_kwonlyargcount,
            value.co_nlocals,
            value.co_stacksize,
            value.co_flags,
            value.co_code.hex(),
            value.co_exceptiontable.hex(),
            list(value.co_names),
            list(value.co_varnames),
            list(value.co_freevars),
            list(value.co_cellvars),
            [_constant(item) for item in value.co_consts],
        ]
    if value is None or kind is bool or kind is int or kind is str:
        return [kind.__name__, value]
    if kind is tuple:
        return ["tuple", [_constant(item) for item in value]]
    raise TypeError("Unrecognized compiler code constant")


def _code_fingerprint(code):
    return hashlib.sha256(
        json.dumps(_constant(code), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ast_data(value):
    if isinstance(value, ast.AST):
        return [
            type(value).__name__,
            [
                [name, _ast_data(child)]
                for name, child in ast.iter_fields(value)
                if not (name == "type_params" and child == [])
            ],
        ]
    if type(value) is list:
        return [_ast_data(child) for child in value]
    return _constant(value)


def _known_definition(
    function,
    namespace,
    code,
    *,
    path=("_tensorify_impl",),
    expected_ast=_KNOWN_AST,
):
    filename = namespace.get("__file__")
    if type(filename) is not str:
        return None
    source = Path(filename).read_bytes()
    tree = ast.parse(source, filename=filename)
    definition = tree
    for name in path:
        definitions = [
            node
            for node in definition.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
        ]
        if len(definitions) != 1:
            return None
        definition = definitions[0]
    fingerprint = hashlib.sha256(
        json.dumps(_ast_data(definition), separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if fingerprint != expected_ast:
        return None
    # Compile for the current Python's code structure without executing source.
    compiled = compile(
        source, filename, "exec", dont_inherit=True, optimize=sys.flags.optimize
    )
    for name in path:
        codes = [
            value
            for value in compiled.co_consts
            if type(value) is CodeType and value.co_name == name
        ]
        if len(codes) != 1:
            return None
        compiled = codes[0]
    if _code_fingerprint(code) != _code_fingerprint(compiled):
        return None
    return definition


def _known_function(function, namespace, *, code=None):
    if code is None:
        code = function.__code__
    return _known_definition(function, namespace, code) is not None


def _known_cache_function(function, namespace, *, code=None):
    if code is None:
        code = function.__code__
    return (
        _known_definition(
            function,
            namespace,
            code,
            path=("FxGraphHashDetails", "__init__"),
            expected_ast=_KNOWN_CACHE_AST,
        )
        is not None
    )


def _rewrite_final_predicate(definition):
    """Limit the known final sweep to real specializations, including derived ones."""
    loop = definition.body[-3]
    condition = loop.body[0].body[1]
    expected = ast.parse(
        "not TensorifyState.should_specialize(name) and k not in tensorified_symbols",
        mode="eval",
    ).body
    if ast.dump(condition.test) != ast.dump(expected):
        return False
    # A backed hint alone is not a specialization. Actual equality guards can
    # pin expressions such as zf + 0.001 without creating a simple replacement.
    # Those guards still require a restart when zf was tensorified elsewhere.
    replacement = ast.parse(
        """not TensorifyState.should_specialize(name) and (
            isinstance(shape_env.replacements.get(k), sympy.core.numbers.Float)
            or any(isinstance(guard.expr, sympy.Equality)
                   and k in guard.expr.free_symbols for guard in shape_env.guards)
        )""",
        mode="eval",
    ).body
    ast.increment_lineno(replacement, condition.test.lineno - 1)
    condition.test = ast.copy_location(replacement, condition.test)
    return True


def _repair(function, namespace, code):
    if (
        type(function) is not FunctionType
        or function.__globals__ is not namespace
        or type(function.__module__) is not str
        or function.__module__ != _MODULE
        or type(function.__name__) is not str
        or function.__name__ != "_tensorify_impl"
        or type(function.__qualname__) is not str
        or function.__qualname__ != "_tensorify_impl"
        or function.__dict__
        or function.__defaults__ is not None
        or function.__kwdefaults__ is not None
    ):
        return None
    definition = _known_definition(function, namespace, code)
    if definition is None or not _rewrite_final_predicate(definition):
        return None
    future = importlib.import_module("__future__")
    future_flags = sum(
        getattr(future, name).compiler_flag for name in future.all_feature_names
    )
    compiled = compile(
        ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
        code.co_filename,
        "exec",
        flags=code.co_flags & future_flags,
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    repaired_code = next(
        value
        for value in compiled.co_consts
        if type(value) is CodeType and value.co_name == "_tensorify_impl"
    )
    # Compile only the verified definition; no source or helper globals execute.
    # Numerical tensorification and the live-SymFloat sweep retain their AST.
    repaired = FunctionType(
        repaired_code,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    update_wrapper(repaired, function)
    repaired.__kwdefaults__ = function.__kwdefaults__
    return repaired


def _repair_cache(function, namespace, code):
    if (
        type(function) is not FunctionType
        or function.__globals__ is not namespace
        or type(function.__module__) is not str
        or function.__module__ != _CACHE_MODULE
        or type(function.__name__) is not str
        or function.__name__ != "__init__"
        or type(function.__qualname__) is not str
        or function.__qualname__ != "FxGraphHashDetails.__init__"
        or function.__dict__
        or function.__defaults__ is not None
        or function.__kwdefaults__ is not None
        or not _known_cache_function(function, namespace, code=code)
    ):
        return None

    def cache_init(self, *args, **kwargs):
        function(self, *args, **kwargs)
        # A structured field separates repaired keys from every valid string
        # user tag, including strings that happen to contain our marker.
        # Preserve the original object without coercion or overloaded operators.
        self.cache_key_tag = (_CACHE_TAG, self.cache_key_tag)

    update_wrapper(cache_init, function)
    cache_init.__kwdefaults__ = function.__kwdefaults__
    return cache_init


def _install_locked():
    global _installed, _status
    if _status == "installed":
        (
            namespace,
            function,
            code,
            cache_namespace,
            owner,
            cache_init,
            cache_code,
            original_cache,
            original_cache_code,
            aot_namespace,
            aot_owner,
        ) = _installed
        if (
            namespace.get("_tensorify_impl") is function
            and function.__code__ is code
            and cache_namespace.get("FxGraphHashDetails") is owner
            and vars(owner).get("__init__") is cache_init
            and cache_init.__code__ is cache_code
            and original_cache.__code__ is original_cache_code
            and aot_namespace.get("AOTAutogradCacheDetails") is aot_owner
            and owner in aot_owner.__mro__
        ):
            return True
        _status = "replaced"
        return False
    if _status != "uninitialized":
        return False
    try:
        from ._bootstrap import torch_compiler_warning_context

        with torch_compiler_warning_context():
            module = importlib.import_module(_MODULE)
            cache_module = importlib.import_module(_CACHE_MODULE)
            aot_module = importlib.import_module(_AOT_MODULE)
    except Exception:
        _status = "unavailable"
        return False
    try:
        if any(
            type(value) is not ModuleType
            for value in (module, cache_module, aot_module)
        ):
            _status = "unsupported"
            return False
        namespace, cache_namespace = vars(module), vars(cache_module)
        aot_namespace = vars(aot_module)
        original = namespace.get("_tensorify_impl")
        owner = cache_namespace.get("FxGraphHashDetails")
        aot_owner = aot_namespace.get("AOTAutogradCacheDetails")
        if (
            type(original) is not FunctionType
            or type(owner) is not type
            or type(aot_owner) is not type
            or owner not in aot_owner.__mro__
        ):
            _status = "unsupported"
            return False
        original_cache = vars(owner).get("__init__")
        if type(original_cache) is not FunctionType:
            _status = "unsupported"
            return False
        code, cache_code = original.__code__, original_cache.__code__
        repaired = _repair(original, namespace, code)
        cache_init = _repair_cache(original_cache, cache_namespace, cache_code)
        if repaired is None or cache_init is None:
            _status = "unsupported"
            return False
        if (
            namespace.get("_tensorify_impl") is not original
            or original.__code__ is not code
            or cache_namespace.get("FxGraphHashDetails") is not owner
            or vars(owner).get("__init__") is not original_cache
            or original_cache.__code__ is not cache_code
            or aot_namespace.get("AOTAutogradCacheDetails") is not aot_owner
            or owner not in aot_owner.__mro__
        ):
            _status = "replaced"
            return False
        installation = (
            namespace,
            repaired,
            repaired.__code__,
            cache_namespace,
            owner,
            cache_init,
            cache_init.__code__,
            original_cache,
            cache_code,
            aot_namespace,
            aot_owner,
        )
    except Exception:
        # Private compiler APIs can change independently of Dendra. Keep
        # their stock implementation whenever the repair cannot be proven.
        _status = "unsupported"
        return False
    # Cache lookup precedes tensorify. Separate repaired artifacts before making
    # the corrected pass visible, while Torch's compiler lock excludes captures.
    owner.__init__ = cache_init
    namespace["_tensorify_impl"] = repaired
    _installed = installation
    _status = "installed"
    return True


def install_tensorify_compatibility():
    """Install both guarded hooks once without changing compiler configuration."""
    with _lock:
        if _status not in ("uninitialized", "installed"):
            return False
    try:
        conversion = importlib.import_module("torch._dynamo.convert_frame")
        compile_lock = vars(conversion).get("compile_lock")
    except Exception:
        return _decline("unavailable")
    if not hasattr(type(compile_lock), "__enter__") or not hasattr(
        type(compile_lock), "__exit__"
    ):
        return _decline("unsupported")
    with ExitStack() as stack:
        try:
            stack.enter_context(compile_lock)
        except Exception:
            return _decline("unavailable")
        # Match Torch's lock order: compiler first, compatibility bookkeeping
        # second. The hooks themselves never take either lock or call back here.
        with _lock:
            return _install_locked()


def _decline(status):
    global _status
    with _lock:
        if _status == "uninitialized":
            _status = status
        return False


def _compatibility_status():
    """Read installation state without importing or activating Torch internals."""
    with _lock:
        return _status
