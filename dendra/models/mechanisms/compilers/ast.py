from __future__ import annotations

import ast
import copy
import inspect
import re
import textwrap
from functools import lru_cache

import sympy as sp

from .source import SourceUnavailableError, safe_source

EXPECTED_FACTORIZATION_ERRORS = (
    SourceUnavailableError,
    SyntaxError,
    StopIteration,
    NotImplementedError,
    ValueError,
    ZeroDivisionError,
    sp.SympifyError,
)


# helper: Python‑AST → SymPy
def _ast_to_sympy(node: ast.AST, local_syms: dict[str, sp.Expr]) -> sp.Expr:
    if isinstance(node, ast.Name):
        return local_syms.setdefault(node.id, sp.symbols(node.id))
    if isinstance(node, ast.Constant):
        return sp.sympify(node.value)
    if isinstance(node, ast.Attribute):  # self.x → self_x
        base = _ast_to_sympy(node.value, local_syms)
        if isinstance(base, sp.Symbol) and base.name == "self":
            return local_syms.setdefault(
                f"self_{node.attr}", sp.symbols(f"self_{node.attr}")
            )
        raise NotImplementedError("Only `self.attr` attributes supported.")
    if isinstance(node, ast.BinOp):
        operators = {
            ast.Add: lambda left, right: left + right,
            ast.Sub: lambda left, right: left - right,
            ast.Mult: lambda left, right: left * right,
            ast.Div: lambda left, right: left / right,
            ast.Pow: lambda left, right: left**right,
        }
        operator = operators.get(type(node.op))
        if operator is None:
            raise NotImplementedError(
                f"Unsupported binary operator: {type(node.op).__name__}"
            )
        left, right = (
            _ast_to_sympy(node.left, local_syms),
            _ast_to_sympy(node.right, local_syms),
        )
        return operator(left, right)
    if isinstance(node, ast.UnaryOp):
        operators = {
            ast.UAdd: lambda operand: +operand,
            ast.USub: lambda operand: -operand,
        }
        operator = operators.get(type(node.op))
        if operator is None:
            raise NotImplementedError(
                f"Unsupported unary operator: {type(node.op).__name__}"
            )
        o = _ast_to_sympy(node.operand, local_syms)
        return operator(o)
    if isinstance(node, ast.Call):  # pow(a, b)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "pow"
            and len(node.args) == 2
            and not node.keywords
        ):
            a, b = (_ast_to_sympy(arg, local_syms) for arg in node.args)
            return a**b
    raise NotImplementedError(f"Unsupported AST node: {ast.dump(node)}")


def _descriptor_function(member):
    if isinstance(member, (classmethod, staticmethod)):
        return member.__func__
    return member


def _runtime_method(cls, method):
    owner = next((base for base in cls.__mro__ if method in base.__dict__), cls)
    return owner, _descriptor_function(owner.__dict__.get(method))


def _method_cache_token(obj_or_src, method):
    """Return a cache token that changes when a runtime method is replaced."""
    if isinstance(obj_or_src, str):
        return None
    cls = obj_or_src if inspect.isclass(obj_or_src) else type(obj_or_src)
    _, function = _runtime_method(cls, method)
    code = getattr(function, "__code__", None)
    try:
        hash((function, code))
    except TypeError:
        return id(function), id(code)
    return function, code


def _function_is_authored_on_class(function, cls, method):
    return getattr(function, "__qualname__", None) == (
        f"{getattr(cls, '__qualname__', cls.__name__)}.{method}"
    )


def _function_nodes_with_owner(tree):
    functions = []

    def visit(node, class_owner=None):
        if isinstance(node, ast.ClassDef):
            class_owner = node.name
        if isinstance(node, ast.FunctionDef):
            functions.append((node, class_owner))
        for child in ast.iter_child_nodes(node):
            visit(child, class_owner)

    visit(tree)
    return functions


def _select_runtime_function(tree, function, method):
    functions_with_owner = _function_nodes_with_owner(tree)
    code = getattr(function, "__code__", None)
    if code is not None:
        line_matches = [
            node
            for node, _ in functions_with_owner
            if node.lineno == code.co_firstlineno
        ]
        if line_matches:
            return line_matches[-1]

    names = {method, getattr(function, "__name__", method)}
    qualname_parts = getattr(function, "__qualname__", "").split(".")
    owner_name = qualname_parts[-2] if len(qualname_parts) > 1 else None
    if owner_name and owner_name != "<locals>":
        owner_matches = [
            node
            for node, class_owner in functions_with_owner
            if node.name in names and class_owner == owner_name
        ]
        if owner_matches:
            return owner_matches[-1]

    name_matches = [node for node, _ in functions_with_owner if node.name in names]
    if name_matches:
        return name_matches[-1]
    raise StopIteration(f"Could not locate source for current method {method!r}.")


def _runtime_function_source(function, method):
    if not inspect.isfunction(function):
        raise NotImplementedError(
            f"Current method {method!r} is not a Python function with source."
        )
    if hasattr(function, "__wrapped__") or getattr(
        getattr(function, "__code__", None), "co_freevars", ()
    ):
        raise NotImplementedError(
            "Decorated current methods and closure-based current methods require "
            "an explicit analytic conductance pair or numerical differentiation."
        )
    src = textwrap.dedent(safe_source(function))
    tree = ast.parse(src)
    return _select_runtime_function(tree, function, method)


def _class_function_source(source_owner, method):
    src = textwrap.dedent(safe_source(source_owner))
    tree = ast.parse(src)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    named_classes = [node for node in classes if node.name == source_owner.__name__]
    candidates = named_classes or classes
    if not candidates:
        raise StopIteration(
            f"Could not locate class source for {source_owner.__name__!r}."
        )
    cls = candidates[-1]
    return next(
        node
        for node in reversed(cls.body)
        if isinstance(node, ast.FunctionDef) and node.name == method
    )


# Shared cached analysis used by the reversal-factorization and conductance-only
# APIs.  The latter can represent an exact zero derivative even when no reversal
# potential exists for a voltage-independent current.
@lru_cache(maxsize=None)
def _analyze_linear_in_v(
    obj_or_src,
    *,
    method: str = "i",
    v_param: str = "v",
    method_token=None,
):
    # 1. obtain the source text of the class
    if isinstance(obj_or_src, str):  # already text
        src = textwrap.dedent(obj_or_src)
        tree = ast.parse(src)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        fn = next(
            node
            for node in reversed(cls.body)
            if isinstance(node, ast.FunctionDef) and node.name == method
        )
    else:  # a class object or instance
        cls = obj_or_src if inspect.isclass(obj_or_src) else type(obj_or_src)
        runtime_owner, runtime_function = _runtime_method(cls, method)
        # Renamed mechanisms are created dynamically with ``type`` and carry
        # an explicit link to the class whose methods they cloned.  Resolve
        # class-level fallback source from that stable owner instead of asking
        # inspect/IPython to reverse-engineer a synthetic alias class.
        source_class = runtime_owner.__dict__.get(
            "_dendra_symbolic_source_class", runtime_owner
        )
        provenance_owner = next(
            (base for base in source_class.__mro__ if method in base.__dict__),
            source_class,
        )
        if source_class is not runtime_owner and runtime_owner.__dict__.get(
            method
        ) is provenance_owner.__dict__.get(method):
            source_owner = provenance_owner
        else:
            # Respect a method replaced on an alias after it was created.
            source_owner = runtime_owner
        try:
            # The actual runtime function is authoritative.  This handles
            # renamed/inherited methods and invalidates stale authored-class
            # assumptions when a current is deliberately hot-swapped.
            fn = _runtime_function_source(runtime_function, method)
        except SourceUnavailableError:
            if source_owner is runtime_owner and not _function_is_authored_on_class(
                runtime_function, runtime_owner, method
            ):
                raise
            fn = _class_function_source(source_owner, method)
    if fn.decorator_list:
        raise NotImplementedError(
            "Decorated current methods require an explicit analytic conductance "
            "pair or numerical differentiation."
        )

    env: dict[str, ast.AST] = {}

    # Inline locally defined temporaries against the environment that existed
    # at each statement.  Deferring all substitution until the return statement
    # makes later reassignments retroactively change earlier computations.
    def substitute(node: ast.AST):
        if isinstance(node, ast.Name) and node.id in env:
            return substitute(copy.deepcopy(env[node.id]))
        for field, val in ast.iter_fields(node):
            if isinstance(val, ast.AST):
                setattr(node, field, substitute(val))
            elif isinstance(val, list):
                setattr(node, field, [substitute(x) for x in val])
        return node

    return_expr = None
    body = fn.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    for stmt in body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if not isinstance(stmt.targets[0], ast.Name):
                raise NotImplementedError("Only simple `name = …` assignments.")
            target = stmt.targets[0].id
            if target in {"self", v_param}:
                raise ValueError(
                    "Cannot symbolically analyze assignment to reserved name "
                    f"{target!r}."
                )
            env[target] = substitute(copy.deepcopy(stmt.value))
        elif isinstance(stmt, ast.Return):
            return_expr = substitute(copy.deepcopy(stmt.value))
            break
        else:
            raise NotImplementedError("Only straight-line code is supported.")

    if return_expr is None:
        raise ValueError("No return statement found in method.")

    expanded = ast.fix_missing_locations(return_expr)

    # 4. convert to SymPy and extract coefficients
    syms: dict[str, sp.Expr] = {v_param: sp.symbols(v_param)}
    expr = sp.expand(_ast_to_sympy(expanded, syms))
    unresolved = sorted(
        symbol.name
        for symbol in expr.free_symbols
        if symbol != syms[v_param] and not symbol.name.startswith("self_")
    )
    if unresolved:
        names = ", ".join(unresolved)
        raise ValueError(
            f"Unresolved bare symbol(s) in current expression: {names}. "
            "Declare mechanism values on `self` so generated conductance code "
            "can resolve them."
        )
    if sp.simplify(sp.diff(expr, syms[v_param], 2)) != 0:
        raise ValueError(f"Expression is not linear in {v_param!r}.")
    A = expr.coeff(syms[v_param])
    B = sp.expand(A * syms[v_param] - expr)

    A, B = sp.simplify(A), sp.simplify(B)
    return A, B


def _dotify(expr: sp.Expr) -> str:
    return re.sub(r"\bself_(\w+)\b", r"self.\1", str(expr))


def linear_conductance_in_v(
    obj_or_src, *, method: str = "i", v_param: str = "v"
) -> str:
    """Return the exact voltage coefficient of a supported affine current."""
    conductance, _ = _analyze_linear_in_v(
        obj_or_src,
        method=method,
        v_param=v_param,
        method_token=_method_cache_token(obj_or_src, method),
    )
    return _dotify(conductance)


def factorize_linear_in_v(obj_or_src, *, method: str = "i", v_param: str = "v"):
    """
    Return (A, B) such that the specified *method* equals **A*v - B*A**.

    *obj_or_src* may be either a class *object* (or instance) **or** a
    source-code string containing exactly one class definition.
    """
    A, B = _analyze_linear_in_v(
        obj_or_src,
        method=method,
        v_param=v_param,
        method_token=_method_cache_token(obj_or_src, method),
    )
    if A == 0:
        raise ZeroDivisionError("A is identically zero; cannot factor B = A*C")
    C = sp.simplify(B / A)

    return _dotify(A), _dotify(C)


# Preserve the cache-control surface used by tests and interactive workflows.
# Both public views share the same underlying analysis cache.
factorize_linear_in_v.cache_clear = _analyze_linear_in_v.cache_clear
factorize_linear_in_v.cache_info = _analyze_linear_in_v.cache_info
linear_conductance_in_v.cache_clear = _analyze_linear_in_v.cache_clear
linear_conductance_in_v.cache_info = _analyze_linear_in_v.cache_info
