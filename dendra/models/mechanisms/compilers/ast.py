from __future__ import annotations

import ast
import re
import textwrap
from functools import lru_cache

import sympy as sp

from .source import safe_source


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
        left, right = (
            _ast_to_sympy(node.left, local_syms),
            _ast_to_sympy(node.right, local_syms),
        )
        return {
            ast.Add: left + right,
            ast.Sub: left - right,
            ast.Mult: left * right,
            ast.Div: left / right,
            ast.Pow: left**right,
        }[type(node.op)]
    if isinstance(node, ast.UnaryOp):
        o = _ast_to_sympy(node.operand, local_syms)
        return {ast.UAdd: +o, ast.USub: -o}[type(node.op)]
    if isinstance(node, ast.Call):  # pow(a, b)
        if isinstance(node.func, ast.Name) and node.func.id == "pow":
            a, b = (_ast_to_sympy(arg, local_syms) for arg in node.args[:2])
            return a**b
    raise NotImplementedError(f"Unsupported AST node: {ast.dump(node)}")


# public API
@lru_cache(maxsize=None)
def factorize_linear_in_v(obj_or_src, *, method: str = "i", v_param: str = "v"):
    """
    Return (A, B) such that the specified *method* equals **A*v - B*A**.

    *obj_or_src* may be either a class *object* (or instance) **or** a
    source-code string containing exactly one class definition.
    """
    # 1. obtain the source text of the class
    if isinstance(obj_or_src, str):  # already text
        src = textwrap.dedent(obj_or_src)
    else:  # a class object
        try:
            src = safe_source(obj_or_src)
        except OSError as e:  # happens e.g. for built‑ins or eval‑crafted
            raise ValueError("Can't retrieve source for the supplied class") from e
        src = textwrap.dedent(src)

    # 2. parse, locate target method, record simple assignments
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method
    )

    env: dict[str, ast.AST] = {}
    return_expr = None
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if not isinstance(stmt.targets[0], ast.Name):
                raise NotImplementedError("Only simple `name = …` assignments.")
            env[stmt.targets[0].id] = stmt.value
        elif isinstance(stmt, ast.Return):
            return_expr = stmt.value
            break
        else:
            raise NotImplementedError("Only straight-line code is supported.")

    if return_expr is None:
        raise ValueError("No return statement found in method.")

    # 3. inline locally‑defined temporaries
    def substitute(node: ast.AST):
        if isinstance(node, ast.Name) and node.id in env:
            return substitute(env[node.id])
        for field, val in ast.iter_fields(node):
            if isinstance(val, ast.AST):
                setattr(node, field, substitute(val))
            elif isinstance(val, list):
                setattr(node, field, [substitute(x) for x in val])
        return node

    expanded = ast.fix_missing_locations(substitute(return_expr))

    # 4. convert to SymPy and extract coefficients
    syms: dict[str, sp.Expr] = {v_param: sp.symbols(v_param)}
    expr = sp.expand(_ast_to_sympy(expanded, syms))
    if sp.simplify(sp.diff(expr, syms[v_param], 2)) != 0:
        raise ValueError(f"Expression is not linear in {v_param!r}.")
    A = expr.coeff(syms[v_param])
    B = sp.expand(A * syms[v_param] - expr)

    A, B = sp.simplify(A), sp.simplify(B)
    if A == 0:
        raise ZeroDivisionError("A is identically zero; cannot factor B = A*C")
    C = sp.simplify(B / A)

    def _dotify(expr: sp.Expr) -> str:
        return re.sub(r"\bself_(\w+)\b", r"self.\1", str(expr))

    return _dotify(A), _dotify(C)
