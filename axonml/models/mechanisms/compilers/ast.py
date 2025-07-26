from __future__ import annotations
import ast, inspect, textwrap, sympy as sp, re
from sympy import symbols, sympify, Poly, expand, factor
from functools import lru_cache

from axonml.helpers import DEBUG
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
        l, r = (
            _ast_to_sympy(node.left, local_syms),
            _ast_to_sympy(node.right, local_syms),
        )
        return {
            ast.Add: l + r,
            ast.Sub: l - r,
            ast.Mult: l * r,
            ast.Div: l / r,
            ast.Pow: l**r,
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
    A = expr.coeff(syms[v_param])
    B = sp.expand(A * syms[v_param] - expr)

    A, B = sp.simplify(A), sp.simplify(B)
    if A == 0:
        raise ZeroDivisionError("A is identically zero; cannot factor B = A*C")
    C = sp.simplify(B / A)

    def _dotify(expr: sp.Expr) -> str:
        return re.sub(r"\bself_(\w+)\b", r"self.\1", str(expr))

    return _dotify(A), _dotify(C)


def replace_v(code_str):
    # Use a regex with word boundaries to ensure only standalone 'v' is replaced.
    # The replacement inserts '(v + v_n) / 2' in place of v.
    return re.sub(r"\bv\b", "(v + v_n) / 2", code_str)


@lru_cache(maxsize=None)
def factor_linear_in_x_from_codeblock(code_str, x_var="v_n"):
    lines = code_str.strip().split("\n")

    # Identify self-prefixed variables
    pattern = r"self\.(\w+)"
    self_vars_all = re.findall(pattern, code_str)
    self_vars_all = set(self_vars_all)
    self_mapping = {var: f"self.{var}" for var in self_vars_all}

    env = {}

    def parse_expr(expr_str):
        # Extract potential variables
        potential_vars = set(re.findall(r"[a-zA-Z_]\w*", expr_str))
        for var in potential_vars:
            if var not in env:
                env[var] = symbols(var, real=True)
        return sympify(expr_str, locals=env)

    final_expr = None

    # Parse line by line
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_no_self = line.replace("self.", "")

        if line_no_self.startswith("return "):
            return_expr_str = line_no_self[len("return ") :].strip()
            final_expr = parse_expr(return_expr_str)
        elif "=" in line_no_self:
            lhs, rhs = line_no_self.split("=", 1)
            var_name = lhs.strip()
            rhs_expr_str = rhs.strip()
            rhs_expr = parse_expr(rhs_expr_str)
            env[var_name] = rhs_expr
        else:
            final_expr = parse_expr(line_no_self)

    if DEBUG:
        print(f"Final expression in mech factorization: {final_expr}")

    if final_expr is None:
        raise ValueError("No final expression or return statement found.")

    if x_var not in env:
        env[x_var] = symbols(x_var, real=True)
    x = env[x_var]

    # Factor the final_expr as A + B*x
    expr_expanded = expand(final_expr)
    p = Poly(expr_expanded, x)

    if p.degree() != 1:
        raise ValueError("Expression is not linear in x.")

    A = p.eval(0)
    B = p.coeff_monomial(x)

    # Now factor each of A and B individually
    A_factor = factor(A)
    B_factor = factor(B)

    # Convert to strings
    A_str = str(A_factor)
    B_str = str(B_factor)

    # Restore self. prefixes
    for var in sorted(self_mapping.keys(), key=len, reverse=True):
        A_str = re.sub(rf"\b{var}\b", self_mapping[var], A_str)
        B_str = re.sub(rf"\b{var}\b", self_mapping[var], B_str)

    return A_str, B_str
