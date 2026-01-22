import ast
import io
import re
import textwrap
import tokenize
from typing import Iterable, List, Set

import torch


def match_derivative_to_states(derivative, states):
    matched = {}
    for state in states:
        for d in derivative:
            s, _ = d.split("'")
            if s == state:
                matched[state] = d
                break
    return matched


# PyTorch operations
TORCH_OPS = set(dir(torch))


class UnderscoreLHS(ast.NodeTransformer):
    """
    An AST NodeTransformer that traverses an AST and prepends an underscore
    to the variable names on the left-hand side of any assignment.
    """

    def _prefix_target(self, target_node):
        """Recursively prefixes the appropriate part of an assignment target."""
        if isinstance(target_node, ast.Name):
            # This is a simple variable name like 'a'.
            target_node.id = "_" + target_node.id
        elif isinstance(target_node, ast.Attribute):
            # This is an attribute like 'obj.value'. We change 'value' to '_value'.
            target_node.attr = "_" + target_node.attr
        elif isinstance(target_node, (ast.Tuple, ast.List)):
            # This is unpacking like 'a, b = ...'. Recurse on each element.
            for element in target_node.elts:
                self._prefix_target(element)
        elif isinstance(target_node, ast.Subscript):
            # This is an item assignment like 'd[k] = v'. Recurse on the variable 'd'.
            self._prefix_target(target_node.value)
        elif isinstance(target_node, ast.Starred):
            # This is a starred assignment like 'a, *b = ...'. Recurse on 'b'.
            self._prefix_target(target_node.value)

        return target_node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Handles simple assignments: a = b"""
        for target in node.targets:
            self._prefix_target(target)
        self.generic_visit(node)  # Ensure we visit children on the right-hand side too
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """Handles annotated assignments: a: int = b"""
        self._prefix_target(node.target)
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        """Handles augmented assignments: a += b"""
        self._prefix_target(node.target)
        self.generic_visit(node)
        return node


def add_underscore_to_lhs(code_string: str) -> str:
    """
    Parses a Python code string, adds an underscore to all variables on the
    left-hand side of assignments, and returns the modified code string.

    Args:
        code_string: A string containing one or more lines of Python code.

    Returns:
        The modified code string.

    Requires Python 3.9+ for ast.unparse().
    """
    try:
        # 1. Parse the string into an Abstract Syntax Tree
        tree = ast.parse(code_string)

        # 2. Instantiate our transformer and have it visit the tree
        transformer = UnderscoreLHS()
        new_tree = transformer.visit(tree)

        # 3. Add line numbers and other metadata back to the new tree
        ast.fix_missing_locations(new_tree)

        # 4. Unparse the modified tree back into a string
        return ast.unparse(new_tree)
    except (SyntaxError, ValueError) as e:
        print(f"Error processing code string: {e}")
        return code_string


def add_underscore_to_states(expression: str, states: List[str]) -> str:
    for state in states:
        expression = re.sub(rf"\b{state}\b", f"_{state}", expression)
    return expression


# -- function parsers & code emitters --
def extract_vars(f: str, exclude: set) -> List[str]:
    """
    Extract variable names from a given string, excluding specified names.

    Parameters
    ----------
    f : str
        The input string from which to extract variable names.
    exclude : set
        A set of variable names to exclude from the result.

    Returns
    -------
    List[str]
        A list of variable names found in the input string, excluding the specified names.
    """
    pattern = r"\b[a-zA-Z_]\w*\b"
    all_variables = set(re.findall(pattern, f))
    filtered_variables = [var for var in all_variables if var not in exclude]
    return filtered_variables


def replace(input_string: str, replace_list: List[str]) -> str:
    """
    Replace occurrences of substrings in the input string with their 'self.' prefixed versions.

    Parameters
    ----------
    input_string : str
        The string in which to replace substrings.
    replace_list : list of str
        A list of substrings to be replaced.

    Returns
    -------
    str
        The modified string with specified substrings replaced by 'self.' prefixed versions.
    """
    for substring in replace_list:
        input_string = re.sub(rf"\b{substring}\b", f"self.{substring}", input_string)
    return input_string


def _find_local_defs(src: str) -> Set[str]:
    """Collect names defined by `def`, `class`, or simple assignment."""
    try:
        tree = ast.parse(textwrap.dedent(src))
    except SyntaxError:
        return set()

    names: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, n):  # def foo():
            names.add(n.name)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, n):  # class Foo:
            names.add(n.name)

        def visit_Assign(self, n):  # x = …
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)

    V().visit(tree)
    return names


# ─────────────────────── main transformation ──────────────────────────
def modify_operations(
    code: str,
    torch_operations: Iterable[str] = TORCH_OPS,
) -> str:
    """
    Prefix bare calls to *torch_operations* with ``torch.`` while preserving
    whitespace, comments, and strings.

    If *code* cannot be parsed as valid Python (e.g. because Hypothesis
    injected unmatched quotes, stray control bytes, etc.), it is returned
    **unchanged**.
    """
    # ── 0. Bail out early on syntactically invalid snippets ────────────
    try:
        ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return code

    # ── 1. local defs for shadowing detection ──────────────────────────
    local_defs = _find_local_defs(code)
    ops = set(torch_operations)

    # ── 2. Tokenise; leave untouched on lexical errors (NULL byte, …) ──
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, UnicodeDecodeError):
        return code

    # Build line-offset table for fast (line, col) → absolute_index
    line_offsets = [0]
    for ln in code.splitlines(keepends=True):
        line_offsets.append(line_offsets[-1] + len(ln))

    def abs_index(pos):
        line, col = pos
        return line_offsets[line - 1] + col

    SIGNIF = {
        tokenize.NAME,
        tokenize.NUMBER,
        tokenize.OP,
        tokenize.STRING,
        tokenize.ERRORTOKEN,
    }
    fstring_start = getattr(tokenize, "FSTRING_START", None)
    fstring_end = getattr(tokenize, "FSTRING_END", None)

    def next_sig(idx):
        j = idx + 1
        while j < len(tokens) and tokens[j].type not in SIGNIF:
            j += 1
        return j

    def prev_sig(idx):
        j = idx - 1
        while j >= 0 and tokens[j].type not in SIGNIF:
            j -= 1
        return j

    out, cursor, fdepth = [], 0, 0

    for i, tok in enumerate(tokens):
        ttype, tstr, (sl, sc), (el, ec), _ = tok

        # Track f-string expression nesting
        if fstring_start is not None and ttype == fstring_start:
            fdepth += 1
        elif fstring_end is not None and ttype == fstring_end:
            fdepth -= 1

        # Absolute positions in the *original* source
        start = abs_index((sl, sc))
        end = abs_index((el, ec))

        # Copy text that lies *before* this token (whitespace, comments …)
        if cursor < start:
            out.append(code[cursor:start])

        # Decide whether to rewrite this NAME
        is_candidate = (
            ttype == tokenize.NAME
            and fdepth == 0
            and tstr in ops
            and tstr not in local_defs
        )
        if is_candidate:
            j = next_sig(i)
            k = prev_sig(i)
            call_follows = j < len(tokens) and tokens[j].string == "("
            dot_before = k >= 0 and tokens[k].string == "."
            string_before = k >= 0 and tokens[k].type == tokenize.STRING
            if call_follows and not dot_before and not string_before:
                out.append(f"torch.{tstr}")
                cursor = end
                continue  # done with this token

        # default: keep token text verbatim
        out.append(code[start:end])
        cursor = end

    # trailing text (e.g. final newline)
    out.append(code[cursor:])
    return "".join(out)
