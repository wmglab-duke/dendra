import pytest
from hypothesis import given
from hypothesis import strategies as st

from dendra.models.mechanisms._solve_utils import modify_operations

TORCH_OPS = {"sin", "cos", "exp"}

CASES = [
    # --- ordinary rewrites -------------------------------------------
    ("y = sin(x)", "y = torch.sin(x)"),
    ("a = cos(b) + exp(c)", "a = torch.cos(b) + torch.exp(c)"),
    # --- should NOT rewrite ------------------------------------------
    ("y = math.sin(x)", "y = math.sin(x)"),  # attribute access
    ("y = obj.sin(x)", "y = obj.sin(x)"),
    ("print('sin(x)')", "print('sin(x)')"),  # string literal
    ("# sin(x) comment", "# sin(x) comment"),  # comment
    ("code = f'{sin(3)}'", "code = f'{sin(3)}'"),  # f-string expression
    (
        "def sin(x): return x\ny = sin(3)",  # local definition shadows torch.sin
        "def sin(x): return x\ny = sin(3)",
    ),
    # multi-line call split with backslash
    ("z = sin(\n        a + b\n)\n", "z = torch.sin(\n        a + b\n)\n"),
]


@pytest.mark.parametrize("src, expected", CASES)
def test_static_cases(src, expected):
    assert modify_operations(src, TORCH_OPS) == expected


# --------------------------------------------------------------------
# Property-based test: never touch quotes or comments
# --------------------------------------------------------------------
QUOTE_CHARS = st.sampled_from(['"', "'"])


@given(st.text(min_size=1), QUOTE_CHARS, st.text(min_size=1))
def test_no_change_inside_strings(prefix, quote, suffix):
    """
    Random string of the form  foo "<prefix> sin(x) <suffix>" bar
    must stay identical because the call is inside quotes.
    """
    code = f"foo {quote}{prefix} sin(x) {suffix}{quote} bar"
    assert modify_operations(code, TORCH_OPS) == code


# --------------------------------------------------------------------
# Sanity: modified code should remain valid Python (round-trip)
# --------------------------------------------------------------------
PYTHON_STRINGS = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)), max_size=30
)
PYTHON_EXPRESSIONS = st.recursive(
    st.one_of(
        st.integers(min_value=-1000, max_value=1000).map(repr),
        st.sampled_from(("x", "y", "z")),
        PYTHON_STRINGS.map(repr),
    ),
    lambda expression: st.one_of(
        st.tuples(
            st.sampled_from(("sin", "cos", "exp", "math.sin", "obj.cos")),
            expression,
        ).map(lambda call: f"{call[0]}({call[1]})"),
        st.tuples(expression, st.sampled_from(("+", "-", "*", "/")), expression).map(
            lambda binary: f"({binary[0]} {binary[1]} {binary[2]})"
        ),
        expression.map(lambda operand: f"-({operand})"),
    ),
    max_leaves=12,
)


@st.composite
def valid_python_snippets(draw):
    expressions = draw(st.lists(PYTHON_EXPRESSIONS, min_size=1, max_size=4))
    label = draw(PYTHON_STRINGS)
    statements = [f"label = {label!r}  # sin(x) inside a comment"]
    for index, expression in enumerate(expressions):
        operation = draw(st.sampled_from(sorted(TORCH_OPS)))
        statements.append(f"value_{index} = {operation}({expression})")
    return "\n".join(statements) + "\n"


@given(valid_python_snippets())
def test_roundtrip_valid_python(code):
    """Generated assignments remain valid after rewriting nested math calls."""
    compile(code, "<src>", "exec")
    transformed = modify_operations(code, TORCH_OPS)
    assert transformed != code  # Each snippet contains at least one bare call.
    compile(transformed, "<dst>", "exec")  # should not raise
