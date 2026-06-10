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
@given(
    st.text(
        alphabet=st.characters(blacklist_categories=("Cs",)), min_size=1, max_size=100
    )
)
def test_roundtrip_valid_python(code):
    """
    If the original snippet parses, the transformed snippet should also parse.
    """
    try:
        compile(code, "<src>", "exec")
    except SyntaxError:
        pytest.skip("Original snippet isn't valid Python")

    transformed = modify_operations(code, TORCH_OPS)
    compile(transformed, "<dst>", "exec")  # should not raise
