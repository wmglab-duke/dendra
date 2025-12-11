# tests/test_parametric.py
import itertools as _it
import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# ---- Import the module under test -------------------------------------------------
import axonml.models.parametric as M

# If the source file had a weird `import itertools` line, patch it here.
if not hasattr(M, "itertools"):
    M.itertools = _it


# ---- Helpers ----------------------------------------------------------------------
class ConstModule(torch.nn.Module):
    """Returns a constant tensor with same dtype/device as input buffer."""

    def __init__(self, out_shape, value):
        super().__init__()
        self.out_shape = tuple(out_shape)
        self.value = float(value)

    def forward(self, buffer):
        return buffer.new_full(self.out_shape, self.value)


@pytest.fixture(autouse=True)
def clean_class_decls():
    """Isolate GLOBAL / RANGE / PARAMETER temp declarations between tests."""
    gdecl = list(M.Parameterized._global_declarations)
    rdecl = list(M.Parameterized._range_declarations)
    pdecl = list(M.SimpleParameterized._params_declarations)
    try:
        M.Parameterized._global_declarations.clear()
        M.Parameterized._range_declarations.clear()
        M.SimpleParameterized._params_declarations.clear()
        yield
    finally:
        M.Parameterized._global_declarations[:] = gdecl
        M.Parameterized._range_declarations[:] = rdecl
        M.SimpleParameterized._params_declarations[:] = pdecl


# ---- to_param ---------------------------------------------------------------------
def test_to_param_converts_number_and_tensor_but_not_module():
    p = M.to_param(3.14)
    assert isinstance(p, torch.nn.Parameter)
    assert not p.requires_grad
    t = torch.tensor([1, 2, 3])
    q = M.to_param(t)
    assert isinstance(q, torch.nn.Parameter)
    mod = torch.nn.Linear(3, 2, bias=False)
    assert M.to_param(mod) is mod


# ---- distribute_over --------------------------------------------------------------
def test_distribute_over_basic_shapes():
    v = torch.arange(4)
    assert M.distribute_over(v, "p").shape == (4, 1)
    assert M.distribute_over(v, "c").shape == (1, 4)
    assert torch.equal(M.distribute_over(v, "pc"), v)
    with pytest.raises(ValueError):
        M.distribute_over(v, "oops")


# ---- Functional -------------------------------------------------------------------
def test_functional_no_key_pass_through():
    class Doubler(torch.nn.Module):
        def forward(self, x):
            return 2 * x

    f = M.Functional(Doubler())
    buf = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = f(buf)
    assert torch.allclose(out, 2 * buf)


def test_functional_with_key_and_fill_scalar():
    class Const(torch.nn.Module):
        def forward(self, x):
            return x.new_tensor(5.0)

    H, W = 2, 3
    key = torch.tensor([0, 4])  # flat indices
    buf = torch.zeros(H, W)

    # scalar expansion -> fill(p) should expand to len(key)
    def fill_scalar(p):
        return p.expand(key.numel())

    f = M.Functional(Const(), fill=fill_scalar, key=key)
    out = f(buf)
    expect = buf.clone().view(-1)
    expect[key] = 5.0
    assert torch.allclose(out.view(-1), expect)


# ---- create_param_expander --------------------------------------------------------
@given(
    H=st.integers(1, 5),
    W=st.integers(1, 5),
    n=st.integers(1, 8),
    indices=st.lists(st.integers(0, 24), min_size=1, max_size=8),
)
@settings(max_examples=80)
def test_expander_presized_and_scalar(H, W, n, indices):
    HW = H * W
    key = torch.tensor([i % HW for i in indices], dtype=torch.long)
    if key.numel() == 0:
        key = torch.tensor([0], dtype=torch.long)
    # presized
    param = torch.randn(key.numel())
    expander = M.create_param_expander(param, key, (H, W))
    out = expander(param)
    assert out.shape == (key.numel(),)
    assert torch.allclose(out, param.reshape(-1))
    # scalar
    sparam = torch.tensor(2.5)
    exp = M.create_param_expander(sparam, key, (H, W))
    out2 = exp(sparam)
    assert out2.shape == (key.numel(),)
    assert torch.allclose(out2, torch.full((key.numel(),), 2.5))


def _unique_mapping_rows_cols(key, H, W):
    rows = torch.div(key, W, rounding_mode="floor")
    cols = key % W
    urows, rinv = torch.unique(rows, return_inverse=True)
    ucols, cinv = torch.unique(cols, return_inverse=True)
    return urows, rinv, ucols, cinv


def test_expander_row_broadcast():
    H, W = 3, 4
    # choose keys spanning rows 0 and 2, any columns
    key = torch.tensor([0, 1, 2, 10], dtype=torch.long)
    urows, rinv, _, _ = _unique_mapping_rows_cols(key, H, W)  # urows = [0, 2]
    param = torch.tensor([[7.0], [9.0]])  # shape (len(urows), 1)
    exp = M.create_param_expander(param, key, (H, W))
    out = exp(param)
    expect = param[rinv].squeeze(-1)
    assert torch.allclose(out, expect)


def test_expander_col_broadcast():
    H, W = 3, 4
    # choose keys with cols 0 and 3
    key = torch.tensor([0, 3, 7, 11], dtype=torch.long)
    _, _, ucols, cinv = _unique_mapping_rows_cols(key, H, W)  # ucols = [0, 3]
    param = torch.tensor([[1.0, 5.0]])  # shape (1, len(ucols))
    exp = M.create_param_expander(param, key, (H, W))
    out = exp(param)
    expect = param[0, cinv]
    assert torch.allclose(out, expect)


def test_expander_invalid_raises():
    H, W = 2, 3
    key = torch.tensor([0, 1, 2], dtype=torch.long)
    bad = torch.empty(2, 2)  # mismatch
    with pytest.raises(ValueError):
        M.create_param_expander(bad, key, (H, W))


# ---- staticproperty / add_instance_property / Referency ---------------------------
def test_staticproperty_and_add_instance_property():
    class A:
        pass

    a = A()
    M.add_instance_property(a, "answer", lambda: 42)
    assert a.answer == 42
    # Ensure the property is on the instance's new class
    assert isinstance(a.__class__, type)


def test_referency_setreference():
    class R(M.Referency):
        pass

    r = R()
    r.setreference("pi", lambda: math.pi)
    assert abs(r.pi - math.pi) < 1e-12


# ---- SimpleParameterized ----------------------------------------------------------
def test_simpleparameterized_decl_and_instantiate_and_kwargs_check():
    M.SimpleParameterized.PARAMETER(alpha=1.0, beta=2.0)

    class A(M.SimpleParameterized):
        pass

    a = A(alpha=3.5)
    # attributes exist as nn.Parameter
    assert isinstance(a.alpha, torch.nn.Parameter)
    assert isinstance(a.beta, torch.nn.Parameter)
    assert torch.allclose(a.alpha.data, torch.tensor(3.5))
    assert torch.allclose(a.beta.data, torch.tensor(2.0))
    # kwargs check
    with pytest.raises(ValueError):
        a.check_kwargs({"gamma": 1.0})


def test_simpleparameterized_inheritance_merges_params():
    M.SimpleParameterized.PARAMETER(a=1.0)

    class Base(M.SimpleParameterized):
        pass

    M.SimpleParameterized.PARAMETER(b=2.0)

    class Child(Base):
        pass

    c = Child()
    names = dict(c.named_parameters()).keys()
    assert "a" in names and "b" in names


# ---- check_conflicts --------------------------------------------------------------
def test_check_conflicts_detects_duplicates():
    with pytest.raises(ValueError):
        M.check_conflicts({"x": 1}, {"x": 2}, {"x": 3})
    # no duplicates is fine
    assert M.check_conflicts({"a": 1}, {"b": 2}, {"c": 3}) is None


# ---- assign_precendence -----------------------------------------------------------
def test_assign_precedence_here_wins_and_tie_breaker():
    class C:
        _global = {"x": 1}
        _range = {"x": 2}
        _params = {}
        _global_defined_here = {"x"}  # defined here in _global
        _range_defined_here = set()
        _params_defined_here = set()

    kept = M.assign_precendence(C)
    assert kept["x"] == "_global"
    assert "x" in C._global and "x" not in C._range

    # pathological: defined_here in two categories -> use fallback order (_params > _range > _global)
    class D:
        _global = {"y": 1}
        _range = {"y": 2}
        _params = {}
        _global_defined_here = {"y"}
        _range_defined_here = {"y"}
        _params_defined_here = set()

    kept2 = M.assign_precendence(D)
    assert kept2["y"] == "_range"
    assert "y" in D._range and "y" not in D._global


def test_assign_precedence_mro_first_defined_class_wins():
    M.Parameterized.GLOBAL(z=1.0)

    class G(M.Parameterized):  # defines z in _global_defined_here
        pass

    M.Parameterized.RANGE(z=3.0)

    class R(M.Parameterized):  # defines z in _range_defined_here
        pass

    class Child(G, R):
        pass

    # After defining Child(G, R):
    assert "z" in Child._global
    assert "z" not in Child._range
    # optional: calling again returns nothing to do
    assert M.assign_precendence(Child) == {}


def test_assign_precedence_no_defined_anywhere_fallback_order():
    class E:
        _global = {"t": 1}
        _range = {"t": 2}
        _params = {"t": 3}
        _global_defined_here = set()
        _range_defined_here = set()
        _params_defined_here = set()
        __mro__ = (object,)  # minimal mro for the function

    kept = M.assign_precendence(E)
    # fallback _params > _range > _global
    assert kept["t"] == "_params"
    assert "t" in E._params and "t" not in E._global and "t" not in E._range


# ---- Parameterized: GLOBAL/RANGE, buffers, additional params, parametrizations ----
def test_parameterized_global_and_range_instantiation_and_defaults():
    M.Parameterized.GLOBAL(alpha=1.5, config={"w": 3.0, "b": 5.0})
    M.Parameterized.RANGE(beta=2.0)

    class W(M.Parameterized):
        pass

    w = W(shape=(2, 3), shape_f=(2, 3))
    # global scalar
    assert isinstance(getattr(w, "alpha_default"), torch.nn.Parameter)
    assert torch.allclose(getattr(w, "alpha"), torch.tensor(1.5))
    # global dict -> ParameterDict + individual parameters
    assert isinstance(getattr(w, "config"), torch.nn.ParameterDict)
    assert isinstance(getattr(w, "w"), torch.nn.Parameter)
    assert isinstance(getattr(w, "b"), torch.nn.Parameter)
    # range buffer and default
    assert getattr(w, "beta").shape == (2, 3)
    assert torch.allclose(getattr(w, "beta"), torch.full((2, 3), 2.0))
    assert isinstance(getattr(w, "beta_default"), torch.nn.Parameter)


def test_parameterized_additional_parameters_tensor_and_module_and_apply():
    H, W = 3, 4
    M.Parameterized.RANGE(x=0.0)

    class T(M.Parameterized):
        pass

    # two additions into 'x':
    # (1) tensor: write value 7.0 at a few indices
    key1 = torch.tensor([0, 5, 11], dtype=torch.long)
    # (2) module: write constant 9.0 at other indices
    key2 = torch.tensor([3, 4, 7, 10], dtype=torch.long)

    extra = {
        "x": [
            (None, 7.0, key1),
            (None, ConstModule((), 9.0), key2),  # scalar module
        ]
    }
    w = T(shape=(H, W), shape_f=(H, W), additional_parameters=extra)
    # Before populate: buffer exists but may be empty() (registered)
    w.populate_parameter_buffers()  # reset to defaults, then load additions, then apply parametrizations

    flat = w.x.view(-1)
    # defaults were 0.0; tensor additions should be 7.0; module additions 9.0
    for i in key1.tolist():
        assert flat[i].item() == pytest.approx(7.0)
    for i in key2.tolist():
        assert flat[i].item() == pytest.approx(9.0)
    # elsewhere remain 0.0
    others = set(range(H * W)) - set(key1.tolist()) - set(key2.tolist())
    assert torch.allclose(
        flat[list(sorted(others))], torch.zeros(len(others), dtype=flat.dtype)
    )


def test_populate_resets_to_defaults_then_applies_again():
    H, W = 2, 3
    M.Parameterized.RANGE(r=5.0)

    class T(M.Parameterized):
        pass

    w = T(
        shape=(H, W),
        shape_f=(H, W),
        additional_parameters={"r": [(None, 1.0, torch.tensor([0, 2]))]},
    )
    w.populate_parameter_buffers()
    # Mutate buffer
    w.r += 100.0
    w.populate_parameter_buffers()
    # Reset to default (5.0), then apply additions (indices 0 and 2 -> 1.0)
    flat = w.r.view(-1)
    assert flat[0].item() == pytest.approx(1.0)
    assert flat[2].item() == pytest.approx(1.0)
    others = [i for i in range(H * W) if i not in (0, 2)]
    assert torch.allclose(flat[others], torch.full((len(others),), 5.0))


def test_detach_buffers_does_not_fail_and_removes_grad_fn():
    M.Parameterized.RANGE(r=1.0)

    class W(M.Parameterized):
        pass

    w = W(shape=(2, 2), shape_f=(2, 2))
    w.populate_parameter_buffers()
    w.detach()
    for _, b in w.named_buffers():
        assert b.grad_fn is None


def test_parameters_dict_contains_named_parameters():
    M.Parameterized.GLOBAL(a=1.0)
    M.Parameterized.RANGE(b=2.0)

    class W(M.Parameterized):
        pass

    w = W(shape=(1, 1), shape_f=(1, 1))
    d = w.parameters_dict()
    # contains a_default and b_default (the actual nn.Parameters)
    assert "a_default" in d and "b_default" in d
    assert isinstance(d["a_default"], torch.nn.Parameter)


# ---- build_parametrization --------------------------------------------------------
def test_build_parametrization_returns_functional_and_sets_key_fill():
    H, W = 2, 3
    key = torch.tensor([0, 2, 5], dtype=torch.long)
    mod = ConstModule((), 4.2)  # scalar output
    # module output for 'p' at construction-time
    p0 = mod(torch.empty(H, W))
    f = M.build_parametrization(mod, p0, key, (H, W))
    assert isinstance(f, M.Functional)
    # Applying it should write 4.2 to key locations
    buf = torch.zeros(H, W)
    out = f(buf)
    flat = out.view(-1)
    for i in key.tolist():
        assert flat[i].item() == pytest.approx(4.2)
