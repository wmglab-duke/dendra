"""Strict-warning tests permit only known upstream compiler warnings."""

import warnings

import pytest
import torch

_NONLEAF_GRAD_MESSAGE = (
    "The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. "
    "Its .grad attribute won't be populated during autograd.backward(). "
    "If you indeed want the .grad field to be populated for a non-leaf Tensor, "
    "use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor "
    "by mistake, make sure you access the leaf Tensor instead. "
    "See github.com/pytorch/pytorch/pull/30531 for more information."
)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("category", [FutureWarning, DeprecationWarning])
@pytest.mark.parametrize(
    "name, module",
    [
        ("script", "torch.jit._script"),
        ("script_method", "torch.jit._script"),
        ("trace", "torch.jit._trace"),
    ],
)
@pytest.mark.parametrize("quoted", [False, True])
def test_known_torchscript_deprecation_is_nonfatal_under_warning_errors(
    category, name, module, quoted
):
    function = f"torch.jit.{name}"
    if quoted:
        function = f"`{function}`"
    warnings.warn_explicit(
        f"{function} is deprecated. Please switch to torch.compile or torch.export.",
        category,
        filename="torch/jit/_script.py",
        lineno=1,
        module=module,
    )


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "suffix", ["", " (Triggered internally at /torch/aten/TensorBody.h:493.)"]
)
@pytest.mark.parametrize(
    "module", ["torch._dynamo.variables.builder", "torch._subclasses.meta_utils"]
)
def test_dynamo_internal_nonleaf_grad_inspection_is_nonfatal(suffix, module):
    warnings.warn_explicit(
        _NONLEAF_GRAD_MESSAGE + suffix,
        UserWarning,
        filename="torch/_dynamo/variables/builder.py",
        lineno=1,
        module=module,
    )


@pytest.mark.filterwarnings("error")
def test_user_nonleaf_grad_access_still_raises():
    leaf = torch.tensor(2.0, requires_grad=True)
    nonleaf = leaf.square()
    with pytest.raises(UserWarning, match=r"\.grad attribute.*not a leaf Tensor"):
        _ = nonleaf.grad


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "message, category, module",
    [
        ("unrelated future warning", FutureWarning, "torch.jit._script"),
        ("unrelated deprecation", DeprecationWarning, "torch.jit._script"),
        ("`torch.jit.script` is deprecated.", FutureWarning, "user_module"),
        ("`torch.jit.script` is deprecated.", RuntimeWarning, "torch.jit._script"),
        ("`torch.jit.script` is unsupported.", FutureWarning, "torch.jit._script"),
        ("unrelated trace warning", FutureWarning, "torch.jit._trace"),
        ("`torch.jit.trace` is deprecated.", FutureWarning, "user_module"),
        ("`torch.jit.trace` is deprecated.", UserWarning, "torch.jit._trace"),
        ("`torch.jit.script` is deprecated.", FutureWarning, "torch.jit._trace"),
        (_NONLEAF_GRAD_MESSAGE, UserWarning, "user_module"),
        (_NONLEAF_GRAD_MESSAGE, FutureWarning, "torch._dynamo.variables.builder"),
        ("unrelated builder warning", UserWarning, "torch._dynamo.variables.builder"),
        ("unrelated meta warning", UserWarning, "torch._subclasses.meta_utils"),
        (
            _NONLEAF_GRAD_MESSAGE + " Extra warning.",
            UserWarning,
            "torch._dynamo.variables.builder",
        ),
    ],
)
def test_other_warnings_remain_errors(message, category, module):
    with pytest.raises(category, match="."):
        warnings.warn_explicit(
            message,
            category,
            filename="warning_source.py",
            lineno=1,
            module=module,
        )
