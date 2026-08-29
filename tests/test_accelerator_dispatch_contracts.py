from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dendra  # noqa: F401 - initialize Dendra before loading integrator modules
from dendra.models.integrators.triton import (
    TRITON_AVAILABLE,
    dhs_bt_solve_cuda,
    thomas_solve_cuda_bt,
)
from dendra.models.integrators.triton._contracts import (
    adjoint_main_blocks,
    copy_rhs_workspace,
    flatten_vmap_solver_batch,
    is_vmap_batched_tensor,
    restore_vmap_solver_batch,
    validate_block_tridiagonal,
    validate_threads,
    validate_tree,
    validate_tree_multi,
    validate_tridiagonal,
)
from dendra.models.networks import netcon_bitpack_ops, netcon_bitpack_ops_triton
from dendra.models.networks.netcon_bitpack_contracts import (
    validate_delivery_structure,
    validate_pack_structure,
)


class _AcceleratorTensor(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True


class _OtherAcceleratorTensor(torch.Tensor):
    @property
    def device(self):
        return torch.device("cuda:1")

    @property
    def is_cuda(self):
        return True


def _accelerator(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.as_subclass(_AcceleratorTensor)


def _other_accelerator(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.as_subclass(_OtherAcceleratorTensor)


def _pack_inputs(n_source: int = 64):
    n_words = (n_source + 62) // 63
    return (
        _accelerator(torch.zeros(n_source, dtype=torch.bool)),
        _accelerator(torch.zeros((4, n_words), dtype=torch.int64)),
        _accelerator(torch.zeros(1, dtype=torch.int64)),
    )


def _delivery_inputs(n_conn: int = 3):
    return (
        _accelerator(torch.zeros((4, 2), dtype=torch.int64)),
        _accelerator(torch.zeros(1, dtype=torch.int64)),
        _accelerator(torch.ones(n_conn, dtype=torch.int64)),
        _accelerator(torch.zeros(n_conn, dtype=torch.int64)),
        _accelerator(torch.ones(n_conn, dtype=torch.int64)),
        _accelerator(torch.zeros(n_conn, dtype=torch.int64)),
        _accelerator(torch.ones(n_conn, dtype=torch.float32)),
        _accelerator(torch.zeros(2, dtype=torch.float32)),
    )


class _FakeKernel:
    def __init__(self):
        self.calls = []
        self.grid = None

    def __getitem__(self, grid):
        self.grid = grid
        return self

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


@pytest.mark.parametrize("threads", [0, -1, 3, 33, 1.5, True])
def test_dhs_lane_contract_rejects_invalid_values(threads):
    with pytest.raises((TypeError, ValueError), match="threads"):
        validate_threads(threads)


def test_tridiagonal_contract_rejects_non_tensor_before_attribute_access():
    a = torch.empty((1, 1))
    c = torch.empty((1, 1))
    d = torch.empty((1, 2))
    with pytest.raises(TypeError, match="b must be a torch.Tensor"):
        validate_tridiagonal(a, object(), c, d)


def test_tridiagonal_contract_rejects_cpu_shape_and_dtype_mismatches():
    tensors = (
        torch.ones((2, 2)),
        torch.ones((2, 3)),
        torch.ones((2, 2)),
        torch.ones((2, 3)),
    )
    with pytest.raises(ValueError, match="CUDA tensors"):
        validate_tridiagonal(*tensors)

    accelerator_tensors = tuple(_accelerator(tensor) for tensor in tensors)
    with pytest.raises(ValueError, match="a must have shape"):
        validate_tridiagonal(_accelerator(torch.ones((2, 1))), *accelerator_tensors[1:])
    with pytest.raises(ValueError, match="same dtype"):
        validate_tridiagonal(accelerator_tensors[0].double(), *accelerator_tensors[1:])


def test_block_contract_rejects_cross_device_inputs():
    lower = _accelerator(torch.zeros((1, 1, 3)))
    main = _accelerator(torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1))
    upper = _other_accelerator(torch.zeros((1, 1, 3)))
    rhs = _accelerator(torch.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="same device"):
        validate_block_tridiagonal(lower, main, upper, rhs)


def test_tree_contract_validates_topology_metadata_without_device_sync():
    data = tuple(_accelerator(torch.ones((2, 3))) for _ in range(3))
    parent = _accelerator(torch.tensor([-1, 0, 0], dtype=torch.int64))
    order = _accelerator(torch.tensor([1, 2, 0], dtype=torch.int64))
    layer_ptr = _accelerator(torch.tensor([0, 2, 3], dtype=torch.int64))

    assert validate_tree(*data, parent, order, layer_ptr, 2) == (2, 3, 2)
    with pytest.raises(ValueError, match="parent_idx and order"):
        validate_tree(*data, parent[:-1], order, layer_ptr, 2)
    with pytest.raises(TypeError, match="layer_ptr"):
        validate_tree(*data, parent, order, layer_ptr.float(), 2)


def test_multi_tree_contract_rejects_partial_grid_launches():
    data = tuple(_accelerator(torch.ones((2, 3))) for _ in range(3))
    p_cat = _accelerator(torch.tensor([-1, 0, 0], dtype=torch.int64))
    order_cat = _accelerator(torch.tensor([1, 2, 0], dtype=torch.int64))
    layer_ptr_cat = _accelerator(torch.tensor([0, 2, 3], dtype=torch.int64))
    warp_metadata = [
        _accelerator(torch.tensor([0, 0], dtype=torch.int64)),
        _accelerator(torch.tensor([0, 0], dtype=torch.int64)),
        _accelerator(torch.tensor([0, 0], dtype=torch.int64)),
        _accelerator(torch.tensor([2, 2], dtype=torch.int32)),
        _accelerator(torch.tensor([0, 1], dtype=torch.int64)),
        _accelerator(torch.tensor([1, 1], dtype=torch.int32)),
    ]

    with pytest.raises(ValueError, match="grid_x must equal"):
        validate_tree_multi(
            *data,
            p_cat,
            order_cat,
            layer_ptr_cat,
            *warp_metadata,
            3,
            2,
            2,
            1,
        )


def test_block_adjoint_transposes_main_blocks_and_rhs_workspace_is_independent():
    main = torch.arange(18, dtype=torch.float64).reshape(1, 2, 3, 3)
    adjoint = adjoint_main_blocks(main)
    assert adjoint.is_contiguous()
    assert torch.equal(adjoint, main.transpose(-1, -2))

    rhs = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3).transpose(0, 1)
    workspace = copy_rhs_workspace(rhs)
    assert workspace.is_contiguous()
    assert workspace.data_ptr() != rhs.data_ptr()
    workspace.zero_()
    assert torch.count_nonzero(rhs) > 0


def test_vmap_batch_detection_is_dynamo_fullgraph_safe():
    def branch_on_transform(tensor):
        if is_vmap_batched_tensor(tensor):
            return tensor + 1
        return tensor - 1

    compiled = torch.compile(branch_on_transform, backend="eager", fullgraph=True)
    value = torch.tensor([2.0])
    torch.testing.assert_close(compiled(value), value - 1)


def test_vmap_solver_batch_helpers_flatten_broadcast_and_restore_without_cuda():
    info = SimpleNamespace(batch_size=3)
    shared = torch.arange(10, dtype=torch.float64).reshape(2, 5)
    mapped = torch.arange(30, dtype=torch.float64).reshape(2, 5, 3)

    (flat_shared, flat_mapped), solver_batch = flatten_vmap_solver_batch(
        info,
        (None, 2),
        shared,
        mapped,
    )

    assert solver_batch == 2
    assert flat_shared.shape == (6, 5)
    assert flat_mapped.shape == (6, 5)
    torch.testing.assert_close(
        flat_shared,
        shared.unsqueeze(0).expand(3, 2, 5).reshape(6, 5),
    )
    torch.testing.assert_close(
        flat_mapped,
        mapped.movedim(2, 0).reshape(6, 5),
    )
    restored = restore_vmap_solver_batch(flat_mapped, 3, solver_batch)
    torch.testing.assert_close(restored, mapped.movedim(2, 0))


def test_vmap_solver_batch_helpers_reject_missing_or_mismatched_solver_batches():
    info = SimpleNamespace(batch_size=3)
    with pytest.raises(ValueError, match="at least one solver operand"):
        flatten_vmap_solver_batch(info, ())

    shared = torch.zeros((2, 5))
    mapped = torch.zeros((3, 4, 5))
    with pytest.raises(ValueError, match="same batch size"):
        flatten_vmap_solver_batch(info, (None, 0), shared, mapped)


def test_pack_structure_rejects_wrong_word_width_and_mixed_devices():
    spikes, history, step = _pack_inputs()
    with pytest.raises(ValueError, match="wrong word width"):
        validate_pack_structure(spikes, history[:, :1], step)
    with pytest.raises(ValueError, match="same device"):
        validate_pack_structure(spikes, history, _other_accelerator(step.cpu()))


def test_delivery_structure_rejects_mismatched_vectors_and_half_precision():
    inputs = list(_delivery_inputs())
    inputs[3] = inputs[3][:-1]
    with pytest.raises(ValueError, match="conn_word_idx must contain"):
        validate_delivery_structure(*inputs)

    inputs = list(_delivery_inputs())
    inputs[-2] = _accelerator(inputs[-2].cpu().half())
    inputs[-1] = _accelerator(inputs[-1].cpu().half())
    with pytest.raises(TypeError, match="float32 or float64"):
        validate_delivery_structure(*inputs)


def test_extension_pack_dispatches_only_after_contract_validation(monkeypatch):
    calls = []
    extension = SimpleNamespace(
        pack_source_spikes=lambda *args: calls.append(args),
    )
    monkeypatch.setattr(netcon_bitpack_ops, "_load_extension", lambda: extension)
    inputs = _pack_inputs()

    assert netcon_bitpack_ops.pack_source_spikes(*inputs) is None
    assert calls == [inputs]

    noncontiguous_history = _accelerator(
        torch.zeros((2, 4), dtype=torch.int64).transpose(0, 1)
    )
    with pytest.raises(RuntimeError, match="packed_history must be contiguous"):
        netcon_bitpack_ops.pack_source_spikes(
            inputs[0], noncontiguous_history, inputs[2]
        )
    assert calls == [inputs]


def test_triton_pack_dispatches_only_after_contract_validation(monkeypatch):
    pack_kernel = _FakeKernel()
    monkeypatch.setattr(
        netcon_bitpack_ops_triton,
        "_define_kernels",
        lambda: (pack_kernel, _FakeKernel()),
    )
    inputs = _pack_inputs()

    assert netcon_bitpack_ops_triton.pack_source_spikes(*inputs) is None
    assert pack_kernel.grid == (2,)
    assert len(pack_kernel.calls) == 1


def test_extension_and_triton_delivery_dispatch_receive_validated_inputs(monkeypatch):
    extension_calls = []
    extension = SimpleNamespace(
        build_delivery=lambda *args: extension_calls.append(args),
    )
    monkeypatch.setattr(netcon_bitpack_ops, "_load_extension", lambda: extension)
    inputs = _delivery_inputs()
    assert netcon_bitpack_ops.build_delivery(*inputs) is None
    assert extension_calls == [inputs]

    delivery_kernel = _FakeKernel()
    monkeypatch.setattr(
        netcon_bitpack_ops_triton,
        "_define_kernels",
        lambda: (_FakeKernel(), delivery_kernel),
    )
    assert netcon_bitpack_ops_triton.build_delivery(*inputs) is None
    assert delivery_kernel.grid == (1,)
    assert len(delivery_kernel.calls) == 1


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_block_solver_preserves_rhs_and_solves_views(noncontiguous):
    lower = torch.randn((1, 2, 3), device="cuda", dtype=torch.float64)
    upper = torch.randn((1, 2, 3), device="cuda", dtype=torch.float64)
    main = torch.randn((1, 3, 3, 3), device="cuda", dtype=torch.float64) * 0.05
    main = main + 4.0 * torch.eye(3, device="cuda", dtype=torch.float64)
    if noncontiguous:
        rhs = torch.randn((1, 3, 6), device="cuda", dtype=torch.float64)[..., ::2]
    else:
        rhs = torch.randn((1, 3, 3), device="cuda", dtype=torch.float64)
    expected = rhs.clone()

    result = thomas_solve_cuda_bt(lower, main, upper, rhs)
    assert torch.equal(rhs, expected)

    dense = torch.zeros((1, 9, 9), device="cuda", dtype=torch.float64)
    for block in range(3):
        start = 3 * block
        dense[:, start : start + 3, start : start + 3] = main[:, block]
    for block in range(2):
        start = 3 * block
        dense[:, start + 3 : start + 6, start : start + 3] = torch.diag_embed(
            lower[:, block]
        )
        dense[:, start : start + 3, start + 3 : start + 6] = torch.diag_embed(
            upper[:, block]
        )
    reference = torch.linalg.solve(dense, expected.reshape(1, 9, 1)).reshape(1, 3, 3)
    torch.testing.assert_close(result, reference, rtol=1e-8, atol=1e-9)


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or not TRITON_AVAILABLE,
    reason="CUDA and Triton are required",
)
def test_block_tree_adjoint_handles_nonsymmetric_main_blocks():
    dtype = torch.float64
    device = torch.device("cuda")
    parent = torch.tensor([-1, 0], dtype=torch.int64, device=device)
    order = torch.tensor([1, 0], dtype=torch.int64, device=device)
    layer_ptr = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    main = torch.tensor(
        [
            [
                [[4.0, 0.2, -0.1], [0.05, 4.5, 0.3], [0.1, -0.2, 5.0]],
                [[3.7, -0.1, 0.2], [0.25, 4.2, -0.05], [-0.2, 0.1, 4.8]],
            ]
        ],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    edge = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.2, 0.3, 0.4]]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    rhs = torch.randn((1, 2, 3), dtype=dtype, device=device, requires_grad=True)

    def solve(main_arg, edge_arg, rhs_arg):
        return dhs_bt_solve_cuda(
            main_arg,
            edge_arg,
            rhs_arg,
            parent,
            order,
            layer_ptr,
            threads=2,
        )

    assert torch.autograd.gradcheck(
        solve, (main, edge, rhs), eps=1e-6, atol=1e-5, rtol=1e-4
    )
