try:
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    from .bt_kernel import thomas_solve_cuda_bt
    from .bt_spd_kernel import (
        solve_bt_spd_cuda,
        solve_bt_spd_cuda_consume,
        solve_bt_spd_cuda_consume_unchecked,
    )
    from .dhs_kernel import dhs_solve_cuda
    from .dhs_kernel_bt import dhs_bt_solve_cuda
    from .dhs_kernel_multi import dhs_solve_multi_cuda
    from .pcr_kernel_thomas import pcr_solve_cuda_t
    from .t_kernel_thomas import thomas_solve_cuda_t
else:
    thomas_solve_cuda_bt = None
    solve_bt_spd_cuda = None
    solve_bt_spd_cuda_consume = None
    solve_bt_spd_cuda_consume_unchecked = None
    dhs_solve_cuda = None
    dhs_bt_solve_cuda = None
    dhs_solve_multi_cuda = None
    thomas_solve_cuda_t = None
    pcr_solve_cuda_t = None
