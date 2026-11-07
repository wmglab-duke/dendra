"""Device compatibility for optional native solver backends."""


def _mps_solver_available(solvers):
    probe = getattr(solvers, "is_mps_available", None)
    return callable(probe) and probe()


def _cpu_topology(sources):
    """Copy accelerator topology only when constructing an owning CPU plan."""
    return tuple(
        source.cpu() if source.device.type == "mps" else source for source in sources
    )
