"""Private solver diagnostics for captured Population timesteps."""

from contextvars import ContextVar

import torch

_active_checks = ContextVar("dendra_solver_failure_capture", default=None)


def _capturing_solver_checks():
    return _active_checks.get() is not None


def _record_solver_check(accepted, solver):
    checks = _active_checks.get()
    if checks is None:
        return False
    label = getattr(solver, "_solver_failure_label", type(solver).__name__)
    checks.append((accepted, label if solver is not None else "State solve"))
    return True


def _first_step_failure(checks):
    # Check order is the order in which the authored transition ran its solves.
    # Each member index is local to the corresponding mechanism's support.
    event = checks[0][0].new_full((2,), -1, dtype=torch.int64)
    for site in reversed(range(len(checks))):
        accepted, _ = checks[site]
        flat = accepted.flatten()
        size = flat.numel()
        if size == 0:
            continue
        # Use integer indices because the indexed Boolean reduction is not
        # supported by every compiler backend.
        if size == 1:
            member = flat[0].to(torch.int64)
        else:
            indices = torch.arange(size, dtype=torch.int64, device=flat.device)
            member = torch.where(flat, size, indices).amin()
        candidate = torch.stack((torch.full_like(member, site), member))
        event = torch.where(member < size, candidate, event)
    return event


def _advance_solver_failure(carry, event):
    # The first field counts completed steps. The rest is the first failure's
    # zero-based chunk step, solver site, and flat local batch index.
    cursor, step, site, member = carry[0].unbind()
    take = (step < 0) & (event[0] >= 0)
    previous = torch.stack((step, site, member))
    failure = torch.where(take, torch.cat((cursor.reshape(1), event)), previous)
    return (torch.cat(((cursor + 1).reshape(1), failure)),)


class _SolverFailureCapture:
    """Collect step checks during FX capture, then check one payload per chunk."""

    def __init__(self):
        self._sites = ()

    @property
    def active(self):
        return bool(self._sites)

    def wrap(self, body):
        def authored(*args):
            checks = []
            token = _active_checks.set(checks)
            try:
                updated, emitted = body(*args)
            finally:
                _active_checks.reset(token)
            # Retain only static descriptions. The graph owns the tensors.
            self._sites = tuple((label, tuple(ok.shape)) for ok, label in checks)
            if checks:
                return updated, (emitted, _first_step_failure(checks))
            return updated, emitted

        return authored

    def initial(self, device):
        if not self.active:
            return ()
        return (torch.tensor((0, -1, -1, -1), dtype=torch.int64, device=device),)

    def check(self, carry):
        if not carry:
            return
        # Transfer the whole payload once, at the host publication boundary.
        _, step, site, member = carry[0].tolist()
        if step >= 0:
            label, shape = self._sites[site]
            indices = []
            for size in reversed(shape):
                member, index = divmod(member, size)
                indices.append(index)
            batch = tuple(reversed(indices))
            raise RuntimeError(
                f"derivimplicit failed to converge in {label} at chunk step {step}, "
                f"batch index {batch}"
            )
