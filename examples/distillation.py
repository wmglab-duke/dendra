import math
from typing import Tuple

import torch
from tqdm.auto import tqdm

from ..dendra.models.backend import Backend as A
from ..dendra.models.callbacks import Recorder
from ..dendra.models.core import Axon


def num_rows_to_zero(A: int, x: float) -> int:
    return int(torch.floor(torch.tensor(x * A)).item())


def random_sinusoid_sum(
    x_student: torch.Tensor,
    x_teacher: torch.Tensor,
    A: int,
    N: int,
    f_bounds: Tuple[float, float],
    *,
    seed: int | None = None,
) -> torch.Tensor:
    if x_student.ndim != 2 or x_student.shape[0] != 1:
        raise ValueError("x_student must have shape (1, n_coords)")
    if x_teacher.ndim != 2 or x_teacher.shape[0] != 1:
        raise ValueError("x_teacher must have shape (1, n_coords)")

    f_min, f_max = f_bounds
    if f_min >= f_max:
        raise ValueError("f_bounds must satisfy f_min < f_max")

    # Ensure generator uses the same device/dtype as `x`.
    if seed is not None:
        g = torch.Generator(device=x_student.device).manual_seed(seed)
    else:
        g = None

    dtype = x_student.dtype
    device = x_student.device

    # Draw random frequencies (A, N) and phases (A, N)
    freqs = torch.empty((A, N), dtype=dtype, device=device).uniform_(
        f_min, f_max, generator=g
    )
    phases = torch.empty((A, N), dtype=dtype, device=device).uniform_(
        0.0, 2 * math.pi, generator=g
    )

    # Reshape for broadcasting: (A, N, 1) × (1, n_coords) → (A, N, n_coords)
    freqs = freqs.unsqueeze(-1)
    phases = phases.unsqueeze(-1)

    # Compute sinusoids and sum over N
    signals_student = torch.sin(2 * math.pi * freqs * x_student + phases).sum(dim=1)
    signals_teacher = torch.sin(2 * math.pi * freqs * x_teacher + phases).sum(dim=1)

    return signals_student, signals_teacher


def random_sinusoid_sum_time(
    t_ms: torch.Tensor,
    A: int,
    N: int,
    f_bounds_Hz: Tuple[float, float],
    *,
    seed: int | None = None,
) -> torch.Tensor:
    f_bounds_kHz = [x / 1000 for x in f_bounds_Hz]

    if t_ms.ndim != 1:
        raise ValueError("t_ms must be 1-D (n_timepoints,)")

    f_min, f_max = f_bounds_kHz
    if f_min >= f_max:
        raise ValueError("f_bounds_kHz must satisfy f_min < f_max")

    # Generator that matches the tensor's device
    g = (
        torch.Generator(device=t_ms.device).manual_seed(seed)
        if seed is not None
        else None
    )

    # Random frequencies and phases: shapes (A, N)
    freqs_kHz = torch.empty((A, N), dtype=t_ms.dtype, device=t_ms.device).uniform_(
        f_min, f_max, generator=g
    )
    phases = torch.empty((A, N), dtype=t_ms.dtype, device=t_ms.device).uniform_(
        0.0, 2 * math.pi, generator=g
    )

    # Broadcast to (A, N, n_timepoints)
    freqs_kHz = freqs_kHz.unsqueeze(-1)
    phases = phases.unsqueeze(-1)
    t_ms = t_ms.unsqueeze(0)  # (1, n_timepoints)

    # Evaluate and sum across N
    signals = torch.sin(2 * math.pi * freqs_kHz * t_ms + phases).sum(dim=1)

    return signals


def generator(
    student: Axon,
    teacher: Axon,
    f_bounds_Hz=(0, 1000),
    f_bounds_s=(5e-5, 5e-4),
    tstop=2.5,
    dt=None,
    scale=100,
    n=1000,
):
    if dt is None:
        dt = A.dt

    device = student.device()
    t = torch.arange(0, tstop, dt, device=device)

    x_student = student.x[0].view(1, -1)  # Ensure x is (1, n_coords)
    x_teacher = teacher.x[0].view(1, -1)  # Ensure x is (1, n_coords)
    n_a = student.np

    for _ in range(n):
        ve_s_student, ve_s_teacher = random_sinusoid_sum(
            x_student, x_teacher, n_a, 5, f_bounds_s
        )
        ve_t = random_sinusoid_sum_time(t, n_a, 5, f_bounds_Hz)

        ve_student = torch.einsum("ac, at -> tac", ve_s_student, ve_t).detach() * scale
        ve_teacher = torch.einsum("ac, at -> tac", ve_s_teacher, ve_t).detach() * scale

        yield ve_student, ve_teacher


def distill(
    student: Axon,
    teacher: Axon,
    params,
    l1_params,
    input_generator,
    n,
    n_t,
    chunk_length,
    criterion,
    optimizer,
    lambduh=0.001,
    alpha=0.75,
    **kwargs,
):
    student.train()
    teacher.eval()

    parameters = list(student.collect_parameters(*params))
    l1_parameters = list(student.collect_parameters(*l1_params))

    for p in parameters:
        p.requires_grad = True
    for p in l1_parameters:
        p.requires_grad = True

    all_p = []
    for p in parameters:
        all_p.append(p)
    for p in l1_parameters:
        if p not in all_p:
            all_p.append(p)

    rec_v_student = Recorder(["v"])
    rec_v_teacher = Recorder(["v"])

    optimizer = optimizer(all_p, **kwargs)

    pbar = tqdm(total=n)

    n_splits = int(n_t / chunk_length)

    for j, (inputs_student, inputs_teacher) in enumerate(input_generator):
        inputs_student = inputs_student.to(student.device())
        inputs_teacher = inputs_teacher.to(teacher.device())
        input_chunks_student = torch.tensor_split(inputs_student, n_splits, dim=0)
        input_chunks_teacher = torch.tensor_split(inputs_teacher, n_splits, dim=0)

        subsample = teacher.nc // student.nc

        for i, (chunk_student, chunk_teacher) in enumerate(
            zip(input_chunks_student, input_chunks_teacher)
        ):
            reinit = i == 0

            rec_v_student.reset()
            rec_v_teacher.reset()

            # Forward pass through the teacher model
            with torch.no_grad():
                if reinit:
                    teacher.initialize()
                teacher.run(
                    ve=chunk_teacher,
                    callbacks=[rec_v_teacher],
                    progressbar=False,
                )
                teacher_outputs = rec_v_teacher.stack("v")[..., ::subsample]
                if torch.isnan(teacher_outputs).any():
                    pbar.set_description(f"Chunk {i}: NaN teacher output; skipping")
                    continue

            # Forward pass through the student model
            if reinit:
                student.initialize()
            else:
                student.detach()
                student.populate()
            student.run(
                ve=chunk_student,
                callbacks=[rec_v_student],
                progressbar=False,
            )

            student_outputs = rec_v_student.stack("v")

            # Compute the distillation loss
            loss = criterion(student_outputs, teacher_outputs)
            if torch.isnan(loss).any():
                pbar.set_description(f"Chunk {i}: NaN loss; skipping")
                continue

            pbar.set_description(f"Chunk {i}: {loss.item():.4f}")
            loss = loss + (
                lambduh
                * alpha
                * torch.sum(torch.stack([torch.abs(p) for p in l1_parameters]))
            )
            loss = loss + (
                lambduh
                / 2
                * (1 - alpha)
                * torch.sum(torch.stack([torch.square(p) for p in l1_parameters]))
            )

            loss = loss / n_splits
            # Backward pass and optimization
            loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            for p in all_p:
                p.clamp_(min=0.0)

        pbar.update(1)
