from typing import Tuple

import torch
import math
from tqdm.auto import tqdm

from .core import Axon
from .callbacks import Recorder
from .backend import Backend as A


def num_rows_to_zero(A: int, x: float) -> int:
    # floor(x * A) rows will be zeroed; you can also use round() if you prefer
    return int(torch.floor(torch.tensor(x * A)).item())


def random_sinusoid_sum(
    x: torch.Tensor,
    A: int,
    N: int,
    f_bounds: Tuple[float, float],
    *,
    seed: int | None = None
) -> torch.Tensor:
    
    if x.ndim != 2 or x.shape[0] != 1:
        raise ValueError("x must have shape (1, n_coords)")

    f_min, f_max = f_bounds
    if f_min >= f_max:
        raise ValueError("f_bounds must satisfy f_min < f_max")

    # Ensure generator uses the same device/dtype as `x`.
    if seed is not None:
        g = torch.Generator(device=x.device).manual_seed(seed)
    else:
        g = None

    # Draw random frequencies (A, N) and phases (A, N)
    freqs  = torch.empty((A, N), dtype=x.dtype, device=x.device)\
                .uniform_(f_min, f_max, generator=g)
    phases = torch.empty((A, N), dtype=x.dtype, device=x.device)\
                .uniform_(0.0, 2 * math.pi, generator=g)

    # Reshape for broadcasting: (A, N, 1) × (1, n_coords) → (A, N, n_coords)
    freqs  = freqs.unsqueeze(-1)
    phases = phases.unsqueeze(-1)

    # Compute sinusoids and sum over N
    signals = torch.sin(2 * math.pi * freqs * x + phases).sum(dim=1)

    return signals


def random_sinusoid_sum_time(
    t_ms: torch.Tensor,
    A: int,
    N: int,
    f_bounds_Hz: Tuple[float, float],
    *,
    seed: int | None = None
) -> torch.Tensor:
    
    f_bounds_kHz = [x/1000 for x in f_bounds_Hz]
    
    if t_ms.ndim != 1:
        raise ValueError("t_ms must be 1-D (n_timepoints,)")

    f_min, f_max = f_bounds_kHz
    if f_min >= f_max:
        raise ValueError("f_bounds_kHz must satisfy f_min < f_max")

    # Generator that matches the tensor's device
    g = torch.Generator(device=t_ms.device).manual_seed(seed) if seed is not None else None

    # Random frequencies and phases: shapes (A, N)
    freqs_kHz = torch.empty((A, N), dtype=t_ms.dtype, device=t_ms.device)\
                  .uniform_(f_min, f_max, generator=g)
    phases    = torch.empty((A, N), dtype=t_ms.dtype, device=t_ms.device)\
                  .uniform_(0.0, 2 * math.pi, generator=g)

    # Broadcast to (A, N, n_timepoints)
    freqs_kHz = freqs_kHz.unsqueeze(-1)
    phases    = phases.unsqueeze(-1)
    t_ms      = t_ms.unsqueeze(0)                     # (1, n_timepoints)

    # Evaluate and sum across N
    signals = torch.sin(2 * math.pi * freqs_kHz * t_ms + phases).sum(dim=1)

    return signals


def generator(
        model: Axon, 
        f_bounds_Hz=(0, 1000), 
        f_bounds_s=(5e-5, 5e-4), 
        tstop=2.5, 
        dt=None, 
        scale=100, 
        n=1000
    ):
    if dt is None:
        dt = A.dt

    device = model.device()
    t = torch.arange(0, tstop, dt, device=device)

    x = model.x()
    n_a = model.n_ax

    for _ in range(n):
        ve_s = random_sinusoid_sum(x, n_a, 5, f_bounds_s)
        ve_t = random_sinusoid_sum_time(t, n_a, 5, f_bounds_Hz)
        
        ve = torch.einsum('ac, at -> tac', ve_s, ve_t) * scale
        ve = ve.unsqueeze(2)

        yield ve
    

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
        randomize_diameters=True,
        **kwargs
    ):

    student.train()
    teacher.eval()

    parameters = student.collect_parameters(*params)
    l1_parameters = student.collect_parameters(*l1_params)

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

    rec_v_student = Recorder(['v'])
    rec_v_teacher = Recorder(['v'])

    optimizer = optimizer(all_p, **kwargs)

    pbar = tqdm(total=n)

    n_splits = int(n_t / chunk_length)

    for j, inputs in enumerate(input_generator):

        if randomize_diameters:
            new_diams = 0.5 + 2.5 * torch.rand(student.n_ax, device=student.device())
            student.set_diameters(new_diams)
            teacher.set_diameters(new_diams)

        inputs = inputs.to(student.device())
        input_chunks = torch.tensor_split(inputs, n_splits, dim=0)

        #if j % 2 == 0:
        #    loss = lambduh * torch.sum(torch.stack([torch.abs(p) for p in l1_parameters])) # L1 regularization
        #    loss.backward()

        if True:
            for i, chunk in enumerate(input_chunks):

                reinit = (i == 0)

                rec_v_student.reset()
                rec_v_teacher.reset()

                # Forward pass through the teacher model
                with torch.no_grad():
                    teacher.run(
                        ve=chunk, 
                        callbacks=[rec_v_teacher], 
                        progressbar=False, 
                        reinit=reinit
                    )
                    teacher_outputs = rec_v_teacher.stack('v')
                    if torch.isnan(teacher_outputs).any():
                        pbar.set_description(f"Chunk {i}: NaN teacher output; skipping")
                        continue

                # Forward pass through the student model
                student.run(
                    ve=chunk, 
                    callbacks=[rec_v_student], 
                    progressbar=False, 
                    reinit=reinit
                )
                student_outputs = rec_v_student.stack('v')

                # Compute the distillation loss
                loss = criterion(student_outputs, teacher_outputs)
                if torch.isnan(loss).any():
                    pbar.set_description(f"Chunk {i}: NaN loss; skipping")
                    continue

                pbar.set_description(f"Chunk {i}: {loss.item():.4f}")
                loss += lambduh * alpha * torch.sum(torch.stack([torch.abs(p) for p in l1_parameters]))
                loss += lambduh / 2 * (1 - alpha) * torch.sum(torch.stack([torch.square(p) for p in l1_parameters]))

                loss = loss / n_splits
                # Backward pass and optimization
                loss.backward()

            
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            for p in all_p:
                p.clamp_(min=0.0)

        pbar.update(1)

    

