"""Shared host samplers retain opt-in diagnostics at later draw boundaries."""

import pytest
import torch

import dendra as dn
from dendra._bootstrap import torch_compiler_warning_context
from dendra.models.mechanisms import Mechanism
from dendra.models.random_parameters import register_random_distribution
from dendra.units import nA
from tests.func.test_poisson_stimulation import DT, DTYPE, _clone, _close

pytestmark = pytest.mark.cpu


def _population(*, raises):
    def sampler(rng, shape, *, device, dtype):
        draw = rng.randn(shape, device=device, dtype=dtype)
        if draw.reshape(-1)[0] > 2.0:
            torch.rand((), device=device)
            if raises:
                raise RuntimeError("conditional custom sampler failure")
        return draw

    def transform(parameters, draw):
        return parameters["mu"] + parameters["sigma"] * draw

    name = f"shared_stimulation_conditional_global_rng_{raises}"
    register_random_distribution(
        name,
        {"mu": 0.0, "sigma": 0.001},
        draw_sampler=sampler,
        transform=transform,
        overwrite=True,
    )

    class ConditionalNoise(Mechanism):
        Mechanism.RANGENOISE("eta", distribution=name, seed=31)
        Mechanism.RANGE(g=0.001, e=-65.0)
        Mechanism.NONSPECIFIC_CURRENT("i")
        Mechanism.AFFINE("i")

        def i_affine_terms(self, v):
            del v
            return self.g, self.g * self.e + self.eta

    cell = dn.Unmyelinated(
        [2.0],
        L=4.0,
        dx=1.0,
        v_init=-45.0,
        dtype=DTYPE,
        integrator=dn.bwd_euler_ub(method="pcr", imem=False),
    )
    cell.insert(ConditionalNoise.rename("probe"))
    cell.train()
    cell.initialize()
    pulse = dn.mono_rect(amp=0.2 * nA, pw=0.045).poisson(
        interval=0.055,
        n=5,
        start=-0.035,
        noise=0.7,
        randomize_every_call=True,
        generator=cell.mech.probe.eta_rng.rng,
    )
    cell[:, 1].inject(pulse)
    cell.initialize()
    cell.integrator._initialize(cell, cell.t.new_tensor(DT), force=True)
    cell.detach()
    return cell, pulse


@pytest.mark.parametrize(
    "check_purity,raises", ((False, False), (True, False), (True, True))
)
def test_conditional_global_rng_draw_respects_opt_in_sampling_diagnostics(
    check_purity, raises
):
    global_rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            torch_compiler_warning_context(),
            dn.ctx(
                JIT=0,
                REQUIRE_GRAD=1,
                IMEM=False,
                DTYPE=DTYPE,
                DEVICE="cpu",
                CHECK_PURITY=check_purity,
            ),
        ):
            source, pulse = _population(raises=raises)
            functional, tensors = dn.func.make_functional(source, dt=DT)
            before = _clone(tensors)
            generator = pulse.generator.get_state().clone()
            checkpoint = torch.get_rng_state().clone()
            # The captured request passes; a later explicit RNG frame takes
            # the conditional branch which the lowering probe cannot exercise.
            functional.sample_noise_draw(tensors.state)
            _close(torch.get_rng_state(), checkpoint)
            if check_purity:
                error = RuntimeError if raises else dn.func.FunctionalizationError
                message = (
                    "conditional custom sampler failure" if raises else "global RNG"
                )
                with pytest.raises(error, match=message):
                    functional.sample_noise_draw_tape(tensors.state, 128)
                _close(torch.get_rng_state(), checkpoint)
            else:
                functional.sample_noise_draw_tape(tensors.state, 128)
                assert not torch.equal(torch.get_rng_state(), checkpoint)
            _close(tensors, before)
            _close(pulse.generator.get_state(), generator)
    finally:
        torch.set_rng_state(global_rng)
        torch.set_num_threads(threads)
