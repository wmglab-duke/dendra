#!/usr/bin/env python3
"""Check an installed Dendra distribution before publishing it.

Install the built wheel in a fresh environment, then run this script with that
environment's Python in isolated mode::

    python -I /absolute/path/to/dendra/scripts/distribution_smoke.py

The checks use CPU execution and do not require dendra-models, native CPU
solvers, a CUDA toolkit, or a GPU. Imports must come from the environment's
installed distribution, rather than an editable checkout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import sysconfig
from importlib.metadata import distribution
from importlib.resources import files
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory


def check_installed_package():
    """Reject source imports and check metadata against the installed module."""
    if not sys.flags.isolated:
        raise RuntimeError("Run this script with python -I to isolate imports.")

    import dendra as dn
    import dendra.models.analysis as analysis

    checkout = Path(__file__).resolve().parents[1]
    prefix = Path(sys.prefix).resolve()
    for module in (dn, analysis):
        module_path = Path(module.__file__).resolve()
        if module_path.is_relative_to(checkout):
            raise RuntimeError(
                f"{module.__name__} imported from checkout: {module_path}"
            )
        if not module_path.is_relative_to(prefix):
            raise RuntimeError(
                f"{module.__name__} is outside the test environment: {module_path}"
            )

    installed = distribution("dendra")
    if installed.version != dn.__version__:
        raise RuntimeError(
            f"Metadata version {installed.version} differs from {dn.__version__}."
        )
    print(f"Installed Dendra {installed.version}: {Path(dn.__file__).resolve()}")
    return dn, analysis, installed


def check_native_source_data() -> None:
    """CUDA JIT source files must ship even in the universal Python wheel."""
    package = files("dendra.models.networks")
    for name in ("netcon_bitpack_kernel.cpp", "netcon_bitpack_kernel.cu"):
        source = package.joinpath(name)
        if not source.is_file() or not source.read_bytes():
            raise RuntimeError(f"Missing or empty runtime native source: {name}")
    print("Runtime C++ and CUDA source data are present.")


def check_optional_solvers(expectation: str | None) -> None:
    """Ensure base and solver-extra jobs exercise their declared environment."""
    present = find_spec("dendra_solvers") is not None
    if expectation is not None and present != (expectation == "present"):
        raise RuntimeError(
            f"Expected dendra-solvers {expectation}; detected "
            f"{'present' if present else 'absent'}."
        )
    if present:
        import dendra_solvers._ext

        print(f"Optional native CPU solvers loaded: {dendra_solvers._ext.__file__}")
    else:
        print("Base installation has no optional native CPU solvers.")


def check_cpu_simulation(dn) -> None:
    """A passive cable must run and remain connected to a trainable parameter."""
    import torch

    from dendra.models.mod import pas

    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu"):
        morphology = dn.Morphology()
        morphology.section("path", L=50.0, diam=2.0, nseg=5)
        cable = dn.Cable.from_morphology(
            morphology, N=1, v_init=-65.0, dtype=torch.float64
        )
        cable.insert(pas, g=0.001, e=-70.0)
        cable.train()
        cable.unfreeze("pas.g")
        cable.initialize()
        cable.run(tstop=0.1, dt=0.025, progressbar=False)
        if not bool(torch.isfinite(cable.v).all()) or not bool((cable.v < -65.0).all()):
            raise RuntimeError("The passive CPU cable did not relax toward -70 mV.")
        (gradient,) = torch.autograd.grad(cable.v.sum(), cable.mech.pas.g_param)
        if not bool(torch.isfinite(gradient).all()) or not bool((gradient < 0).all()):
            raise RuntimeError("Passive conductance did not supply a finite gradient.")
    print("CPU cable simulation and conductance gradient succeeded.")


def check_functional_cpu_simulation(dn) -> None:
    """Lower and capture a connected model using the installed distribution."""
    import torch

    from dendra.models.mod import expsyn, pas

    dt = 0.025

    def initialized_network():
        cell = dn.SingleCompartment(N=1, C=1, v_init=-65.0, dtype=torch.float64)
        cell.insert(pas, g=0.001, e=-70.0)
        cell.insert(expsyn.rename("exc"), tau=2.0, e=0.0)
        source = dn.NetStim(N=1, start=0.0, interval=1.0, noise=0.0, seed=17)
        network = dn.Network({"cell": cell}, netstim=source)
        network.connect_one_to_one(
            source[:], cell[:, :1], cell.mech.exc, weight=0.001, delay=dt
        )
        network.eval()
        network.initialize(dt)
        return network

    with dn.ctx(JIT=0, DEVICE="cpu"):
        network = initialized_network()
        draws = dn.func.sample_runtime_draws(network)
        with dn.ctx(CHECK_PURITY=True):
            plan, tensors = dn.func.make_functional(network, dt=dt)
        reference = initialized_network()
        plan.extract(reference)
        compiled = torch.compile(plan.prepare_and_step, backend="eager", fullgraph=True)
        state, _ = compiled(
            tensors.parameters, tensors.constants, tensors.state, draws=draws
        )
        reference.step(draws=draws)
        torch.testing.assert_close(
            state["populations"]["cell"]["integrator"]["v"],
            reference.populations["cell"].v,
        )
    print("Compiled functional Network step matches native draw replay.")


def check_descriptor_gradient(analysis) -> None:
    """Check a stable synthetic crossing branch against the hard descriptor."""
    import torch

    base = torch.tensor(
        [-2.0, -2.0, -2.0, 2.0, -2.0, -2.0, -2.0, 3.0, -2.0, -2.0, -2.0, 1.0, -2.0],
        dtype=torch.float64,
    )[:, None, None]
    offset = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    soft = analysis.differentiable_spike_timing(base + offset, 1.0, V_spk=0.0)
    hard = analysis.hard_spike_timing(base, 1.0, V_spk=0.0)
    if not bool(soft["valid"].all()) or int(hard["count_comp"].item()) != 3:
        raise RuntimeError("The synthetic three-spike timing descriptor is invalid.")
    torch.testing.assert_close(
        soft["span_frequency_hz"].detach(),
        hard["span_frequency_hz"],
        rtol=0.0,
        atol=1e-12,
    )
    (gradient,) = torch.autograd.grad(soft["span_frequency_hz"].sum(), offset)
    if not bool(torch.isfinite(gradient)) or float(gradient.abs()) <= 1e-8:
        raise RuntimeError("The spike-timing descriptor has no useful derivative.")

    step = 1e-5
    lower = analysis.hard_spike_timing(base - step, 1.0, V_spk=0.0)
    upper = analysis.hard_spike_timing(base + step, 1.0, V_spk=0.0)
    if not (
        lower["branch_signature"]
        == hard["branch_signature"]
        == upper["branch_signature"]
    ):
        raise RuntimeError("The synthetic finite difference changed event branch.")
    finite_difference = (
        upper["span_frequency_hz"].sum() - lower["span_frequency_hz"].sum()
    ) / (2.0 * step)
    torch.testing.assert_close(gradient, finite_difference, rtol=1e-6, atol=1e-8)
    print(
        "Spike-timing value parity and stable hard finite-difference gradient succeeded."
    )


def check_cli(installed) -> None:
    """Check both entry points away from the source checkout."""
    entry_points = [
        entry
        for entry in installed.entry_points
        if entry.group == "console_scripts" and entry.name == "dendra"
    ]
    if len(entry_points) != 1 or entry_points[0].value != "dendra.cli:main":
        raise RuntimeError(
            "The distribution is missing its dendra console entry point."
        )
    executable = Path(sysconfig.get_path("scripts")) / (
        "dendra.exe" if sys.platform == "win32" else "dendra"
    )
    if not executable.is_file():
        raise RuntimeError(f"The installed console script is missing: {executable}")
    with TemporaryDirectory(prefix="dendra-cli-smoke-") as directory:
        for command in (
            [sys.executable, "-I", "-m", "dendra", "--help"],
            [str(executable), "--help"],
        ):
            result = subprocess.run(
                command,
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            if "doctor" not in result.stdout:
                raise RuntimeError(f"The CLI help did not list doctor: {result.stdout}")
    print("Module and console-script entry points succeeded.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-solvers",
        choices=("present", "absent"),
        help="require the presence or absence of the optional dendra-solvers package",
    )
    args = parser.parse_args(argv)
    dn, analysis, installed = check_installed_package()
    check_native_source_data()
    check_optional_solvers(args.expect_solvers)
    check_cpu_simulation(dn)
    check_functional_cpu_simulation(dn)
    check_descriptor_gradient(analysis)
    check_cli(installed)
    print("Installed distribution smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
