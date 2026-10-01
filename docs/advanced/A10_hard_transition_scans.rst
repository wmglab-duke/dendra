.. _hard-transition-scans:

Validating hard transition scans (experimental)
================================================

A hard descriptor can change discontinuously as a stimulus amplitude changes.
For example, an action-potential count can jump from two to three and later
return to two. A transition scan locates the stimulus intervals containing
these jumps so that they can be used in a noise-averaged hard descriptor.

Use this workflow when a complete hard descriptor may have several positive or
negative jumps along one stimulus axis. For an ordinary activation threshold,
chronaxie, or paired-pulse threshold, start with
:ref:`macroscopic-descriptor-training`; those workflows use the complete hard
threshold and its final inactive/active bracket directly.

A finite scan can find and refine transitions that it samples. It cannot prove
that a narrower pair of opposite jumps is absent between samples. The checks
on this page therefore provide evidence about a declared stimulus range and
resolution, rather than a completeness certificate.

Inputs and terminology
----------------------

The caller supplies a deterministic function ``evaluate(amplitude)`` that
runs the complete hard protocol and returns one finite scalar:

* :math:`a` is the scalar stimulus amplitude, in the caller's units.
* :math:`H(a)` is the complete hard descriptor, such as an integer spike
  count or a binary response.
* A *transition bracket* :math:`[\ell_j,u_j]` contains an observed jump.
  Its signed size is :math:`\Delta_j=H(u_j)-H(\ell_j)`.
* ``tolerance`` is the maximum requested width :math:`u_j-\ell_j`, in
  stimulus-amplitude units.
* ``scan_depth`` adds dyadic samples to every interval in the supplied
  amplitude grid. Depth :math:`d` divides each interval into :math:`2^d`
  subintervals before transition refinement.

Every evaluator call must start from the same initialized state and use the
same stimulus waveform, recording sites, duration, and hard measurement rule.
The scan helpers run it under ``torch.no_grad()``.

Complete runnable workflow
--------------------------

This example contains one positive and one negative hard transition. It runs a
coarse scan, repeats it at a finer resolution, challenges the finer result at
independent offsets, and propagates the observed root brackets into bounds on
a Gaussian expected-hard value.

.. code-block:: python

   from dendra.models.analysis.hard_transition_table import (
       discover_hard_transitions,
   )
   from dendra.models.analysis.hard_transition_validation import (
       challenge_transition_scan,
       compare_transition_scans,
       gaussian_expected_hard_bracket_bounds,
   )


   def hard_descriptor(amplitude: float) -> float:
       """A deterministic stand-in for one complete hard simulation."""
       return (
           1.0
           + 1.0 * (amplitude >= 0.31)
           - 2.0 * (amplitude >= 0.73)
       )


   scan_grid = [0.0, 0.25, 0.50, 0.75, 1.0]

   coarse = discover_hard_transitions(
       hard_descriptor,
       scan_grid,
       tolerance=1e-5,
       scan_depth=0,
   )
   fine = discover_hard_transitions(
       hard_descriptor,
       scan_grid,
       tolerance=1e-5,
       scan_depth=2,
   )
   if not coarse.valid or not fine.valid:
       raise RuntimeError("At least one observed transition was not refined")

   comparison = compare_transition_scans(coarse, fine)
   if not comparison.consistent:
       raise RuntimeError(f"The finer scan changed: {comparison.reasons}")

   # With the default two fractions, this requests two probes per interval
   # in fine.scan_amplitudes, plus two endpoint reproducibility checks.
   challenge = challenge_transition_scan(
       fine,
       hard_descriptor,
       max_evaluations=64,
   )
   if challenge.status != "inconclusive":
       raise RuntimeError(f"Transition scan failed: {challenge.status}")

   bounds = gaussian_expected_hard_bracket_bounds(
       fine,
       center=0.55,
       noise_std=0.20,
   )

   for transition in fine.transitions:
       print(
           f"root in [{transition.lower_amplitude:.6f}, "
           f"{transition.upper_amplitude:.6f}], "
           f"signed jump = {transition.jump:+.0f}"
       )
   print(
       "Gaussian expected hard value: "
       f"{bounds.midpoint:.6f} "
       f"in [{bounds.lower:.6f}, {bounds.upper:.6f}]"
   )

``fine.valid`` means that every transition exposed by that scan was refined to
the requested tolerance. ``comparison.consistent`` means the two finite scans
found compatible observed jumps. ``challenge.status == "inconclusive"`` means
the additional probes did not contradict the table. None of these results
rules out a transition between all sampled points.

Using a Dendra hard protocol
----------------------------

Replace ``hard_descriptor`` with a wrapper around the complete simulation and
hard measurement. A model factory is a simple way to guarantee the same
initialized state on every call:

.. code-block:: python

   def evaluate(amplitude: float) -> float:
       # These three functions belong to the user's protocol.
       model = make_initialized_model()
       voltage_mV = run_complete_stimulus_protocol(
           model,
           amplitude=amplitude,
       )
       value = measure_hard_descriptor(voltage_mV)
       return float(value)


   table = discover_hard_transitions(
       evaluate,
       amplitudes=predeclared_amplitude_grid,
       tolerance=amplitude_tolerance,
       scan_depth=initial_scan_depth,
       max_evaluations=evaluation_budget,
   )

Reusing one model is also possible when the callback restores the complete
initialized state before each trial. Do not let an earlier amplitude trial
alter channel state, stimulus state, random draws, or recorder contents used by
a later trial. If the protocol intentionally includes randomness, fix its
realization or define a deterministic hard summary before scanning.

Interpreting and challenging a table
------------------------------------

Outside the observed root brackets, ``challenge_transition_scan`` compares
new hard reruns with the signed-step reconstruction

.. math::

   \widehat{H}(a)
   =H(a_0)+\sum_{j:u_j<a}\Delta_j,

where :math:`a_0` is the first scan amplitude, :math:`H(a_0)` is its measured
hard value, :math:`u_j` is the upper endpoint of bracket :math:`j`, and
:math:`\Delta_j` is that transition's signed jump. A probe inside a root
bracket is ambiguous because either adjacent hard value is compatible with
the current root tolerance.

.. list-table:: Result interpretation
   :header-rows: 1
   :widths: 27 33 40

   * - Result
     - Meaning
     - Action
   * - ``table.valid is False``
     - At least one observed bracket did not reach ``tolerance``.
     - Increase ``max_bisection_steps`` or the evaluation budget, or relax a
       scientifically unnecessary tolerance.
   * - ``comparison.consistent is False``
     - The finer scan changed an observed value, jump, or bracket sequence.
     - Inspect ``comparison.reasons`` and use the finer scan as the new
       starting point.
   * - ``status="scan_contradicted"``
     - An unambiguous challenge probe disagreed with the reconstructed steps.
     - Add samples around the reported mismatch and rebuild the table.
   * - ``status="protocol_changed"``
     - A rerun of a scan endpoint changed its hard value.
     - Fix model-state restoration or other nondeterminism before continuing.
   * - ``status="inconclusive"``
     - Every finite challenge probe was compatible with the table.
     - Record the scan range, resolution, challenge points, and remaining
       finite-resolution limitation.

Use ``interval_indices`` to spend the challenge budget on selected original
scan intervals. To test known suspicious locations, set
``probe_fractions=()`` and supply ``probe_amplitudes``. The function checks the
entire requested budget, including optional endpoint reruns, before evaluating
the protocol.

Gaussian expectation bounds
---------------------------

For a Gaussian stimulus :math:`A\sim\mathcal{N}(\mu,\sigma^2)`, the observed
transition table represents

.. math::

   \mathbb{E}[H(A)]
   =H(a_0)+\sum_j\Delta_j
   \Phi\!\left(\frac{\mu-r_j}{\sigma}\right),

where :math:`r_j` is the unknown root inside
:math:`[\ell_j,u_j]` and :math:`\Phi` is the standard-normal cumulative
distribution function. ``gaussian_expected_hard_bracket_bounds`` evaluates
both endpoints of every observed root bracket to bound the numerical effect of
that root uncertainty. Its midpoint estimate uses
:math:`r_j=(\ell_j+u_j)/2`.

These bounds account only for the observed brackets. Choose a scan range that
covers the stimulus probability relevant to the application, and separately
assess the Gaussian probability outside that range. Offset challenges and
deeper scans strengthen the evidence within the range; they cannot exclude an
arbitrarily narrow unsampled transition.

Once a transition set has been validated for the declared protocol, use
``gaussian_expected_hard_from_transition_roots`` to build the graph-connected
expected-hard training objective described in
:ref:`macroscopic-descriptor-training`.
