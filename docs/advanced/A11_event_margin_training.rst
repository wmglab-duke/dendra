.. _event-margin-training:

Training event presence with voltage margins (experimental)
============================================================

``dendra.models.analysis.event_margin_training`` supplies a local training
direction for creating or removing a declared spike event. It is intended for
an objective whose hard value changes only when an event appears or
disappears. The returned loss reports the complete hard task loss on its
forward pass and uses a voltage-crossing margin on its backward pass.

This direction is experimental. It is not the derivative of a hard spike
count, which is locally constant away from an event transition, and it does
not guarantee that a finite optimizer step will improve the hard task. Rerun
the complete hard protocol after every proposal. For AP width, timing among
existing spikes, rate trajectories, thresholds, chronaxie, and paired-pulse
recovery, use the supported workflows in
:ref:`macroscopic-descriptor-training`.

Concepts supplied by the protocol
---------------------------------

The analysis functions do not choose a stimulation protocol or assign spikes
to stimuli. The caller defines:

* A **complete hard protocol**, which initializes the model, applies the
  stimulus, detects events, assigns them to response opportunities, and
  returns the scientific task loss.
* An **event slot**, meaning one declared opportunity for an event at selected
  recording sites and within a selected response window. Each slot represents
  zero versus at least one crossing. Use separate slots when several spikes
  in one response window matter.
* A **hard event label** :math:`e_j\in\{0,1\}` and target label
  :math:`y_j\in\{0,1\}` for each slot :math:`j`.
* A **carrier trace**, meaning the differentiable voltage replay used to
  generate the training direction. By default, every mismatched slot must be
  event-absent in its carrier trace.

For a missing desired event, the current event-absent trial can normally be
its carrier. To remove an unwanted event, use a nearby, predeclared stimulus
probe at which that same event is absent. Following the already developed
crossing of an unwanted spike can mostly shift its sampled crossing time and
need not raise the hard activation threshold.

The voltage supplied to ``event_crossing_margin`` has shape
``(*slots, time, site)``. ``valid_pairs`` is a Boolean tensor broadcastable to
``(*slots, time - 1, site)``. It selects the sample pairs and sites belonging
to each response opportunity.

Signed crossing margin
----------------------

For adjacent voltage samples :math:`V_{t,c}` and :math:`V_{t+1,c}` at site
:math:`c`, define

.. math::

   m_{t,c}=\min\left\{
   \theta-V_{t,c},\;
   V_{t+1,c}-\theta,\;
   V_{t+1,c}-V_{t,c}-\Delta t\,d_{\min}
   \right\}.

Here :math:`\theta` is the hard voltage threshold, :math:`\Delta t` is the
simulation time step, and :math:`d_{\min}` is an optional minimum upstroke
slope. Omit the third term when the hard event rule has no upstroke gate. For
slot :math:`j`, the function returns

.. math::

   m_j=\max_{(t,c)\in W_j}m_{t,c},

where :math:`W_j` is the set of sample-pair and site combinations selected by
``valid_pairs``. A positive margin identifies a sampled upward crossing. A
negative margin measures how far the closest selected pair remains from
satisfying all crossing conditions. At exactly zero, use the hard decision:
for example, ``V_previous == threshold`` does not satisfy the strict rule
``V_previous < threshold <= V_next``.

The maximum and minimum operations are piecewise differentiable. Autograd
follows the pair and site selected in the current replay; the result records
their indices for inspection.

Complete minimal tensor example
-------------------------------

The following script is deliberately small so that it can be run without a
neuron model. The scalar ``q`` controls the second voltage sample. The complete
hard task wants one currently missing event. The hard-forward loss retains the
hard mismatch value of one while its gradient proposes increasing ``q``.

.. code-block:: python

   import torch

   from dendra.models.analysis.event_margin_training import (
       event_crossing_margin,
       hard_forward_event_training_loss,
   )


   def trace(parameter: torch.Tensor) -> torch.Tensor:
       # Shape: (slot=1, time=2, site=1), in millivolts.
       return torch.stack(
           (parameter.new_tensor(-1.0), parameter - 0.7)
       ).reshape(1, 2, 1)


   q = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
   valid_pairs = torch.ones((1, 1, 1), dtype=torch.bool)

   crossing = event_crossing_margin(
       trace(q),
       threshold_mV=0.0,
       valid_pairs=valid_pairs,
   )
   hard_events = crossing.hard_crossed.detach()       # tensor([False])
   target_events = torch.tensor([True])
   hard_task_loss = (hard_events != target_events).double().sum()

   training = hard_forward_event_training_loss(
       hard_task_loss,
       crossing.margin,
       hard_events,
       target_events,
       temperature_mV=0.5,
   )
   gradient, = torch.autograd.grad(training.loss, q)

   assert training.loss.detach() == hard_task_loss
   assert gradient < 0       # Gradient descent increases q toward a crossing.

   proposed_q = (q - 0.75 * gradient).detach()
   proposed_event = event_crossing_margin(
       trace(proposed_q),
       threshold_mV=0.0,
       valid_pairs=valid_pairs,
   ).hard_crossed
   print("hard event before:", hard_events.item())
   print("hard event after: ", proposed_event.item())

This example demonstrates the API and sign convention. A real training step
still requires the complete hard event detector, including its refractory
rule, response assignment, and any constraints on other events.

Hard-forward training loss
--------------------------

Let :math:`M=\{j:e_j\ne y_j\}` contain the mismatched event slots and define
:math:`s_j=2y_j-1`, so :math:`s_j=1` requests an event and :math:`s_j=-1`
requests its removal. For fixed nonnegative event weights :math:`w_j`, voltage
scale :math:`\tau>0`, and optional target margin :math:`\kappa\geq0`, the trace
objective is

.. math::

   R(q)=
   \frac{\sum_{j\in M}w_j\,
   \operatorname{softplus}\!\left((\kappa-s_jm_j(q))/\tau\right)}
   {\sum_{j\in M}w_j}.

Here :math:`q` denotes the trainable model parameters, :math:`\tau` is
``temperature_mV``, :math:`\kappa` is ``margin_target_mV``, and zero weight
excludes a slot. The event weights are fixed protocol choices rather than
learned values; equal weights are the usual starting point, while unequal
weights can express declared task priorities. The function returns

.. math::

   L_{\mathrm{train}}
   =\operatorname{stopgrad}(L_H)
   +\lambda\left[R-\operatorname{stopgrad}(R)\right],

where :math:`L_H` is the scalar loss from the complete hard protocol,
:math:`\lambda` is ``backward_scale``, and ``stopgrad`` means that the enclosed
value is detached from autograd. Consequently,

.. math::

   L_{\mathrm{train}}=L_H
   \quad\text{on the forward pass},\qquad
   \nabla_qL_{\mathrm{train}}=\lambda\nabla_qR
   \quad\text{on the backward pass}.

The trace objective is dimensionless. One reverse-mode pass, also called a
vector-Jacobian product or VJP, reaches every model parameter connected to the
carrier voltage.

Creating and removing events
----------------------------

.. list-table:: Carrier choice for a mismatched slot
   :header-rows: 1
   :widths: 20 27 31 22

   * - Hard label
     - Target
     - Carrier
     - If unavailable
   * - Event absent
     - Event present
     - The current inactive trial, with the same site and response window.
     - Report no event-direction update for that slot.
   * - Event present
     - Event absent
     - A nearby stimulus probe where the corresponding event is absent.
     - Report no event-direction update for that slot.
   * - Labels match
     - No change
     - No carrier is needed; the slot receives zero trace weight.
     - Continue with the remaining mismatches.

A carrier is suitable only when its complete response window was recorded,
the window does not begin above threshold because of an earlier spike, and the
event assignment is unambiguous. Its hard detector must use the same recording
site, voltage threshold, upstroke rule, and response window as the training
protocol. Predeclare a bounded ladder of nearby stimulus probes and a maximum
acceptable distance from the training stimulus. If no acceptable inactive
carrier occurs in that ladder, omit the slot and report the uncovered hard
mismatch.

Do not extend the response window beyond the recorded trace. A peak at the
last sample or a response that has not returned sufficiently far from its peak
indicates that a longer simulation is required. If a window can contain
several task-relevant events, assign separate slots or use a descriptor that
models the event train.

Model training loop
-------------------

The following template shows the responsibilities around the analysis calls.
Names such as ``run_complete_hard_protocol`` and ``run_carrier_voltage`` are
functions supplied by the user's model and stimulation protocol.

.. code-block:: python

   for update in range(number_of_updates):
       # 1. Measure the current scientific objective without differentiation.
       current = run_complete_hard_protocol(model)
       # current.loss: finite scalar
       # current.events: Boolean tensor with one value per declared slot

       # 2. Select current or nearby inactive carriers using a fixed ladder.
       carrier = choose_inactive_carrier(
           model,
           current.events,
           target_events,
       )
       if carrier is None:
           record_no_update(update, "no valid inactive carrier")
           continue

       # 3. Make the final carrier replay differentiable. If one voltage trace
       #    serves several slots, expand it over the leading slot dimension.
       voltage_mV = run_carrier_voltage(model, carrier, requires_grad=True)
       slot_voltage_mV = voltage_mV.unsqueeze(0).expand(
           number_of_slots, -1, -1
       )
       margins = event_crossing_margin(
           slot_voltage_mV,
           threshold_mV=event_threshold_mV,
           valid_pairs=slot_pair_and_site_mask,
           dv_threshold_mV_per_ms=minimum_upstroke_mV_per_ms,
           dt_ms=dt_ms,
       )

       try:
           direction = hard_forward_event_training_loss(
               current.loss,
               margins.margin,
               current.events,
               target_events,
               event_weights=event_weights,
               temperature_mV=temperature_mV,
           )
       except ValueError as error:
           record_no_update(update, str(error))
           continue

       # 4. Save the parameters, propose one optimizer step, and then measure
       #    the complete hard protocol again from its initialized state.
       saved_parameters = copy_trainable_parameters(model)
       optimizer.zero_grad()
       direction.loss.backward()
       optimizer.step()
       proposed = run_complete_hard_protocol(model)

       # 5. The hard task and declared collateral constraints decide whether
       #    the proposal is kept. Rejected parameters are restored exactly.
       if not accept_hard_proposal(current, proposed):
           restore_trainable_parameters(model, saved_parameters)

       # Rebuild hard labels, masks, carriers, and the autograd graph on the
       # next iteration, including after an accepted branch change.

When different slots require different carrier simulations, consume the VJP
from each carrier before running the next stateful simulation. Each helper
call normalizes by its own selected weight sum. If carrier :math:`p` represents
weight :math:`W_p` in one global objective, use
``backward_scale = W_p / sum(W_all_carriers)`` for that call before accumulating
the parameter gradients.

Outputs and common rejection cases
----------------------------------

``event_crossing_margin`` returns:

``margin``
   One graph-connected signed margin per slot.
``hard_crossed``
   The sampled crossing decision for this primitive. The complete hard
   protocol remains authoritative for refractory handling and event ownership.
``pair_index`` and ``site_index``
   The currently selected sample pair and site. Record them when diagnosing a
   branch change; they are not fixed event identifiers.

``hard_forward_event_training_loss`` returns:

``loss``
   The exact hard task loss in forward and the voltage-margin direction in
   backward.
``trace_objective``
   The graph-connected softplus objective :math:`R`.
``mismatched_events``
   The Boolean slots for which the hard and target labels differ.
``local_sensitivity_fraction``
   The magnitude of the local softplus derivative divided by its maximum.
   A very small value means that the carrier is too far from the crossing
   boundary to provide a useful local penalty.

The functions reject malformed inputs immediately. The most actionable cases
are:

.. list-table:: Common failure handling
   :header-rows: 1
   :widths: 32 31 37

   * - Failure
     - Meaning
     - Action
   * - ``inactive-carrier``
     - A positively weighted mismatch has a nonnegative carrier margin.
     - Choose a probe at which that event is absent, or omit the slot.
   * - ``inactive_carrier_saturated``
     - The local penalty derivative is below
       ``min_local_sensitivity_fraction``.
     - Look for a closer valid inactive carrier. Do not enlarge the voltage
       temperature solely to conceal a distant carrier.
   * - ``valid sample pair``
     - A slot mask selects no adjacent sample pair and site.
     - Correct the response window or recording-site mask.
   * - Shape or Boolean-label error
     - Margins, hard labels, targets, or weights do not describe the same
       slots.
     - Make the slot axes and ownership rule explicit before retrying.
   * - Zero gradient with matching labels
     - The hard task already matches for every positively weighted slot.
     - No event-presence update is required.

Validating proposed directions
------------------------------

Before using an event-margin direction in an optimization study:

1. Express the gradient in the same parameter coordinates that the optimizer
   updates. For a positive physical parameter
   :math:`p=p_0\exp(q)`, use
   :math:`\partial L/\partial q=p\,\partial L/\partial p`.
2. Predeclare a small grid of feasible step lengths along the proposed descent
   direction. An opposite-direction or random-direction control can help
   establish whether the proposal is informative.
3. Rerun the complete hard protocol from the same initialized state at every
   proposed setting. Record the hard task loss, event identities, extra
   spikes, and any required physiological constraints.
4. Accept a step according to the declared hard task. A spike moving to an
   adjacent sample pair is expected and does not by itself invalidate an
   otherwise useful step.
5. Repeat the hard measurement beyond the training window when moving an event
   toward the recording boundary could create an apparent improvement.
6. Recompute hard labels, carrier choice, masks, and the voltage-margin
   direction at every accepted parameter setting.

If no predeclared descent step improves the hard task, report that the method
did not provide an immediately useful direction at that operating point.
Several loss-neutral steps may be evaluated for a plateaued count only when
the complete hard loss is remeasured and the local direction is rebuilt after
every accepted step.

This construction represents zero versus at least one crossing in each slot.
It does not provide a general gradient for spike count, sustained firing rate,
or frequency following. Frequency-following optimization and universal
event-birth or event-removal directions remain experimental.
