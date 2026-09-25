# Algorithm rationale and interview evidence

The project combines established algorithms with a project-owned physical task,
observation/action boundary, recurrent training pipeline and controlled study.
It is not presented as a new reinforcement-learning algorithm.

PPO learns the privileged state teacher through physical interaction. PPO alternates
rollout sampling and multiple minibatch updates to a policy surrogate objective
([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)). The implemented
rl-games configuration uses PPO clipping 0.2, GAE lambda 0.95, discount 0.99,
512 environments, a 32-step horizon, four optimization epochs and a shared
3×256 actor/critic MLP. Curriculum and reset perturbations are project decisions.

The teacher sees aperture-relative geometry; the deployed visual student does
not. CNN depth features and proprioception feed a 128-unit GRU, then collective
thrust and body-rate commands. A fixed rate controller allocates motor thrust.
The student receives no gap coordinates, phase flags or teacher output during
evaluation. No visual-PPO fine-tuning or VLA is used in this version.

### Reward and terminal outcomes

The original task reward uses a distance-progress term `4 × (d_old - d_new)`
per policy step, a `-0.01` step cost, one-time `+10` complete-crossing and `+30`
stable-success bonuses, and a `-15` failure penalty. It also penalizes squared
change in the applied normalized action (`0.002 × ||a_t-a_(t-1)||²`) and gives a
small post-crossing goal-distance/speed term. The distance target switches from
a via-aperture path to direct goal distance at crossing. Because of that phase
switch and the other terms, this is described as distance-progress shaping, not
as policy-invariant potential-based shaping. Traversal tilt is not directly
penalized in the original reward; success still requires the explicit stable
recovery hold.

The optional `dense_v2` post-crossing feedback is a development diagnostic. It
was not adopted into the failed formal-v3 attempt, which retained the original
reward. Its results remain separate from the formal evidence.

BC fits teacher actions on teacher rollouts. DAgger instead queries the expert
on states induced by the current student and aggregates the data, addressing the
observation-distribution dependence on earlier actions
([Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)). Here round 0
is teacher-controlled BC and later DAgger rounds are pure student control, with
all outcomes retained. This does not imply monotonic success improvements.

Sequence training uses the full causal episode prefix without gradients, then
masked 64-step targets. Padding and hidden-state resets cannot carry information
across episode boundaries. Stored teacher targets, applied actions and pre-action
observations are separately recorded. The replay audit found at most 0.00025
stored/replayed action discrepancy; streamed/batched evaluation agrees closely.

Pilot behavior is evidence of a learned skill, but not yet a controlled causal
comparison: BC initially scored 0/100 show and 0/100 held-out; the final DAgger
pilot scored 98/100 and 88/100 after 196,608 labels / 12,000 updates. Both data
and training budgets grew, so equal-budget BC and memory comparisons are needed
before attributing the entire improvement to DAgger or GRU. Those comparisons
are frozen for seeds 11/22/33, with 820,548-parameter GRU and 822,692-parameter
four-depth-frame stack. Equal update counts do not imply identical valid-label
presentations or FLOPs; actual presentations and timing will be reported.

Own-state estimation is simulated from noisy/delayed ground truth, not VIO.
The task uses static apertures; no real-flight transfer is claimed. Tests cover
full rotor-envelope crossing, 200Hz contact/swept checks, recovery hold criteria,
causal sensor delay/dropouts, independent mass/impulse conditions, information
boundaries, recurrent resets and checkpoint provenance. Pilot validation cases
are reused across checkpoints; they are not new independent samples each time.
The untouched final-test manifest is reserved for all fixed-budget formal models.
