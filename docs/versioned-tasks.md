# Versioned task selection

The base catalogue returned by `list_task_specs()` retains its existing task
IDs, prompts, aliases, reset defaults and evaluators. Explicit variants are
registered and listed separately. Registration checks all normalized names,
aliases and prompts before publishing any key. Repeating the same definition
is allowed; changing a definition under an existing key fails.

A variant may reuse its canonical base task's prompt. That shared prompt
continues to select the base task. It must not be a variant name or alias.
Select a variant by its exact canonical ID and explicit version:

```python
from abc_sim import register_task_spec, select_task_spec
from abc_sim.task_variants import BottleThroughputTask

condition = BottleThroughputTask(bottle_count=6, horizon_seconds=30)
register_task_spec(condition.task_spec())
spec = select_task_spec(condition.task_id, version="v1")
```

Pass the selected spec or its registered name to `abc_sim.make_env`. The
`SimTaskSpec` records `version`, `base_task` and `horizon_seconds`; original
specs leave them unset. Native `step()` reports truncation at the variant's
elapsed simulated-time horizon. Low-level users of `advance_physics()` retain
control over stepping and must honor the declared horizon themselves.
Versioned reset overrides can add native randomization options, but cannot
change the fixed fields in the definition. They require randomized resets.

The existing short-task identities and ranges are retained:

| Definition | Identity | Counts | Horizons |
| --- | --- | --- | --- |
| `BottleThroughputTask` | `throughput_bottles{N}_{S}s_v1` | 2–6 bottles | 30 or 60 s |
| `LegoThroughputTask` | `throughput_lego{N}percolor_{S}s_v1` | 1–9 per native color | 30 or 60 s |
| `NutsBoltsThroughputTask` | `throughput_nuts{N}_bolts{B}_{S}s_v1` | 1–20 of each | 30 or 60 s |

These definitions reuse native randomizers and evaluators without changing
their sampling or reward formulas. `validate_reset()` checks realized native
count metadata; `progress_counts()` reads the evaluator's existing counts.
Collection cadence, trajectory grouping and RL-specific reward transforms
remain caller responsibilities. Defining or registering a variant does not
add it to the base benchmark roster.

## Conservative spelling v1

`free_placement_spelling_task()` defines only
`spell_random_2_free_placement_60s_v1`, with evaluator identity
`abc-spelling-free-placement-v1`. It reuses the two-letter
`unique-uppercase-v1` seeded goal and pose randomization. The original
`spell_random_2/3/4` and fixed-word spelling tasks retain `spell_word`.

The migrated conservative geometry reads toward decreasing scene Y, considers
all projected occupants and admissible endpoint/gap interpretations, and
credits only placements shared by every interpretation. Missing endpoints
remain ambiguous; a wrong occupant can locate another correct letter.
The score is correct placements divided by the maximum of the target length
and admissible row occupancy. For target `AB`, row `XB` earns ½; rows `_B`
and `A_` earn zero.

The native hook samples direct table support once after every physics tick.
Geometry and original spelling diagnostics are scored every 17 ticks;
evaluator reads return the latest scheduled result and never sample again.
The first result therefore requires 17 ticks, including terminal-drain paths.
Fresh baseline positions and support tracking are bound after completed reset,
goal/model rebinding and reset settling. Ordinary tasks use the native loop
without per-tick observer work.
The public `PhysicsObservedTaskEvaluator` protocol declares reset binding,
physics observation and scheduled result reads; the environment and batched
capability guard use this contract.

The v1 contact exclusion set is **finger/linkage subtrees** selected by the
existing native gripper helper. It does not exclude wrist, palm or camera
contact. This is the shipped historical behavior; broadening the exclusion set
requires a distinct semantic version and qualification. Direct table support,
low velocity and uninterrupted 0.1-second settling remain required.

The result records the exact frame, support evidence, configuration and
conservative score under `spelling_placement`. `original_spell_reward` and
`original_spell_success` retain the original evaluator's separate diagnostics.
The new result uses the native float32 `TaskEvalResult` reward container; the
unrounded conservative scalar remains in the saved score. Historical
downstream reward labels and results are not renamed.

Qpos-only replay and `BatchedWarpYAMEnv` explicitly reject this contact-dependent
variant. They cannot reconstruct its measured support history. This restriction
does not change original spelling task support.

## Validation scope

The focused unit tests cover registry collisions, all existing short-task
count/horizon definitions, original catalogue lookup, geometry counterexamples
and invalid evidence. Native tests require the declared MuJoCo dependencies
and cover primitive contact loads, settling, the real environment hook/reset
boundary, native-score separation and unsupported replay routes. Existing
`test_spelling_goals.py` and `test_sorting_counts.py` remain the original-goal,
default-reset and native-score compatibility checks.

No plate variant, new task family, SDK/contact patch, simulator asset change
or learning experiment is included.
