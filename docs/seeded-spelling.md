# Optional seeded spelling goals

The existing `spell_cat`, `spell_dog`, and other fixed-word tasks retain their
prompts, rewards, metrics, and episode limits. To sample a sequence of two,
three, or four distinct letters on reset, pass a custom `SimTaskSpec` directly
to the environment factory:

```python
from abc_sim import SimTaskSpec, make_env

spec = SimTaskSpec(
    name="sampled_spelling",
    env_task="blocks",
    prompt="spell the sampled letter sequence",
    evaluator_name="spell_word",
    evaluator_options=(("sequence_length", 3),),
    reset_request=(("sequence_length", 3),),
)
env = make_env(task=spec, render_cameras=False)
observation, info = env.reset(seed=42)
print(observation["prompt"])
print(info["randomization"].metadata["spelling_goal"])
env.close()
```

The evaluator and reset request must specify the same length. Supply an
explicit nonnegative integer reset seed. The goal uses a separate random
stream, so adding a goal does not change block placements for that seed, and
scene-placement retries do not change the sampled sequence. Each letter has
one physical block, so sequences contain no repeated letters. The supported
lengths retain the two-, three-, and four-block scope of these optional goals.

Reset metadata records the realized uppercase `sequence`, its `seed`, and the
matching prompt. The native spelling evaluator reads that exact sequence on
every reset; its row, orientation, spacing, success, and partial-reward rules
remain the same. Scoring before binding a new reset goal is an error.

In random mode, the `word` metric has a world axis so batched worlds can retain
different targets. In fixed-word mode it remains a scalar string. A separate
evaluator discovers metric axes without changing a live episode's goal or
success history. A custom spec does not add an entry to the base task catalogue.

Use `env.set_task(other_spelling_spec)` to switch spelling goals on the same
blocks scene. This replaces the task's reset request, prompt, and evaluator;
switching back to `env.set_task("spell_cat")` clears the random-sequence request
and restores the fixed-task prompt. An explicit `make_env(..., prompt=...)`
override is retained when switching tasks. Sampled reset prompts still describe
the realized goal during random-sequence episodes, as they do on initial reset.
