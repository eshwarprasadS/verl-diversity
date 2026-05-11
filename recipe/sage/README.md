# SAGE Recipe for verl-diversity

SAGE (Self-Augmented Generation and Evaluation) recipe adapted for verl 0.7.1.

Source: https://github.com/BaohaoLiao/SAGE

## Mechanism

When all rollouts for a prompt receive zero reward, SAGE self-generates 3-level
progressive hints from the reference solution, then re-rolls with hints injected
into the prompt. Hints are only used during training -- they are never used at
validation or inference time.

### Hint levels

- **Level 1**: Minimal -- points to the key concept or approach
- **Level 2**: Medium -- provides direction on method or intermediate steps
- **Level 3**: Detailed -- gives substantial guidance while still requiring the
  student to complete the solution

### Strategies

- `sage` (on-policy): Roll first, identify zero-reward prompts, generate hints,
  re-roll with progressively stronger hints until solved or level 3 exhausted.
- `sage-light` (off-policy): Track per-prompt accuracy across training steps.
  Escalate/de-escalate hint level before each roll based on accuracy thresholds.

## Files

| File | Description |
|------|-------------|
| `hint_trainer.py` | `RayHintTrainer` subclass of `RayPPOTrainer` with hint logic |
| `main_hint.py` | Hydra entry point |
| `prompt.py` | Hint generation prompt templates |
| `reward_tracker.py` | Per-prompt reward and hint state tracking |
| `config/hint_trainer.yaml` | Config overlay (inherits `ppo_trainer.yaml`) |

## Usage

```bash
python -m recipe.sage.main_hint \
    trainer.method=sage \
    actor_rollout_ref.model.path=<model_path> \
    data.train_files=<train_data> \
    data.val_files=<val_data>
```

### Sage-light mode

```bash
python -m recipe.sage.main_hint \
    trainer.method=sage-light \
    trainer.hint_accuracy_min_threshold=0.0 \
    trainer.hint_accuracy_max_threshold=0.35 \
    actor_rollout_ref.model.path=<model_path> \
    data.train_files=<train_data> \
    data.val_files=<val_data>
```

## Configuration

Key config knobs (set via Hydra overrides):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.method` | `sage` | `sage` or `sage-light` |
| `trainer.hint_accuracy_min_threshold` | `0.0` | Accuracy below which hint level escalates (sage-light) |
| `trainer.hint_accuracy_max_threshold` | `0.35` | Accuracy above which hint level de-escalates (sage-light) |

## Dependencies

Requires `json5` for robust JSON parsing of hint payloads:

```bash
pip install json5
```

## Notes

- This recipe does NOT modify any verl core files.
- The dataset must include a `solution` (or `answer` or `reward_model.ground_truth`)
  field so that hints can be generated from the reference solution.
- Checkpointing saves and restores the reward tracker state alongside the model.
