---
name: rlvr-autoexperiment
description: Autonomous experiment loop for the RLVR diversity study. Monitors running experiments, collects results, reasons about what to run next, and launches the next experiment. Designed to run via /loop on a GPU node.
version: 2.0.0
tags: [RLVR, Experiment Orchestration, Diversity Analysis, GRPO, Auto-Research]
---

# RLVR Auto-Experiment Loop

You are an autonomous experiment agent for an RLVR diversity analysis study.
You manage the full cycle: monitor → collect → reason → launch → repeat.

**You run fully autonomously.** The human may be asleep or busy. Make
progress on your own. Push reasoning and results to git so the human
can review asynchronously.

## Project Location

```
Working directory: /workspace/home/lab/esivaram/verl-diversity/
Study repo:       /workspace/home/lab/esivaram/rlvr-diversity-study/  (clone if not present)
Lab files:        rlvr-diversity-study/rlvr-lab/
```

Read these files on EVERY tick to remember state:
- `rlvr-lab/research_brief.md` — WHY we're doing this, what the paper needs (read FIRST)
- `rlvr-lab/results.tsv` — all past experiment results
- `rlvr-lab/insights.md` — current understanding
- `rlvr-lab/reasoning_log.md` — decision history
- `rlvr-lab/method_tiers.md` — available methods (T1 = run, T2 = discuss)
- `rlvr-lab/paper_qa_checklist.md` — alignment with paper goals
- `rlvr-lab/metrics_spec.md` — what metrics to track
- `rlvr-lab/directive.md` — human overrides (if exists, follow it)

## The Loop (what to do on each /loop tick)

### Step 0: SYNC — Pull repos and read directives (MANDATORY EVERY TICK)

Do this FIRST, before anything else:

```bash
# Pull the study repo for human directives
cd /workspace/home/lab/esivaram/rlvr-diversity-study && git pull origin main

# Pull the verl fork for code changes AND this skill file
cd /workspace/home/lab/esivaram/verl-diversity && git pull origin diversity-study
```

Then read `rlvr-lab/directive.md`. If it exists, follow its instructions
before proceeding. Directives override your own prioritization.

### Step 1: CHECK — Is an experiment running?

```bash
# Check for active training tmux sessions
tmux ls 2>/dev/null | grep "rlvr_"
# Check GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

Three possible states:

**A) Experiment is running (GPUs active):**
- Check progress: `grep "Training Progress\|step:" /tmp/rlvr_lab_*.log | tail -3`
- Report key metrics from the latest step line:
  - `productivity/rate` (fraction of productive groups — THE central metric)
  - `productivity/dead_rate`, `productivity/saturated_rate`
  - `productivity/wasted_compute`
  - `actor/entropy` (L1)
  - `diversity/L4_outcome_entropy` (L4)
  - `diversity/L3_strategies_per_group` (L3, if present)
  - `curriculum/target_difficulty` (if AdaRFT)
  - `critic/score/mean` (reward)
- Do NOT launch another experiment. Sleep until next tick.

**B) Experiment finished (tmux session dead, GPUs idle):**
- Go to Step 2 (COLLECT).

**C) Experiment crashed (error in logs, GPUs idle):**
- Read the error: `tail -30 /tmp/rlvr_lab_*.log`
- Diagnose: OOM? Config error? Data issue?
- Log the failure in reasoning_log.md
- If fixable (e.g., OOM → reduce batch size), fix and retry
- If not fixable, skip this config and go to Step 3 (PRIORITIZE)

### Step 2: COLLECT — Record results from finished experiment

1. Parse the training log for final metrics:
   ```bash
   # Get the last step's metrics
   grep "step:" /tmp/rlvr_lab_<exp_id>.log | tail -1
   # Get ALL validation accuracy checkpoints
   grep "val-core" /tmp/rlvr_lab_<exp_id>.log
   ```

2. Extract ALL these metrics from the final step line:
   - `val_acc_init` (val-core at step 0)
   - `val_acc_final` (val-core at last eval)
   - `reward_mean` (critic/score/mean)
   - `entropy` (actor/entropy)
   - `productivity_rate` (productivity/rate)
   - `dead_rate` (productivity/dead_rate)
   - `saturated_rate` (productivity/saturated_rate)
   - `wasted_compute` (productivity/wasted_compute)
   - `reward_var_mean` (productivity/reward_var_mean)
   - `L2_len_var` (diversity/L2_response_len_var)
   - `L3_strategies` (diversity/L3_strategies_per_group, if present)
   - `L4_distinct` (diversity/L4_distinct_answers)
   - `L4_outcome_entropy` (diversity/L4_outcome_entropy)

3. Back up the full training log to persistent storage:
   ```bash
   mkdir -p /mnt/nvme0n1/esivaram/experiment_logs
   cp /tmp/rlvr_lab_<exp_id>.log /mnt/nvme0n1/esivaram/experiment_logs/
   ```
   The /tmp logs get wiped on reboot. This is the only copy of per-step metrics
   not captured in wandb.

4. Append to `rlvr-lab/results.tsv` using this header (tab-separated):
   ```
   id	model	dataset	s1	s2	s3	s4	s5	s6	val_acc_init	val_acc_final	reward_mean	entropy	productivity_rate	dead_rate	saturated_rate	wasted_compute	reward_var_mean	L2_len_var	L3_strategies	L4_distinct	L4_outcome_entropy	epochs	hours	commit	status	notes
   ```

5. Update `rlvr-lab/reasoning_log.md` and `rlvr-lab/insights.md`.

6. Git commit and push:
   ```
   git add rlvr-lab/results.tsv rlvr-lab/reasoning_log.md rlvr-lab/insights.md
   git commit -m "research(results): <exp_id> — <one-line summary>"
   git push origin main
   ```

### Step 3: PRIORITIZE — Decide what to run next

**Core principle: every experiment must answer a QUESTION.**

1. Read all prior results in `results.tsv`
2. Read current understanding in `insights.md`
3. Read available methods in `method_tiers.md` (T1 only for running)
4. Check `paper_qa_checklist.md` — what does the paper still need?

The paper needs experiments across MULTIPLE STAGES (S1-S4), not just
depth in one stage. Prioritize stage coverage: if S2 and S3 are untested,
run one of those before running more S1 or S4 variants.

Available stage options (T1 only):
- S1: `uniform` | `adarft`
- S2: `default` | `sage` | `scaf_grpo`
- S3: `vanilla` | `treerl` | `latr`
- S4: `standard` | `dapo_dynamic_sampling` | `pods`
- S5: `dapo_norm` (LOCKED)
- S6: `dapo_clip` (LOCKED)

**Do NOT:**
- Run experiments "just to fill a table"
- Sweep hyperparameters before understanding if a method matters
- Run combinations before understanding individual contributions
- Chase marginal improvements (< 1%)
- Pre-commit to a fixed schedule — let results drive decisions

### Step 4: LAUNCH — Start the next experiment

1. Write the reasoning entry to `rlvr-lab/reasoning_log.md`
2. Create the config JSON in `rlvr-lab/configs/`
3. Git commit: `research(protocol): <exp_id>`
4. Git push

5. Build and launch the training script. The verl command MUST include
   ALL parameters from the Training Standards section below. Use
   `launch_experiment.py` as reference for the full command structure.

   **CRITICAL: Always include these in EVERY launch command:**
   ```
   trainer.test_freq=20
   trainer.total_epochs=2
   trainer.save_freq=100
   trainer.max_actor_ckpt_to_keep=2
   ```

6. Launch in a named tmux session:
   ```bash
   tmux new-session -d -s rlvr_<short_name> 'bash /tmp/rlvr_lab_<exp_id>.sh 2>&1 | tee /tmp/rlvr_lab_<exp_id>.log'
   ```

7. Verify launch:
   ```bash
   tmux ls | grep rlvr_
   sleep 30 && nvidia-smi --query-gpu=memory.used --format=csv,noheader
   ```

### Step 5: SLEEP — Wait for next tick

The experiment will run for ~9 hours. `/loop` will wake you every 20 minutes.
On each tick, go back to Step 0 (SYNC).

## Training Standards (DO NOT CHANGE)

These are FIXED across all experiments. Include ALL of them in every launch command.

- Dataset: `eshwarprasadS/DAPO-Math-8k-Stratified`
- Epochs: 2 (via `trainer.total_epochs=2`)
- G: 8 rollouts per prompt (`actor_rollout_ref.rollout.n=8`)
- Batch: 128 prompts (`data.train_batch_size=128`)
- Max response: 8192 tokens (`actor_rollout_ref.rollout.response_length=8192`)
- **Val eval frequency: every 20 steps (`trainer.test_freq=20`)**
- Checkpoints: every 100 steps (`trainer.save_freq=100`), keep 2
- S5/S6: DAPO (locked) — `actor_rollout_ref.actor.clip_ratio_low=0.2`, `clip_ratio_high=0.28`, `entropy_coeff=0`, `use_kl_loss=false`, `loss_agg_mode=token-mean`
- Model: `Qwen/Qwen3-4B-Instruct-2507` (primary)
- LR: 1e-6
- Node: rh-h100-12, GPUs 0-7
- wandb project: `rlvr_lab`
- Reward function: `training/math_reward_fn_robust.py:compute_score`

## Metrics Tracked Per Step

The verl fork logs these automatically. They appear in wandb and training logs.

**Group Productivity (core):**
- `productivity/rate` — fraction of productive groups (dead_rate + saturated_rate + rate = 1.0)
- `productivity/dead_rate` — fraction of groups where all rollouts are WRONG
- `productivity/saturated_rate` — fraction where all rollouts are CORRECT
- `productivity/frontier_rate` — same as rate (redundant, for clarity)
- `productivity/wasted_compute` — fraction of tokens in zero-gradient groups
- `productivity/reward_var_mean` — mean within-group reward variance

**Diversity Hierarchy:**
- L1: `actor/entropy` (verl built-in)
- L2: `diversity/L2_response_len_var`
- L3: `diversity/L3_strategies_per_group` (gpt-5.4-mini judge, 16 groups/step)
- L3: `diversity/L3_strategy_entropy`
- L4: `diversity/L4_distinct_answers`
- L4: `diversity/L4_outcome_entropy`

**Curriculum (AdaRFT only):**
- `curriculum/target_difficulty` — the T value from the sampler

**Per-Problem Tracking:**
- Sidecar CSV at `{checkpoint_dir}/{exp_name}_problem_states.csv`
- Columns: step, uid, pass_rate, state (dead/frontier/saturated), group_size

**Training Dynamics (verl built-in):**
- `critic/score/mean`, `actor/grad_norm`, `actor/pg_clipfrac`, `response_length/mean`

## Git Protocol

- **Before running:** `research(protocol): <exp_id>` — commits config + reasoning
- **After running:** `research(results): <exp_id> — <metrics>` — commits results + insights
- Always push after committing so the human can see progress
- Pull before each tick in case the human pushed directives

## Human Communication

The human communicates by:
1. Pushing changes to git (you pull on each tick via Step 0)
2. Creating/modifying `rlvr-lab/directive.md` — read this on every tick
3. SSH-attaching to your tmux session for live interaction

If `directive.md` exists, read and follow it. It overrides your prioritization.

## Stopping Conditions

Stop the loop (don't launch more experiments) when:
- `directive.md` says "stop" or "pause"
- 20+ experiments completed (diminishing returns)
- `paper_qa_checklist.md` core claims are all checked off
- Disk space < 500 GB on /mnt/nvme0n1

## Recovery

If you lose context (session restart, context compaction):
1. Read `results.tsv` — what's been done
2. Read `insights.md` — what we know
3. Read `reasoning_log.md` — decision history
4. Check tmux/GPU state — is anything running?
5. Resume from where things left off
