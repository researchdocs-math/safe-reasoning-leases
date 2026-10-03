# Safe Reasoning Leases

Reproducible experiments for state-dependent reasoning deadlines in edge-agent control. The runtime combines a recoverability certificate, a backup watchdog, generation fencing, and a current-state action gate. The two synthetic storage systems include actuator inertia and bounded sensor errors.

## Contents

- `experiments/core.py`: plant dynamics, safety leases, numerical action prediction, and seeded scenarios.
- `experiments/simulate.py`: parallel, trace-driven closed-loop experiments and independent fixed-timeout calibration.
- `experiments/verify.py`: randomized certificate checks and a reproducible inertia counterexample.
- `experiments/inference.py`: the local Qwen3 agent, prompts, tool protocol, and GPU profiling.
- `experiments/live.py`: live inference with a wall-clock-paced simulator and a separate watchdog thread.
- `experiments/analyze.py`: seed-level summaries and paired bootstrap contrasts.
- `experiments/audit_records.py`: integrity checks against frozen input hashes and independent verification of recorded live transitions.
- `experiments/check_reproduction.py`: numerical comparison with the archived reference outputs.
- `results/`: recorded inference profiles and live trajectories, including unsuccessful workflows.
- `expected/`: reference results for deterministic comparisons.
- `configs/`: the frozen experiment protocol and pinned model revisions.
- `SHA256SUMS`: integrity checksums for the release files.

## Reproduce the numerical results

Use Python 3.12. The CPU workflow runs on macOS, Linux, or Windows. Run the following commands from this directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/verify.py
python experiments/check_prediction.py
python experiments/audit_records.py
python experiments/simulate.py --phase tune --workers 10
python experiments/simulate.py --phase test --workers 10
python experiments/analyze.py
python experiments/check_reproduction.py
```

On Windows, activate with `.venv\Scripts\activate`. Set `--workers` to the available CPU capacity. Ten workers were used on the 12-core M3 Pro. Numerical comparisons allow relative tolerance `1e-9` and absolute tolerance `1e-10` for platform-level floating-point differences.

The `tune` phase uses 24 development seeds per cell to choose a fixed timeout from 1, 2, 4, 8, 12, and 18 seconds and either a held-command or backup response on expiry (12 configurations). The `test` phase uses 100 disjoint seeds for each model, plant, and latency factor. Each episode lasts 180 simulated seconds. The input timing traces are sampled with fixed seeds. The 2× and 4× factors are synthetic delay stress. Timing replay uses a fixed numerical proposal generator so that differences isolate scheduling rather than model output quality.

The analysis produces `aggregate_summary.csv`, `paired_contrasts.csv`, `cell_summary.csv`, `cell_contrasts.csv`, `selected_fixed.csv`, `latency_summary.csv`, `live_summary.csv`, and `summary.json` in `results/`. Aggregate intervals first average the 12 model–plant–delay cells within a seed, then resample seeds 10,000 times. Constraint failure is evaluated at the episode level.

## Remeasure local inference

GPU profiling uses Apple silicon and MLX. Install the optional dependencies and download the pinned checkpoints:

```sh
python -m pip install -r requirements-gpu.txt
python experiments/download_models.py
```

The recorded models are `mlx-community/Qwen3-1.7B-4bit` and `mlx-community/Qwen3-4B-4bit`. Their exact revisions are in `configs/model_manifest.json`. The download includes each checkpoint's model card and license information. The recorded environment used an Apple M3 Pro, 18 GB unified memory, macOS, and Python 3.12.

Profiling resumes by record identifier. To collect a fresh dataset, move the two existing `results/*_profile.jsonl` files into a separate run directory, then run:

```sh
python experiments/inference.py --model Qwen3-1.7B-4bit --n 120
python experiments/inference.py --model Qwen3-4B-4bit --n 120
```

A profile record contains model and scenario identifiers, telemetry-context length, elapsed time, generated token count, parsed action, status, full tool transcript, and observed peak MLX memory. Each model receives 40 jobs for each of 0, 16, and 64 telemetry records. Greedy decoding uses at most 96 tokens per turn and six turns per job. Each turn prefills the complete prompt without reusing a key/value cache across turns. All outcomes remain in the timing dataset.

For a fresh live run, move `results/live.jsonl` to a separate run directory and run:

```sh
python experiments/live.py --model Qwen3-4B-4bit --seeds 10 --speed 3
python experiments/analyze.py
```

The simulator advances three simulated seconds per wall-clock second. Every live record includes plant trajectories, complete job outcomes, and watchdog scheduling lateness. Policy order is randomized within each seed–plant block. Fresh wall-clock measurements vary with hardware, system load, and runtime versions; the archived profiles are the inputs for reproducing the reported numerical replay results.

## Interpreting the measurements

All protected schedulers share the same watchdog, backup command, generation check, and actuator admission test. The lease deadline additionally reserves 0.5 simulated seconds for handoff. Its safety argument assumes the stated discrete dynamics, correct error bounds, nonnegative withdrawal, and a bounded independent handoff.

Tracking service is the fraction of simulated time within 0.08 of the target. Discarded inference is elapsed occupancy in timed-out, obsolete, or inadmissible jobs. An episode-censored job contributes occupancy but is not classified as discarded. These are time measurements, not electrical-energy measurements. Model agreement in the profiles means agreement with the specified numerical action predictor.

`verify.py` checks backup invariance and leased handoff over randomly selected admissible states. Its timing measurements reflect the local machine; the trajectory counterexample and seeded invariant checks are deterministic. `check_reproduction.py` validates the numerical replay and statistical outputs against `expected/`.

## License and citation

The experiment software and generated experiment records are released under the MIT License. Citation metadata is provided in `CITATION.cff`. Model checkpoints and dependencies retain their respective upstream licenses.
