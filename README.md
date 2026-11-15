# Zeno-free but late: ordinal event triggers in master–slave Lorenz synchronisation

Code, simulated data and LaTeX source for the article

> J. Padilla-Villanueva, *Zeno-free but late: ordinal event triggers with a guaranteed minimum inter-event time in master–slave Lorenz synchronisation*, Zenodo, 2026. [doi:10.5281/zenodo.24263807](https://doi.org/10.5281/zenodo.24263807)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24263807.svg)](https://doi.org/10.5281/zenodo.24263807)

## Abstract

**Background.** Event triggers built on windowed Kendall τ_b fire when the mean absolute change D_E of ordinal concordance on a fixed edge set leaves a band. For re-anchoring (R1), hysteresis (R2) and their κ-qualified forms (R3), a sampled minimum inter-event time (MIET) has been proved and machine-checked in Lean 4, together with exact invariance of the event sequence under strictly increasing sensor distortion. Whether these guarantees yield useful closed-loop control had not been tested. **Methods.** We inserted R1, R2 and two κ-qualified variants (R3a, R3b) as the “when to update” layer of a two-layer master–slave Lorenz synchronisation loop with a saturated sliding-mode correction. We compared them with continuous updating, a Girard-type dynamic magnitude trigger and a naive ordinal level trigger. A rank-based correction layer (V) tested distortion invariance of the whole loop. All parameters (W = 20, p = 1, θ = 0.40, δ = 0.20, θ_κ = 0.5) were fixed a priori. We used 20 seeds and four sensor scenarios (clean, Gaussian noise, monotone distortion, drift), with paired Wilcoxon tests and Holm adjustment. **Results.** The guarantees held exactly. Across 375 runs with a measurable inter-event time there were 0 violations of the bound 2h. The observed minimum was 3h, and the largest one-step change of τ_b was 4/W = 0.20: the bound is attained but never exceeded. The event sequence and the error of V were identical with and without distortion in all 20 seeds. Control performance, however, was poor. The magnitude trigger had lower error than every ordinal rule in every scenario and in 20 of 20 seeds (two-sided p = 1.9×10^−6, Holm-adjusted p = 7.6×10^−6). A periodic schedule with a comparable update budget also outperformed R1 (clean RMS 5.93 vs 12.81). **Conclusion.** In this loop, a Zeno-free ordinal trigger fires too late to serve as a fine tracking trigger. It is better viewed as a structural alarm with guaranteed event separation.

## Contents

| Path | Description |
|---|---|
| `simulation/bench.py` | Master–slave Lorenz plant, sensor scenarios, sliding-mode correction and baseline policies A, B, C |
| `simulation/cancellation.py` | Cancellation identity on a fixed edge set: D_E, D̄_E, C_E, κ_E |
| `simulation/run_experiment.py` | Rules R1, R2, R3a, R3b, rank policies V and VA, calibration, verification checks, statistics and periodic diagnostic |
| `data/reference_results_v1.csv` | Results of policies A, B, C from the earlier bench (reproduced exactly by `run_experiment.py`) |
| `results/runs.csv` | One row per run (method × scenario × seed) |
| `results/summary.csv` | Per-scenario, per-method summary |
| `results/tests.csv` | Paired Wilcoxon tests with Holm adjustment |
| `results/periodic_diagnostic.csv` | Periodic schedule P with a budget comparable to R1 |
| `results/run_summary.txt` | Console summary of the run |
| `article/` | `main.tex`, `refs.bib`, `figures/`, `generated/` (tables and number macros), `scripts/make_assets.py`, `main.pdf` |

### Columns of `results/runs.csv`

`seed`; `scenario` (clean, noise, distortion, drift); `method` (A, B, C, R1, R2, R3a, R3b, V, VA); `rms` (true RMS synchronisation error over t ∈ [6, 20]); `n_updates` (control updates on [0, 20], including the initial one); `min_iet`, `med_iet` (minimum and median inter-event time); `n_events_post` (events with t ≥ 6); `diverged`; `n_post_nan`; `n_eval`, `n_eval_nan`, `n_eval_ties` (ordinal evaluations, undefined τ_b, windows with ties); `max_step_dtau` (largest one-step change of τ_b); `bound` (guaranteed MIET); `bound_ok` (1 = bound respected, 0 = violated, `no_IET` = at most one event). `periodic_diagnostic.csv` has the same columns plus `period_steps`.

## Reproduction

Requirements: Python 3 (tested with 3.13) and the packages in `requirements.txt`; a TeX distribution with `pdflatex`, `bibtex` and `latexmk` for the article.

```bash
python3 -m pip install -r requirements.txt

# 1. Simulation (about 4 minutes on one CPU core); writes results/*.csv
python3 simulation/run_experiment.py

# 2. Figures, tables and number macros from results/*.csv
python3 article/scripts/make_assets.py

# 3. Article PDF
cd article && latexmk -pdf main.tex
```

Steps 2 and 3 use the CSV files shipped in `results/`, so step 1 is optional. The simulation is deterministic: seeds 0–19 (evaluation) and 100–104 (calibration) fix all initial conditions and noise realisations.

## Related work

- Companion preprint with the proofs: J. Padilla-Villanueva, *Minimum inter-event time for ordinal (Kendall τ_b) event triggers: proofs, counterexamples and an open conjecture*, Zenodo, [doi:10.5281/zenodo.24260444](https://doi.org/10.5281/zenodo.24260444).
- Lean 4 formalisation of the discrete core: [github.com/johelpadilla/ordinal-miet-lean](https://github.com/johelpadilla/ordinal-miet-lean).

## License

Code: MIT (see `LICENSE`). Article text and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

See `CITATION.cff`.
