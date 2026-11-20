# Reproducibility Artifact

**Paper:** *When Does Adjustment Help Under Simpson's Paradox? A Decision-Error Study on the Risk-Difference Scale*

**Authors:** Blinded for review. Author names, affiliations and corresponding-author details are provided on the separate title page submitted through the journal system. This artifact will be updated with the final author list and the Zenodo DOI before camera-ready submission.

**Short running title:** A decision-error study of adjustment under Simpson's paradox

---

## 1. What this study does (method in one paragraph)

This study asks *when statistical adjustment reduces decision error*, rather than when it improves bias or coverage, under Simpson's-paradox conditions on the risk-difference (RD) scale. A three-band decision rule is used: with threshold δ, a treatment is recommended when the estimated RD is below −δ, avoided when above +δ, and otherwise declared uncertain (δ = 0.01; the band is closed). The primary outcome is the three-band **decision error** rate, P(decision(θ̂) ≠ decision(θ)); **sign error** is a comparator. Four estimators are contrasted on identical replicate data: **B0** crude RD; **A1** direct standardisation to the sample distribution of the covariate; **A2** standardisation to the treated distribution (estimand θ_t); **A3** g-computation from a deliberately misspecified main-effects logistic model. Three Monte Carlo experiments are run: **Exp 1** varies the crude–truth relation over 24 scenarios (13 sign-reversing, 5 same-sign bias, 6 unconfounded; 2,000 replicates each); **Exp 2** probes three boundaries on a fixed-truth skeleton (nothing to adjust for, wrong adjustment set, three definitions of a wrong decision); **Exp 3** crosses two binary covariates in eight scenarios, adding an omitted-covariate arm (A1Z1). A closed-form normal approximation, decision error ≈ Φ(−((|θ| − δ)/σ + b/σ)), is shown to reproduce all 184 cells and yields a design nomogram.

**This is a pure Monte Carlo simulation study.** It uses no neural network, no pretrained weights, no external dataset and no GPU. `torch` is optional and is used only to fix its random seed in `utils/seed.py`.

## 2. Software and hardware environment

Recorded in the manuscript's `ENVIRONMENT.lock` and pinned in `requirements.txt`:

| Component | Version in the paper |
|---|---|
| Python | 3.11.1 |
| NumPy | 2.3.5 |
| pandas | 3.0.0 |
| matplotlib | 3.10.8 |
| PyYAML | ≥ 6.0 (config parsing) |
| torch | optional, only for seed fixing |

The study is CPU-only and hardware-independent; the reported run completed Experiment 1 in ≈ 40 s and the Experiment 3 simulation in ≈ 2 min on the authors' machine. Reproducibility comes from fixed `SeedSequence` substreams, not from a particular GPU. A checkout that only has NumPy, pandas and matplotlib can still run: `utils/io.py` contains a dependency-free YAML subset reader used automatically when PyYAML is absent.

To install (in your own environment; this artifact never downloads anything at runtime):

```bash
python3 -m pip install -r requirements.txt
```

## 3. Data and pretrained weights

- **Datasets:** none. All data are generated at runtime by the DGPs in `src/simulate.py`, `src/simulate_exp3.py` and `src/exp2.py` from the frozen parameter files in `configs/`. No dataset is bundled and no download link is required for the simulations.
- **Replicate-level records:** Experiment 1 (48,000 rows) and Experiment 3 (96,000 rows) accompany the submission as a reproducibility package. Experiment 2 replicate-level files are not yet in the package (manuscript Sec. 5.5, Appendix F); its cells are regenerated from the seed specification in Sec. 3.6.
- **Public deposition:** Zenodo deposition is planned before submission. **[PLACEHOLDER — insert reserved DOI here.]** Until then, the package is available from the corresponding author.
- **Pretrained weights:** not applicable. No model weights exist in this study; there is nothing to download.

No resource is fetched from the network by any script in this repository.

## 4. One-command quick start

From the repository root:

```bash
# Environment check + 29 artifact self-checks (fast, no simulation)
python3 scripts/verify.py

# Full reproduction (Exp 1 + Exp 3 + evaluation + figures)
bash scripts/run_all.sh

# Or run each piece independently:
bash scripts/run_exp1.sh          # 8 scenario blocks, 2000 reps each
bash scripts/run_exp2.sh          # Exp 2 boundary probes (seed regeneration)
bash scripts/run_exp3.sh          # Exp 3, 8 scenarios, 2000 reps each
python3 scripts/evaluate.py       # metrics tables
python3 scripts/ablation.py       # strategy contrasts
python3 scripts/plot.py           # Figures 1-4, S1-S5
```

**Tiny smoke test (does NOT reproduce paper numbers):**

```bash
python3 src/run_pipeline.py --mode check
python3 src/run_pipeline.py --mode simulate --smoke --repetitions 3 --output-dir outputs/smoke
python3 src/run_pipeline.py --mode analyze --output-dir outputs/smoke
python3 src/run_exp3.py --smoke --repetitions 3 --output-dir outputs/smoke_exp3
```

Every run accepts `--config`, `--scenarios`, `--output-dir`, `--repetitions`, `--delta` and `--seed-base` on the command line; the shell wrappers accept `REPS` and `PY` environment variables. **All paths are relative to the repository root — absolute paths are rejected by design** (`utils/io.py::resolve_path`).

## 5. Reported metrics (placeholders to be filled after the local GPU/CPU rerun)

These slots mirror Tables 1–6 of the manuscript. Values are intentionally left as placeholders until the artifact is rerun in the final environment; fill them from `outputs/metrics_eval.csv` and `outputs/exp3/metrics_exp3.csv`.

| Quantity | Paper value | Artifact rerun |
|---|---|---|
| Exp 1 REV family crude decision error | 0.979 [0.946, 1.000] | `TBD` |
| Exp 1 REV family A1 decision error | 0.036 [0.011, 0.066] | `TBD` |
| Exp 1 REV family A2 decision error | 0.003 [0.001, 0.005] | `TBD` |
| Exp 1 REV A1 paired difference vs B0 | −0.943 [−0.979, −0.900] | `TBD` |
| Exp 1 CON crude decision error | 0.000 [0.000, 0.001] | `TBD` |
| Exp 1 NULL crude vs A1 | 0.056 vs 0.054 | `TBD` |
| Exp 1 A3 decision error (REV) | 0.064 [0.030, 0.100] | `TBD` |
| Φ approximation fit (184 cells) | r ≥ 0.995 | `TBD` |
| Exp 2 no-confounding cells worse | 9 / 9 | `TBD` |
| Exp 2 collider/mediator worse | 36 / 36 per arm | `TBD` |
| Exp 3 near-threshold A1 error (n=200) | 0.380 | `TBD` |
| Exp 3 A1Z1 bias | +0.13 | `TBD` |
| `verify.py` checks passed | 29 / 29 | `TBD` |

## 6. Reproduction notes

- **Seeds.** Repetition *r* uses `SeedSequence([seed_s, r])`; the coverage bootstrap uses `SeedSequence([seed_s, r, 1])`. Scenario seeds are `exp1_seed_base + scenario index` (Exp 1/3), `exp2_grid_base + 100000·cell + replicate` (E1), and `exp2_struct_base + 2000000·arm + 2000·cell + replicate` (E2). All bases are in `configs/run_config.yaml`.
- **Frozen configuration.** The scenario table and seeds were frozen before execution; SHA-256 `c7ee2cd29ea87cc5`.
- **Estimator identity check.** With one binary covariate, saturated g-computation and A1 agree exactly; `scripts/verify.py` and the analyze mode report the maximum discrepancy (expected 0 to machine precision).
- **Truths.** `θ = Σ_c P(C=c)(p1c − p0c)`; `θ_t` weights by `P(C=c | A=1)`. A2 is always compared against `θ_t`, not `θ`. For SC18–SC20 the closed form `θ = 0.02 − 0.22π` is verified.
- **Empty-cell rule.** E2's pre-registered primary rule is `empty_arm_only`; alternatives (`all_haldane`, `degenerate_only`, `raw_skip`) are selectable in config.
- **Known scope limits (from manuscript Sec. 5.5).** Binary covariates only; RD scale only; parameters set by design; Exp 2 replicate-level records not yet packaged. These are declared limitations, not artifact defects.

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: numpy / pandas / matplotlib / yaml` | dependencies not installed | `python3 -m pip install -r requirements.txt` |
| `Absolute paths are not allowed` | a path like `/home/...` or `C:\...` was passed | use paths relative to repo root; edit YAML or pass relative `--output-dir` |
| `No replicate CSVs found` | analyze/evaluate run before simulate | run the matching `--mode simulate` or `run_exp1.sh` first |
| `Expected a mapping at top level of ...` | YAML file corrupted | restore from the repository; validate indentation (spaces, no tabs) |
| `family` parsed as `None` for NULL scenarios | YAML 1.1 treats bare `NULL` as null | keep the quotes: `family: "NULL"` |
| Figures empty | metrics CSV missing columns | run `scripts/evaluate.py` first, then `scripts/plot.py` |
| Numbers differ from the paper | different library versions or reps | match pinned versions in `requirements.txt`; use `REPS=2000`; check `verify.py` |
| `torch` warnings | torch absent or CUDA unavailable | harmless; torch is optional and only seeded if importable |

## 8. BibTeX

```bibtex
@article{adjustment_decision_error_simpson_2026,
  title   = {When Does Adjustment Help Under Simpson's Paradox? A Decision-Error Study on the Risk-Difference Scale},
  author  = {The Authors},
  journal = {[PLACEHOLDER — insert target journal]},
  year    = {2026},
  note    = {Blinded review version. Author list, journal and DOI to be inserted before submission.},
  doi     = {[PLACEHOLDER — insert reserved Zenodo DOI]}
}
```

## 9. Artifact checklist

- [x] Fixed directory layout: `src/`, `scripts/`, `configs/`, `utils/`, `README.md`, `requirements.txt`, `.gitignore`, `LICENSE`
- [x] No datasets, weights or large files generated; download/DOI slots declared as placeholders
- [x] No hard-coded absolute paths; all inputs from YAML / CLI, absolute paths rejected
- [x] Seeds fixed for `random`, `numpy` (`SeedSequence`) and optional `torch`
- [x] Independent scripts: training/simulation, evaluation, ablation, plotting, verification
- [x] README contains all 11 required sections (title/authors, method, environment, data, weights link, one-command run, metric placeholders, BibTeX, reproduction, troubleshooting, checklist)
- [x] Sandbox smoke test only (3 reps); no full-scale training performed here
- [x] Changes delivered in 4 confirmed batches; no git commit performed
- [x] `verify.py`: 29/29 checks pass
- [ ] Final local GPU/CPU rerun with paper versions and `REPS=2000`; fill Section 5 placeholders
- [ ] Insert final author list, target journal, reserved Zenodo DOI in Sections 1, 3 and 8

## License

MIT — see `LICENSE`.
