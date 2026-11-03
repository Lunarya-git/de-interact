# Drug–Excipient Incompatibility Prediction: DE-INTERACT reproduction, leakage audit, and a graph cross-attention extension

Reproduces the ANN of *Patel et al., "DE-INTERACT", Int. J. Pharm. 637 (2023) 122839* (881-bit PubChem fingerprints of drug + excipient →
compatible / incompatible), audits its evaluation for **duplicate-pair and chemical-structure leakage**, and compares it under splits that test
generalisation to **unseen drugs / excipients** against stronger baselines and a proposed **symmetric hybrid graph + fingerprint network with
bidirectional atom-level cross-attention (DE-CrossNet)**.

> **Project status (please read).** Implemented and ready: data audit/cleaning, PaDEL fingerprints, CID/structure-level split machinery with leakage assertions, the paper ANN,
> fingerprint baselines (concat-MLP, Siamese-FP, RF, entity-prior), metrics, experiment runner (E1–E4 *v1*), tests.
> **Plan corrected in v3 (documentation only — no code changed; awaiting approval):** realistic 20 h schedule (11.3 h planned, 125 Optuna trials instead of 200), chemistry-only cold-start grouping with
> fingerprint collisions evaluated separately, tuning on a pool of molecules that never appears in the final test sets, concrete graph-batching/masking/memory specification, 4 core ablations instead of 16, and a verified reference list.
> **Not implemented yet:** DE-CrossNet, graph baselines, new split logic (`chem` unit, entity pools), tuning, benchmark gate, v2 experiments. No result tables exist. Details: [`PROJECT_DESIGN.md`](PROJECT_DESIGN.md).
> **Known code/doc discrepancy:** `split.leakage_unit: structure` (shipped default) still merges molecules with identical PubChem fingerprints; the corrected primary unit `chem` is implemented in Stage 0a. Until then, cached splits and `split_summary.csv` are legacy.

## Research question and hypotheses
Does the paper's performance reflect chemistry-based generalisation, or memorisation of duplicated pairs and of individual molecules? Does a role-symmetric model
that sees atom-level structure and explicitly attends across the two molecules generalise better than the paper's concatenation MLP, *at equal tuning effort*?
H1 cross-molecule atom attention helps on unseen molecules · H2 the gain comes from the interaction module, not just the graph or regularisation ·
H3 graph + fingerprint ≥ either alone · H4 gains are bounded by identity shortcuts (diagnosed by `entity_prior`). A tie within seed noise is reported as "no evidence of improvement".

## Final architecture (planned): DE-CrossNet
```
SMILES ─► molecular graph ─► GINE encoder (shared θ) ─► atom embeddings ┐
SMILES ─► PubChem FP (881) ─► FP-MLP ───────────────────────────────────┤  per molecule A and B (same weights)
         bidirectional multi-head cross-attention (A↔B atoms, shared weights) ◄┘
         attention + mean readout ⊕ FP embedding ─► m_A, m_B
         z = [m_A+m_B, m_A⊙m_B, |m_A−m_B|] ─► MLP ─► P(incompatible)      (f(A,B)=f(B,A) by construction)
```
≈ 0.9 M parameters (paper ANN 3.9 M). Not a new algorithm: an adaptation of DDI co-attention designs (SSI-DDI, SA-DDI) to drug–excipient pairs plus the paper's fingerprint, with one-factor ablations.

## Models and experiments
| Model | Input | Status |
|---|---|---|
| `paper_ann` — DE-INTERACT ANN (3×1024, softmax-2, BCE, RMSprop) | FP concat | implemented |
| `concat_mlp`, `concat_mlp_reg` — same family, unified protocol (± dropout / tuning) | FP concat | implemented (tuning planned) |
| `siamese_fp` — shared encoder + symmetric pair features (v1 proposal, now a baseline; config key still `siamese`) | FP | implemented |
| `gnn_siamese` — graph encoder, no cross-attention | graph | planned |
| **`de_crossnet`** — proposed | graph + FP | planned |
| `rf`, `entity_prior` (identity-only diagnostic) | FP / IDs | implemented |

| Exp | Purpose | Splits / seeds | Status |
|---|---|---|---|
| E0 | benchmark gate (time, memory, epochs) | – | planned |
| E1 | reproduce the paper protocol (leaky row split, full-data metrics) | `paper_rows`, 5 seeds | code ready |
| E2 | quantify leakage (paper model, clean split, train-only oversampling) | `random_pair`, 5 seeds | code ready |
| T | Optuna tuning on validation PR-AUC inside pool D, seeds 100/101, trials = 5 × #dimensions (40/30/30/25 = **125**), wall-clock capped | 3 split types | planned |
| E3 | main comparison: `de_crossnet`, `gnn_siamese`, `siamese_fp`, `concat_mlp_reg`, `concat_mlp` (+ `rf`, `entity_prior`) | 4 splits × 10 seeds = 200 GPU runs | v1 code ready (4 models, 5 seeds); v2 planned |
| E4 | ablations: tier 1 = 4 core variants (`no_fp`, `self_attn` [capacity-matched control], `ordered_head`, `largest_fragment`), 10 seeds = 120 runs; tier 2 (conditional) = 2 design checks, 5 seeds | 3 splits | v1 code ready (2 variants); v2 planned |
| E5 | sensitivity: CID vs `chem` vs `parent` units; **fingerprint-collision-stratified metrics** (post hoc); label-conflict exclusion; tuned vs default | 3 splits × 5 seeds (60 runs) | planned |
| E6 | attention case studies | – | planned |
Budget (RTX 3050, **estimates until E0**; each row = runs × s/run × 1.1): E0 0.4 · E1/E2 0.4 · T 3.6 · E3 2.3 · E4-T1 2.8 · E5 0.9 · E4-T2 0.7 (conditional) · E6 0.2 = **11.3 h planned**; 16.95 h if every timing is 1.5× too optimistic; 3.05 h hard reserve of the 20 h — see `PROJECT_DESIGN.md` §13/§15.

## Leakage controls (enforced by assertions in the current code; v3 changes are planned, see below)
Duplicate rows removed (unique CID pairs); **structure groups** hold CIDs of one chemical structure out together. *Current code (legacy):* merges identical SMILES / stereo-stripped SMILES / InChIKey connectivity **and identical PubChem fingerprint** (799 → 766 groups) — wrong for fingerprint-colliding but chemically different molecules.
*Corrected primary unit `chem` (Stage 0a, measured by `scripts/analysis_chem_groups.py`):* identical SMILES / stereo-stripped SMILES / InChIKey connectivity only → **783 groups**, 3,570 unique pairs, 2 label-conflicting group-pairs; the **16 fingerprints shared by 33 distinct chemical groups** (10 fingerprint-pair keys with conflicting labels) are reported separately, never merged. With CID-keyed cold splits 19.6 % of test excipients had a structure twin in training; `random_pair` splits on unordered structure-pairs; training-only class weighting/oversampling;
model, epoch and threshold selected on validation only; test used once. `split.leakage_unit` in the config currently selects `structure` (legacy default) | `cid` | `parent`; planned: `chem` (default).
**Planned evaluation protocol (§11–12):** a fixed, label-blind evaluation-only entity pool E (25 % of drug and 25 % of excipient groups) is never used in tuning; tuning runs inside the remaining pool D; final test entities come only from E. See `PROJECT_DESIGN.md` §3.1, §11, §12.

## Repository layout
```
configs/default.yaml     every hyper-parameter, split rule, experiment definition
data/raw/                drug_excipient_data.csv (never modified)
data/processed/          generated: pairs.csv, molecules.csv, fingerprints.npz, structure_groups.csv, structure_audit.json,
                         splits/, split_summary.csv, audit.md/json   (the shipped copy already contains them)
src/deint/               data.py features.py structure.py splits.py models.py train.py metrics.py
                         baselines.py plots.py experiments.py utils.py
scripts/                 01_audit_and_clean.py 02_generate_fingerprints.py 03_make_splits.py
                         04_run_experiments.py 05_aggregate_results.py run_all.sh
tests/                   test_pipeline.py (synthetic tests + 1 real PaDEL check)
results/                 generated: <experiment>/<model>__<split>__seed<k>/ and results/summary/
PROJECT_DESIGN.md        full technical study, viva guide, schedule, six-page report plan
```
Added in v3 (analysis only): `scripts/analysis_chem_groups.py` → `data/processed/chem_group_preview.json`.
Planned additions after approval: `src/deint/graphs.py`, `gnn.py`, `tune.py`, `scripts/00_benchmark.py`, `scripts/06_tune.py`, `configs/tuned/`, `chem` unit + entity pools in `structure.py`/`splits.py`.

## Prerequisites and installation
Python ≥ 3.10; **Java (JRE ≥ 8) on PATH** (PaDEL is a Java program; check `java -version`); NVIDIA GPU optional (RTX 3050 recommended), everything also runs on CPU.
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu124    # pick your CUDA build at pytorch.org
python -c "import torch; print(torch.cuda.is_available())"             # True on the laptop
```
Dataset: copy `drug_excipient_data.csv` into `data/raw/` (path configurable in `configs/default.yaml`). The shipped `data/processed/` already holds the audit, fingerprints and structure groups, so stages 1–2 can be skipped.

## Run
```bash
bash scripts/run_all.sh cuda        # or: auto | cpu       (Windows: run the python commands below by hand)
```
Step by step:
```bash
python scripts/01_audit_and_clean.py                 # audit + de-duplicate (seconds)
python scripts/02_generate_fingerprints.py           # PaDEL, 799 molecules (~9 min, cached/resumable)
python scripts/03_make_splits.py                     # structure groups + all splits + leakage assertions (seconds)
python scripts/04_run_experiments.py -e all --device cuda
python scripts/05_aggregate_results.py               # tables + comparison figures
pytest tests -q                                      # optional
```
Individual experiments:
```bash
python scripts/04_run_experiments.py -e E1_reproduction
python scripts/04_run_experiments.py -e E3_generalization --models siamese concat_mlp --splits cold_drug --seeds 0 1
python scripts/04_run_experiments.py -e E2_leakage_audit --device cpu --force     # --force archives old runs
```
Note: the experiments currently runnable are the **v1** definitions in `configs/default.yaml` (`E3`: concat_mlp, siamese[=siamese_fp], rf, entity_prior; `E4`: siamese_sum, concat_mlp_reg).
The v2 programme (graph models, tuning, extended ablations) is specified in `PROJECT_DESIGN.md` §13 and becomes runnable after implementation.

## Outputs
`results/<exp>/<model>__<split>__seed<k>/`: `config.yaml, metrics.json, metrics.txt, training_log.txt, classification_report.txt, training_history.csv, predictions_test.csv` and (seed 0 only) `confusion_matrix.png, roc_curve.png, precision_recall_curve.png, training_curves.png` (600 DPI), `checkpoint_*.pt`.
`results/summary/`: `summary.md, all_runs.csv, E3_E4_summary.csv, paired_differences.csv`, figures. Audits: `data/processed/audit.md|json`, `structure_audit.json`, `split_summary.csv`, `fingerprint_report.json`.

## Reproducibility
Config-driven; seed *k* fixes split and initialisation; splits are cached per (kind, leakage unit, seed) in `data/processed/splits/`; runs are resumable (`metrics.json` present ⇒ skipped; `--force` archives to `results/_archive/`); raw data never modified. Tuning seeds (100, 101) are disjoint from evaluation seeds (0–9); in the planned protocol tuning additionally never touches the evaluation-only entity pool E.

## Testing
`pytest tests -q` runs synthetic tests (cleaning, split leakage + injected-leak detector, symmetry, metrics, checkpoints, end-to-end smoke). The structure-group and grouped-split tests were added in this revision;
in the sandbox, 26 of 27 tests (incl. the end-to-end smoke test) passed after this change; the one real-PaDEL test could not run there because `padelpy` was not installed, so run the full suite once on your machine. The smoke test sets `split.leakage_unit: cid` because it uses synthetic data without structure groups.

## Troubleshooting
* `java: command not found` / PaDEL errors → install a JRE, re-run stage 2 (cache keeps finished chunks).
* `FileNotFoundError: structure_groups.csv` → run `python scripts/03_make_splits.py` (builds it), or set `split.leakage_unit: cid`.
* `RuntimeError: No valid ... split` → lower `split.min_pos_eval`.
* `CUDA out of memory` is not expected for the current models (< 4 M parameters); the planned DE-CrossNet has token-budget batching, chunked attention and an OOM guard specified in `PROJECT_DESIGN.md` §8.2–8.3, §15.1. Use `--device cpu` as fallback.
* `No results found` in stage 5 → run stage 4 first. Windows: `run_all.sh` needs Git-Bash/WSL.
* Stage 3 needs `rdkit` (in `requirements.txt`).

## Known limitations
SMILES (not PubChem SDF) feed PaDEL; fingerprints cannot be verified against PubChem offline; cold splits have few positives (high variance, `cold_both` ≈ 15); labels mix literature incompatibilities with "listed on label ⇒ compatible" assumptions;
10 structure-pairs carry conflicting labels (kept, analysed in E5c); GPU timings are estimates until the E0 benchmark (no GPU was available; no graph model exists yet). Full list: `PROJECT_DESIGN.md` §18.

## Literature
The reference list in `PROJECT_DESIGN.md` §20 contains only works verified against primary/indexing sources (DE-INTERACT, SSI-DDI, SA-DDI, MHCADDI, Lin et al. 2023, D-MPNN, GIN, GINE/Hu et al. 2020, PaDEL). Unverifiable items from earlier drafts were removed and must not be cited.
