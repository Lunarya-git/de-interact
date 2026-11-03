# Drug–Excipient Incompatibility Prediction: DE-INTERACT Reproduction, Leakage Audit, and DE-CrossNet Deep Learning Extension

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![CUDA 12.1+](https://img.shields.io/badge/CUDA-12.1%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)]()

> **Third-Year B.Tech AI & DS Deep Learning Semester Project**  
> Reproduces the Artificial Neural Network of *Patel et al., "DE-INTERACT", Int. J. Pharm. 637 (2023) 122839*, audits its evaluation methodology for **duplicate-pair and chemical-structure leakage**, and proposes **DE-CrossNet**: a role-symmetric hybrid architecture fusing molecular graphs and PubChem fingerprints via bidirectional atom-level cross-attention.

---

## Table of Contents
1. [Project Overview & Core Contributions](#project-overview--core-contributions)
2. [Model Architectures](#model-architectures)
3. [Leakage Prevention & Entity-Pool Methodology](#leakage-prevention--entity-pool-methodology)
4. [Windows Installation & CUDA Setup (NVIDIA RTX 3050)](#windows-installation--cuda-setup-nvidia-rtx-3050)
5. [Dataset Placement & Preprocessing](#dataset-placement--preprocessing)
6. [Step-by-Step Execution Guide](#step-by-step-execution-guide)
7. [RTX 3050 Schedule & 20-Hour Compute Budget](#rtx-3050-schedule--20-hour-compute-budget)
8. [Testing & Verification](#testing--verification)
9. [Artifacts, Logging & Checkpoint Resuming](#artifacts-logging--checkpoint-resuming)
10. [Troubleshooting](#troubleshooting)
11. [Scientific Limitations & Negative Result Interpretation](#scientific-limitations--negative-result-interpretation)

---

## 1. Project Overview & Core Contributions

Patel et al. (2023) introduced DE-INTERACT, an Artificial Neural Network predicting drug–excipient incompatibility from concatenated 881-bit PubChem fingerprints. The paper reported 98.4% accuracy and 98% recall on 3,728 literature-derived interaction pairs (3,549 compatible, 179 incompatible) across 739 unique nodes. However, our dataset audit of the supplied CSV revealed:
1. **Duplicate-Pair Leakage:** The raw supplied dataset contains 10,754 rows with 7,145 exact duplicate rows, collapsing to 3,609 unique CID pairs (3,358 compatible, 251 incompatible, 6.95% positive class) across 799 unique molecules (520 drugs, 283 excipients, 4 dual-role). Evaluating on random row splits leaks identical molecular pairs into both train and test partitions.
2. **Chemical-Structure Leakage:** Molecules often exist under multiple database identifiers (isomers, anomers, salts, hydrates). Random CID-based splits leave 19.6% of test molecules with structural twins in training.
3. **Role Asymmetry:** The original concatenation model treats $(A, B)$ and $(B, A)$ differently, failing physical permutation symmetry.
4. **Lack of Interaction Modeling:** Concatenating fingerprints forces dense layers to learn complex cross-molecular interactions without geometric or atom-level inductive bias.

### Our Contributions
- **Audited Reproduction (E1, E2):** Faithfully reproduced Patel et al.'s ANN and measured the performance drop when duplicate-pair leakage is eliminated.
- **Chemistry-Only Grouping (`chem`):** Implemented strict grouping based on canonical SMILES, stereo-stripped SMILES, and InChIKey connectivity (783 unique chemical groups). Fingerprint collisions are separated, never merged into chemical definitions.
- **Independent Entity-Pool Tuning (Protocol T):** Created a label-blind partition separating development pool D from evaluation pool E (25% held out). Hyperparameter optimization never observes evaluation pool entities.
- **DE-CrossNet Proposed Architecture:** Designed a role-symmetric hybrid deep learning architecture combining shared GINE graph encoders, PubChem fingerprint projections, and bidirectional atom-level cross-attention.
- **GNN Siamese Baseline & Factorial Ablations (E3, E4):** Implemented a graph-only Siamese model and capacity-matched self-attention controls to verify whether performance gains genuinely stem from cross-molecule interaction modeling.

---

## 2. Model Architectures

```
─────────────────────────────────────────────────────────────────────────────
                             DE-CrossNet ARCHITECTURE
─────────────────────────────────────────────────────────────────────────────

   Drug SMILES ────────► Molecular Graph ──► Shared GINE Encoder (θ) ──► H_A [N_A, d_h] ──┐
   Excipient SMILES ───► Molecular Graph ──► Shared GINE Encoder (θ) ──► H_B [N_B, d_h] ──┤
                                                                                          │
   ┌──────────────────────────────────────────────────────────────────────────────────────┘
   │
   ▼
Bidirectional Multi-Head Atom Cross-Attention (Shared Weights, N_x layers):
   H̃_A = LayerNorm( H_A + Softmax( Q_A · K_Bᵀ / √d ) · V_B )  [Masked]
   H̃_B = LayerNorm( H_B + Softmax( Q_B · K_Aᵀ / √d ) · V_A )  [Masked]
   │
   ├─► Attention + Mean Readout:
   │   m_A = Linear( [AttnPool(H̃_A) ‖ MeanPool(H̃_A)] )  ∈ R^{d_h}
   │   m_B = Linear( [AttnPool(H̃_B) ‖ MeanPool(H̃_B)] )  ∈ R^{d_h}
   │
   ├─► Fingerprint Branch:
   │   Drug PubChem FP (881-bit)      ──► Shared FP-MLP ──► f_A ∈ R^{d_fp}
   │   Excipient PubChem FP (881-bit) ──► Shared FP-MLP ──► f_B ∈ R^{d_fp}
   │
   ▼
Combined Molecule Embeddings:
   z_A = [m_A ‖ f_A],   z_B = [m_B ‖ f_B]
   │
   ▼
Symmetric Interaction Head:
   pair = [ z_A + z_B ‖ z_A ⊙ z_B ‖ |z_A − z_B| ]    (Permutation symmetric: f(A,B) ≡ f(B,A))
   │
   ▼
MLP Classification Head ──► Logit ──► Sigmoid ──► P(Incompatible)
─────────────────────────────────────────────────────────────────────────────
```

### Complete Model Suite

| Model Identifier | Architecture Description | Input Modality | Parameters |
| :--- | :--- | :--- | :--- |
| `de_crossnet` | **Proposed:** GINE encoder + Cross-Attention + PubChem FP + Symmetric Head | Molecular Graph + 881-bit FP | 2,054,022 (~2.05 M) |
| `gnn_siamese` | **Graph Baseline:** GINE encoder + Readout + Symmetric Head (No FP, No Cross-Attn) | Molecular Graph | 1,171,718 (~1.17 M) |
| `siamese` (`siamese_fp`) | **FP Baseline:** Shared MLP encoder [512, 256, 128] + Symmetric Head [128, 64] | 881-bit PubChem FP | 681,473 (~0.68 M) |
| `concat_mlp` | **Reproduced MLP:** 5-layer MLP [1024, 1024, 512, 256, 128] with unified BCE loss | Concatenated FP (1762-d) | 3,905,537 (~3.91 M) |
| `concat_mlp_reg` | Same 5-layer MLP with 0.3 Dropout & AdamW weight decay | Concatenated FP (1762-d) | 3,905,537 (~3.91 M) |
| `paper_ann` | Paper-faithful protocol: 3 × 64, Softmax-2, RMSprop, oversampled | Concatenated FP (1762-d) | 121,282 (~121 K) |
| `rf` | Balanced Random Forest (300 estimators) | Concatenated FP (1762-d) | Classical |
| `entity_prior` | Naive Bayes empirical entity incompatibility rate diagnostic | Molecule CID only | Classical |

### Core Ablation Variants (E4)
- `de_crossnet_no_fp`: Graph encoder + cross-attention with the fingerprint branch removed ($d_{\text{fp}} = 0$). Tests whether 881-bit structural keys provide complementary signal to graphs.
- `de_crossnet_self_attn`: Cross-attention blocks replaced with intra-molecular self-attention of identical dimension and head count. **Key capacity-matched control** testing if gains are interaction-driven or just transformer refinement.
- `de_crossnet_ordered_head`: Symmetric interaction terms $[z_A+z_B, z_A \odot z_B, |z_A-z_B|]$ replaced with ordered concatenation $[z_{\text{drug}}, z_{\text{exc}}]$.
- `largest_fragment`: Input molecular graphs stripped of counter-ions and solvates, keeping only the largest heavy-atom component.

---

## 3. Leakage Prevention & Entity-Pool Methodology

To prevent data snooping and false optimism, three strict leakage barriers are enforced:

### A. Chemical Grouping (`chem`)
- Legacy bug corrected: identical PubChem fingerprints are **never** used as a grouping criterion.
- `chem` combines identical SMILES, stereo-stripped SMILES, and InChIKey connectivity into single chemical groups (**783 groups**).
- Fingerprint collisions (16 fingerprints shared by 33 distinct chemical groups) are tracked separately and analyzed post-hoc in E5.
- Cold-start splits hold out entire `chem` groups: no test drug or excipient shares a chemical group with training molecules.

### B. Split Taxonomy
- `paper_rows`: Row-level stratified split (leaky by design; reproduces the paper's protocol).
- `random_pair`: Stratified random split of unique chemical pairs (no duplicate leakage; entity overlap permitted).
- `cold_drug`: Whole drug entities held out (no test drug ever seen in training in any role).
- `cold_excipient`: Whole excipient entities held out.
- `cold_both`: Strict disjoint molecule sets for train, val, and test (only pairs whose both constituents belong to the split partition are retained).

### C. Independent Tuning Pool (Protocol T)
- Dataset is partitioned into Development Pool D (75%) and Evaluation Pool E (25% of drug-only and excipient-only groups; dual-role entities stay in D).
- Hyperparameter tuning and architecture exploration are conducted **exclusively within Pool D**.
- Final test sets in E3, E4, and E5 are drawn strictly from Pool E, guaranteeing that final evaluation molecules were never observed during tuning.

---

## 4. Windows Installation & CUDA Setup (NVIDIA RTX 3050)

### System Requirements
- **OS**: Windows 10/11 64-bit
- **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Driver**: NVIDIA Driver $\ge$ 528.33 (CUDA 12.x support)
- **Java**: JRE $\ge$ 8 installed and on PATH (for PaDEL-Descriptor PubChem fingerprints)
- **Python**: Python 3.11 (Recommended) or Python 3.12 (with NumPy `<2.0.0` C-ABI compatibility)

### Automated 1-Step Setup (Recommended)
We provide an automated setup script that cleanly wipes any stale `.venv` (preventing C-extension version mismatch between Python 3.11 and 3.12), creates a fresh `.venv`, installs PyTorch with CUDA 12.1 acceleration, installs pinned dependencies, and verifies GPU detection:

```powershell
# Open PowerShell in project directory and run:
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

To specify an explicit Python binary (e.g., Python 3.11 or 3.12):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -PythonExe "py -3.11"
```

---

### Manual Setup (Step-by-Step)
If you prefer setting up manually from your terminal:

```powershell
# 1. Remove old virtual environment to eliminate compiled C-ABI conflicts
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }

# 2. Create clean virtual environment (using Python 3.11 or 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and build tools
python -m pip install --upgrade pip setuptools wheel

# 4. Install CUDA-enabled PyTorch (CUDA 12.1 for RTX 3050 Ampere architecture)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Install project requirements (with pinned NumPy <2.0 for RDKit compatibility)
pip install -r requirements.txt

# 6. Verify environment and GPU detection
python -c "import torch, numpy, pandas, sklearn, rdkit, optuna; print('CUDA Ready:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 7. Run test suite
pytest tests -v
```

---

## 5. Dataset Placement & Preprocessing

The raw dataset must be located at `data/raw/drug_excipient_data.csv`. The processed tables, fingerprints, and structure groupings are generated deterministically:

```powershell
# Stage 1: Data Audit & Cleaning
python scripts/01_audit_and_clean.py
# Outputs: data/processed/pairs.csv (3,609 pairs), molecules.csv (799 CIDs), audit.json

# Stage 2: PubChem 881-bit Fingerprint Generation (uses PaDEL with chunk caching)
python scripts/02_generate_fingerprints.py
# Outputs: data/processed/fingerprints.npz, fingerprint_status.csv

# Stage 3: Chemistry-Only Structure Grouping & Entity Pools
python scripts/03_make_splits.py
# Outputs: data/processed/structure_groups.csv (783 chem groups), entity_pools_seed42.json, split_summary.csv
```

---

## 6. Step-by-Step Execution Guide

### Automated All-in-One Execution (Recommended)
Run the complete pipeline with strict error checking, benchmark gating, hyperparameter tuning on Pool D, and automated checkpoint handling:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Device cuda
```

**Key Execution Features:**
- **Strict Error Handling:** Explicitly verifies `$LASTEXITCODE` after every Python invocation and terminates immediately on failure, preventing cascading errors.
- **Benchmark Gate E0:** Benchmarks GPU throughput per model, estimates total programme runtime (including Stage T), and stops for user approval before long runs. Use `-AutoApprove` for automated runs.
- **Integrated Stage T (Protocol T):** Executes 125 Optuna trials across 4 models inside Entity Pool D, freezes winning hyperparameters to `configs/tuned/` and `results/tuned/`, and automatically loads them for E3/E4. (Pass `-SkipTuning` if you wish to run evaluation using default hyperparameters).

---

### Modular Individual Execution

#### Step 1: Run the Comprehensive Test Suite
Execute the 42 test functions / 53 test cases (synthetic data, no GPU needed):
```powershell
pytest tests -v
```

#### Step 2: Stage 0 — GPU Benchmark Gate (E0)
Measure runtime per epoch and peak VRAM allocation on your RTX 3050:
```powershell
python scripts/00_benchmark.py --device cuda
```
*Outputs: `results/E0_benchmark.json` containing measured throughput and updated schedule estimates.*

#### Step 3: Reproduce Paper Protocol (E1) & Leakage Audit (E2)
```powershell
# E1: Paper reproduction (leaky row split, 3 seeds)
python scripts/04_run_experiments.py -e E1_reproduction --device cuda

# E2: Leakage audit (duplicate-free random_pair split, 3 seeds)
python scripts/04_run_experiments.py -e E2_leakage_audit --device cuda
```

#### Step 4: Hyperparameter Tuning on Entity Pool D (Protocol T)
Tune the learning rate, weight decay, dropout, and GNN depth within Pool D:
```powershell
# Tune DE-CrossNet (budget: 40 trials)
python scripts/06_tune.py --model de_crossnet --trials 40 --device cuda

# Or tune all models in sequence
python scripts/06_tune.py --all-models --device cuda
```
*Outputs: Winning parameter files saved to `results/tuned/<model>.yaml`.*

#### Step 5: Stage E3 — Main Model Comparison
Run all models across all 4 split types across 5 seeds:
```powershell
python scripts/04_run_experiments.py -e E3_generalization --device cuda
```

#### Step 6: Stage E4 — Prioritized Component Ablations
Evaluate `de_crossnet_no_fp`, `de_crossnet_self_attn`, `concat_mlp_reg`, and `siamese_sum`:
```powershell
python scripts/04_run_experiments.py -e E4_ablation --device cuda
```

#### Step 7: Stage E5 — Sensitivity & Fingerprint Collision Analysis
Analyze the impact of fingerprint collisions, label conflicts, and leakage units:
```powershell
python scripts/07_sensitivity.py --device cuda
```

#### Step 8: Stage E6 — Cross-Attention Case Studies
Generate atom-level cross-attention heatmaps for validated drug–excipient pairs:
```powershell
python scripts/06_attention.py --checkpoint results/E3_generalization/de_crossnet__cold_drug__seed0/checkpoint_val_pr_auc.pt --device cuda
```
*Outputs: Attention maps saved to `results/E6_attention/<drug_cid>_<exc_cid>_attn.png`.*

#### Step 9: Stage 5 — Aggregate Results & Generate Publication Figures
Compile summary tables, paired differences, and publication-ready 600 DPI figures:
```powershell
python scripts/05_aggregate_results.py
```
*Outputs: `results/summary/summary.md`, `all_runs.csv`, `fig_E3_E4_pr_auc.png`, `fig_E1_vs_E2_leakage.png`.*

---

## 7. RTX 3050 Schedule & 20-Hour Compute Budget

On an NVIDIA RTX 3050 (4 GB VRAM), the total runtime budget is **20 hours**. With early stopping (patience 15), mixed-precision AMP, and token-budget batching (batch size 32), the planned execution takes **~11.3 hours**, leaving a 43% safety margin.

| Stage | Experiment Scope | Total Runs / Trainings | Estimated GPU Time | VRAM Peak |
| :--- | :--- | :--- | :--- | :--- |
| **E0** | Benchmark & throughput gate | 12 short benchmark runs | 0.2 hours | ~1.8 GB |
| **E1 / E2** | Paper reproduction & leakage audit | 6 runs (3 seeds each) | 0.4 hours | ~0.8 GB |
| **T** | Tuning on Pool D (125 trials) | ~430 early-stopped runs | 3.6 hours | ~2.4 GB |
| **E3** | Main model comparison (4 splits × 5 seeds) | 120 runs (GPU + CPU) | 3.2 hours | ~2.4 GB |
| **E4** | Prioritized component ablations (3 splits × 5 seeds) | 60 runs | 2.5 hours | ~2.2 GB |
| **E5** | Sensitivity & post-hoc collision audit | Post-hoc + 12 sensitivity runs | 0.8 hours | ~1.5 GB |
| **E6** | Attention extraction & case studies | 6 inference passes | 0.1 hours | ~1.2 GB |
| **Margin** | 20% Reserve for failures / reruns | - | 4.0 hours | - |
| **Total** | **Full Project Programme** | **~630 trainings** | **15.8 / 20.0 h** | **< 2.5 GB** |

### Memory & Stability Safeguards
- **Mixed Precision:** Uses `torch.cuda.amp.autocast(dtype=torch.float16)` for cross-attention and GINE layers.
- **VRAM Cap:** Peak allocated memory stays below 2.5 GB (safe for 4 GB VRAM laptops).
- **OOM Auto-Recovery:** `gnn_train.py` intercepts CUDA OOM exceptions, empties GPU cache (`torch.cuda.empty_cache()`), and logs warnings without crashing.
- **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` prevents gradient explosion in attention layers.

---

## 8. Testing & Verification

The test suite comprises 42 test functions across `tests/test_pipeline.py`, yielding 53 collected test items when parametrized across splits and architectures. All tests execute on self-contained synthetic data without requiring network access or external Java installations.

Run tests:
```powershell
pytest tests -v
```

Execution Status:
- **52 PASSED, 1 SKIPPED, 0 FAILED** (completed in ~53s on CPU).
- The 1 skipped test (`test_padel_pubchem_fingerprint`) gracefully skips if `padelpy` or Java is absent from the runtime environment.

### Verified Test Areas
1. **Data Cleaning:** Deterministic deduplication, label mapping (`Compatible` $\to$ 0, `Imcompatible` $\to$ 1), conflicting label detection.
2. **Leakage Invariance:** Injected leak detector catches pair and entity leakage; verifies 0% entity overlap in cold splits.
3. **Chemistry Grouping:** Verifies stereoisomers merge into identical `chem_group`; asserts fingerprint collisions do **not** merge distinct chemical structures.
4. **Entity Pool Independence:** Asserts that development pool D and evaluation pool E have 0 overlapping molecules and 0 overlapping chemical groups.
5. **Stage T Tuning Protocol:** Verifies hyperparameter sampling, multi-split validation on Pool D, and winning config freezing to `configs/tuned/` and `results/tuned/`.
6. **Model Permutation Symmetry:** Proves $f(A, B) \equiv f(B, A)$ within $10^{-6}$ numerical tolerance for `de_crossnet`, `gnn_siamese`, and `siamese_fp` via shared `LayerNorm` and symmetric readout pooling; confirms failure for asymmetric controls (`concat_mlp`, `de_crossnet_ordered_head`).
7. **Padding Invariance:** Injected noise into padding atoms does not alter model predictions.
8. **End-to-End Smoke Tests:** Simulates complete training, checkpoint saving, metric logging, resuming, and result aggregation on synthetic data.

---

## 9. Artifacts, Logging & Checkpoint Resuming

Every execution produces self-contained, auditable directories:

```
results/
├── E1_reproduction/
│   └── paper_ann__paper_rows__seed0/
│       ├── config.yaml               # Exact run parameters
│       ├── metrics.json              # Full numerical metrics
│       ├── metrics.txt               # Formatted metric tables
│       ├── training_log.txt          # Per-epoch loss and validation metrics
│       ├── classification_report.txt # Scikit-learn classification report
│       ├── training_history.csv      # Epoch-by-epoch trajectory
│       ├── predictions_test.csv      # Test pair predictions and probabilities
│       ├── checkpoint_val_pr_auc.pt  # PyTorch model state dict
│       ├── confusion_matrix.png      # 600 DPI confusion matrix
│       ├── roc_curve.png             # ROC curve with ROC-AUC
│       └── precision_recall_curve.png# PR curve with PR-AUC
├── summary/
│   ├── summary.md                    # Auto-generated markdown report
│   ├── all_runs.csv                  # Aggregated tabular results
│   ├── E3_E4_summary.csv             # Mean ± std across seeds
│   ├── paired_differences.csv        # Seed-matched bootstrap differences
│   └── fig_E3_E4_pr_auc.png          # Grouped publication bar chart
└── _archive/                         # Archived prior runs (when using --force)
```

### Resuming & Safe Re-runs
- Completed runs containing `metrics.json` are **automatically skipped**.
- Re-running commands will resume from where they left off without redundant compute.
- Passing `--force` archives existing results to `results/_archive/<run_id>/` before re-running.

---

## 10. Troubleshooting

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| `java: command not found` | JRE not found on system PATH | Install OpenJDK 17 or JRE 8+, add `bin` to Windows System PATH. |
| `CUDA out of memory` | Batch size too large for 4 GB VRAM | Reduce `batch_size: 16` in `configs/default.yaml` under `gnn_protocol`. |
| `FileNotFoundError: structure_groups.csv` | Preprocessing stage 3 skipped | Run `python scripts/03_make_splits.py`. |
| `RuntimeError: No valid split found` | `min_pos_eval` too strict for split seed | Decrease `min_pos_eval: 6` in `configs/default.yaml`. |
| `Permission denied` on venv activation | PowerShell execution policy restricted | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. |

---

## 11. Scientific Limitations & Negative Result Interpretation

In accordance with rigorous academic integrity:
1. **Absence of Evidence $\neq$ Compatibility:** The dataset derives negative labels ("Compatible") largely from co-formulation in approved drug products. Unreported incompatibilities may exist.
2. **Imbalance & Positive Count:** Only 250 positive (incompatible) pairs exist in the cleaned dataset. Cold splits yield ~15–40 test positives, necessitating multi-seed reporting and confidence intervals.
3. **Negative Result Interpretation:** If DE-CrossNet does not statistically outperform Siamese-FP or Random Forest under cold splits, this is **not** an experimental failure: it establishes that global functional group presence (fingerprints) or identity priors dominate drug–excipient incompatibility at current dataset sizes, and that localized atom-level cross-attention requires larger corpora to generalize.

---
*Developed for B.Tech AI & Data Science Deep Learning Project, 2026.*
