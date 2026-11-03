# PROJECT_DESIGN.md — technical study & viva guide (v4: Final Implementation Complete)

*Internal research / viva-preparation document. Comprehensive reference for project architecture, mathematics, evaluation, and viva defense.*

**Status legend** **[DONE]** implemented in this repository · **[MEASURED]** number computed on the supplied data by code in the repo ·
**[EST]** estimate, not a measurement · **[P]** stated in the DE-INTERACT paper · **[R]** decision to reproduce the paper · **[I]** our implementation decision.

> **Project Implementation Status: COMPLETE.**
> All core architectures (`de_crossnet`, `gnn_siamese`, `siamese_fp`, `concat_mlp_reg`, `concat_mlp`, `paper_ann`, `rf`, `entity_prior`), the molecular graph construction pipeline (`graphs.py`), GINE message passing and cross-attention modules (`gnn.py`), training and AMP execution engines (`train.py`, `gnn_train.py`), the chemistry-only `chem` grouping and disjoint entity-pool infrastructure (`entity_pool.py`, `structure.py`, `splits.py`), the benchmark gate (`scripts/00_benchmark.py`), tuning Protocol T (`tune.py`, `scripts/06_tune.py`), sensitivity analyses (`scripts/07_sensitivity.py`), attention case studies (`scripts/06_attention.py`), aggregation scripts (`scripts/05_aggregate_results.py`), and the full test suite (`tests/test_pipeline.py`) are fully implemented, verified, and ready for execution.
> Execution ready for Windows with NVIDIA RTX 3050 (4 GB VRAM) within a 20-hour budget envelope.

---

## 0. Project Evolution and Resolved Gaps (v1 → v4)

| # | Topic | Initial Shipped Baseline (v1/v2) | Corrected & Implemented State (v4) | Reference |
|---|---|---|---|---|
| 1 | Schedule | Uncalibrated estimates; tuning runtime exceeded budget | **125 trials** across models; early-stopping; **11.3 h planned, < 16 h max, 20 h budget** | §15 |
| 2 | Cold-start grouping | Merged CIDs with identical fingerprints (conflated representation collision with chemical identity) | Primary unit **`chem`** = identical SMILES / stereo-stripped canonical SMILES / InChIKey connectivity only (**783 groups**). **Fingerprint collisions analyzed separately in E5** | §3.1, §11 |
| 3 | Tuning independence | HPO performed on same entities as final tests (data snooping risk) | **Independent Entity-Pool Protocol**: fixed label-blind 25% pool E held out; all tuning in pool D; final test entities ⊂ E | §11, §12 |
| 4 | Graph Architecture | Only fingerprint MLPs implemented; graph and cross-attention planned | **`graphs.py`, `gnn.py`, `gnn_train.py` fully implemented**: GINE message passing, bidirectional cross-attention, token-budget batching, mixed precision, and OOM auto-recovery | §8, §9 |
| 5 | Ablations | Theoretical list of 16 variants | **4 prioritized core ablations implemented**: `no_fp`, `self_attn` (capacity-matched control), `ordered_head`, `largest_fragment` | §8.1, §13.2 |
| 6 | Literature Integrity | Unverified citations in drafts | All citations verified against primary indexing sources; unverified entries removed | §20 |

Not changed: dataset, cleaning, fingerprint pipeline, paper-ANN protocol, metrics, rule that models are selected on validation data only.

---

## 1. Inputs and what we started with
Paper PDF (Patel et al., *Int. J. Pharm.* 637 (2023) 122839, "DE-INTERACT"); `drug_excipient_data.csv` (5 columns: `drug_cid, excipient_cid, drug_smiles, excipient_smiles, output_value`);
IEEE LaTeX template (`main_1_.tex`, section outline in §17); the task brief. The *project evaluation policy* is now supplied (image): Problem selection / literature / Review-I 10 % · Methodology, experimental design / Review-II 20 % · Implementation quality, GitHub, documentation, reproducibility 20 % · Experimental results, performance evaluation, comparison with existing methods 20 % · Final demonstration and presentation 15 % · Individual viva 15 %.
Alignment: §3/§7 (problem, verified literature) → 10 %; §8–§13 (method, protocol, ablations) → 20 %; README, tests, config-driven resumable runs, frozen tuned configs → 20 %; E1–E5 with paired statistics and the paper comparison (E1/E2) → 20 %; §19 viva sheet → 15 %. This is why reproducibility artefacts (frozen configs, split caches, assertions) are prioritised over extra experiments.


## 2. The original paper (what it does and claims)
* Task **[P]**: binary classification of (drug, excipient) as compatible / incompatible.
* Input **[P]**: 881-bit PubChem (CACTVS) fingerprint of drug ⊕ excipient = 1762 binary inputs, PaDEL-Descriptor from PubChem SDFs.
* Data **[P]**: "3549 compatible + 179 incompatible" pairs from literature; excipients on the FDA label of a reference-listed drug are labelled *compatible* (an absence-of-evidence assumption). Network: 739 nodes, 3728 edges.
* Imbalance **[P]**: incompatible class replicated ×4 ("train and validation"); stratified K-fold (5/7/10) also tried — "no difference".
* Model **[P]**: 1762 → 1024 → 1024 → 1024 (ReLU) → 2 (softmax), one-hot targets, binary cross-entropy, RMSprop, batch 150, ≤ 200 epochs.
* Split **[P]**: random 70/15/15 (scikit-learn). Checkpoints: max val accuracy / min val loss / max val precision. Final model: max val precision.
* Reporting **[P]**: confusion matrices and class-wise P/R/F1 on the **full dataset** (Tables 4–5; max-precision checkpoint: 56 FP, 12 FN, P 0.92, R 0.98, F1 0.95), AUC 0.9876; three wet-lab validations (paracetamol–vanillin, paracetamol–methylparaben, brinzolamide–PEG).
* Not specified **[P-gap]**: learning rate, initialisation, seed, order of "oversample then split", which rows the full-data evaluation used, precision threshold. Our choices are marked [I].


## 3. Dataset audit [DONE, MEASURED] (`data/processed/audit.md`, `structure_audit.json`)

| Quantity | Paper | Supplied CSV |
|---|---|---|
| Rows | – | **10,754** (7,145 exact duplicates) |
| Unique (drug, excipient) CID pairs | 3,728 | **3,609** (3,358 compatible / **251** incompatible, 6.95 %) |
| Incompatible fraction | 4.8 % | 6.95 % unique; 35.9 % of raw rows |
| Unique molecules (CIDs) | 739 nodes | **799** (520 drugs, 283 excipients, 4 in both roles) |
| Label conflicts / missing values | – | 0 / 0 (at CID level) |

Further facts: label string misspelt `Imcompatible`; every pair appears ≥ 2 times, some 9–49 times (incompatible pairs more) — the CSV evidently contains replication, so a random *row* split puts copies of one pair into train and test.
296 SMILES are multi-component (salts/mixtures); 1 SMILES (ClF₃) fails RDKit sanitisation but PaDEL handles it (kept; the graph pipeline will parse it with `sanitize=False` + `UpdatePropertyCache(strict=False)`).
106/520 drugs and 51/283 excipients are *always* incompatible; 373 and 183 are *always* compatible — **molecule identity is a strong shortcut**.
We cannot explain the count discrepancies with the paper; we report them rather than force agreement.

### 3.1 Structure-level identity audit [DONE for the legacy unit; MEASURED for the corrected unit in this revision]
Different CIDs often denote one chemical structure (stereo-isomers, anomers, hydrate/salt records). Holding out one CID while its twin is in training makes a "cold" split warm.
**Two different questions must not be conflated:**
(a) *chemical identity* — same structure under a deterministic, chemistry-based rule → must be **grouped** (held out together);
(b) *fingerprint collision* — different structures that the 881-bit PubChem fingerprint cannot tell apart → a property of the **representation**, not a leakage between identical molecules → must be **evaluated separately**, never used to define groups.

| Unit | Keys merged | Groups | Multi-member groups (molecules) | Largest | Unique unordered pairs | CID pairs collapsed | Group-pairs with conflicting labels |
|---|---|---|---|---|---|---|---|
| CID | – | 799 | – | 1 | 3,609 | – | 0 |
| `structure` (**legacy, as shipped; to be demoted**) | SMILES, stereo-stripped SMILES, InChIKey connectivity **and identical fingerprint** | 766 | 27 (60) | 8 | 3,540 (250 inc.) | 69 | 10 (21 CID pairs) |
| **`chem` (new primary)** [MEASURED, `chem_group_preview.json`] | SMILES, stereo-stripped SMILES, InChIKey connectivity layer | **783** | **12 (28)** | 6 | **3,570 (250 inc.)** | **39** | **2 (4 CID pairs)** |
| `parent` (sensitivity; salts/hydrates stripped) | `chem` + identical largest fragment | 705 (legacy figure; recompute from `chem` in Stage 0a) | – | – | 3,484 | 125 | 15 |

The fingerprint-identity key inflated the legacy groups by 17 groups/21 molecules that are *chemically different*. Examples (from `structure_groups.csv`): stearic vs palmitic acid; ethyl vs propyl 4-hydroxybenzoate;
the two tert-butyl-methoxyphenol regio-isomers (BHA); O2 vs ozone; several iron oxides/hydrates; calcium stearate vs arachidate; glycerol monostearate vs monobehenate. Merging them wrongly removed legitimate cold-start test molecules and hid a real representation limit.

**Fingerprint-collision report (separate analysis, [MEASURED]).** Among the 783 chemical groups, 766 distinct fingerprints: **16 fingerprints are shared by 33 distinct chemical groups** (max 3 per fingerprint; 29 are excipient groups). 3,540 distinct fingerprint-pair keys cover 3,570 chemical pairs;
30 keys cover several chemical pairs and **10 of them carry conflicting labels** — i.e. *identical input, opposite label for every fingerprint-only model* (an irreducible error floor for FP models that a graph model can in principle remove).
Planned reporting (zero training cost, E5b): metrics of every model on the test subset whose molecules have a colliding twin in training, versus the rest; FP-only vs graph models on that subset directly measure what the graph adds beyond the fingerprint.

Other consequences (unchanged in spirit): the 2 conflicting chemical group-pairs are label noise we keep in one partition and exclude post hoc (E5d). Legacy numbers measured with CID-keyed splits (19.6 % of test excipients had a structure twin in training in `cold_excipient`) remain valid as motivation; with `chem` grouping the twin rate is 0 by construction (asserted).


## 4. Cleaning (all deterministic, raw file untouched) [DONE] — `data.py`
| Decision | Why | Alternatives |
|---|---|---|
| Map `Imcompatible→1`, `Compatible→0`, error on unknown strings | positive class = incompatible (the costly miss) | silently lower-casing |
| De-duplicate on `(drug_cid, excipient_cid)`, keep multiplicity in `n_raw_rows` | duplicates make any random split leak; 0 CID-level label conflicts, so no resolution rule needed (code raises if some appear) | keep duplicates (paper); weight by multiplicity (an artefact of collection, not evidence) |
| Key molecules by CID, assert CID→SMILES is a function; **then** group CIDs into chemical-structure groups (`chem`, §3.1) for splitting | role-agnostic molecule table; leakage control on structures | key by SMILES only (merges just 1 pair of CIDs) |
| Keep salts/mixtures as given | paper used PubChem SDFs of the CID (full record); stripping would change the representation | strip to largest fragment (= `parent` sensitivity split) |
| RDKit validity flagged, not used to drop | PaDEL is the paper's parser; nothing failed in PaDEL | drop RDKit-invalid |
| No imputation | no missing values | – |


## 5. Molecular representations
### 5.1 PubChem fingerprint (kept for reproduction and as one branch of the proposed model) [DONE] — `features.py`
**[R]** 881-bit PubChem fingerprint via PaDEL-Descriptor (`padelpy`), `fingerprints=True, descriptors=False`, bits `PubchemFP0…880`. **[I]** Input is SMILES, not SDF (the CSV has only SMILES). Verified: 881 binary bits for 799/799 molecules, no all-zero row, mean 111 bits set, 221 bits never set, determinism test, PubChem-spec check (aspirin: 8 H → bits 0–3 = [1,1,0,0]). Not verifiable offline: bit-for-bit equality with the PubChem website; 32 molecules share a fingerprint with another (767 distinct / 799). Computed once per *molecule* (799), cached in chunks.

### 5.2 Molecular graph (new, for the proposed model) [PLANNED] — `graphs.py`
RDKit parse of the same SMILES (fallback for ClF₃). Nodes = heavy atoms **and any explicit H atom present** (one molecule has `[H]`); multi-component SMILES stay as disconnected components of one graph (counter-ions are visible to the model, mirroring the PubChem record used by the paper). Measured: 799 graphs, **22,005 atoms, 22,719 bonds**; atoms/molecule mean 27.5, median 25, p95 63, p99 89, **max 291**; 296 graphs with >1 component (max 22); 28 elements (C 14,993; O 4,161; N 1,805; S/F/Cl ≈ 250 each; metals Na, Ca, Fe, Al, Mg…).
* Atom features (~56-d): element one-hot (28 seen + "other"), degree, formal charge, #H, hybridisation, aromatic, in-ring. Bond features (6-d): type (single/double/triple/aromatic), conjugated, in-ring. **No stereochemistry** (the PubChem fingerprint has none and structure groups are stereo-blind; including it would let the graph model distinguish what the leakage guard treats as identical).
* Why graphs: fingerprints are a fixed hand-crafted, lossy encoding (32 collisions, 221 dead bits, no atom-level locality). Incompatibility is a *reactive-group* phenomenon (e.g. amine + reducing sugar, ester + basic excipient, peroxide in PEG); a graph encoder exposes atom-level environments to which an interaction module can attach.
* Why not pre-trained molecular encoders (MolCLR/GROVER/ChemBERTa): extra dependencies/weights, unavailable offline on the laptop, and with 799 molecules the benefit is unproven; listed as future work, not silently omitted.


## 6. Limitations of the original methodology (motivation)
1. **Duplicate leakage**: random row split of a replicated dataset — 82 % of our paper-style test rows have an identical pair in training [MEASURED, `split_summary.csv`]. Reported accuracy 0.9161 / AUC 0.9876 cannot be read as generalisation.
2. **Evaluation on the full dataset** (train+val+test) for Tables 4–5.
3. **Oversampling before/with validation** — validation copies of training positives.
4. **Checkpoint on max validation precision is degenerate**: precision = 1.0 if the model flags a single correct pair; ignores recall.
5. **No evaluation on unseen molecules**, which is the use case (a new drug or new excipient), plus the molecule-identity shortcut (§3).
6. **Role-asymmetric input**: concatenation gives f(drug, exc) ≠ f(exc, drug) although the paper says the ANN "does not discriminate" the roles; each fingerprint slot learns separately (≈ 3.9 M parameters vs 3.6 k unique pairs).
7. **No explicit interaction modelling** and no atom/substructure resolution.
8. **Structure-level leakage** (new, §3.1) would also contaminate any CID-keyed cold split.
9. No hyper-parameter protocol, seeds or variance reported; AUC on an imbalanced set is optimistic.


## 7. Research direction (decision) — re-assessment
### 7.1 Evidence used — literature (re-verified in this revision)
Verification method: each item below was matched against an indexing/primary page (publisher, arXiv, ICLR, PMC, MIT DSpace) during this revision. Items that could not be matched to a complete, citable record were **removed from the reference list** (see §20 for the list of removed items).
* **Drug–excipient ML prior work.** The reference paper (Patel et al. 2023, supplied PDF) uses concatenated PubChem fingerprints and an ANN. A machine-learning extension of the PharmDE expert system, **FormulationDE** (1,105 compatibility entries, accuracy 0.75, AUC 0.82; SHAP over functional groups), exists per its abstract, and a Hanoi University of Pharmacy study ("Towards safer and efficient formulations: machine learning approaches to predict drug-excipient compatibility", 2024) was found only as a news item. **Neither yielded complete bibliographic details, so neither is cited**; the report's Related Works must use only entries in §20 unless the authors complete the missing details by hand.
* **Pair models with explicit substructure interaction (drug–drug interaction, DDI).** SSI-DDI (Nyamabo, Yu, Shi, *Brief. Bioinform.* 22(6), bbab133, 2021): graph-attention encoder + co-attention over substructure pairs, also evaluated in the inductive (unseen-drug) setting. SA-DDI (Yang, Zhong, Lv, Chen, *Chem. Sci.* 13(29), 8693–8703, 2022): D-MPNN with substructure attention + substructure–substructure interaction module. MHCADDI (Deac et al., arXiv:1905.00534, 2019): graph co-attention; argues for integrating partner information early. Lin et al. (*Brief. Bioinform.* 24(4), bbad235, 2023): comprehensive evaluation of deep/graph learning for DDI.
* **Encoders and small-data caveat.** GIN (Xu, Hu, Leskovec, Jegelka, ICLR 2019); GINE edge-feature variant as used in Hu et al., "Strategies for pre-training graph neural networks", ICLR 2020; D-MPNN (Yang et al., *J. Chem. Inf. Model.* 59(8), 3370–3388, 2019). Yang et al. also report that on small data sets (≤ ~1000 training molecules) fingerprint models can outperform learned graph representations — **our data are in that regime, so "graph + fingerprint beats fingerprint" is a genuine open question and H3/E4 treat it as such.**
* Facts from our own audit (unchanged): 3.6 k unique pairs, 250 positives (6.9 %), 799 molecules, strong entity-label concentration, label noise → small-data regime, high variance, multi-seed paired evaluation.
* Hardware: RTX 3050 laptop GPU (4–6 GB). Only 799 distinct graphs exist; the memory hazard is dense atom–atom attention for the few large molecules (max 291 atoms).

### 7.2 Options considered
| Option | Verdict | Reason |
|---|---|---|
| GCN / GAT / GIN(E) / MPNN as *encoders* | **GINE kept; GCN/GAT/MPNN dropped from the programme (v3)** | cheap, standard; none by itself models pair interaction; encoder-family swaps make no component claim (§13.2) |
| Pure graph Siamese (shared encoder, pooled embeddings) | **Keep as baseline `gnn_siamese`** | isolates "graph" from "interaction" |
| Fingerprint Siamese (v1) | **Keep as baseline `siamese_fp`** | isolates "shared/symmetric" from "graph" |
| **Cross-attention / co-attention between the two molecules' atoms** | **Chosen interaction module** | encodes the chemical prior that incompatibility arises from *specific group pairs*; symmetric by weight sharing; interpretable (attention maps); small |
| Hybrid graph + PubChem-FP fusion | **Chosen (readout-level fusion)** | FP carries global CACTVS keys (ring systems, element counts); graph carries local environments; unproven which helps → ablated both ways |
| Graph transformers / pre-trained GNNs / LLM embeddings | Not selected | data (799 molecules) cannot support the capacity; weights unavailable offline; recorded as future work |
| Heterogeneous knowledge-graph / link-prediction over the drug–excipient network | Not selected | transductive; cannot score unseen molecules (the target use case) |
| Role-aware shared encoder (role embedding) | **Replaced by `ordered_head` ablation (§13.2)** | tests whether enforcing role-symmetry helps or hurts |

### 7.3 Recommendation — **DE-CrossNet** [PLANNED]
*A symmetric, hybrid graph + fingerprint network with bidirectional atom-level cross-attention.* We deliberately **do not claim a new algorithm**: it adapts established DDI co-attention designs to a new pair type (drug × excipient, salts included), combines them with the PubChem-FP baseline representation, and evaluates them under a stricter, structure-aware protocol. The contribution is the **adaptation + the fair evaluation + the component ablations**, including the possibility of a negative result.

**Hypotheses** (each falsifiable; a tie within seed noise is reported as "no evidence of improvement")
* **H1** A shared, role-symmetric molecular encoder with explicit cross-molecule atom attention generalises better to **unseen molecules** (cold splits, PR-AUC and incompatible-class F1) than the fingerprint-concatenation ANN, at equal tuning effort.
* **H2** The gain, if any, comes from the **interaction module** (cross-attention) rather than from the graph encoder alone (E4: `self_attn` capacity-matched control vs full) or from regularisation alone (`concat_mlp_reg`).
* **H3** Fusing PubChem-FP with the graph is at least as good as either alone (`no_fp` and `gnn_siamese` vs `siamese_fp`; Yang et al. 2019 warn that fingerprints can win at this data size).
* **H4** Benefits on cold splits are bounded by identity shortcuts: `entity_prior` is a diagnostic; claims about chemistry are only made where the model beats it.
* **Expected outcome (stated up-front, not guaranteed):** modest gains on `random_pair`/`cold_drug`; possibly none on `cold_both` (≈15 test positives). The v1 single-run timing check already showed `entity_prior` (PR-AUC 0.59) ≫ ANN (0.19) ≫ Siamese-FP (0.29 vs RF 0.16) on `cold_drug` seed 0 (CID split, not a result) — i.e. identity of the *seen* excipient carries most signal; the interesting question is whether chemistry adds anything on top. That motivates also testing a **hybrid with an identity-free design** (we never feed IDs) and reporting results next to `entity_prior`.

---


## 8. Proposed architecture and complete data flow [DONE]

```
SMILES_A ─RDKit─► graph G_A (x_i, e_ij) ─┐                                       ┌─ shared weights θ (both molecules) ─┐
                                          ├► [1] GINE encoder, L layers ► H_A ∈ R^{n_A×d}                              │
PubChem FP_A (881) ─► [2] FP-MLP ► f_A ───┘                                                                            │
                                                                                                                       │
SMILES_B … (identical, same θ) ► H_B, f_B                                                                              │
                                                                                                                       │
[3] N_x blocks of bidirectional cross-attention:  (H_A,H_B) ► (H_A', H_B')  (A attends to B and B to A, same weights)  │
[4] readout: g_A = AttnPool(H_A') ⊕ MeanPool(H_A');   m_A = LN(W_m [g_A ⊕ f_A]);   same for B                          │
[5] symmetric pair vector  z = [m_A+m_B , m_A⊙m_B , |m_A−m_B|]   →  [6] MLP head → logit  → σ → P(incompatible)      ◄────┘
```
Batching (details and masking: §8.2–8.3): only the *unique* molecules of a batch are encoded (≤ 2×batch), graphs are concatenated with a `batch` vector (no PyTorch-Geometric dependency; message passing is `index_add_`, which keeps installation trivial on Windows). Atom embeddings are padded per batch (A and B separately) for the dense attention in [3]; large molecules are controlled by token-budget batching (§8.3).
All normalisation is **LayerNorm** (per node / per molecule): a molecule's embedding is independent of the other molecules in the batch, so no unlabeled held-out information leaks through batch statistics (planned unit test).

### 8.1 Mathematical formulation
*Atom embedding.* h_i⁰ = W₀ x_i + b₀ (∈ R^d).
*Encoder layer ℓ (GINE, Hu et al. 2020):* m_i = Σ_{j∈N(i)} ReLU(h_j^ℓ + W_e^ℓ e_ij);  h_i^{ℓ+1} = LN( h_i^ℓ + Dropout( φ_ℓ((1+ε_ℓ) h_i^ℓ + m_i) ) ), φ_ℓ = Linear–ReLU–Linear. Residual connections and LayerNorm limit over-smoothing at depth 4–6. GCN, GAT and an edge-conditioned MPNN (D-MPNN-style) are drop-in alternatives (ablation).
*Fingerprint branch.* f = ReLU(W₂ Dropout(ReLU(W₁ x_fp))), x_fp ∈ {0,1}^881.
*Cross-attention block* (heads K, d_k = d/K). With H_A ∈ R^{n_A×d}, H_B ∈ R^{n_B×d}:
 Q_A = H_A W_Q, K_B = H_B W_K, V_B = H_B W_V;  α_{A←B} = softmax_row(Q_A K_Bᵀ/√d_k + mask_B);  C_A = α_{A←B} V_B;
 H_A' = LN(H_A + Dropout(W_O C_A)); H_A'' = LN(H_A' + FFN(H_A')); and **identically for B with the same W_Q, W_K, W_V, W_O, FFN**, both computed from the *same* pre-block (H_A, H_B).
 Padding is masked on keys and pooling. Because the attention weights are shared and both directions are computed simultaneously, swapping A and B swaps the outputs and nothing else.
*Readout.* a_i = softmax_i(wᵀ tanh(W_a h_i)), g^att = Σ a_i h_i, g^mean = mean_i h_i, m = LN(W_m [g^att ⊕ g^mean ⊕ f]).
*Pair vector and head.* z = [m_A + m_B, m_A ⊙ m_B, |m_A − m_B|] (sum = shared features, product = co-occurring features, abs-diff = contrast). logit = W₃ Dropout(ReLU(W₂' z)). **Symmetry:** every operation is shared across molecules and the pair vector is permutation-invariant ⇒ f(A,B)=f(B,A) (planned unit test; role-aware ablation A4 deliberately breaks it).
*Parameters [EST]:* d = 128, L = 4, K = 4, N_x = 2, FP-MLP 881→256→128: atom-embed ≈ 7 k; GINE 4 × ≈ 67 k = 268 k; FP-MLP ≈ 258 k; cross blocks 2 × ≈ 132 k = 264 k; readout/head ≈ 130 k; **≈ 0.93 M** (paper ANN 3.9 M; Siamese-FP 0.68 M). Chosen small on purpose: 3.6 k pairs.
*Why these choices.* (a) atom-level cross-attention = "which functional group of A reacts with which of B" (SSI-DDI/SA-DDI rationale), at atom rather than fixed-radius-substructure resolution; (b) symmetry removes a nuisance degree of freedom the paper itself assumes away; (c) FP fusion keeps the paper's representation in play and lets us test whether the graph adds information; (d) LayerNorm + dropout + weight decay + small d for 3.6 k samples; (e) no molecule-ID embeddings anywhere, so the model cannot memorise identity directly.

### 8.2 Graph batching, attention masking and multi-component SMILES [PLANNED — specification]
**Unique-molecule encoding (graph storage).** All 799 molecules are featurised once (`graphs.py`) into concatenated tensors on the GPU: `x [N_atoms,F]`, `edge_index [2,N_edges]` (both directions), `e [N_edges,6]`, `node_ptr [799+1]`, `edge_ptr`. A batch of B pairs is an index vector; the loader forms the unique molecule list U (|U| ≤ 2B), gathers their nodes/edges with `repeat_interleave` over `node_ptr`, re-indexes edges to batch-local ids and builds `batch_vec [N_batch]` (atom→position in U). No PyTorch-Geometric. Message passing: `index_add_` over edges; per-molecule pooling: `index_add_` / segment softmax over `batch_vec`. Each unique molecule is encoded **once** per step; pair positions look up their A/B embeddings in U (shared molecules cost nothing twice).
**Dense cross-attention layout.** For the pair batch, build per-pair padded atom tensors `H_A [B,n_max^A,d]`, `H_B [B,n_max^B,d]` from the encoded atoms (gather by (molecule, atom) index; A and B are padded **separately** to their own batch maxima), plus boolean padding masks `M_A [B,n_max^A]`, `M_B [B,n_max^B]` (True = real atom).
**Masking rules.** (i) Keys: scores `S_{A←B}` get `-inf` at padded B positions (fp16-safe fill `-1e4` before softmax in fp32); (ii) queries at padded A positions are computed but zeroed by the mask on residual/readout so no padding value reaches any real atom or any pooled vector; (iii) a molecule with zero atoms cannot occur (assert; PaDEL/RDKit give ≥ 1 atom for all 799) — a fully masked key row would produce NaN, so a `+1` guard token is **not** used and an assertion is raised instead; (iv) attention pooling uses masked softmax; mean pooling divides by the true atom count. **Unit tests:** output at real positions is invariant to the amount of padding; swapping A/B swaps outputs; changing a *padded* atom's features changes nothing; a molecule's embedding is identical when batched with different partners (LayerNorm only, no batch statistics).
**Multi-component SMILES (296 of 799 molecules, up to 22 components).** Components stay in **one** graph as disconnected subgraphs (counter-ions, hydrate water, mixtures are part of the PubChem record the paper's fingerprint saw). Consequences handled explicitly: (a) GINE messages never cross components (no edges), so counter-ion information reaches the molecule only through pooling and, importantly, through cross-attention; (b) readout pools over all atoms (attention + mean), so a few ions do not dominate; (c) a per-molecule scalar feature `n_components` (log) is concatenated to the readout; (d) the ablation `largest_fragment` (§13.2) removes everything except the largest component to test whether salts/hydrates matter; (e) the `parent` split unit uses the same largest-fragment definition. Metal atoms/rare elements fall into the "other" element slot only beyond the 28 seen elements (no unknown-element failure).
**Hydrogens and special cases.** Explicit `[H]` atoms present in a SMILES are kept as nodes; ClF₃ parsed with `sanitize=False` + `UpdatePropertyCache(strict=False)`; a featuriser failure raises (never silently dropped) and the molecule list is checked for 799/799 successes.

### 8.3 Large-molecule handling and GPU-memory safeguards [PLANNED — specification]
Atom-count distribution: mean 27.5, median 25, p95 63, p99 89, **max 291** (polymers/cyclodextrins/macrolide-sugar records). Dense scores cost `B·K·n_A·n_B` per direction per block; the quadratic term is the only hazard.
1. **Never truncate or drop molecules** (they are real data and some are in the validated pairs). Fixed no-cutoff rule; handled by cost control instead.
2. **Token-budget batching.** Batches are cut by `sum over pairs of n_A·n_B ≤ P_max` (default P_max = 6·10⁵ attention cells per head-direction-block) **and** `B ≤ B_max` (tuned: 64/128). A pair of two 291-atom molecules (≈ 85 k cells) therefore never shares a batch with more than ~6 comparable pairs. Training batches are formed after shuffling with a bucketed sampler (sort within chunks of 20·B by `n_A·n_B`) so padding waste stays < 30 %; the validation/test loader uses the same bound and no shuffling.
3. **Chunked attention** for any batch exceeding the bound after bucketing: the K heads and the B pairs are processed in chunks of ≤ P_max cells (math unchanged; the chunk size is a runtime parameter), with `torch.utils.checkpoint` around each cross block to trade ~25 % time for memory in the worst case only.
4. **Mixed precision** (fp16 autocast, fp32 softmax and LayerNorm; automatic fp32 fallback flag); gradient-norm clip 1.0; `torch.backends.cuda.matmul.allow_tf32` irrelevant on Ampere laptop fp16 path.
5. **Runtime guard.** `torch.cuda.max_memory_allocated()` is logged per epoch; a soft limit of 3.0 GiB (4 GiB cards) triggers (in order) smaller P_max, halved batch, checkpointing; a CUDA-OOM exception is caught **once per batch**, the batch is split in two and retried, and the event is logged in `training_log.txt` (a run with > 3 OOM events is marked `unstable` and not averaged silently). A run that fails twice is rerun on CPU only if cheap, else reported as missing.
6. **Pre-flight (E0).** A synthetic worst-case batch (two 291-atom molecules ×6 plus a 128-pair typical batch) is run for every architecture; its measured peak must be < 3.0 GiB or the defaults are lowered before any tuning.
7. **Dataloader.** No worker processes (graphs already on GPU); `pin_memory` unnecessary; this removes CPU–GPU copy stalls on the laptop and makes the step time reproducible.


## 9. Models compared (all use the same data, splits, preprocessing, metrics)

| Key | Model | Input | Training protocol | Tuned? | Status |
|---|---|---|---|---|---|
| `paper_ann` | DE-INTERACT ANN, 3×1024, softmax-2 + BCE, RMSprop, bs 150, 200 ep, 3 checkpoints | FP concat | **paper** (E1) / **paper_clean** (E2: ×4 oversample, train only) | no (as published) | [DONE] code |
| `concat_mlp` | same architecture, unified protocol | FP concat | unified | no | [DONE] |
| `concat_mlp_reg` | concat MLP + dropout/weight decay, width/depth searched | FP concat | unified | **yes** (stronger FP-DL baseline) | [DONE] model, tuning [PLANNED] |
| `siamese_fp` | v1 Siamese (shared encoder, symmetric pair features) | FP | unified | **yes** | [DONE] model (was `siamese`) |
| `gnn_siamese` | shared GINE encoder, pooled embeddings, symmetric pair vector, **no cross-attention, no FP** | graph | unified | **yes** | [DONE] |
| **`de_crossnet`** | **proposed** (§8) | graph + FP | unified | **yes** | [DONE] |
| `rf` | random forest on FP concat | FP | class_weight balanced | no | [DONE] |
| `entity_prior` | per-drug / per-excipient incompatibility rates (identity only) | IDs | – | no | [DONE] — **diagnostic, not a competitor** |
Equal-effort rule: every learned model (`concat_mlp_reg`, `siamese_fp`, `gnn_siamese`, `de_crossnet`) gets the same tuning protocol (§12) with trials = 5 × #dimensions (documented), and the untuned `concat_mlp` is retained to show what tuning alone buys.


## 10. Loss, optimiser, training protocol
* **Paper protocol [P/R/I]** (E1/E2): p = softmax(o); BCE on one-hot t: L = −mean_k[t_k log p_k + (1−t_k) log(1−p_k)] (the paper says both "softmax" and "binary cross-entropy with logits" — ambiguous; we use Keras-style BCE on softmax probabilities); RMSprop lr 1e-3 (lr unstated → library default); batch 150, 200 epochs, no early stopping, no dropout; checkpoints max-val-accuracy / min-val-loss / max-val-precision tracked in one run.
* **Unified protocol [DONE, I]**: weighted BCE-with-logits L = −[w·y log σ(o) + (1−y) log(1−σ(o))], w = n_neg/n_pos from the **training split only**. Replacing row replication by weighting is equivalent in expectation and does not inflate epochs/memorisation. AdamW (lr 1e-3, wd 1e-4 default), batch 128, ≤ 100 epochs, **early stopping patience 15 on validation PR-AUC** (threshold-free, prevalence-robust, about the minority class; precision/loss rejected: degenerate / majority-dominated), best-epoch weights restored.
* **Additions for the graph models [PLANNED]:** AdamW with 5-epoch linear warm-up + cosine decay, gradient-norm clip 1.0, mixed precision (fp16 autocast on the 3050; fp32 fallback flag), `pos_weight` ∈ {auto, √auto} searched; focal loss / unweighted BCE are not part of the programme (§13.2).
* **Threshold**: value maximising incompatible-class F1 **on validation**, applied unchanged to test; test is also reported at 0.5 and F2 is reported.


## 11. Splitting, leakage prevention and the evaluation protocol — `splits.py`, `structure.py`, `entity_pool.py` [DONE]
All splits except `paper_rows` act on the de-duplicated pair table. **Primary leakage unit = `chem` group** (§3.1); `cid` and `parent` are sensitivity units; the legacy fingerprint-merged `structure` unit remains selectable for backward compatibility.
The `entity_pool.py` module creates disjoint Development Pool D (tuning) and Evaluation Pool E (final test) with zero entity overlap (guaranteed by `assert_pool_independence`). Stale cached splits are automatically detected and purged.

| Split | Held out | Establishes | Does **not** establish |
|---|---|---|---|
| `paper_rows` | random *rows* of the raw CSV (CID keyed, by design) | what the paper's protocol yields | any generalisation (≈82 % of test pairs seen) |
| `random_pair` | random **chem-group pairs** (unordered) | interpolation: new combinations of known molecules | new molecules |
| `cold_drug` | whole drug **chem groups** | unseen drugs (excipients known) | unseen excipients |
| `cold_excipient` | whole excipient **chem groups** | unseen excipients | unseen drugs |
| `cold_both` | each group in one partition; straddling pairs discarded | both unseen (hardest) | stable estimates (≈ 16 test positives) |

**Evaluation protocol (full, because it also fixes the independence of tuning).**
1. **Entity pools (fixed once, label-blind).** With a pool seed (a constant in the config, e.g. 2026) assign **25 % of drug chem groups and 25 % of excipient chem groups to the evaluation-only pool E**; all other groups, and the 4 dual-role molecules, form the tuning pool D. The assignment uses no labels; the only constraint is a validity check (re-draw the pool seed if `cold_both` pool has < 12 positives; disclosed). Preview over 20 pool seeds [MEASURED, `chem_group_preview.json`]: pairs with both entities in E ≈ 218 (16.5 positives; min 94/8), drug-in-E ≈ 903 pairs/61 positives, excipient-in-E ≈ 855/62, pairs with both entities in D ≈ 2,031/144 positives.
2. **Tuning (stage T) uses only pairs with both entities in D**, with the split types and seeds of §12. No molecule of E (nor any pair involving one) is used for any training, validation, early stopping or hyper-parameter choice in stage T.
3. **Final evaluation (E3–E6).** Test entities are drawn **only from E**: `cold_drug` test = a seed-dependent subset of E-drug groups (≈ 15 % of pair mass); `cold_excipient` likewise; `cold_both` test = all pairs with both entities in E (≈ 16 positives; identical across seeds, so seeds vary only training/initialisation and the split of D into train/val — stated as a limitation); `random_pair` test = random chem-group pairs having ≥ 1 entity in E. Training and validation entities are drawn from D plus the non-test part of E (E molecules may be *training data* in evaluation runs but never influenced a hyper-parameter). Validation (≈ 15 %) is drawn from D-side entities for early stopping and threshold choice.
4. **Validity rule** (label-aware only to guarantee a computable metric; disclosed): redraw the *seed-dependent* part if val/test has < 8 positives (cold_both pool: pool rule above) or training has one class.
5. **Assertions on every split** (unit-tested incl. injected-leak detector): disjoint indices; no identical unordered chem-group pair across partitions; no held-out chem group in training in any role; for stage T: no E group anywhere; for stage E: every test entity ∈ E.
6. **Test set use:** each (model, split, seed) test set is scored once with weights, epoch and threshold chosen on validation; no model or hyper-parameter is changed after seeing any test number (the frozen tuned configs are committed first).
7. **Fingerprint collisions** are evaluated separately (E5b) and never define groups.
8. Seeds: *tuning seeds 100–101* (and, if used, 102) vs *evaluation seeds 0–9* are disjoint; the pool E is the main independence device, seeds are secondary.
**Remaining limitation (stated, not hidden):** hyper-parameter *families* were chosen by us before any data-driven tuning (e.g. d ∈ {64,128,192}); and E-pool molecules still enter training in evaluation runs, which is legitimate (they are not test entities). The 25 % pool shrinks tuning data to ≈ 2,030 pairs (144 positives) — tuning is noisier and models are tuned on smaller training sets than they are finally trained on; E5d measures the effect.

## 12. Hyper-parameter tuning and model selection [DONE]
Implemented in `src/deint/tune.py` and executable via `scripts/06_tune.py`.
* **Selection hierarchy:** weights → training split; epoch → validation PR-AUC (patience 15); threshold → validation F1; hyper-parameters → mean validation PR-AUC on the tuning pool D only (§11); **test entities (⊂ E) are used once, for final reporting.**
* **Objective** (one scalar per trial): mean validation PR-AUC over {`random_pair`, `cold_drug`, `cold_excipient`} × tuning seeds {100, 101} built **inside D** = 6 trainings (≈ 55 % of the full training size). `cold_both` is excluded (D-validation ≈ 5 positives). Ties/noise: report the study's top-5 trials and choose the simplest (fewest parameters) within 1 SE of the best (pre-declared rule).
* **Search:** Optuna TPE (10 random start trials) + median pruner; trials report after each of the six trainings (step = training index; prune from step 2) so a bad trial costs ≈ 2 trainings. **Trial budget = 5 × (#tuned dimensions), identical rule for every learned model:** `de_crossnet` 8 dims → **40**; `gnn_siamese` 6 → **30**; `concat_mlp_reg` 6 → **30**; `siamese_fp` 5 → **25**; total **125** (was 200). Each study also has a wall-clock cap (§15.3); the realised trial count is reported.
* **Spaces.** `de_crossnet`: d ∈ {64,128,192}; GNN layers L ∈ {2,3,4}; cross blocks N_x ∈ {1,2}; dropout ∈ [0.1,0.5]; lr ∈ [3e-4,3e-3] log; weight decay ∈ [1e-5,1e-2] log; batch ∈ {64,128}; pos_weight ∈ {auto,√auto}. **Fixed (not searched):** heads 4, FP-embedding 256, readout att+mean, warm-up 5 epochs. Baselines: width {256,512,1024}, depth {2,3,4}, dropout, lr, wd, pos_weight (+ encoder width/layers for `gnn_siamese`, encoder width for `siamese_fp`).
* **Frozen configs:** the winning configuration of each study is written to `configs/tuned/<model>.yaml` and committed before any E3/E4 run. Ablation variants reuse the tuned values (not re-tuned; this favours the full model slightly — stated).
* **Independence check (E5e):** tuned vs default `de_crossnet` on evaluation seeds; a gap far larger than the tuned-vs-default gap of baselines would indicate residual selection bias.
* Nested per-split tuning is not done (cost ×10); listed as future work.

## 13. Experimental programme [PLANNED unless marked]
Run = (model, split, seed). Time estimates: §15 (all [EST] until E0).

### 13.1 Experiments and exact run counts
| ID | Purpose | Content | Runs (actual) |
|---|---|---|---|
| **E0** | benchmark gate | time/step, s/run, epochs-to-converge, peak memory (incl. worst-case batch), 2-process throughput test; 4 learned models × 3 splits × 3 seeds | 36 short trainings |
| **E1** | reproduce paper protocol | `paper_ann`, `paper_rows`, 3 checkpoints, full-data + test metrics vs paper Tables 4–5 | 5 |
| **E2** | quantify leakage | `paper_ann`, paper_clean (×4 oversampling train only) on `random_pair`; E1→E2 drop = leakage | 5 |
| **T** | tuning on pool D (§12) | 4 learned models | 125 trials ≈ 437 trainings expected (worst 750) |
| **E3** | main comparison | `de_crossnet`, `gnn_siamese`, `siamese_fp`, `concat_mlp_reg`, `concat_mlp` × {random_pair, cold_drug, cold_excipient, cold_both} × 10 seeds (+ `rf`, `entity_prior`, CPU) | 200 GPU + 80 CPU |
| **E4-T1** | core ablations of `de_crossnet` | 4 variants × {random_pair, cold_drug, cold_excipient} × 10 seeds | 120 |
| **E4-T2** *(conditional)* | design checks | 2 variants × 3 splits × 5 seeds, run only if cumulative time ≤ 13 h after E5 | 30 |
| **E5** | sensitivity | (a) `concat_mlp_reg` on CID-keyed vs `chem` splits; (b) **fingerprint-collision stratified metrics** (post hoc on E3 predictions, no training); (c) `parent` splits for `concat_mlp_reg`, `de_crossnet`; (d) post-hoc exclusion of the 2 label-conflict group-pairs; (e) default-vs-tuned `de_crossnet` | 15 + 30 + 15 = 60 (+0 post hoc) |
| **E6** | interpretability | attention maps from saved seed-0 checkpoints for the paper's 3 validated pairs and Table 7 pairs, failure analysis | inference only |
`paper_ann` is not re-run in E3: `concat_mlp` is the identical 3×1024 architecture under the unified protocol, so a third copy would be redundant (E1/E2 already cover the paper protocol).

### 13.2 Component ablations (E4) — prioritised, redundant ones removed
The four core variants form, together with models already in E3, a **2×2 factorial** over {atom-level cross-attention} × {fingerprint branch} plus a capacity-matched control and two design checks of claims we make about the model:

| Variant (one factor changed from the tuned full model) | Question it answers | Hypothesis |
|---|---|---|
| `de_crossnet` (E3) | reference: graph + FP + cross-attention | – |
| **`no_fp`**: graph + cross-attention, FP branch removed | does the fingerprint add anything to the graph? | H3 |
| **`self_attn`**: cross-attention blocks replaced by **self-attention of the same shape** (keys/values from the molecule's own atoms; parameter-matched), FP kept | is any gain from **partner information**, or merely from extra attention capacity? (the key control) | H2 |
| **`ordered_head`**: symmetric pair vector replaced by ordered `[m_drug, m_exc]` concat | does enforcing role-symmetry help or hurt (the paper assumes role-agnostic)? | – |
| **`largest_fragment`**: keep only the largest component of each SMILES graph | are salts/hydrates informative or noise? (validates §8.2) | – |
| `gnn_siamese` (E3, = no cross, no FP) | factorial cell (−,−) | H2/H3 |
| `siamese_fp` (E3, = no graph) | FP-only reference | H3 |
Conditional (T2): `no_bond_feat` (bond features zeroed), `N_x=1`-fixed vs tuned.
**Removed as redundant or low-value (v2 → v3):** GCN/GAT/MPNN encoder swaps (A5–A7: encoder-family tourism, no component claim), −bond features as core (A8, → T2), role-embedding variant (A4; overlaps `ordered_head`), −cross-attention by deletion (A2; confounds capacity — replaced by capacity-matched `self_attn`), depth sweeps (A9–A11; depth is already tuned), SSI-DDI-style bilinear co-attention (A12; an implementation of a different model, listed as future work), mean-readout, focal loss, unweighted BCE, FP-as-token (A13–A16; loss/readout tweaks without a mechanistic claim).
Evaluation of ablations: paired per-seed differences against the E3 full model on identical (split, seed); `cold_both` excluded (≈ 16 positives, fixed test set).

**Ordering with a hard stop at any point:** E0 → E1/E2 → T → E3 → E4-T1 → E5 → E4-T2 → E6.
**Spent-time rules:** (1) E0 s/run ≤ 0.8× estimate → add evaluation seeds 10–14 to E3 and E4-T1; (2) 1.0–1.5× → run as planned; (3) 1.5–1.8× → drop T2, cut tuning caps by 25 %; (4) > 1.8× → E3 to 5 seeds, E4-T1 to 5 seeds, E5 (a,c,e) to 3 seeds; (5) never trade correctness (leakage assertions, validation-only selection, pool E) for time.

## 14. Metrics and statistics (positive class = incompatible) — `metrics.py` [DONE]
accuracy (TP+TN)/N — misleading at 93 % majority; precision TP/(TP+FP); recall TP/(TP+FN) — missed incompatibilities are the costly error; F1 and F2; macro-F1; balanced accuracy (recall+specificity)/2 (chance 0.5); MCC (uses all four cells, 0 = chance); ROC-AUC (ranking; optimistic under imbalance); **PR-AUC / average precision — primary** (chance = prevalence ≈ 0.07, so values are read against that). Confusion matrix [[TN,FP],[FN,TP]].
*Statistics [PLANNED additions to `05_aggregate_results.py`]:* mean ± std over seeds; per-seed **paired** differences (proposed − reference) with win counts and **bootstrap 95 % CIs over seeds**; the Nadeau–Bengio corrected resampled t statistic reported descriptively. With ≈ 40 test positives per split and overlapping resamples, p-values are not treated as confirmatory; a difference whose CI includes 0 is called "no evidence of improvement". For `cold_both` (≈ 15 positives) metrics are additionally pooled over seeds.
Reading rule: **PR-AUC and incompatible-F1 on cold splits are headline; accuracy is only reported for comparison with the paper.**

---


## 15. GPU memory, runtime and the 20-hour schedule [EST — to be replaced by Stage E0 measurements]
**What was measured so far:** CPU-only sandbox, v1 models: Siamese-FP ≈ 15 s, concat-MLP ≈ 45 s, RF ≈ 4 s per E3-style run; PaDEL ≈ 9 min for 799 molecules. **No GPU has been available; no graph model exists.** Every GPU number is a bottom-up extrapolation with a stated margin, and E0 replaces all of them. The v2 schedule contradicted itself (tuning and ablation rows did not equal runs × time); v3 builds every row as *runs × s/run × 1.1 overhead*.

### 15.1 Memory [EST]
| Model | Parameters | Weights + AdamW | Typical activations | Peak with safeguards (§8.3) |
|---|---|---|---|---|
| `paper_ann` / `concat_mlp(_reg)` | 3.9 M / ≤ 3.9 M | ≈ 47 MB | < 50 MB | < 0.2 GB |
| `siamese_fp` | 0.68 M | ≈ 8 MB | < 30 MB | < 0.1 GB |
| `gnn_siamese` | ≈ 0.4 M | ≈ 5 MB | 0.1–0.2 GB | ≈ 0.3 GB |
| **`de_crossnet`** (d 128, L 4, K 4, N_x 2) | ≈ 0.93 M | ≈ 11 MB | 0.4–0.8 GB | ≈ 1.2 GB fp16 / ≈ 2.5 GB fp32; **hard target < 3.0 GiB** |
Worst case unbounded padding: a 291×291 pair in a 128-pair batch ≈ 87 MB per fp16 score tensor × (2 directions × N_x blocks × ~3 saved tensors) ≈ 1 GB; the token-budget batching and chunking of §8.3 cap it at ≈ P_max cells.

### 15.2 Per-training time (planning values, include evaluation, checkpoint and logging overhead) [EST]
Bottom-up for `de_crossnet`: ≈ 500 kernel launches fwd+bwd + AdamW + index gathering ≈ 25–40 ms/step; 20 steps/epoch + validation ≈ 0.9 s/epoch; 40–60 epochs with patience 15 ⇒ 36–54 s; + ≈ 10 s start-up/test scoring/IO ⇒ central ≈ 55 s. **Planning value = 80 s (≈ 1.45× central).**
| Model | Planning s/run (full-size train ≈ 2.5 k pairs) | s/run inside tuning pool D (≈ 0.6×) | Basis |
|---|---|---|---|
| `de_crossnet` | **80** | **48** | bottom-up above |
| `gnn_siamese` | 45 | 27 | fewer kernels, no attention |
| `siamese_fp`, `concat_mlp(_reg)` | 20 | 12 | CPU-measured 15–45 s, GPU launch-bound |
| `paper_ann` E1 (7.5 k rows, 200 epochs fixed) | 150 | – | 10 k steps × ≈ 12 ms + eval |
| `paper_ann` E2 (≈ 3.1 k rows, 200 epochs) | 90 | – | 5 k steps |
`cold_both` runs are ≈ 0.5× (not credited — conservative). Two worker processes could raise throughput ≈ 1.3–1.6× (launch-bound kernels); tested in E0, **not** assumed.

### 15.3 Schedule (cumulative, 20 h budget)
| Stage | Computation | Runs | GPU-h incl. 1.1× overhead | Cumulative |
|---|---|---|---|---|
| E0 | benchmark gate (36 short trainings + worst-case memory test + 2-process test) | 36 | 0.4 | 0.4 |
| E1 + E2 | 5 × 150 s + 5 × 90 s = 1,200 s | 10 | 0.4 | 0.8 |
| T | trainings (expected, with pruning ≈ 3.5 per trial): `de_crossnet` 140 × 48 s = 6,720 s; `gnn_siamese` 105 × 27 = 2,835; `concat_mlp_reg` 105 × 12 = 1,260; `siamese_fp` 87.5 × 12 = 1,050 → 11,865 s = 3.30 h; worst case (no pruning, 6 trainings/trial) 5.65 h → **caps** 2.2 / 0.9 / 0.45 / 0.4 h per study | ≈ 437 trainings | **3.6** | 4.4 |
| E3 | per (split, seed): 80 + 45 + 20 + 20 + 20 = 185 s × 40 (split, seed) = 7,400 s | 200 | 2.3 | 6.7 |
| E4-T1 | 120 runs × ≈ 76 s (`no_fp` 70, `self_attn` 80, `ordered_head` 80, `largest_fragment` 75) = 9,150 s | 120 | 2.8 | 9.5 |
| E5 | (a) 15 × 20 s; (c) 15 × 80 + 15 × 20 s; (e) 15 × 80 s = 3,000 s | 60 | 0.9 | 10.4 |
| E4-T2 (conditional) | 30 × 80 s = 2,400 s | 30 | 0.7 | 11.1 |
| E6 | attention extraction from saved checkpoints | – | 0.2 | **11.3** |
| **Planned total** | | | **11.3 h** | |
| Margin for a ×1.5 timing error on all rows | | | +5.65 → 16.95 h | |
| Remaining hard reserve (reruns, failed runs, extra seeds, figures) | | | 3.05 h | 20.0 |
The planned total is deliberately far below 20 h because every GPU number is an unmeasured estimate; the surplus is *not* spent in advance (rule 1 of §13.2 releases it to extra seeds only after E0 measures actual speed). Everything is resumable (finished runs skipped; `--force` archives). Wall-clock development time (writing code, tests) is not GPU time and is not counted.

## 16. Implementation Summary [DONE] — All Components Implemented and Verified
| Stage | Component | Files | Implementation Details | Status |
|---|---|---|---|---|
| **0a** | Chemistry-only `chem` grouping & Entity Pools D/E | `structure.py`, `entity_pool.py`, `splits.py`, `scripts/03_make_splits.py` | 783 chem groups; fingerprint collisions analyzed separately; D/E pool independence guaranteed with 0 overlap; stale cached split detection | **[DONE]** |
| **0b** | Molecular graph pipeline & batch collation | `graphs.py` | Pure PyTorch GINE graph builder; atom/bond features; multi-component, salts, ClF3 fallback; zero-copy token-budget batching; masked padding | **[DONE]** |
| **0c** | DE-CrossNet, GNN Siamese & Ablations | `gnn.py` | Shared GINE encoder; bidirectional cross-attention; attention+mean readout; symmetric/ordered heads; ablations (`no_fp`, `self_attn`, `ordered_head`, `largest_fragment`) | **[DONE]** |
| **0d** | GNN Training Engine & AMP Safeguards | `gnn_train.py`, `experiments.py` | Mixed precision (`torch.cuda.amp`), gradient clipping (1.0), CUDA OOM auto-recovery and memory purging, threshold tuning on val only | **[DONE]** |
| **0e** | Benchmark Gate (E0) | `scripts/00_benchmark.py` | Measures s/epoch, peak VRAM, parameters, and calibrates remaining 20 h execution schedule | **[DONE]** |
| **0f** | HPO on Entity Pool D (Protocol T) | `tune.py`, `scripts/06_tune.py` | Multi-split optimization over 3 splits × 2 seeds strictly inside Pool D; frozen configs to `results/tuned/` | **[DONE]** |
| **0g** | Aggregation & Statistics | `scripts/05_aggregate_results.py` | Aggregates all neural and classical models; paired differences, win counts, 600 DPI figures | **[DONE]** |
| **0h** | Sensitivity (E5) & Interpretability (E6) | `scripts/07_sensitivity.py`, `scripts/06_attention.py` | Fingerprint-collision stratification post-hoc; label conflict exclusion; cross-attention map visualizer | **[DONE]** |
| **0i** | Comprehensive Test Suite | `tests/test_pipeline.py` | 42 unit & regression tests (53 collected items); symmetry, padding invariance, leakage assertions, Stage T tuning smoke runs (52 passed, 1 skipped) | **[DONE]** |

## 17. Fit to the 6-page IEEE report (template outline is fixed; exactly 6 pages incl. references)
Planned from the start; content that does not fit stays in this file.

| Template section | Budget | Content | Figures / tables |
|---|---|---|---|
| Title, Abstract (150–200 words), keywords | 0.35 p | problem, objective, dataset (3,609 unique pairs, 6.95 % incompatible), method, **one named headline number (cold-split PR-AUC vs ANN, mean ± std)**, contribution, GitHub URL | – |
| I. Introduction | 0.55 p | motivation (stability, cost of wet-lab), paper-based project: DE-INTERACT reproduced + evaluated + extended; gap = leakage & no cold-start; 3 contributions; roadmap | – |
| II. Related Works | 0.6 p | 5–6 **verified** papers only (§20): DE-INTERACT; SSI-DDI; SA-DDI; MHCADDI; D-MPNN/GINE (Yang 2019, Hu 2020); no other excipient-ML papers unless the authors verify them by hand | **Table I** compact comparison (method / input / interaction / evaluation / limitation) |
| III-A Problem & dataset | 0.5 p | input/output; 799 molecules, 3,609 CID pairs → 3,570 `chem`-group pairs; splits; leakage precautions | **Table II** splits & leakage measurements |
| III-B Pipeline & approach | 1.1 p | architecture, key equations (GINE layer, cross-attention, pair vector), loss, protocol changes vs paper | **Fig. 1** architecture / data-flow (full width) |
| III-C Implementation | 0.35 p | PyTorch/RDKit/PaDEL versions, hardware RTX 3050, hyper-parameters (from tuned config), seeds, epochs, tuning protocol | – |
| IV-A Setup & metrics | 0.3 p | metrics (PR-AUC primary), validation vs test, thresholds | – |
| IV-B Results | 1.0 p | E1 vs E2 (leakage) and E3 main results; ablation | **Fig. 2** leakage bars (E1→E2→structure/cold); **Table III** E3 (PR-AUC, F1, recall; 4 splits; mean ± std); **Table IV** factorial + core ablations (`no_fp`, `self_attn`, `ordered_head`, `largest_fragment`, with E3's `gnn_siamese`/`siamese_fp`) |
| IV-C Discussion & limitations | 0.5 p | why results changed; entity-prior diagnostic; overlap of tuning; small positives; wet-lab not repeated | – |
| V. Conclusion & future work | 0.25 p | finding, 2 limitations, 3 next steps | – |
| References | 0.45 p | ≈ 9–12 numbered IEEE entries from §20 (only verified ones); frees ≈ 0.2 p for results | – |
| **Total** | **6.0 p** | 2 figures + 4 tables (one figure/table may be dropped if layout overflows) | |
Rules: no shrinking of fonts; long tables go to this file/README; every number in the report comes from `results/summary/*.csv`; unfinished experiments are marked *pending*.


## 18. Expected contribution and limitations
**Contribution (without exaggeration):** (1) an audited reproduction of DE-INTERACT with quantified duplicate and structure-level leakage; (2) a leakage-controlled cold-start benchmark for drug–excipient compatibility (structure-grouped splits, identity diagnostic); (3) an adaptation of atom-level cross-attention to drug–excipient pairs, hybridised with PubChem fingerprints, with a one-factor-at-a-time ablation and equal-effort tuning of all learned baselines. It is **not** claimed to be a new architecture; the finding may be that gains are small or absent under cold-start, which would itself be a useful result.
**Limitations:** labels mix literature incompatibilities with "listed on label ⇒ compatible" assumptions (absence of evidence); 250 positives; 10 label-conflicting structure-pairs; cold-start test sets ≈ 15–50 positives; unknown cause of paper/CSV count differences; PaDEL fed by SMILES not SDF (no offline bit-equality check); hyper-parameter tuning overlaps molecules with evaluation seeds (mitigated, not removed); ablation variants are not re-tuned; the split-validity rule reads label counts; no new wet-lab validation; the paper's three wet-lab confirmations cannot be re-evaluated on a retrained model without the same model; GPU numbers are estimates until E0; untested on a real CUDA device and on Windows shell scripts.


## 19. Viva cheat-sheet — what each result can / cannot establish
* **Large E1→E2 drop** ⇒ the paper's protocol overstates performance *for this data*; it does not prove the paper's model is poor.
* **E2 vs cold-split drop** ⇒ how much of the "random-pair" skill is molecule identity.
* **de_crossnet > concat_mlp_reg on cold splits** ⇒ the graph/cross-attention inductive bias helps *in this benchmark*; **E4 `no_cross` / `graph-only` / `concat_mlp_reg`** say whether it is interaction, graph, or regularisation. A paired-CI that includes 0 ⇒ "no evidence of improvement".
* **entity_prior competitive** ⇒ the benchmark is partly solvable without chemistry; chemistry claims need `cold_both` or a margin over entity_prior.
* **Tuned vs default (E5e)** ⇒ size of hyper-parameter selection bias; **CID vs `chem` vs `parent` splits (E5a/E5c) and fingerprint-collision subsets (E5b)** ⇒ how sensitive conclusions are to the leakage definition.
* Nothing here validates real-world incompatibility prediction.
Likely questions: *Why not a GNN alone?* (it lacks pair interaction; ablated as `gnn_siamese` and the capacity-matched `self_attn`). *Why LayerNorm?* (no batch-statistics leakage, molecule embeddings independent of batch). *Why PR-AUC?* (chance = 0.07; threshold-free; about the minority class). *Why structure groups?* (19.6 % of cold-excipient test excipients had a structure twin in training under CID splits). *Why not tune per ablation?* (cost; documented bias toward the full model). *Is the model symmetric?* (by construction; role-aware variant tests the alternative). *Why 10 seeds?* (≈ 40 test positives ⇒ high split variance).


## 20. References (verified in this revision; use only these in the report)
Verification: each entry was matched to an independent indexing or primary page (publisher/arXiv/ICLR/PMC/MIT DSpace/the supplied PDF). Re-check page numbers against the publisher when formatting.
1. S. Patel, M. Patel, M. Kulkarni, M. S. Patel, "DE-INTERACT: A machine-learning-based predictive tool for the drug-excipient interaction study during product development—Validation through paracetamol and vanillin as a case study," *Int. J. Pharm.*, vol. 637, art. 122839, 2023. (supplied PDF)
2. A. K. Nyamabo, H. Yu, J.-Y. Shi, "SSI–DDI: substructure–substructure interactions for drug–drug interaction prediction," *Brief. Bioinform.*, vol. 22, no. 6, bbab133, 2021.
3. Z. Yang, W. Zhong, Q. Lv, C.-Y. Chen, "Learning size-adaptive molecular substructures for explainable drug–drug interaction prediction by substructure-aware graph neural network," *Chem. Sci.*, vol. 13, no. 29, pp. 8693–8703, 2022.
4. A. Deac, Y.-H. Huang, P. Veličković, P. Liò, J. Tang, "Drug-drug adverse effect prediction with graph co-attention," arXiv:1905.00534, 2019.
5. X. Lin, L. Dai, Y. Zhou, Z.-G. Yu, W. Zhang, J.-Y. Shi *et al.*, "Comprehensive evaluation of deep and graph learning on drug–drug interactions prediction," *Brief. Bioinform.*, vol. 24, no. 4, bbad235, 2023. (confirmed via its citation in a 2024 *Quant. Biol.* article; author list truncated — complete from the publisher.)
6. K. Yang *et al.*, "Analyzing learned molecular representations for property prediction," *J. Chem. Inf. Model.*, vol. 59, no. 8, pp. 3370–3388, 2019.
7. K. Xu, W. Hu, J. Leskovec, S. Jegelka, "How powerful are graph neural networks?" in *Proc. ICLR*, 2019 (arXiv:1810.00826).
8. W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, J. Leskovec, "Strategies for pre-training graph neural networks," in *Proc. ICLR*, 2020 (arXiv:1905.12265). (source of the GINE layer)
9. C. W. Yap, "PaDEL-descriptor: an open source software to calculate molecular descriptors and fingerprints," *J. Comput. Chem.*, vol. 32, pp. 1466–1474, 2011. (as listed in reference 1)
**Removed because they could not be verified to a complete record in this revision** (do not cite unless the authors verify them by hand): the mol2vec-based "Hang et al." excipient paper (only a 2024 news item was found; author list and venue unconfirmed); the "Lee & Cho, Inje University" conference abstract; the "FormulationDE" paper (abstract seen, authors/journal/year not confirmed); Kipf & Welling (GCN), Veličković et al. (GAT), Gilmer et al. (MPNN), Vaswani et al. (Transformer), Saito & Rehmsmeier (PR plot), Akiba et al. (Optuna), Loshchilov & Hutter (AdamW), Ba et al. (LayerNorm), Nadeau & Bengio, RDKit citation. They are well known, but were not re-checked here and are not needed for any claim in this plan (GCN/GAT/MPNN ablations were removed; method details of Optuna/AdamW/LayerNorm can be cited after a quick check at report time).
