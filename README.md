# OryqevaMPNN

**Data-Centric Direct Preference Optimization for Protein Binder Sequence Design**

> Status: Manuscript in preparation | Code: this repository | Area: Computational Biology / Protein Design / Preference Alignment
> Keywords: inverse folding, protein binder design, Direct Preference Optimization, data curation, ProteinMPNN

## TL;DR

OryqevaMPNN fine-tunes ProteinMPNN via a two-stage DPO-then-SFT procedure on a rigorously verified, sequence-structure-matched preference corpus, achieving a statistically significant improvement in AlphaFold2-predicted interface confidence over vanilla ProteinMPNN, using an unmodified architecture and a single consumer GPU.

## Background and Motivation

Preference-optimization methods for protein inverse folding typically assume that scaling the preference-pair corpus improves downstream performance. We instead test a data-centric hypothesis: that structural fidelity of the preference corpus is a critical and underexamined determinant of training signal quality. Auditing a 40,040-pair combined corpus, we found 70-90% of pairs had a sequence that did not match its stated PDB structure. Training exclusively on the verified 3,680-pair remainder yields a significant, reproducible improvement over the base model.

## Key Designs

1. Data integrity audit: exact-match verification between every stated preference-pair sequence and its associated PDB chain; only verified pairs are retained for training.
2. Two-stage fine-tuning: DPO with an auxiliary SFT term on the preferred sequence (beta = 0.5), followed by a second SFT pass on the 368 unique successful sequences (lr = 2e-6).
3. Unmodified architecture: identical ProteinMPNN backbone (1.66M parameters) throughout, so the comparison isolates data quality and training objective, not model capacity.

## Key Experimental Results

21-target held-out benchmark (AlphaFold2 i_pTM, 5 sequences per target per model):

| Model | Mean i_pTM | Parameters |
|---|---|---|
| ProteinMPNN (baseline) | 0.424 | 1.66M |
| SFT-only | 0.491 | 1.66M |
| DPO-only | 0.488 | 1.66M |
| OryqevaMPNN (DPO + SFT) | 0.504 | 1.66M |

Paired t-test vs. ProteinMPNN: t(20) = 2.65, p = 0.015; 15/21 (71%) targets favored OryqevaMPNN.

Ablation: SFT-only and DPO-only are statistically indistinguishable (p = 0.88) - data quality, not the specific preference-optimization algorithm, drives the improvement. Combining both stages gives the strongest result.

Mechanistic insight: OryqevaMPNN's designs shift toward charged residues (D, E, K, R: 40.3% to 48.2%), independently matching a finding reported for ResiDPO/EnhancedMPNN (Univ. of Washington) on an unrelated base model and preference signal.

## Method

The following objective is implemented in oryqeva_mpnn.py (Equations 4-6 of the manuscript).

Cross-context policy and reference contrasts (Equation 4):

delta_q = l_q_impl(y_w | x_w) - l_q_impl(y_l | x_l), for q in {theta, ref}

Here x_w and x_l are the preferred and dispreferred structural inputs; they are not assumed identical, since pairs use distinct PDB paths and frequently differ in length.

Stage-1 preference-plus-supervised objective (Equation 5):

L_stage1 = -log sigma[ beta * (delta_theta - delta_ref) ] - lambda * l_bar_theta(y_w | x_w)

where sigma is the logistic function, beta = 0.5, lambda = 1.0. The frozen reference weights are copied from the policy at the start of each 400-step round. This objective differs from standard DPO, which corresponds to lambda = 0; we refer to it as cross-context preference fine-tuning, since the two sides of a pair are not conditioned on the same backbone.

Stage-2 supervised refinement objective (Equation 6):

L_stage2 = -l_bar_theta(y | x), for y in D_preferred_unique

The final stage minimizes negative mean preferred-sequence log-probability over the de-duplicated preferred examples (1,200 steps, Adam lr = 2e-6).

## Installation

git clone https://github.com/AhSalama-90/oryqeva-mpnn.git
cd oryqeva-mpnn
pip install -r requirements.txt

## Usage

python src/oryqeva_mpnn.py train --pairs data/train_pairs_clean.json --output-prefix checkpoints/oryqeva_mpnn
python src/oryqeva_mpnn.py design --checkpoint checkpoints/oryqeva_mpnn_best.pt --pdb target.pdb --binder-chain A --target-chain B --binder-len 80 --n-samples 5

## Repository Structure

oryqeva-mpnn/
  src/oryqeva_mpnn.py          train, sft-finetune, design commands
  checkpoints/oryqeva_mpnn_best.pt   final checkpoint
  data/train_pairs_clean.json  3,680 verified pairs
  benchmark/
    oryqeva_mpnn_results.json
    proteinmpnn_results.json
    leakage_audit.md
    dpo_methodology_note.md
    mechanistic_insights.md

## Limitations

All evaluation is computational (AlphaFold2 i_pTM); no wet-lab binding assays have been performed. Preferred and dispreferred sequences within each pair are independently-designed binders of different lengths, not samples from a fixed backbone. Training data covers 21 targets; generalization to structurally dissimilar targets is untested.

## Citation

Citation details will be added upon publication.

## License

MIT - see LICENSE.
