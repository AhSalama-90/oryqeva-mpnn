# DPO Preference Pair Methodology: A Note on Structural Conditioning

In response to a methodological review, we audited how preferred and dispreferred sequences are paired in our DPO training data.

## What we verified

Every one of the 3,680 training pairs compares a preferred and dispreferred sequence for the **same biological target** (confirmed after correcting for inconsistent target-name capitalization/formatting across sources, e.g. "PD-L1" vs "Pdl1" vs "pd-l1", which initially appeared as 1,286 "different-target" pairs but were in fact 100% same-target once normalized).

## A real methodological difference from standard practice

The standard approach in the DPO-for-inverse-folding literature (e.g. ResiDPO, EnerBridge-DPO, Protein Inverse Folding From Structure Feedback) generates preferred and dispreferred sequences by sampling multiple candidates **from the same fixed backbone structure**, so that the only difference between the pair is the sequence itself.

Our dataset does not meet this condition. Because preferred and dispreferred sequences come from independent real wet-lab binder design experiments, they frequently correspond to **different binder backbones of different lengths** for the same target — e.g., for Cas9, preferred/dispreferred binder lengths in our data range from 64 to 86 residues even within pairs for the same target.

This means the log-probability difference the model learns from may partly reflect differences in backbone design (length, fold) rather than sequence quality alone.

## Why we made this trade-off

The alternative (sampling preferred/dispreferred pairs from one fixed backbone) requires generating multiple candidate sequences computationally and scoring them with a folding model — which uses synthetic, model-generated labels rather than real experimental outcomes. We instead prioritized training exclusively on genuine wet-lab binding results, at the cost of this structural confound.

## Implication

This is a genuine limitation of the training signal, not a data corruption issue (distinct from the sequence/PDB mismatch problem addressed separately). It should be considered when interpreting what OryqevaMPNN has specifically learned, and is a natural target for future work (e.g., restricting training to pairs with closely matched binder lengths, or incorporating structure-matched synthetic pairs alongside the experimental ones).
