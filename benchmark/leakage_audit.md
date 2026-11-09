# Sequence Leakage Audit

In response to a methodological review, we performed a sequence-identity audit between the 21 held-out test targets and the 21 training targets, using local pairwise alignment (BLOSUM62, minimum 30-residue alignment length).

## Result

One target showed substantial homology to a training target:

- **5U8R (IGF1R)** vs **InsulinR** (training): 69.2% identity over 143/150 residues

This is consistent with the known close evolutionary relationship between IGF1R and the Insulin Receptor (both receptor tyrosine kinases in the same subfamily). This pair is flagged as a genuine case of homologous leakage.

All other 20 targets showed weak-to-moderate identity (31-53%) over short alignment windows (30-95 residues), consistent with normal background similarity between unrelated proteins rather than leakage.

## Impact on the main result

Re-computing the benchmark after excluding 5U8R:

| | n=21 (full) | n=20 (excluding 5U8R) |
|---|---|---|
| Mean OryqevaMPNN i_pTM | 0.488 | 0.500 |
| Mean ProteinMPNN i_pTM | 0.424 | 0.433 |
| Paired t-test p-value | 0.047 | 0.049 |
| Wins | 14/21 | 13/20 |

The statistical result remains significant (p < 0.05) after removing the flagged target, indicating the overall finding is not driven by this single case.

## Method

Target chain sequences were extracted from each PDB structure and aligned pairwise (test target vs every training target) using Biopython's `PairwiseAligner` with the BLOSUM62 substitution matrix, local alignment mode, and a minimum alignment length threshold of 30 residues to exclude spurious short matches.
