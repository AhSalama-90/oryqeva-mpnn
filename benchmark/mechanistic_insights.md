# Mechanistic Insights: Amino Acid Composition Shift

Following the analysis approach used in ResiDPO (Xue et al., 2025, EnhancedMPNN), we compared the amino acid composition of OryqevaMPNN's designs against vanilla ProteinMPNN's, across the same 21 held-out targets.

## Result

| Category | ProteinMPNN | OryqevaMPNN | Change |
|---|---|---|---|
| Charged (D,E,K,R) | 40.31% | 48.16% | +7.85 pp |
| Ambiguous (A,S,T,Q) | 12.28% | 11.38% | -0.90 pp |

Largest individual increases: Lysine (K, +5.85pp), Glutamic acid (E, +2.96pp)
Largest individual decreases: Leucine (L, -4.73pp), Glycine (G, -4.63pp), Alanine (A, -2.04pp)

## Interpretation

This matches the compositional shift independently reported by ResiDPO/EnhancedMPNN (University of Washington Institute for Protein Design), which also found increased favoring of charged residues (E, K, R) and reduced use of ambiguous residues (A, S, T, Q) after preference-based fine-tuning — despite that work using a completely different base model (LigandMPNN), preference signal (AlphaFold pLDDT rather than experimental binding outcomes), and training method (ResiDPO rather than standard DPO+SFT).

The convergence of two independent preference-optimization approaches on the same compositional direction suggests this shift reflects a genuine, generalizable signal — charged residues likely reduce sequence-structure ambiguity at designed interfaces, which both experimental binding success and AlphaFold-predicted confidence may reward in similar ways.
