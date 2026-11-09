# OryqevaMPNN

Code, cleaned training data, and analysis scripts accompanying a manuscript currently under peer review. This repository exists to support independent verification and reuse, in line with the journal's code/data availability requirements.

## Contents

- `src/oryqeva_mpnn.py` — unified training and design CLI (DPO, SFT, sequence generation)
- `data/train_pairs_clean.json` — 3,680 verified preference pairs (sequence-to-structure match confirmed) used for training
- `scripts/` — figure-generation and statistical analysis scripts (bootstrap, Welch's t-test, leave-one-out, ROC/PR)
- `results/tables/` — numerical results underlying the manuscript's figures and statistics

## Reproducing the analysis

```bash
pip install -r requirements.txt
python src/oryqeva_mpnn.py train --pairs data/train_pairs_clean.json --output-prefix checkpoints/oryqeva_mpnn
python src/oryqeva_mpnn.py design --checkpoint checkpoints/oryqeva_mpnn_best.pt --pdb target.pdb --binder-chain A --target-chain B --binder-len 80 --n-samples 5
```

`scripts/make_all_figures.py` requires the output tables in `results/tables/` and a cleaned preference-pair dataset (see `data/train_pairs_clean.json`); adjust the `F` variable at the top of the script to point to your data.

## Model

Built on ProteinMPNN (Dauparas et al., 2022), fine-tuned via a two-stage preference-optimization and supervised fine-tuning procedure described in the manuscript.

## Citation

Citation details will be added upon publication.

## License

MIT — see LICENSE.
