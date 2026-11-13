# Training data

`train_pairs_clean.json` contains the 3,680 verified preference pairs used for training. Each record references a backbone structure under `data/training_structures/`.

The 1,939 unique training backbone structures (481 MB uncompressed) are distributed as a split gzip archive to stay within GitHub file-size limits. To reconstruct:

    cd data
    cat training_structures.tar.gz.part_* > training_structures.tar.gz
    tar -xzf training_structures.tar.gz
    rm training_structures.tar.gz training_structures.tar.gz.part_*

This creates `data/training_structures/` with all 1,939 PDB files, matching the paths recorded in `train_pairs_clean.json`.
