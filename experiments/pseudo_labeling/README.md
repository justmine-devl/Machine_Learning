# Pseudo-labeling

First run `generate_pseudo_labels.py`, then `train_with_pseudo_labels.py` with `config_pseudo.yaml`. Selected windows retain their audio path and chunk index; training mixes soft pseudo targets with labeled recordings using `pseudo_labeling.labeled_ratio`.
