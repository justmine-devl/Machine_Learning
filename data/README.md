# Data placement

The BirdCLEF dataset is not redistributed here. The default root configuration expects:

```text
data/
|-- train_metadata.csv
|-- train_audio/
|   |-- <primary_label>/<filename>.ogg
|   `-- <filename>.ogg              # flat layout is also supported
|-- unlabeled_soundscapes/
|-- test_soundscapes/
`-- sample/                         # tiny redistributable examples only
```

Required metadata columns are `filename` and `primary_label`. `secondary_labels` is optional and may contain a stringified Python list. Run:

```bash
python scripts/prepare_data.py --config config.yaml
```

This writes `metadata_with_folds.csv`, `classes.txt`, and `dataset_summary.csv` to `outputs/processed/` by default. Experiment YAML files may use different paths; their `data` sections are authoritative.

Raw audio and generated datasets are ignored by Git.
