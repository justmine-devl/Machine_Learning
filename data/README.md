# Data placement

The BirdCLEF dataset is not redistributed in this repository. Configure paths in `config.yaml` and use this local layout:

```text
data/
├── metadata.csv
├── classes.txt
├── audio/
│   ├── species_id/recording.ogg   # class folders are supported
│   └── recording.ogg              # a flat directory is also supported
├── unlabeled_audio/
└── sample/                        # only tiny, license-compatible examples
```

Required metadata columns are `primary_label` plus `filename` or `filepath`. `secondary_labels` is optional. `scripts/prepare_data.py` creates recording-level train/validation CSVs and a sorted class list under `data/processed/`.

Raw audio, generated arrays, and processed datasets are ignored by Git. Verify the source dataset's license before adding any sample.
