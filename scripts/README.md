# Scripts

```bash
python scripts/prepare_data.py --config config.yaml
python scripts/train_baseline.py --config config.yaml
python scripts/train_sed.py --config config.yaml
python scripts/generate_pseudo_labels.py --config config.yaml --checkpoint outputs/sed/best.pt
python scripts/train_student.py --config config.yaml --soft-targets outputs/pseudo/soft_targets.npy
python scripts/evaluate.py --pred outputs/preds.npy --target outputs/targets.npy --classes outputs/classes.txt
python scripts/infer_ensemble.py --config config.yaml --models-dir weights/openvino --audio-dir data/test_audio
python scripts/make_report_plots.py --metrics outputs/metrics_history.csv --out-dir reports/figures
```
