# Report-to-code experiment map

| Report section | Unified support | Original evidence |
|---|---|---|
| Dataset and Exploratory Analysis | `dataset.py`, `prepare_data.py`, notebook 01 | NinhMinhHieu notebook |
| Data Preprocessing | `audio.py`, `spectrogram.py`, notebooks 01–02 | Hieu, Quyen, Hung, Quan |
| Model Architecture | `models.py`, baseline/SED scripts, notebooks 02–03 | Quyen, Hung, Quan, Huy |
| Training Strategy | `losses.py`, `training.py`, `config.yaml` | Quyen AUC study; Hung and Quan BCE/focal training |
| Semi-Supervised and Teacher-Student | `pseudo_labeling.py`, `distillation.py`, pseudo/student scripts, notebooks 04–05 | Quyen and Hung pseudo-label pipelines; Quan student loss |
| Inference and Post-Processing | `ensemble.py`, ensemble script, notebook 06 | Quan regular/shifted OpenVINO ensemble; Huy power adjustment |
| Experimental Results | `metrics.py`, `evaluate.py`, `reports/tables/report_results.csv` | Group report tables and member plots |

## Important interpretation notes

The report's ablation and final metrics are transcribed as reported results, not regenerated results. Reproduction requires the original dataset splits and checkpoints, which are not committed. Macro AUC excludes classes without both positive and negative validation examples, matching the report.

The unified implementation does not claim every notebook variant is behavior-identical. It standardizes shared concepts; experiment-specific details such as SoftAUC, voice activity inspection, exact OpenVINO graphs, and staged frame-level pseudo-label datasets remain in `legacy/` for manual review.
