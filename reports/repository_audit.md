# Pre-refactor repository audit

The audit covered the 48 files originally stored in five member folders.

## Inventory by member

| Member | Files | Classification |
|---|---:|---|
| Hoang Bao Huy | 3 | 2 notebooks, 1 requirements file; voice/noise analysis, post-processing, SED inference and plots |
| Le The Quyen | 2 | 1 notebook, 1 requirements file; preprocessing cache, model/training, AUC losses, pseudo-labeling and inference |
| Nguyen Cong Hung | 19 | 17-package/script files plus runner and requirements; configuration, paths, audio, datasets, models, losses/metrics, training, pseudo-labeling, inference and plotting |
| Nguyen Hoang Quan | 22 | Installable-style SED project with 7 modules, 3 scripts, 2 tests, config/docs, one training notebook, and placeholders |
| Ninh Minh Hieu | 2 | 1 analysis/end-to-end notebook, 1 requirements file |

## Overlap and duplication

- Four implementations load/resample audio and crop or pad it to five seconds.
- Quyen, Hung, Quan, and Huy each define a Timm/SED-style model; multiple versions repeat attention blocks and initialization helpers.
- BCE/focal loss, sigmoid inference, AUC calculation, and threshold metrics appear in several locations.
- Quyen, Hung, and Quan independently implement pseudo-label or teacher/student logic.
- Huy and Quan both implement ensemble/post-processing concepts; Quan includes the regular/shifted OpenVINO flow and Huy explores power adjustment.

The unified package selects small report-level interfaces. Behavior-specific originals remain under `legacy/` when merging would alter experiment semantics.

## Paths, imports, and dependencies

- All four original notebooks with competition data references contain hard-coded `/kaggle/input` or `/kaggle/working` paths. Huy's post-processing notebook also references `/kaggle/input/.../sed{i}.pth`.
- Cong Hung's `paths.py` searches fixed Kaggle locations as fallbacks. This is useful on Kaggle but unsuitable as the unified default.
- Hieu, Huy, and Quyen each had the same generic `requirements.txt` (`openai`, pandas, plotting tools), which does not describe their notebook imports such as Torch, librosa, timm, OpenCV, and soundfile.
- Quan and Hung had the most complete dependency lists; these were reconciled in the root requirements. OpenVINO is optional because it is only needed for deployment inference.
- Original notebook code is stateful and contains environment-install cells, so imports can fail when cells are run out of order. The unified notebooks import tested modules instead.

## Artifacts and migration decisions

- Embedded notebook outputs/plots make Huy's voice-separation notebook roughly 22 MB; they are preserved only as original research evidence.
- No datasets or model checkpoints were present in the original tracked folders.
- Placeholder result directories and notebook caches are not part of the new implementation.
- The original five folders were copied to `legacy_original_snapshot/` before any write, verified by file count and total byte count, then moved unchanged into `legacy/`.
