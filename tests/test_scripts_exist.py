from pathlib import Path


def test_expected_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    expected = [
        'prepare_data.py',
        'train_baseline.py',
        'train_sed.py',
        'generate_pseudo_labels.py',
        'train_student.py',
        'evaluate.py',
        'infer_ensemble.py',
        'make_report_plots.py',
    ]
    for name in expected:
        assert (root / 'scripts' / name).exists()
