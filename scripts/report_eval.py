"""
Aggregate per-subject evaluation NPZ files and report mean metrics.

Each eval_*.py saves:
    PATH_ASSETS/results2/{exp_tag}/{subject_tag}.npz           (model-based methods)
    PATH_ASSETS/results2/{exp_tag}/{subject_tag}_tracking.npz  (spatial tracker)
    PATH_ASSETS/results2/{exp_tag}/{subject_tag}_pointmap.npz  (spatial tracker)

Body pose keys (all methods):
    framewise_mpjpe                     : (N,) MPJPE in metres
    framewise_pa_mpjpe                  : (N,) PA-MPJPE in metres

Blade keys (spatial tracker only):
    framewise_blade_chamfer_distance          : (N,) Chamfer distance in metres
    framewise_blade_f1_score                  : (N,) F1 score [0, 1]
    framewise_blade_chamfer_distance_centered : (N,) centred Chamfer distance in metres
    framewise_blade_f1_score_centered         : (N,) centred F1 score [0, 1]

Usage:
    python report_eval.py --exp_tag ajahr
    python report_eval.py --exp_tag motionbert
    python report_eval.py --exp_tag spatial_tracker_v2
"""

import sys
sys.path.append('..')

import os
import glob
import numpy as np

from utils.constants import PATH_ASSETS
from utils.eval_utils import eval_mean_valid

# Static display config for known metric keys: (scale, unit, column label)
METRIC_CONFIG = {
    'mpjpe':                           (1000, 'mm', 'MPJPE'),
    'pa_mpjpe':                        (1000, 'mm', 'PA-MPJPE'),
    'blade_chamfer_distance':          (1000, 'mm', 'CD'),
    'blade_chamfer_distance_centered': (1000, 'mm', 'CD-cent'),
    # backward-compat single-threshold keys
    'blade_f1_score':                  (100,  '%',  'F1'),
    'blade_f1_score_centered':         (100,  '%',  'F1-cent'),
}

BODY_METRICS = ['mpjpe', 'pa_mpjpe']


def _metric_cfg(key):
    """Return (scale, unit, label) for any metric key, including dynamic per-threshold F1 keys."""
    if key in METRIC_CONFIG:
        return METRIC_CONFIG[key]
    # Dynamic blade F1 score keys: blade_f1_score_100, blade_f1_score_centered_30, etc.
    if 'f1_score' in key:
        thresh_mm = key.rsplit('_', 1)[-1]
        if 'centered' in key:
            return (100, '%', f'F1-c@{thresh_mm}')
        return (100, '%', f'F1@{thresh_mm}')
    return (1, '', key)


def _blade_sort_key(k):
    """Sort blade metrics: non-centered group first (CD then F1@…), then centered group (CD-cent then F1-c@…).
    Within each F1 group, sort descending by threshold (largest first)."""
    is_centered = 'centered' in k
    is_f1 = 'f1_score' in k
    try:
        thresh = int(k.rsplit('_', 1)[-1])
    except ValueError:
        thresh = 0
    return (is_centered, is_f1, -thresh)


def _discover_blade_metrics(sample_keys):
    """Return ordered blade metric keys present in sample_keys (framewise_* stripped)."""
    blade_keys = [
        k.replace('framewise_', '')
        for k in sample_keys
        if k.startswith('framewise_blade_')
    ]
    return sorted(blade_keys, key=_blade_sort_key)



def print_table(title, metric_keys, rows, agg_fw, valid_counts):
    """Print a section table for a group of metrics.

    rows        : list of (stem, n_total, values)
    agg_fw      : {'n': n_total_all, k: mean_value, ...}
    valid_counts: {k: n_valid_all} — valid frame count per metric across all subjects
    """
    cfg = [_metric_cfg(k) for k in metric_keys]
    col_w = 10
    name_w = 16
    n_w = 8

    header = f"{'':>{name_w}} {'N':>{n_w}}" + "".join(
        f"  {f'{c[2]} ({c[1]})':>{col_w}}" for c in cfg
    )
    sep = "=" * len(header)

    print(f"\n{title}")
    print(sep)
    print(header)
    print("-" * len(header))

    for subject_tag, n_total, values, valid_per_metric in rows:
        row = f"{subject_tag:<{name_w}} {n_total:>{n_w}}"
        for (scale, _, _), v in zip(cfg, values):
            row += f"  {v * scale:>{col_w}.2f}"
        print(row)
        n_invalid = {k: n_total - valid_per_metric[k] for k in metric_keys}
        if any(v > 0 for v in n_invalid.values()):
            valid_row = f"  {'invalid':<{name_w - 2}} {'':>{n_w}}"
            for k in metric_keys:
                label = str(n_invalid[k]) if n_invalid[k] > 0 else ''
                valid_row += f"  {label:>{col_w}}"
            print(valid_row)

    print(sep)

    fw_row = f"{'Mean (by frame)':<{name_w}} {agg_fw['n']:>{n_w}}"
    for k, (scale, _, _) in zip(metric_keys, cfg):
        fw_row += f"  {agg_fw[k] * scale:>{col_w}.2f}"
    print(fw_row)

    n_invalid_all = {k: agg_fw['n'] - valid_counts[k] for k in metric_keys}
    if any(v > 0 for v in n_invalid_all.values()):
        valid_row = f"{'Invalid samples':<{name_w}} {'':>{n_w}}"
        for k in metric_keys:
            label = str(n_invalid_all[k]) if n_invalid_all[k] > 0 else ''
            valid_row += f"  {label:>{col_w}}"
        print(valid_row)

    print(sep)


def _avg_blade_pairs(fw, n_frames):
    """Average alternating blade1/blade2 entries [b1_f0, b2_f0, b1_f1, b2_f1, ...]
    into one per-frame value. Invalid (-1) entries are excluded from the average."""
    arr = np.array(fw, dtype=float).reshape(n_frames, 2)
    v1, v2 = arr[:, 0], arr[:, 1]
    both  = (v1 >= 0) & (v2 >= 0)
    only1 = (v1 >= 0) & (v2 < 0)
    only2 = (v1 < 0)  & (v2 >= 0)
    result = np.full(n_frames, -1.0)
    result[both]  = (v1[both] + v2[both]) / 2
    result[only1] = v1[only1]
    result[only2] = v2[only2]
    return result


def main():
    results_dir = os.path.join(PATH_ASSETS, 'eval')
    if not os.path.exists(results_dir):
        print(f"Error: results directory not found: {results_dir}")
        return

    npz_paths = sorted(glob.glob(os.path.join(results_dir, '*.npz')))
    npz_paths = [p for p in npz_paths if os.path.basename(p) != 'all_subjects.npz']

    if not npz_paths:
        print(f"No per-subject NPZ files found in {results_dir}")
        return

    print(f"\nExp: {len(npz_paths)} file(s) found")

    # Load all NPZs
    loaded = []
    for path in npz_paths:
        subject_tag = os.path.splitext(os.path.basename(path))[0]
        data = np.load(path)
        loaded.append((subject_tag, data))

    # Determine which metric groups are present
    sample_keys = set(loaded[0][1].keys())
    blade_metrics = _discover_blade_metrics(sample_keys)

    active_groups = [('Body pose metrics', BODY_METRICS)]
    if blade_metrics:
        active_groups.append(('Blade metrics', blade_metrics))

    all_agg_fw = {}  # accumulate means across groups for LaTeX line

    for title, metric_keys in active_groups:
        rows = []
        per_subject = {k: [] for k in metric_keys}  # k -> list of framewise arrays

        for subject_tag, data in loaded:
            n_frames = len(data['framewise_mpjpe'])
            values = []
            valid_per_metric = {}
            for k in metric_keys:
                fw = data[f'framewise_{k}']

                # If the blade metrics are stored as alternating blade1/blade2 values, average them into one per-frame value.
                if len(fw) == 2 * n_frames:
                    fw = _avg_blade_pairs(fw, n_frames)
                per_subject[k].append(fw)
                values.append(eval_mean_valid(fw))
                valid_per_metric[k] = int(np.sum(fw >= 0))
            rows.append((subject_tag, n_frames, values, valid_per_metric))

        # Frame-weighted aggregate
        cat = {k: np.concatenate(per_subject[k]) for k in metric_keys}
        agg_fw = {k: eval_mean_valid(cat[k]) for k in metric_keys}
        agg_fw['n'] = len(cat[metric_keys[0]])

        # Per-metric valid sample counts
        valid_counts = {k: int(np.sum(cat[k] >= 0)) for k in metric_keys}

        all_agg_fw.update(agg_fw)
        print_table(title, metric_keys, rows, agg_fw, valid_counts)


    print()

    # LaTeX row: MPJPE & PA-MPJPE & CD & F1@100 & F1@50 & CD-cent & F1-c@50 & F1-c@30
    latex_cols = [
        ('mpjpe',                           1000, '.2f'),
        ('pa_mpjpe',                        1000, '.2f'),
        ('blade_chamfer_distance',          1000, '.2f'),
        ('blade_f1_score_100',               100, '.2f'),
        ('blade_f1_score_50',                100, '.2f'),
        ('blade_chamfer_distance_centered', 1000, '.2f'),
        ('blade_f1_score_centered_50',       100, '.2f'),
        ('blade_f1_score_centered_30',       100, '.2f'),
    ]
    latex_vals = []
    for k, scale, fmt in latex_cols:
        v = all_agg_fw.get(k, float('nan'))
        latex_vals.append(format(v * scale, fmt))
    print('LaTeX row:')
    print(' & '.join(latex_vals) + ' \\\\')


if __name__ == '__main__':
    main()
