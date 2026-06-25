"""
Evaluation script for scripts_release/alignment_new.py output.

Loads per-clip H5 results produced by alignment_new.py and evaluates against
ground truth from the released RSP3D dataset:
  - Body : MPJPE and PA-MPJPE
  - Blade: Chamfer distance and F1 score (root-relative and centred variants)
"""

import sys
sys.path.append('.')
sys.path.append('..')

import os
import numpy as np
import h5py
import argparse
from tqdm import tqdm

from utils.io_utils import prepare_h5_reader, load_frame_from_h5
from data.dataset import RSP3D
from utils.constants import (PATH_DATASET, PATH_ASSETS,
                              H36M_JOINTS, INVALID_JOINTS_BY_SUBJECT,
                              get_eval_model_free_cfg)
from utils.eval_utils import eval_3d_pose, eval_blade, eval_mean_valid


def main():
    parser = argparse.ArgumentParser(description='Evaluate alignment_new.py output')
    parser.add_argument('--subject_tag', type=str, default=None,
                        help='Only evaluate this subject prefix, e.g. P8')
    parser.add_argument('--step', type=int, default=1,
                        help='Frame sampling step')
    args = parser.parse_args()

    cfg = get_eval_model_free_cfg(args)

    # Per-subject validity mask (exclude amputated joints)
    valid_eval = {}
    for csubject, list_joints in INVALID_JOINTS_BY_SUBJECT.items():
        cvalid = np.ones(len(H36M_JOINTS), dtype=bool)
        cvalid[[k for k, v in H36M_JOINTS.items() if v in list_joints]] = False
        valid_eval[csubject] = cvalid

    results_all = {
        'mpjpe': [], 'pa_mpjpe': [],
        'blade_chamfer_distance': [], 'blade_chamfer_distance_centered': [],
    }
    for t in cfg.blade_chamfer_threshold:
        results_all[f'blade_f1_score_{int(round(t * 1000))}'] = []
    for t in cfg.blade_chamfer_threshold_centered:
        results_all[f'blade_f1_score_centered_{int(round(t * 1000))}'] = []
    num_missing = 0

    print("Loading dataset metadata...")
    dataset = RSP3D(root_release=PATH_DATASET, subject_tag=args.subject_tag)
    print(f"Found {len(dataset.actionlist)} clips")

    for caction in dataset.actionlist:
        subject_id  = caction['subject_id']
        camera_id   = caction['camera_id']
        action_name = caction['action_name']
        meta_key    = caction['meta_key']

        print(f"\n{'='*80}")
        print(f"  {subject_id} / {camera_id} / {action_name}")
        print(f"{'='*80}")

        h5_path = os.path.join(PATH_ASSETS, 'results', 'hybrid_alignment',
                               subject_id, f"{meta_key}.h5")
        if not os.path.exists(h5_path):
            print(f"  Skipping: H5 not found: {h5_path}")
            continue

        datalist = dataset.load_data_for_clip(meta_key, step_frame=args.step)
        print(f"  {len(datalist)} frames")

        camera = dataset.meta_info[meta_key]['camera']
        intr = np.array([
            [camera.fx, 0,         camera.cx],
            [0,         camera.fy, camera.cy],
            [0,         0,         1        ],
        ], dtype=np.float32)

        with h5py.File(h5_path, 'r') as h5f:
            reader = prepare_h5_reader(h5f)

            for data in tqdm(datalist, desc="Evaluating"):
                clip_frame_idx = data['clip_frame_idx']
                csubject       = data['subject_id']

                frame_data = load_frame_from_h5(h5f, clip_frame_idx, reader)
                if frame_data is None:
                    num_missing += 1
                    continue

                body_3d_est  = frame_data['body_3d_est']   # (17, 3)
                blade_3d_est = frame_data['blade_3d_est']  # (N, 3)

                # GT — transform to camera space
                skeleton_3d_gt = data['skeleton_3d'].copy()   # (17, 3)
                skeleton_valid = np.where(np.isnan(skeleton_3d_gt[:, 0]), 0, 1).astype(np.float32)
                blade_3d_gt    = data['blade_3d'].copy()       # (M, 2, 3)
                blade_valid_gt = np.where(np.isnan(blade_3d_gt[..., 0]), 0, 1).astype(np.float32)

                if camera.has_extrinsics:
                    skeleton_3d_gt = camera.world_to_camera(skeleton_3d_gt)
                    blade_3d_gt    = camera.world_to_camera(
                        blade_3d_gt.reshape(-1, 3)).reshape(-1, 2, 3)

                has_blade2 = 'blade2_3d' in data
                if has_blade2:
                    blade2_3d_gt    = data['blade2_3d'].copy()
                    blade2_valid_gt = np.where(np.isnan(blade2_3d_gt[..., 0]), 0, 1).astype(np.float32)
                    if camera.has_extrinsics:
                        blade2_3d_gt = camera.world_to_camera(
                            blade2_3d_gt.reshape(-1, 3)).reshape(-1, 2, 3)
                    blade2_3d_est = frame_data.get('blade2_3d_est', None)

                skeleton_valid_eval = np.logical_and(
                    skeleton_valid.astype(bool), valid_eval[csubject.split('_')[0]])
                eval_joints = skeleton_valid_eval.nonzero()[0].tolist()

                # Body metrics
                mpjpe, pa_mpjpe = eval_3d_pose(
                    pred=body_3d_est[np.newaxis],
                    target=skeleton_3d_gt[np.newaxis],
                    root_idx=cfg.root_idx,
                    eval_joints=eval_joints,
                    valid_joints=None,
                )
                results_all['mpjpe'].append(mpjpe[0])
                results_all['pa_mpjpe'].append(pa_mpjpe[0])

                # Blade metrics
                blade_results = eval_blade(
                    blade_3d_gt=blade_3d_gt,
                    blade_valid_gt=blade_valid_gt,
                    blade_3d_est=blade_3d_est,
                    skeleton_3d_gt=skeleton_3d_gt,
                    skeleton_3d_est=body_3d_est,
                    root_idx=cfg.root_idx,
                    num_samples_along=cfg.blade_num_samples_along,
                    num_samples_across=cfg.blade_num_samples_across,
                    thre_fscore=cfg.blade_chamfer_threshold,
                    thre_fscore_centered=cfg.blade_chamfer_threshold_centered,
                )
                for k, v in blade_results.items():
                    results_all[f'blade_{k}'].append(v)

                if has_blade2 and blade2_3d_est is not None:
                    blade2_results = eval_blade(
                        blade_3d_gt=blade2_3d_gt,
                        blade_valid_gt=blade2_valid_gt,
                        blade_3d_est=blade2_3d_est,
                        skeleton_3d_gt=skeleton_3d_gt,
                        skeleton_3d_est=body_3d_est,
                        root_idx=cfg.root_idx,
                        num_samples_along=cfg.blade_num_samples_along,
                        num_samples_across=cfg.blade_num_samples_across,
                        thre_fscore=cfg.blade_chamfer_threshold,
                        thre_fscore_centered=cfg.blade_chamfer_threshold_centered,
                    )
                    for k, v in blade2_results.items():
                        results_all[f'blade_{k}'].append(v)

    if num_missing > 0:
        print(f"\nWarning: {num_missing} frames missing in H5")

    n_frames = len(results_all['mpjpe'])
    subject_tag = args.subject_tag if args.subject_tag else 'all'
    print(f"\n{'='*60}")
    print(f"Evaluation Results — hybrid_alignment  [{subject_tag}]")
    print(f"  Frames evaluated : {n_frames}")
    print(f"  MPJPE            : {eval_mean_valid(results_all['mpjpe'])*1000:.2f} mm")
    print(f"  PA-MPJPE         : {eval_mean_valid(results_all['pa_mpjpe'])*1000:.2f} mm")
    print(f"  Blade CD         : {eval_mean_valid(results_all['blade_chamfer_distance'])*1000:.2f} mm")
    for t in cfg.blade_chamfer_threshold:
        k = f'blade_f1_score_{int(round(t * 1000))}'
        print(f"  Blade F1@{int(round(t*1000)):3d}mm   : {eval_mean_valid(results_all[k]):.4f}")
    print(f"  Blade CD (cent)  : {eval_mean_valid(results_all['blade_chamfer_distance_centered'])*1000:.2f} mm")
    for t in cfg.blade_chamfer_threshold_centered:
        k = f'blade_f1_score_centered_{int(round(t * 1000))}'
        print(f"  Blade F1-c@{int(round(t*1000)):3d}mm : {eval_mean_valid(results_all[k]):.4f}")
    print(f"{'='*60}")

    # Save numeric results
    save_path = os.path.join(PATH_ASSETS, 'eval', f'{subject_tag}.npz')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict = {}
    for metric_name, values in results_all.items():
        save_dict[f'framewise_{metric_name}'] = np.array(values)
        save_dict[f'mean_{metric_name}']      = eval_mean_valid(values)
        save_dict[f'nsamples_{metric_name}']  = len(values)
    save_dict['blade_chamfer_thresholds']          = np.array(cfg.blade_chamfer_threshold)
    save_dict['blade_chamfer_thresholds_centered'] = np.array(cfg.blade_chamfer_threshold_centered)
    np.savez(save_path, **save_dict)
    print(f"  Saved → {save_path}")


if __name__ == '__main__':
    main()
