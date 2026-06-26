
import sys
sys.path.append('.')
sys.path.append('..')
import os
import numpy as np
import h5py
import cv2
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


from utils.constants import PATH_ASSETS, PATH_DATASET, H36M_JOINTS, INVALID_JOINTS_BY_SUBJECT, H36M_JOINT_PAIRS, get_eval_model_free_cfg
from utils.io_utils import prepare_h5_reader, load_frame_from_h5
from data.dataset import RSP3D, create_clip_subset
from data.io_utils import load_frame
from data.sequence_dataloader import create_sequence_dataloader
from utils.vis_utils import visualize_skeleton_2d
from utils.camera import apply_intrinsics_transform_to_2d, project_3d_with_intrinsics
from utils.align_utils import load_model_base_and_preprocess, extract_3d_from_point_map, align_with_bbox_scale_and_pnp, align_with_root_aligned_and_scale





def visualize_alignment(image_rgb, body_3d_model_base, blade_3d_est,
                        skeleton_valid, intr_resized, frame_idx, vis_dir,
                        crop_bbox, resize_scale, blade2_3d_est=None):
    # --- Prepare cropped image ---
    x_min, y_min, x_max, y_max = crop_bbox
    scale_fx, scale_fy = resize_scale
    crop = image_rgb[int(y_min):int(y_max), int(x_min):int(x_max)]
    crop_h = int(round((y_max - y_min) * scale_fy))
    crop_w = int(round((x_max - x_min) * scale_fx))
    crop = cv2.resize(crop, (crop_w, crop_h))
    img_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_bgr.shape[:2]

    # --- 2D Projection ---
    body_2d_model_base = project_3d_with_intrinsics(body_3d_model_base, intr_resized)  # (17, 2)
    blade_2d_est       = project_3d_with_intrinsics(blade_3d_est,        intr_resized)  # (N, 2)

    img_vis = img_bgr.copy()
    img_vis = visualize_skeleton_2d(img_vis, body_2d_model_base, skeleton_valid,
                                    joint_color=(200, 200, 0), bone_color=(200, 200, 0))
    for pt in blade_2d_est:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img_w and 0 <= y < img_h:
            cv2.circle(img_vis, (x, y), 3, (255, 255, 0), -1)
    if blade2_3d_est is not None and len(blade2_3d_est) > 0:
        blade2_2d_est = project_3d_with_intrinsics(blade2_3d_est, intr_resized)
        for pt in blade2_2d_est:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < img_w and 0 <= y < img_h:
                cv2.circle(img_vis, (x, y), 3, (0, 165, 255), -1)

    os.makedirs(vis_dir, exist_ok=True)
    cv2.imwrite(os.path.join(vis_dir, f'frame_{frame_idx:04d}_2d.jpg'), img_vis)

    # --- 3D Plot ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    body_mb_vis = body_3d_model_base[skeleton_valid > 0.5]
    if len(body_mb_vis) > 0:
        ax.scatter(body_mb_vis[:, 0], body_mb_vis[:, 2], -body_mb_vis[:, 1],
                   c='cyan', s=40, alpha=0.9, marker='^', label='model_base body')
        for ja, jb in H36M_JOINT_PAIRS:
            if skeleton_valid[ja] and skeleton_valid[jb]:
                pts = body_3d_model_base[[ja, jb]]
                ax.plot(pts[:, 0], pts[:, 2], -pts[:, 1], 'c-', linewidth=1.5, alpha=0.7)

    if len(blade_3d_est) > 0:
        ax.scatter(blade_3d_est[:, 0], blade_3d_est[:, 2], -blade_3d_est[:, 1],
                   c='yellow', s=20, alpha=0.7, label='blade1')

    if blade2_3d_est is not None and len(blade2_3d_est) > 0:
        ax.scatter(blade2_3d_est[:, 0], blade2_3d_est[:, 2], -blade2_3d_est[:, 1],
                   c='orange', s=20, alpha=0.7, label='blade2')

    pelvis = body_3d_model_base[0]
    pad = 1.5
    ax.set_xlim3d([pelvis[0] - pad, pelvis[0] + pad])
    ax.set_ylim3d([pelvis[2] - pad, pelvis[2] + pad])
    ax.set_zlim3d([-pelvis[1] - 1.2, -pelvis[1] + 1.2])
    ax.set_xlabel('X (right)')
    ax.set_ylabel('Z (depth)')
    ax.set_zlabel('Y (up)')
    ax.set_title(f'Frame {frame_idx} - Alignment')
    ax.view_init(elev=5, azim=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fig.savefig(os.path.join(vis_dir, f'frame_{frame_idx:04d}_3d.png'), dpi=100)
    plt.close(fig)



def write_alignment_h5(dst_path: str, frame_indices: list, results: list):
    """
    Write per-frame alignment results to H5.

    frame_indices : list of clip_frame_idx (int)
    results       : list of dicts, each with:
                      body_3d_est   (17, 3) float32
                      blade_3d_est  (K, 3)  float32  — K may vary
                      blade2_3d_est (M, 3)  float32  — optional
    """
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    has_blade2 = any('blade2_3d_est' in r for r in results)

    with h5py.File(dst_path, 'w') as h5f:
        h5f.create_dataset('frame_indices',
                           data=np.array(frame_indices, dtype=np.int64))

        # Fixed-shape: body pose
        body_arr = np.stack([r['body_3d_est'] for r in results]).astype(np.float32)
        h5f.create_dataset('body_3d_est', data=body_arr,
                           compression='gzip', compression_opts=4)

        # Variable-length: blade(s)
        vl = h5f.create_group('variable_length')
        for key in (['blade_3d_est'] + (['blade2_3d_est'] if has_blade2 else [])):
            parts   = [r.get(key, np.empty((0, 3), dtype=np.float32)) for r in results]
            lengths = np.array([len(p) for p in parts], dtype=np.int64)
            flat    = np.concatenate(parts) if any(len(p) > 0 for p in parts) \
                      else np.empty((0, 3), dtype=np.float32)
            kg = vl.create_group(key)
            kg.create_dataset('data',    data=flat.astype(np.float32),
                              compression='gzip', compression_opts=4)
            kg.create_dataset('lengths', data=lengths)


def load_alignment_h5(dst_path: str) -> dict:
    """Load per-frame results from H5; returns dict mapping clip_frame_idx (int) -> result dict."""
    if not os.path.exists(dst_path):
        return {}
    results = {}
    with h5py.File(dst_path, 'r') as h5f:
        frame_indices = h5f['frame_indices'][:]
        body_arr      = h5f['body_3d_est'][:]

        vl           = h5f['variable_length']
        blade_data   = vl['blade_3d_est']['data'][:]
        blade_lens   = vl['blade_3d_est']['lengths'][:]
        has_blade2   = 'blade2_3d_est' in vl
        if has_blade2:
            blade2_data = vl['blade2_3d_est']['data'][:]
            blade2_lens = vl['blade2_3d_est']['lengths'][:]

        b_off, b2_off = 0, 0
        for i, fidx in enumerate(frame_indices):
            r = {
                'body_3d_est':  body_arr[i],
                'blade_3d_est': blade_data[b_off: b_off + blade_lens[i]],
            }
            b_off += blade_lens[i]
            if has_blade2:
                r['blade2_3d_est'] = blade2_data[b2_off: b2_off + blade2_lens[i]]
                b2_off += blade2_lens[i]
            results[int(fidx)] = r
    return results


def align_segment(recording_dataloader, h5f_model_free, h5f_model_base, dir_output, cfg, verbose=False, recording_dataset=None):
    reader_mf = prepare_h5_reader(h5f_model_free)
    reader_mb = prepare_h5_reader(h5f_model_base)

    # Keys needed from model_base (only joints, no vertices etc.)
    keys_mb = ['joints3d_h36m', 'scaled_focal_length', 'focal_length']
    # Keys needed from model_free frame-0 (track correspondence setup)
    keys_mf_frame0 = ['scale_fx', 'scale_fy', 'mask_bbox', 'body_tracks_3d', 'intrinsics']
    # Load all keys from model_free per-frame (needed for point-map extraction)
    keys_mf_frame = None

    # Determine output path from recording_dataset (set at clip level)
    caction    = recording_dataset.actionlist[0]
    subject_id_clip = caction['subject_id']
    meta_key_clip   = caction['meta_key']
    dst_h5_path = os.path.join(dir_output, subject_id_clip, f"{meta_key_clip}.h5")


    all_frame_indices: list = []
    all_results:       list = []

    for batch_idx, (input_batch, meta_batch, gt_batch) in enumerate(tqdm(recording_dataloader, desc="Aligning batches")):
        subject_id = meta_batch['subject_id'][0]
        camera_id = meta_batch['camera_id'][0]

        # Determine valid joints based on subject-specific amputations
        valid_for_eval = np.ones(len(gt_batch['skeleton_3d'][0]), dtype=bool)
        invalid_joint_idx = [k for k, v in H36M_JOINTS.items() if v in INVALID_JOINTS_BY_SUBJECT[subject_id.split('_')[0]]]
        for invalid_idx in invalid_joint_idx:
            valid_for_eval[invalid_idx] = 0

        frame0_idx = int(meta_batch['clip_frame_idx'][0])
        frame0_est_model_free = load_frame_from_h5(h5f_model_free, frame0_idx, reader_mf, keys=keys_mf_frame0)
        frame0_est_model_base = load_frame_from_h5(h5f_model_base, frame0_idx, reader_mb, keys=keys_mb)
        if frame0_est_model_free is None or frame0_est_model_base is None:
            print(f"  Warning: frame {frame0_idx} missing in H5, skipping batch")
            continue

        # Reconstruct intrinsics transform from model-free metadata
        scale_fx = float(frame0_est_model_free['scale_fx'])
        scale_fy = float(frame0_est_model_free['scale_fy'])
        mask_x_min, mask_y_min, mask_x_max, mask_y_max = frame0_est_model_free['mask_bbox']
        intrinsics_transform = np.array([[scale_fx, 0, -scale_fx * mask_x_min],
                                         [0, scale_fy, -scale_fy * mask_y_min],
                                         [0, 0, 1]], dtype=np.float32)
        intr_resized = intrinsics_transform @ input_batch['intrinsics'][0].numpy()


        batch_ref_body_2d = apply_intrinsics_transform_to_2d(points_2d=input_batch['joints2d'].numpy(),
                                                            intrinsics_transform=intrinsics_transform)

        has_blade2 = 'blade2_3d' in gt_batch
        # first interation to load all frames and compute scale_ratio, second iteration to do alignment with loaded data and save results



        for iidx in range(len(meta_batch['frame_idx'])):
            gt_skeleton_valid = gt_batch['skeleton_valid'][iidx].numpy()  # (17,)
            gt_skeleton_valid = np.logical_and(gt_skeleton_valid > 0.5, valid_for_eval)  # (17,)
            valid_for_align = valid_for_eval

            frame_idx      = int(meta_batch['frame_idx'][iidx])
            clip_frame_idx = int(meta_batch['clip_frame_idx'][iidx])


            frame_data_model_free = load_frame_from_h5(h5f_model_free, clip_frame_idx, reader_mf, keys=keys_mf_frame)
            frame_data_model_base = load_frame_from_h5(h5f_model_base, clip_frame_idx, reader_mb, keys=keys_mb)

            intr_est = frame_data_model_free['intrinsics']

            # Load and preprocess model_base body for this frame (used as scale/alignment reference)
            body_3d_model_base, _, body_2d_on_crop_model_base, _ = load_model_base_and_preprocess(
                frame_est_model_base=frame_data_model_base,
                valid_joints= valid_for_eval,
                img_h=cfg.image_shape[0],
                img_w=cfg.image_shape[1],
                intr_original=input_batch['intrinsics'][iidx].numpy(),
                intr_transform=intrinsics_transform,
                root_idx=cfg.root_idx
            )


            # Step 1 — Extract 3D estimates from model_free point-map
            ref_body_2d = batch_ref_body_2d[iidx]  # (17, 2)
            if gt_batch['skeleton_valid'][iidx].numpy().sum() < 5:
                ref_body_2d = body_2d_on_crop_model_base
                valid_for_align = valid_for_eval
                print(f"  Warning: frame {frame_idx} has very few valid GT joints, using projected model_base joints as reference for 3D extraction")

            H, W = frame_data_model_free['depth'].shape[:2]
            mask_blade = input_batch['mask_blade'][iidx].numpy()
            mask_blade = mask_blade[int(mask_y_min):int(mask_y_max), int(mask_x_min):int(mask_x_max)]
            mask_blade = cv2.resize(mask_blade.astype(np.uint8), (W, H),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)

            mask_blade2_crop = None
            if has_blade2:
                mask_blade2_raw = input_batch['mask_blade2'][iidx].numpy()
                mask_blade2_raw = mask_blade2_raw[int(mask_y_min):int(mask_y_max), int(mask_x_min):int(mask_x_max)]
                mask_blade2_crop = cv2.resize(mask_blade2_raw.astype(np.uint8), (W, H),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)

            body_3d_est, blade_3d_est, blade2_3d_est = extract_3d_from_point_map(
                frame_data=frame_data_model_free,
                ref_skeleton_2d=ref_body_2d,
                skeleton_valid=valid_for_align,
                mask_blade=mask_blade,
                blade_grid_size=cfg.blade_grid_size,
                blade_depth_stability_patch=cfg.blade_depth_stability_patch,
                blade_depth_stability_thresh=cfg.blade_depth_stability_thresh,
                mask_blade2=mask_blade2_crop
            )

            # Step 3: Two-stage alignment
            # Concatenate blade and blade2 so the same transform is applied to both
            n_blade_pts = len(blade_3d_est)
            blade_all_3d_est = np.concatenate([blade_3d_est, blade2_3d_est], axis=0)  if has_blade2  else blade_3d_est
            

            # Stage 1 — rough scale via bounding-box comparison (brings scale close to 1)
            blade_all_3d_est, body_3d_est, _ = align_with_bbox_scale_and_pnp(blade_3d_est=blade_all_3d_est,
                                                        body_3d_est=body_3d_est,
                                                        intr_est=intr_est,
                                                        intr_resized=intr_resized,
                                                        skeleton_3d_ref=body_3d_model_base,
                                                        skeleton_valid=valid_for_align,
                                                        scale_ratio=None, verbose=False)

            blade_all_3d_est, body_3d_est = align_with_root_aligned_and_scale(body_3d_est=body_3d_est,
                                                                        blade_3d_est=blade_all_3d_est,
                                                                        body_3d_ref=body_3d_model_base,
                                                                        skeleton_valid=valid_for_align,
                                                                        root_idx=cfg.root_idx)

            blade_3d_est = blade_all_3d_est[:n_blade_pts]
            blade2_3d_est = blade_all_3d_est[n_blade_pts:] if has_blade2 else None


            # Accumulate results for H5 write at end of clip
            frame_result = {
                'body_3d_est':  body_3d_model_base,  # (17, 3)
                'blade_3d_est': blade_3d_est,         # (N_blade, 3)
            }
            if blade2_3d_est is not None:
                frame_result['blade2_3d_est'] = blade2_3d_est

            all_frame_indices.append(clip_frame_idx)
            all_results.append(frame_result)

            if verbose:
                meta_key  = meta_batch['meta_key'][iidx]
                clip_meta = recording_dataset.meta_info[meta_key]
                img_rgb   = load_frame(clip_meta['path_video'], clip_frame_idx)
                visualize_alignment(
                    image_rgb=img_rgb.copy(),
                    body_3d_model_base=body_3d_model_base,
                    blade_3d_est=blade_3d_est,
                    skeleton_valid=valid_for_align,
                    intr_resized=intr_resized,
                    frame_idx=frame_idx,
                    vis_dir=os.path.join('vis_alignment', f'{subject_id}_{camera_id}'),
                    crop_bbox=(mask_x_min, mask_y_min, mask_x_max, mask_y_max),
                    resize_scale=(scale_fx, scale_fy),
                    blade2_3d_est=blade2_3d_est if has_blade2 else None
                )

    if all_frame_indices:
        write_alignment_h5(dst_h5_path, all_frame_indices, all_results)
        print(f"  Saved {len(all_frame_indices)} frames → {dst_h5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run alignment on the released RSP3D dataset')
    parser.add_argument('--subject_tag', type=str, default=None, help='Subject tag filter, e.g., P2_1')
    parser.add_argument('--len_video_segment', type=int, default=60, help='Batch size (frames per segment)')
    parser.add_argument('--step', type=int, default=1, help='Step size for processing frames')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    cfg = get_eval_model_free_cfg(args)

    MODEL_FREE_TAG = 'spatial_tracker_v2'
    MODEL_BASE_TAG = 'sam3d_body'

    dir_output = os.path.join(PATH_ASSETS, 'results', 'hybrid_alignment')

    recording_dataset = RSP3D(
        root_release=PATH_DATASET,
        subject_tag=args.subject_tag,
        load_sam_masks=True,
        load_alphapose=True,
    )

    print(f"Found {len(recording_dataset.actionlist)} actions in dataset")

    for caction in recording_dataset.actionlist:
        print(f"\n{'='*80}")
        print(f"Processing action: {caction['action_name']}")
        print(f"  Subject: {caction['subject_id']}, Camera: {caction['camera_id']}")
        print(f"  Frames: {caction['start_frame_idx']} - {caction['end_frame_idx']}")
        print(f"{'='*80}\n")

        if 'high knee' not in caction['action_name'].lower():
            continue

        sub_recording = create_clip_subset(
            recording_dataset,
            meta_key=caction['meta_key'],
            step_frame=args.step,
        )
        print(f"  Loaded {len(sub_recording.datalist)} frames for this action")

        _, dataloader = create_sequence_dataloader(
            dataset=sub_recording,
            use_gt_joints2d=False,
            use_gt_blade2d=False,
            batch_size=args.len_video_segment,
            shuffle=False,
            num_workers=0,
            load_image=args.verbose,
            load_body_masks=False,
        )

        h5_path_mf = os.path.join(PATH_ASSETS, 'results', MODEL_FREE_TAG,
                                   f"{caction['meta_key']}.h5")
        h5_path_mb = os.path.join(PATH_ASSETS, 'results', MODEL_BASE_TAG,
                                   f"{caction['meta_key']}.h5")
        if not os.path.exists(h5_path_mf):
            print(f"  Skipping: model_free H5 not found: {h5_path_mf}")
            continue
        if not os.path.exists(h5_path_mb):
            print(f"  Skipping: model_base H5 not found: {h5_path_mb}")
            continue

        with h5py.File(h5_path_mf, 'r') as h5f_mf, h5py.File(h5_path_mb, 'r') as h5f_mb:
            align_segment(dataloader, h5f_mf, h5f_mb, dir_output,
                          cfg=cfg, verbose=args.verbose, recording_dataset=sub_recording)
