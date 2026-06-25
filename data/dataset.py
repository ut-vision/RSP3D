"""
Dataset class for the released Blade dataset.

Mirrors the interface of data/recording.py but loads from the released
flat directory structure written by scripts/generate_release.py:

  {root_release}/
    P1/
      videos/       {slug}_{cam}_{start}_{end}.mp4
      annotations/  {slug}_{cam}_{start}_{end}.json
    P2/
      ...

Annotation JSON layout:
  meta   : subject_id, camera_id, action_name, start_frame, end_frame, fps
  camera : width, height, fx, fy, cx, cy, dist_coeffs, R, t
  frames : [ { frame_idx, skeleton_3d (17,3) m, blade_3d (M,2,3) m,
               blade2_3d (M,2,3) m  ← P7/P8 only }, ... ]

All 3D coordinates are in metres. JSON null encodes NaN (invalid frames).

datalist entries mirror recording.py and add:
  clip_frame_idx : 0-based frame index into the released clip video
                   = frame_idx - start_frame
  meta_key       : key into self.meta_info (unique per clip)

meta_info[meta_key]:
  path_video    : str           path to the .mp4 clip
  camera        : CameraParams
  fps           : float
  start_frame   : int           original start frame (= clip offset)
  dir_sam_masks : str           path to SAM2 mask directory (shared across clips)
  sam2_h5       : dict | None   H5 handle info per tag; None if load_sam_masks=False

Loading images:
  image_rgb = load_frame(data['meta_info']['path_video'],
                         data['clip_frame_idx'])
  IMPORTANT: use clip_frame_idx (not frame_idx) — each released clip starts at 0.

Detection lookup (bbox, vitpose, alphapose, SAM masks):
  All detection files use the original frame_idx as key, not clip_frame_idx.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional

sys.path.append('.')
sys.path.append('..')

from utils.camera import CameraParams
from utils.constants import PATH_ASSETS
from utils.vis_utils import visualize_skeleton_2d, visualize_blade_2d
from data.io_utils import load_frame, try_load_sam2_h5_info


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _none_to_nan(obj):
    """Recursively replace JSON null (None) with float NaN."""
    if obj is None:
        return float('nan')
    if isinstance(obj, list):
        return [_none_to_nan(x) for x in obj]
    return obj


def _load_array(data) -> np.ndarray:
    """Convert a nested list (with possible None/null) to float32 ndarray."""
    return np.array(_none_to_nan(data), dtype=np.float32)


def camera_from_dict(d: dict) -> CameraParams:
    """Reconstruct a CameraParams from the dict saved in the annotation JSON."""
    return CameraParams(
        image_width=d['width'],
        image_height=d['height'],
        fx=d['fx'],
        fy=d['fy'],
        cx=d['cx'],
        cy=d['cy'],
        dist_coeffs=np.array(d['dist_coeffs'], dtype=np.float64) if d.get('dist_coeffs') is not None else None,
        R=np.array(d['R'],         dtype=np.float64) if d.get('R') is not None else None,
        t=np.array(d['t'],         dtype=np.float64) if d.get('t') is not None else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RSP3D:
    """
    Dataset for the released Blade dataset.

    __init__ only populates actionlist and meta_info (fast, no per-frame data).
    Per-frame data is loaded on demand via load_data_for_clip(), which is called
    by create_clip_subset() — mirroring the Recording / create_subset_with_selected_data
    pattern in recording.py.

    Parameters
    ----------
    root_release : str or Path
    subject_tag : str, optional  — only load this subject prefix, e.g. 'P1'
    load_bbox_detections : bool
    load_vitpose : bool
    load_alphapose : bool
    load_sam_masks : bool
    """

    def __init__(
        self,
        root_release,
        subject_tag: Optional[str] = None,
        load_bbox_detections: bool = False,
        load_vitpose: bool = False,
        load_alphapose: bool = False,
        load_sam_masks: bool = False,
    ):
        self.root_release         = Path(root_release)
        self.load_bbox_detections = load_bbox_detections
        self.load_vitpose         = load_vitpose
        self.load_alphapose       = load_alphapose
        self.load_sam_masks       = load_sam_masks

        self.datalist   = []
        self.actionlist = []
        self.meta_info  = {}   # meta_key (clip stem) → dict

        # Cache detection JSONs per video_tag (shared across clips of same camera)
        self._det_cache: dict = {}

        for subject_dir in sorted(self.root_release.iterdir()):
            if not subject_dir.is_dir() or subject_dir.name.startswith('_'):
                continue
            if subject_tag and subject_dir.name != subject_tag:
                continue

            annot_dir = subject_dir / 'annotations'
            video_dir = subject_dir / 'videos'
            if not annot_dir.is_dir():
                continue

            for annot_path in sorted(annot_dir.glob('*.json')):
                self._scan_clip(annot_path, video_dir)

    # ── private ───────────────────────────────────────────────────────────────

    def _scan_clip(self, annot_path: Path, video_dir: Path):
        """Read meta + camera from one annotation JSON; populate actionlist and meta_info."""
        stem       = annot_path.stem
        path_video = str(video_dir / f"{stem}.mp4")

        with open(annot_path, 'r') as f:
            annot = json.load(f)

        meta_dict   = annot['meta']
        subject_id  = meta_dict['subject_id']
        camera_id   = meta_dict['camera_id']
        action_name = meta_dict['action_name']
        start_frame = meta_dict['start_frame']
        end_frame   = meta_dict['end_frame']
        fps         = float(meta_dict['fps'])
        camera      = camera_from_dict(annot['camera'])
        meta_key = stem

        # SAM2 H5 lives under PATH_RELEASE_ASSETS/detections/sam2/{meta_key}_{tag}.h5
        # try_load_sam2_h5_info(dir, tag) looks for {parent(dir)}/{basename(dir)}_{tag}.h5
        dir_sam_masks = os.path.join(PATH_ASSETS, 'detections', 'sam2', meta_key)

        if self.load_sam_masks:
            sam2_h5 = {tag: try_load_sam2_h5_info(dir_sam_masks, tag)
                       for tag in ('bodyblade', 'blade')}
            if 'P7' in subject_id or 'P8' in subject_id:
                sam2_h5['blade2'] = try_load_sam2_h5_info(dir_sam_masks, 'blade2')
        else:
            sam2_h5 = None

        self.meta_info[meta_key] = {
            'path_video':    path_video,
            'camera':        camera,
            'fps':           fps,
            'start_frame':   start_frame,
            'annot_path':    str(annot_path),   # stored so load_data_for_clip can re-read frames
            'dir_sam_masks': dir_sam_masks,
            'sam2_h5':       sam2_h5,
        }
        self.actionlist.append({
            'subject_id':      subject_id,
            'camera_id':       camera_id,
            'action_name':     action_name,
            'start_frame_idx': start_frame,
            'end_frame_idx':   end_frame,
            'meta_key':        meta_key,
        })

    def _get_detections(self, meta_key: str) -> dict:
        """Load and cache per-clip detection JSONs (keyed by clip_frame_idx)."""
        if meta_key in self._det_cache:
            return self._det_cache[meta_key]

        det = {}
        det_root = os.path.join(PATH_ASSETS, 'detections')

        if self.load_bbox_detections:
            path = os.path.join(det_root, 'bbox', f"{meta_key}.json")
            det['bbox'] = json.load(open(path)) if os.path.exists(path) else None
            if det['bbox'] is None:
                print(f"  Warning: BBox detection file not found: {path}")

        if self.load_vitpose:
            path = os.path.join(det_root, 'vitpose', f"{meta_key}.json")
            det['vitpose'] = json.load(open(path)) if os.path.exists(path) else None
            if det['vitpose'] is None:
                print(f"  Warning: ViTPose detection file not found: {path}")

        if self.load_alphapose:
            path = os.path.join(det_root, 'alphapose', f"{meta_key}.json")
            det['alphapose'] = json.load(open(path)) if os.path.exists(path) else None
            if det['alphapose'] is None:
                print(f"  Warning: AlphaPose detection file not found: {path}")

        self._det_cache[meta_key] = det
        return det

    # ── public ────────────────────────────────────────────────────────────────

    def load_data_for_clip(self, meta_key: str, step_frame: int = 1) -> list:
        """
        Load per-frame GT + detection data for one clip.  Analogous to
        Recording.load_data_for_a_video().

        Returns a datalist (list of dicts) that can be assigned to a subset object.
        Detection lookup uses original frame_idx; clip_frame_idx is also stored
        for released-clip video seeking.
        """
        clip_meta   = self.meta_info[meta_key]
        start_frame = clip_meta['start_frame']

        with open(clip_meta['annot_path'], 'r') as f:
            annot = json.load(f)

        frames_raw  = annot['frames']
        subject_id  = annot['meta']['subject_id']
        camera_id   = annot['meta']['camera_id']
        action_name = annot['meta']['action_name']

        det = self._get_detections(meta_key)

        last_bbox      = [0, 0, 0, 0]
        last_vitpose   = np.zeros((17, 3), dtype=np.float32)
        last_alphapose = np.zeros((16, 3), dtype=np.float32)

        datalist = []
        for i, frame_dict in enumerate(frames_raw):
            if step_frame > 1 and i % step_frame != 0:
                continue

            frame_idx      = frame_dict['frame_idx']
            clip_frame_idx = frame_idx - start_frame
            data_dict = {
                'frame_idx':      frame_idx,
                'clip_frame_idx': clip_frame_idx,
                'skeleton_3d':    _load_array(frame_dict['skeleton_3d']),
                'blade_3d':       _load_array(frame_dict['blade_3d']),
                'subject_id':     subject_id,
                'camera_id':      camera_id,
                'action_name':    action_name,
                'meta_key':       meta_key,
            }
            if frame_dict.get('blade2_3d') is not None:
                data_dict['blade2_3d'] = _load_array(frame_dict['blade2_3d'])

            # Detection JSONs are keyed by clip_frame_idx (re-indexed in extract_clip_data.py)
            str_idx = str(clip_frame_idx)

            if self.load_bbox_detections and det.get('bbox') is not None:
                if str_idx in det['bbox']:
                    last_bbox = det['bbox'][str_idx][0]['bbox_xyxy']
                data_dict['bbox_xyxy'] = last_bbox

            if self.load_vitpose and det.get('vitpose') is not None:
                if str_idx in det['vitpose']:
                    last_vitpose = np.array(det['vitpose'][str_idx]['keypoints_2d'],
                                            dtype=np.float32)
                data_dict['vitpose_2d'] = last_vitpose

            if self.load_alphapose and det.get('alphapose') is not None:
                if str_idx in det['alphapose']:
                    last_alphapose = np.array(det['alphapose'][str_idx]['keypoints_2d'],
                                              dtype=np.float32)
                data_dict['alphapose_2d'] = last_alphapose

            datalist.append(data_dict)

        return datalist

    # ── public helpers ────────────────────────────────────────────────────────

    def load_image(self, data_dict: dict) -> np.ndarray:
        """
        Load the RGB frame for a datalist entry.

        Uses clip_frame_idx (not frame_idx) because each released clip
        starts at index 0.

        Returns
        -------
        np.ndarray  (H, W, 3) uint8 RGB
        """
        meta  = self.meta_info[data_dict['meta_key']]
        return load_frame(meta['path_video'], data_dict['clip_frame_idx'])

    def get_camera(self, data_dict: dict) -> CameraParams:
        return self.meta_info[data_dict['meta_key']]['camera']

    def visualize_sample(self, data_idx: int, dir_vis: str = './visualizations/') -> str:
        """
        Project GT skeleton and blade onto the video frame and save as PNG.

        Returns the path to the saved image.
        """
        cdata     = self.datalist[data_idx]
        frame_idx = cdata['frame_idx']
        camera    = self.get_camera(cdata)

        to_handle_blade2 = 'blade2_3d' in cdata

        image_rgb = self.load_image(cdata)          # (H, W, 3) uint8 RGB

        skeleton_3d   = cdata['skeleton_3d'].copy()          # (17, 3)   m
        skeleton_valid = np.where(np.isnan(skeleton_3d[:, 0]), 0, 1)  # (17,)

        blade_3d   = cdata['blade_3d'].copy()                # (M, 2, 3) m
        blade_valid = np.where(np.isnan(blade_3d[:, :, 0]), 0, 1)     # (M, 2)

        if to_handle_blade2:
            blade2_3d    = cdata['blade2_3d'].copy()          # (M, 2, 3) m
            blade2_valid = np.where(np.isnan(blade2_3d[:, :, 0]), 0, 1)

        # Project to 2D
        if camera.has_extrinsics:
            skeleton_3d = camera.world_to_camera(skeleton_3d)
            blade_3d    = camera.world_to_camera(blade_3d.reshape(-1, 3))
            if to_handle_blade2:
                blade2_3d = camera.world_to_camera(blade2_3d.reshape(-1, 3))

        skeleton_2d = camera.project(skeleton_3d)                            # (17, 2)
        blade_2d    = camera.project(blade_3d.reshape(-1, 3)).reshape(-1, 2, 2)  # (M, 2, 2)
        if to_handle_blade2:
            blade2_2d = camera.project(blade2_3d.reshape(-1, 3)).reshape(-1, 2, 2)

        # Build visualisation (BGR for OpenCV)
        image_vis = image_rgb[..., ::-1].copy()
        image_vis = visualize_skeleton_2d(image_vis, skeleton_2d, skeleton_valid)
        image_vis = visualize_blade_2d(image_vis, blade_2d, blade_valid)
        if to_handle_blade2:
            image_vis = visualize_blade_2d(image_vis, blade2_2d, blade2_valid,
                                           point_color=(255, 0, 255), edge_color=(255, 0, 255))

        subject_id  = cdata['subject_id'].split('_')[0]
        path_vis = os.path.join(
            dir_vis, subject_id,
            f"{cdata['action_name']}_{cdata['camera_id']}_{frame_idx}.jpg",
        )
        os.makedirs(os.path.dirname(path_vis), exist_ok=True)

        image_vis = cv2.resize(image_vis, (image_vis.shape[1] // 3, image_vis.shape[0] // 3))
        cv2.imwrite(path_vis, image_vis)
        return path_vis


def create_clip_subset(dataset: RSP3D, meta_key: str, step_frame: int = 1) -> RSP3D:
    """
    Load per-frame data for one clip and return a lightweight RSP3D-like object.

    Analogous to Recording.create_subset_with_selected_data(): RSP3D.__init__
    only builds actionlist/meta_info; this function is where actual frame data
    is read (via load_data_for_clip).

    Parameters
    ----------
    dataset    : RSP3D (init-only, datalist may be empty)
    meta_key   : clip identifier matching actionlist entry's 'meta_key'
    step_frame : keep every Nth frame (default 1 = all frames)
    """
    subset = RSP3D.__new__(RSP3D)
    subset.root_release         = dataset.root_release
    subset.load_bbox_detections = dataset.load_bbox_detections
    subset.load_vitpose         = dataset.load_vitpose
    subset.load_alphapose       = dataset.load_alphapose
    subset.load_sam_masks       = dataset.load_sam_masks
    subset._det_cache           = dataset._det_cache

    subset.meta_info  = dataset.meta_info   # shared reference (read-only)
    subset.actionlist = [a for a in dataset.actionlist if a['meta_key'] == meta_key]
    subset.datalist   = dataset.load_data_for_clip(meta_key, step_frame)
    
    return subset


if __name__ == '__main__':
    import argparse
    from utils.constants import PATH_DATASET

    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_tag', type=str, default=None,
                        help="Only load this subject, e.g. 'P1'. Default: all subjects.")
    args = parser.parse_args()


    # ── Visualisation example ────────────────────────────────────────────────
    PATH_DATASET = '/home/fylwen/WS/blade_project/assets/release'

    dataset = RSP3D(PATH_DATASET, subject_tag=args.subject_tag)
    print(f"Loaded {len(dataset.actionlist)} action segments\n")
    for caction in dataset.actionlist:
        print(caction)

    for caction in dataset.actionlist:
        subset = create_clip_subset(dataset, caction['meta_key'], step_frame=30)

        # Visualise every frame in subset (already step=30)
        for i in range(len(subset.datalist)):
            path = subset.visualize_sample(i, dir_vis='./visualizations/')
            print(path)

