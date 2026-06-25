"""
Sequence DataLoader for the released RSP3D dataset.

Mirrors SpatialTrackDataset / create_spatial_tracker_v2_dataloader but works
with data/dataset.py (RSP3D) instead of data/recording.py (Recording).

Key differences from spatialtrack_dataloader.py:
  - meta_info keyed by meta_key (clip stem), not f"{subject_id}_{camera_id}"
  - Frame loading uses clip_frame_idx (0-based within each released clip video)
  - All detections and SAM2 masks also use clip_frame_idx (re-indexed by extract_clip_data.py)
"""
import torch
import sys
sys.path.append('.')
sys.path.append('..')

import numpy as np
import cv2
import h5py
from torch.utils.data import Dataset, DataLoader

from data.dataset import RSP3D
from data.io_utils import load_frame, load_mask_first_from_h5
from utils.process_utils import generate_silhouette_mask


class SequenceDataset(Dataset):
    """
    Dataset wrapping RSP3D (or a create_clip_subset view) for alignment inference.

    Produces the same (input_dict, meta_dict, gt_dict) triple as
    SpatialTrackDataset so that align_segment() works unchanged.
    """

    def __init__(
        self,
        dataset: RSP3D,
        body_joint_radius: int = 20,
        use_gt_joints2d: bool = False,
        use_gt_blade2d: bool = False,
        load_image: bool = True,
        load_body_masks: bool = True,
    ):
        self.dataset           = dataset
        self.use_gt_joints2d   = use_gt_joints2d
        self.use_gt_blade2d    = use_gt_blade2d
        self.body_joint_radius = body_joint_radius
        self.load_image        = load_image
        self.load_body_masks   = load_body_masks

        self._h5_handles: dict = {}

    def _load_mask(self, frame_idx: int, tag: str, h5_info) -> np.ndarray:
        """Return mask[0] (h, w) from H5. frame_idx is clip_frame_idx (re-indexed)."""
        
        self._h5_handles[tag] = h5py.File(h5_info['path'], 'r')
        mask = load_mask_first_from_h5(self._h5_handles[tag], h5_info, frame_idx)
        return mask

    def __len__(self) -> int:
        return len(self.dataset.datalist)

    def __getitem__(self, idx: int):
        data           = self.dataset.datalist[idx]
        frame_idx      = data['frame_idx']        # original absolute index
        clip_frame_idx = data['clip_frame_idx']   # 0-based index into released clip
        subject_id     = data['subject_id']
        camera_id      = data['camera_id']
        meta_key       = data['meta_key']

        meta_info  = self.dataset.meta_info[meta_key]
        path_video = meta_info['path_video']
        camera     = meta_info['camera']

        # ── GT geometry ────────────────────────────────────────────────────
        skeleton_3d    = data['skeleton_3d'].copy()           # (17, 3) m
        skeleton_valid = np.where(np.isnan(skeleton_3d[:, 0]), 0, 1).astype(np.float32)

        blade_3d    = data['blade_3d'].copy()                 # (M, 2, 3) m
        blade_valid = np.where(np.isnan(blade_3d[..., 0]), 0, 1).astype(np.float32)

        has_blade2 = 'blade2_3d' in data
        if has_blade2:
            blade2_3d    = data['blade2_3d'].copy()
            blade2_valid = np.where(np.isnan(blade2_3d[..., 0]), 0, 1).astype(np.float32)

        # ── Project to camera space + 2D ───────────────────────────────────
        if camera.has_extrinsics:
            skeleton_3d = camera.world_to_camera(skeleton_3d)
            blade_3d    = camera.world_to_camera(blade_3d.reshape(-1, 3))
            if has_blade2:
                blade2_3d = camera.world_to_camera(blade2_3d.reshape(-1, 3))

        skeleton_2d = camera.project(skeleton_3d)                                # (17, 2)
        blade_2d    = camera.project(blade_3d.reshape(-1, 3)).reshape(-1, 2, 2)  # (M, 2, 2)
        blade_3d    = blade_3d.reshape(-1, 2, 3)
        if has_blade2:
            blade2_2d = camera.project(blade2_3d.reshape(-1, 3)).reshape(-1, 2, 2)
            blade2_3d = blade2_3d.reshape(-1, 2, 3)

        # ── Image (use clip_frame_idx for released clip) ────────────────────
        
        if self.load_image:
            image_rgb   = load_frame(path_video, clip_frame_idx)
            orig_height, orig_width = image_rgb.shape[:2]
        else:
            orig_height, orig_width = 2160, 3840
            image_rgb = np.zeros((orig_height, orig_width, 3), dtype=np.uint8)


        # ── Scale 2D coords ────────────────────────────────────────────────
        skeleton_2d = skeleton_2d
        blade_2d    = blade_2d
        if has_blade2:
            blade2_2d = blade2_2d

        # ── Adjusted intrinsics ────────────────────────────────────────────
        intrinsics = np.array([
            [camera.fx, 0,   camera.cx],
            [0,  camera.fy, camera.cy],
            [0,  0,  1],
        ], dtype=np.float32)

        # ── Optional detection fields (scaled) ─────────────────────────────
        bbox_xyxy    = data.get('bbox_xyxy', None)
        vitpose_2d   = data.get('vitpose_2d', None)
        alphapose_2d = data.get('alphapose_2d', None)
        if bbox_xyxy    is not None: bbox_xyxy    = np.array(bbox_xyxy,    dtype=np.float32)
        if vitpose_2d   is not None: vitpose_2d   = np.array(vitpose_2d,   dtype=np.float32)
        if alphapose_2d is not None: alphapose_2d = np.array(alphapose_2d, dtype=np.float32)

        # ── joints2d ───────────────────────────────────────────────────────
        if self.use_gt_joints2d or alphapose_2d is None:
            joints2d       = skeleton_2d.copy()
            joints2d_valid = skeleton_valid.copy()
        else:
            joints2d       = alphapose_2d[:, :2].copy()
            joints2d_valid = skeleton_valid.copy()

        # ── SAM body+blade mask ─────────────────────────────────────────────
        sam2_h5       = meta_info.get('sam2_h5') or {}

        if self.load_body_masks:
            # SAM2 H5 is keyed by clip_frame_idx (re-indexed in extract_clip_data.py)
            mask_body_blade = self._load_mask(clip_frame_idx, 'bodyblade', sam2_h5.get('bodyblade'))
        else:
            mask_body_blade = np.zeros((orig_height, orig_width), dtype=bool)

        mask_body_blade = mask_body_blade.astype(np.uint8)
        for j in range(len(joints2d)):
            if joints2d_valid[j] > 0:
                x, y = int(round(joints2d[j, 0])), int(round(joints2d[j, 1]))
                cv2.circle(mask_body_blade, (x, y), radius=self.body_joint_radius, color=1, thickness=-1)
        mask_body_blade = mask_body_blade.astype(bool)

        # ── blade mask ─────────────────────────────────────────────────────
        if self.use_gt_blade2d:
            mask_blade, _ = generate_silhouette_mask((orig_height, orig_width), blade_2d)
            mask_blade = mask_blade > 0
            if has_blade2:
                mask_blade2, _ = generate_silhouette_mask((orig_height, orig_width), blade2_2d)
                mask_blade2 = mask_blade2 > 0
        else:
            mask_blade = self._load_mask(clip_frame_idx, 'blade', sam2_h5.get('blade'))
            if has_blade2:
                mask_blade2 = self._load_mask(clip_frame_idx, 'blade2', sam2_h5.get('blade2'))
                
        # ── Pack outputs ───────────────────────────────────────────────────
        input_dict = {
            'image_rgb':        image_rgb,
            'intrinsics':       intrinsics,
            'joints2d':         joints2d,
            'joints2d_valid':   joints2d_valid,
            'mask_body_blade':  mask_body_blade.astype(np.float32),
            'mask_blade':       mask_blade.astype(np.float32),
        }
        if has_blade2:
            input_dict['mask_blade2'] = mask_blade2.astype(np.float32)
        if bbox_xyxy    is not None: input_dict['bbox_xyxy']    = bbox_xyxy
        if vitpose_2d   is not None: input_dict['vitpose_2d']   = vitpose_2d
        if alphapose_2d is not None: input_dict['alphapose_2d'] = alphapose_2d

        meta_dict = {
            'video_path':      path_video,
            'subject_id':      subject_id,
            'camera_id':       camera_id,
            'frame_idx':       frame_idx,       # original absolute index
            'clip_frame_idx':  clip_frame_idx,  # 0-based in released clip — used by H5/SAM2 lookups
            'meta_key':        meta_key,
        }

        gt_dict = {
            'skeleton_3d':    skeleton_3d,
            'skeleton_valid': skeleton_valid,
            'skeleton_2d':    skeleton_2d,
            'blade_3d':       blade_3d,
            'blade_2d':       blade_2d,
            'blade_valid':    blade_valid,
            'intrinsics':     intrinsics,
        }
        if has_blade2:
            gt_dict['blade2_3d']    = blade2_3d
            gt_dict['blade2_2d']    = blade2_2d
            gt_dict['blade2_valid'] = blade2_valid

        return input_dict, meta_dict, gt_dict



def collate_fn(batch):
    """
    Custom collate function for batching samples.

    Handles variable-sized blade edges by padding to the maximum size in the batch.
    Ensures blade_2d and blade_3d are aligned with the same padding.

    Args:
        batch: List of tuples (input, meta, gt) from ImageDataset.__getitem__

    Returns:
        Tuple of (batched_input, batched_meta, batched_gt)
    """
    # Unpack batch into separate lists
    inputs = [item[0] for item in batch]
    metas = [item[1] for item in batch]
    gts = [item[2] for item in batch]

    # Batch inputs
    batched_input = {}
    for key in inputs[0].keys():
        #if key=='image_rgb':
        #    batched_input[key] = torch.stack([inp[key] for inp in inputs], dim=0)  # (B, 3, H, W)
        #else:
        batched_input[key] = torch.from_numpy(np.stack([inp[key] for inp in inputs], axis=0))  # (B, ...)
    batched_meta = {key: [] for key in metas[0].keys()}
    for cmeta in metas:
        for ckey in cmeta.keys():
            batched_meta[ckey].append(cmeta[ckey])
            
    # Batch groundtruth with padding for blade edges
    batched_gt = {key: [] for key in gts[0].keys()}
    has_blade2 = 'blade2_3d' in gts[0]


    
    # Find maximum number of blade edges in this batch
    max_edges = max(gt['blade_3d'].shape[0] for gt in gts)
    if has_blade2:
        max_edges2 = max(gt['blade2_3d'].shape[0] for gt in gts)

    blade_keys = ['blade_3d', 'blade_2d', 'blade_valid']
    if has_blade2:
        blade_keys += ['blade2_3d', 'blade2_2d', 'blade2_valid']

    for cgt in gts:
        # Skeleton data (fixed size, no padding needed)
        for ckey in cgt.keys():
            if ckey not in blade_keys:
                batched_gt[ckey].append(torch.from_numpy(cgt[ckey].astype(np.float32)))

        # Blade data (variable size, needs padding)
        blade_3d = torch.from_numpy(cgt['blade_3d'].astype(np.float32))  # (M, 2, 3)
        blade_2d = torch.from_numpy(cgt['blade_2d'].astype(np.float32))  # (M, 2, 2)
        blade_valid = torch.from_numpy(cgt['blade_valid'].astype(np.float32))  # (M, 2)
        num_edges = blade_3d.shape[0]

        # Pad both blade_2d and blade_3d to max_edges to keep them aligned
        if num_edges < max_edges:
            padding_3d = torch.zeros((max_edges - num_edges, 2, 3), dtype=blade_3d.dtype)
            padding_2d = torch.zeros((max_edges - num_edges, 2, 2), dtype=blade_2d.dtype)
            blade_3d = torch.cat([blade_3d, padding_3d], dim=0)
            blade_2d = torch.cat([blade_2d, padding_2d], dim=0)
            padding_valid = torch.zeros((max_edges - num_edges, 2), dtype=torch.float32)
            blade_valid = torch.cat([blade_valid, padding_valid], dim=0)

        batched_gt['blade_3d'].append(blade_3d)
        batched_gt['blade_2d'].append(blade_2d)
        batched_gt['blade_valid'].append(blade_valid)

        if has_blade2:
            blade2_3d = torch.from_numpy(cgt['blade2_3d'].astype(np.float32))  # (M, 2, 3)
            blade2_2d = torch.from_numpy(cgt['blade2_2d'].astype(np.float32))  # (M, 2, 2)
            blade2_valid = torch.from_numpy(cgt['blade2_valid'].astype(np.float32))  # (M, 2)
            num_edges2 = blade2_3d.shape[0]

            if num_edges2 < max_edges2:
                padding_3d = torch.zeros((max_edges2 - num_edges2, 2, 3), dtype=blade2_3d.dtype)
                padding_2d = torch.zeros((max_edges2 - num_edges2, 2, 2), dtype=blade2_2d.dtype)
                blade2_3d = torch.cat([blade2_3d, padding_3d], dim=0)
                blade2_2d = torch.cat([blade2_2d, padding_2d], dim=0)
                padding_valid = torch.zeros((max_edges2 - num_edges2, 2), dtype=torch.float32)
                blade2_valid = torch.cat([blade2_valid, padding_valid], dim=0)

            batched_gt['blade2_3d'].append(blade2_3d)
            batched_gt['blade2_2d'].append(blade2_2d)
            batched_gt['blade2_valid'].append(blade2_valid)

    for ckey in batched_gt.keys():
        batched_gt[ckey] = torch.stack(batched_gt[ckey], dim=0)  # (B, ...)

    return batched_input, batched_meta, batched_gt

def create_sequence_dataloader(
    dataset: RSP3D,
    use_gt_joints2d: bool = False,
    use_gt_blade2d: bool = False,
    body_joint_radius: int = 20,
    batch_size: int = 60,
    shuffle: bool = False,
    num_workers: int = 0,
    load_image: bool = True,
    load_body_masks: bool = True,
    **kwargs,
):
    """
    Create a DataLoader over *dataset* (RSP3D or create_clip_subset view).

    Returns
    -------
    (SequenceDataset, DataLoader)
    """
    torch_dataset = SequenceDataset(
        dataset=dataset,
        body_joint_radius=body_joint_radius,
        use_gt_joints2d=use_gt_joints2d,
        use_gt_blade2d=use_gt_blade2d,
        load_image=load_image,
        load_body_masks=load_body_masks,
    )
    dataloader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **kwargs,
    )
    return torch_dataset, dataloader
