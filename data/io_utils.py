import os
import h5py
import numpy as np
import cv2
from pathlib import Path
from typing import Union


def load_frame(
    video_path: Union[str, Path],
    frame_idx: int,
    as_rgb: bool = True
) -> np.ndarray:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx < 0 or frame_idx >= frame_count:
            raise IndexError(
                f"Frame index {frame_idx} out of bounds [0, {frame_count})"
            )

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_idx}")

        if as_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame
    finally:
        cap.release()


def try_load_sam2_h5_info(dir_sam_masks: str, tag: str) -> dict | None:
    """
    If PATH_DETECTION_SAM2/{subject_camera}_{tag}.h5 exists and contains 'mask_first',
    read the index arrays (frame_to_pos, mask shape) — all picklable plain Python/numpy —
    close the file, and return them together with the path.  Returns None when not present.
    """
    parent = os.path.dirname(dir_sam_masks)
    key    = os.path.basename(dir_sam_masks)
    h5_path = os.path.join(parent, f'{key}_{tag}.h5')
    if not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, 'r') as h5f:
        if 'mask_first' not in h5f:
            return None
        frame_to_pos = {int(fi): i for i, fi in enumerate(h5f['frame_indices'])}
        mask_h = int(h5f.attrs['mask_h'])
        mask_w = int(h5f.attrs['mask_w'])
    return {
        'path':         h5_path,
        'frame_to_pos': frame_to_pos,
        'mask_h':       mask_h,
        'mask_w':       mask_w,
    }


def load_mask_first_from_h5(h5f, h5_info: dict, frame_idx: int) -> np.ndarray | None:
    """
    Read mask[0] for *frame_idx* from an already-open H5 file handle.

    Returns a (mask_h, mask_w) bool array, or None if the frame is not present.
    h5f must be an open h5py.File opened on h5_info['path'].
    """
    pos = h5_info['frame_to_pos'].get(int(frame_idx))
    if pos is None:
        return None
    h, w = h5_info['mask_h'], h5_info['mask_w']
    packed = h5f['mask_first'][pos]
    return np.unpackbits(packed)[:h * w].reshape(h, w).astype(bool)
