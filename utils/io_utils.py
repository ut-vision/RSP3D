import numpy as np
import h5py


def prepare_h5_reader(h5f: h5py.File) -> dict:
    """
    Pre-compute lookup structures for fast random access into a results H5 file.
    Call once per open file; pass the returned reader to load_frame_from_h5.

    Returns
    -------
    dict with:
      'frame_to_pos' : {frame_idx (int) -> row position (int)}
      'vl_offsets'   : {key (str) -> cumulative offset array (N+1,) int64}
                       for each variable-length dataset (e.g. body/blade tracks)
    """
    frame_to_pos = {int(fi): i for i, fi in enumerate(h5f['frame_indices'])}
    vl_offsets = {}
    vl = h5f.get('variable_length')
    if vl is not None:
        for key in vl.keys():
            lengths = vl[f'{key}/lengths'][:]
            vl_offsets[key] = np.concatenate([[0], np.cumsum(lengths)])
    return {'frame_to_pos': frame_to_pos, 'vl_offsets': vl_offsets}


def load_frame_from_h5(
    h5f: h5py.File,
    frame_idx: int,
    reader: dict,
    keys: list | None = None,
) -> dict | None:
    """
    Load arrays for one frame into a plain dict.

    Returns None if frame_idx is not present in the H5 file.

    Parameters
    ----------
    keys : list of str, optional
        If given, only these keys are loaded (avoids reading large arrays
        that are not needed, e.g. vertices when only joints3d_h36m is used).
        If None, all keys are loaded.

    Handles three storage layouts transparently:
      - Fixed-shape datasets     : direct row read h5f[key][pos]
      - Variable-length datasets : offset-indexed reconstruction from
                                   variable_length/{key}/data + lengths
      - Padded spatial datasets  : row read + crop back to original shape
                                   using padded/{key}_orig_shapes metadata
    """
    pos = reader['frame_to_pos'].get(int(frame_idx))
    if pos is None:
        return None

    keys_set = set(keys) if keys is not None else None
    frame_data = {}
    padded_group = h5f.get('padded')

    for key in h5f.keys():
        if key in ('frame_indices', 'variable_length', 'padded'):
            continue
        if keys_set is not None and key not in keys_set:
            continue
        arr = h5f[key][pos]
        if padded_group is not None and f'{key}_orig_shapes' in padded_group:
            orig_shape = tuple(padded_group[f'{key}_orig_shapes'][pos])
            arr = arr[tuple(slice(0, s) for s in orig_shape)]
        frame_data[key] = arr

    vl = h5f.get('variable_length')
    if vl is not None:
        for key, offsets in reader['vl_offsets'].items():
            if keys_set is not None and key not in keys_set:
                continue
            s, e = int(offsets[pos]), int(offsets[pos + 1])
            frame_data[key] = vl[f'{key}/data'][s:e]

    return frame_data
