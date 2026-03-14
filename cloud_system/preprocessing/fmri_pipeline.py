"""
fMRI Preprocessing Pipeline.
Extracted from codes/fmri.py FMRIDataLoader.load_fmri_volume().

Converts a 4D NIfTI fMRI scan into a tensor ready for the CNN-LSTM.
Steps: Load 4D NIfTI → sample/pad to 30 timeframes →
       extract 3 orthogonal slices per frame → resize 64×64 → normalise [0,1]
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Optional


def preprocess_fmri(filepath: str,
                    spatial_shape: tuple = (64, 64, 3),
                    time_steps: int = 30) -> Optional[np.ndarray]:
    """
    Load and preprocess a 4D fMRI NIfTI file.
    Exact logic from codes/fmri.py FMRIDataLoader.load_fmri_volume().

    Parameters
    ----------
    filepath : path to .nii or .nii.gz file
    spatial_shape : (H, W, n_slices) — default (64, 64, 3) for 3 orthogonal slices
    time_steps : number of temporal frames to sample (default 30)

    Returns
    -------
    np.ndarray of shape (H, W, 3, time_steps) or None on error
    """
    try:
        import nibabel as nib
        img = nib.load(filepath)
        data = img.get_fdata()

        # Handle 3D images (single time point)
        if data.ndim == 3:
            data = data[..., np.newaxis]

        n_timepoints = data.shape[3] if data.ndim == 4 else 1

        # Sample or pad to get exactly time_steps frames
        if n_timepoints >= time_steps:
            indices = np.linspace(0, n_timepoints - 1, time_steps, dtype=int)
            data = data[..., indices]
        else:
            pad_size = time_steps - n_timepoints
            padding = np.repeat(data[..., -1:], pad_size, axis=3)
            data = np.concatenate([data, padding], axis=3)

        target_h, target_w, n_slices = spatial_shape
        result = np.zeros((target_h, target_w, n_slices, time_steps),
                          dtype=np.float32)

        for t in range(time_steps):
            vol = data[..., t]

            # 3 orthogonal slices: axial, coronal, sagittal
            mid_z = vol.shape[2] // 2
            mid_y = vol.shape[1] // 2
            mid_x = vol.shape[0] // 2

            axial = vol[:, :, mid_z]
            coronal = vol[:, mid_y, :]
            sagittal = vol[mid_x, :, :]

            for i, slice_2d in enumerate([axial, coronal, sagittal]):
                factors = [target_h / slice_2d.shape[0],
                           target_w / slice_2d.shape[1]]
                result[:, :, i, t] = zoom(slice_2d, factors, order=1)

        # Normalise to [0, 1]
        v_min, v_max = result.min(), result.max()
        if v_max > v_min:
            result = (result - v_min) / (v_max - v_min)

        return result.astype(np.float32)

    except Exception as e:
        print(f"[fMRI Pipeline] Error processing {filepath}: {e}")
        return None


def preprocess_fmri_bytes(file_bytes: bytes,
                          filename: str = "scan.nii.gz",
                          spatial_shape: tuple = (64, 64, 3),
                          time_steps: int = 30) -> Optional[np.ndarray]:
    """
    Preprocess fMRI from raw bytes (for API uploads).
    Writes to a temp file then calls preprocess_fmri().
    """
    import tempfile, os
    suffix = ".nii.gz" if filename.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return preprocess_fmri(tmp_path, spatial_shape, time_steps)
    finally:
        os.unlink(tmp_path)
