"""
MRI Preprocessing Pipeline.
Extracted from codes/mri.py DataLoader.load_volume().

Converts a NIfTI structural MRI scan into a tensor ready for the 3D-CNN.
Steps: Load NIfTI -> extract 10 axial slices -> resize 128x128 -> normalise [0,1]
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Optional


def preprocess_mri(filepath: str,
                   n_slices: int = 10,
                   target_size: int = 128) -> Optional[np.ndarray]:
    """
    Load and preprocess a structural MRI NIfTI file.
    Exact logic from codes/mri.py DataLoader.load_volume().

    Parameters
    ----------
    filepath : path to .nii or .nii.gz file
    n_slices : number of evenly spaced axial slices (default 10)
    target_size : resize each slice to target_size × target_size (default 128)

    Returns
    -------
    np.ndarray of shape (n_slices, target_size, target_size, 1) or None on error
    """
    try:
        import nibabel as nib
        img = nib.load(filepath)
        data = img.get_fdata()

        # If 4D (e.g. fMRI accidentally), average across time
        if data.ndim == 4:
            data = np.mean(data, axis=3)

        # Extract n_slices from the middle 80% of the brain (skip empty
        # top/bottom) for more informative anatomy
        total_slices = data.shape[2]
        margin = int(total_slices * 0.1)  # skip 10% top and bottom
        start = max(margin, 0)
        end = min(total_slices - margin, total_slices) - 1
        indices = np.linspace(start, end, n_slices, dtype=int)
        slices = data[:, :, indices]

        # Resize each slice to target_size × target_size
        zoom_factors = (target_size / slices.shape[0],
                        target_size / slices.shape[1],
                        1)
        slices = zoom(slices, zoom_factors, order=1)

        # Normalise to [0, 1]
        v_min, v_max = slices.min(), slices.max()
        if v_max > v_min:
            slices = (slices - v_min) / (v_max - v_min + 1e-8)

        # Reshape from (H, W, n_slices) → (n_slices, H, W, 1)
        slices = np.transpose(slices, (2, 0, 1))
        slices = slices[..., np.newaxis]

        return slices.astype(np.float32)

    except Exception as e:
        print(f"[MRI Pipeline] Error processing {filepath}: {e}")
        return None


def preprocess_mri_bytes(file_bytes: bytes,
                         filename: str = "scan.nii.gz",
                         n_slices: int = 10,
                         target_size: int = 128) -> Optional[np.ndarray]:
    """
    Preprocess MRI from raw bytes (for API uploads).
    Writes to a temp file then calls preprocess_mri().
    """
    import tempfile, os
    suffix = ".nii.gz" if filename.endswith(".gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return preprocess_mri(tmp_path, n_slices, target_size)
    finally:
        os.unlink(tmp_path)
