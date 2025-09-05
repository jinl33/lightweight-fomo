import torch
import pywt
import numpy as np
from pathlib import Path
import nibabel as nib


def pywt_decompose_array(arr, levels=2):
    packet = pywt.WaveletPacketND(torch.tensor(arr).numpy(),
                                'haar', axes=(-3,-2,-1), maxlevel=levels)
    coeffs = np.concatenate([n.data for n in packet.get_level(levels)], axis=0)
    return coeffs / 8.0


def process_subject(subject_dir, out_dir, levels=2):
    """Process all Numpy files in a subject directory with wavelet decomposition."""
    in_dir = Path(subject_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for npy in in_dir.rglob("*.npy"):
        if npy.stem.endswith('_meta') or npy.stem.endswith('pkl'):
            continue
            
        print(f"Processing {npy}...")
        arr = np.load(str(npy))
        wave = pywt_decompose_array(arr, levels=levels)

        rel = npy.relative_to(in_dir).with_suffix(".npy")
        dest = out_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, wave)

        meta = {
            "original_shape": arr.shape,
            "wavelet_shape": wave.shape,
            "levels": levels
        }
        np.save(dest.with_name(dest.stem + "_meta.npy"), meta)
        print(f"Saved to {dest}")


if __name__ == "__main__":
    process_subject("data/preprocessed/fomo_60k/sub_1",
                   "data/wavelet/fomo_60k/sub_1")
