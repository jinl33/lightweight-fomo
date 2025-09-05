import numpy as np
import glob
from pathlib import Path


def get_wavelet_data(data_path):
    """Load wavelet coefficients from directory."""
    npy_files = list(Path(data_path).rglob("*.npy"))
    data_files = [f for f in npy_files if not f.stem.endswith('_meta')]
    
    data = []
    for f in data_files:
        try:
            # Load wavelet coefficients
            wave_data = np.load(f)
            # Load metadata
            meta = np.load(f.with_name(f.stem + '_meta.npy'), allow_pickle=True).item()
            data.append({
                'data': wave_data,
                'meta': meta,
                'path': str(f)
            })
        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue
            
    return data


def prepare_data(data_list):
    """Prepare data for training by converting to tensors."""
    import torch
    
    # Convert numpy arrays to tensors
    tensors = [torch.from_numpy(d['data']).float() for d in data_list]
    
    # Add channel dimension if needed
    tensors = [t.unsqueeze(0) if t.ndim == 3 else t for t in tensors]
    
    return tensors
