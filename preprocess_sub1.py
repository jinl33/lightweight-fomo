import argparse
import os
from pathlib import Path
import numpy as np
from functools import partial
from batchgenerators.utilities.file_and_folder_operations import (
    join,
    save_pickle,
    maybe_mkdir_p as ensure_dir_exists,
)
from yucca.functional.preprocessing import preprocess_case_for_training_without_label
from yucca.functional.utils.loading import read_file_to_nifti_or_np


def process_single_scan(scan_path, target_dir):
    """Process a single scan for pretraining data."""
    scan_name = os.path.splitext(os.path.splitext(os.path.basename(scan_path))[0])[0]
    
    try:
        preprocess_config = {
            "normalization_operation": ["volume_wise_znorm"],
            "crop_to_nonzero": True,
            "target_orientation": "RAS",
            "target_spacing": [1.0, 1.0, 1.0],
            "keep_aspect_ratio_when_using_target_size": False,
            "transpose": [0, 1, 2],
        }
        
        images, image_props = preprocess_case_for_training_without_label(
            images=[read_file_to_nifti_or_np(scan_path)], **preprocess_config
        )
        image = images[0]

        save_path = join(target_dir, scan_name)
        np.save(save_path + ".npy", image)
        save_pickle(image_props, save_path + ".pkl")
        
        print(f"Successfully processed {scan_path}")
    except Exception as e:
        print(f"Error processing {scan_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path", type=str, required=True, help="Path to sub_1 data")
    parser.add_argument("--out_path", type=str, required=True, help="Path to store preprocessed data")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    ensure_dir_exists(out_path)

    # Process all .nii.gz files in the input directory
    for scan_path in in_path.rglob("*.nii.gz"):
        process_single_scan(str(scan_path), str(out_path))


if __name__ == "__main__":
    main()
