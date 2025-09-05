from huggingface_hub import snapshot_download
from pathlib import Path

if __name__ == "__main__":
    subject_id = "sub_1"
    repo_id = "FOMO25/FOMO-MRI"
    src_folder = f"fomo-60k/{subject_id}"

    local_root = Path("data/")
    local_root.mkdir(parents=True, exist_ok=True)

    # Download only files under fomo-60k/sub_1/*
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        local_dir=str(local_root),
        local_dir_use_symlinks=False,
        allow_patterns=[f"{src_folder}/*"]
    )

    print(f"Downloaded files into directory tree under {local_root}/{subject_id}/")

    # Print out the session/file structure
    for p in Path(local_root / subject_id).rglob("*"):
        rel = p.relative_to(local_root)
        print(rel)
