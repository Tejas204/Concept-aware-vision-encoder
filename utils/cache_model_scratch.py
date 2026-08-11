#!/usr/bin/env python3

from pathlib import Path
from huggingface_hub import snapshot_download

# Hugging Face repository
MODEL_ID = "google/siglip-so400m-patch14-384"

# Destination directory
DEST_DIR = Path("/scratch/common_models") / "siglip-so400m-patch14-384"

DEST_DIR.parent.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(DEST_DIR),
    local_dir_use_symlinks=False,  # Store actual files instead of symlinks
    resume_download=True,
)

print(f"Model downloaded to: {DEST_DIR}")