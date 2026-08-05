#!/bin/bash
set -euo pipefail

# Optional: activate your virtual environment
# source /path/to/venv/bin/activate

# Optional: use your Hugging Face token if not logged in
# export HF_TOKEN=your_huggingface_token

python3 utils/cache_model_local.py