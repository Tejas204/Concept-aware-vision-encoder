#!/bin/bash

set -e

# Install Miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    cd "$HOME"
    wget -O miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"

# Create environment if necessary
if ! conda env list | awk '{print $1}' | grep -qx "mlu"; then
    conda create -y -n mlu python=3.11
fi

conda activate mlu

cd "$HOME/Concept-aware-vision-encoder"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<EOF
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
EOF