#!/bin/bash

set -e  # Exit on any error

apt-get update

apt-get install -y build-essential jq

apt-get update

wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

export HF_HUB_ENABLE_HF_TRANSFER=1
echo "Setup complete! Python environment is ready."

# uv venv
# source .venv/bin/activate

# uv pip install colpali-engine[train]


# vidore-benchmark evaluate-retriever \
#     --model-class colidefics3 \
#     --model-name vidore/colSmol-256M \
#     --dataset-name vidore/docvqa_test_subsampled \
#     --dataset-format qa \
#     --split test