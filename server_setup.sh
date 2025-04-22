#!/bin/bash

set -e  # Exit on any error

apt-get update

apt-get install -y build-essential jq

apt-get update

wget -qO- https://astral.sh/uv/install.sh | sh

echo "Setup complete! Python environment is ready."

# source $HOME/.local/bin/env

# uv venv
# source .venv/bin/activate
# export HF_HUB_ENABLE_HF_TRANSFER=1

# uv pip install colpali-engine[train] dotenv hf-transfer wandb


# vidore-benchmark evaluate-retriever \
#     --model-class colidefics3 \
#     --model-name vijay-ravichander/ColSmol-256-Distill-Col-mon \
#     --dataset-name vidore/docvqa_test_subsampled \
#     --dataset-format qa \
#     --split test

# sudo apt-get update
# sudo apt-get install -y python3-dev build-essential