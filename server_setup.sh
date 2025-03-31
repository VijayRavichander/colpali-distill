#!/bin/bash

set -e  # Exit on any error

apt-get update

apt-get install -y build-essential jq

apt-get update

wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "Setup complete! Python environment is ready."

# uv venv
# source .venv/bin/activate

# uv pip install colpali-engine[train]



