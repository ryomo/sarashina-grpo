#!/bin/bash

# Install nbstripout and configure it for the current git repository
uv run nbstripout --install
git config filter.nbstripout.clean "\"$(uv run which python)\" -m nbstripout --drop-empty-cells %f"

echo "nbstripout installed and configured for the current git repository"
