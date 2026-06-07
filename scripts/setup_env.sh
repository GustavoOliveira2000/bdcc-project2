#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="bigdata_312"
KERNEL_DISPLAY_NAME="Python 3.12 (Big Data Env)"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIXI_FILE="$ROOT_DIR/pixi.toml"

if ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL https://pixi.sh/install.sh | bash

    PIXI_BIN="$HOME/.pixi/bin"
    if [[ ":$PATH:" != *":$PIXI_BIN:"* ]]; then
        export PATH="$PIXI_BIN:$PATH"
    fi

    if ! command -v pixi >/dev/null 2>&1; then
        echo "Pixi installation finished, but 'pixi' is still not in PATH."
        echo "Try running: export PATH=\"$HOME/.pixi/bin:$PATH\""
        exit 1
    fi
fi

pixi install --manifest-path "$PIXI_FILE"

pixi run --manifest-path "$PIXI_FILE" python -m ipykernel install \
    --user \
    --name="$ENV_NAME" \
    --display-name="$KERNEL_DISPLAY_NAME"

echo "=========================================================="
echo "SUCCESS: Setup complete!"
echo "=========================================================="
echo "1. Refresh your JupyterLab browser tab."
echo "2. Open your notebook and select the '$KERNEL_DISPLAY_NAME' kernel."
echo "3. To run commands inside the Pixi environment, use:"
echo "   pixi run --manifest-path $PIXI_FILE <command>"
echo "=========================================================="
