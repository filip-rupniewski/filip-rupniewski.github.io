#!/bin/bash
source /home/filip/myenv/bin/activate
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
python3 "$SCRIPT_DIR/pytacz_gui4.0.py"
deactivate
