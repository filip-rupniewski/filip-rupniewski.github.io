#!/bin/bash
source /home/filip/myenv/bin/activate
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
python3 "$SCRIPT_DIR/pytacz_gui3.4.py"
deactivate
