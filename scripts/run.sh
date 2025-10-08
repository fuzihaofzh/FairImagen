#!/bin/bash
# Fair Image Generation - Minimal Version
# Usage: ./scripts/run.sh "data=test11,protect=[gender]" "proc=fpca,remove,enoise=0.6"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

python src/main.py "$1" "$2"
