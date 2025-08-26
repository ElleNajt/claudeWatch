#!/usr/bin/env bash
# ClaudeWatch wrapper script

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Activate ClaudeWatch virtual environment 
source "$PROJECT_ROOT/venv/bin/activate"

# Set config path if not already set
if [ -z "$CLAUDE_WATCH_CONFIG" ]; then
    # Check for .claudewatch file in current directory first
    if [ -f ".claudewatch" ]; then
        export CLAUDE_WATCH_CONFIG="$(cat .claudewatch)"
    else
        # Default config
        export CLAUDE_WATCH_CONFIG="$PROJECT_ROOT/configs/joe_hudson_vs_sycophantic.json"
    fi
fi

# Run the hook (add project root to PYTHONPATH for proper imports)
cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT" python3 src/hooks/claude_watch_hook.py "$@"
