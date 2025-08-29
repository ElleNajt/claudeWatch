#!/usr/bin/env bash
# ClaudeWatch wrapper script

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Activate ClaudeWatch virtual environment 
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
else
    echo "Warning: Virtual environment not found at $PROJECT_ROOT/venv" >&2
fi

# Set analysis mode for better sycophancy detection
export CLAUDE_WATCH_ANALYSIS_MODE="conversation_context"

# Set config path if not already set
if [ -z "$CLAUDE_WATCH_CONFIG" ]; then
    # Check for .claudewatch file in current directory first
    if [ -f ".claudewatch" ]; then
        export CLAUDE_WATCH_CONFIG="$(cat .claudewatch)"
    else
        # Default config - uses 30-feature diverse coaching model
        export CLAUDE_WATCH_CONFIG="$PROJECT_ROOT/configs/diverse_coaching.json"
    fi
fi

# Validate config file exists
if [ ! -f "$CLAUDE_WATCH_CONFIG" ]; then
    echo "Error: Config file not found: $CLAUDE_WATCH_CONFIG" >&2
    exit 1
fi

# Run the hook (add project root to PYTHONPATH for proper imports)
cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT" python3 src/hooks/claude_watch_hook.py "$@"
