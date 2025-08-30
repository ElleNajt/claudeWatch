#!/usr/bin/env bash
# ClaudeWatch wrapper script

# Enhanced logging for debugging
LOG_FILE="/tmp/claudewatch_wrapper.log"
echo "$(date): ClaudeWatch wrapper started with args: $*" >> "$LOG_FILE"
echo "$(date): Current directory: $(pwd)" >> "$LOG_FILE"
echo "$(date): USER: $USER, HOME: $HOME" >> "$LOG_FILE"

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "$(date): SCRIPT_DIR: $SCRIPT_DIR" >> "$LOG_FILE"
echo "$(date): PROJECT_ROOT: $PROJECT_ROOT" >> "$LOG_FILE"

# Activate ClaudeWatch virtual environment 
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    echo "$(date): Activating virtual environment" >> "$LOG_FILE"
    source "$PROJECT_ROOT/venv/bin/activate"
    echo "$(date): Virtual environment activated" >> "$LOG_FILE"
else
    echo "$(date): Warning: Virtual environment not found at $PROJECT_ROOT/venv" >> "$LOG_FILE"
    echo "Warning: Virtual environment not found at $PROJECT_ROOT/venv" >&2
fi

# Set analysis mode for better sycophancy detection
export CLAUDE_WATCH_ANALYSIS_MODE="conversation_context"

# Set config path if not already set
echo "$(date): Looking for .claudewatch file in: $(pwd)" >> "$LOG_FILE"
if [ -z "$CLAUDE_WATCH_CONFIG" ]; then
    # Check for .claudewatch file in current directory first
    if [ -f ".claudewatch" ]; then
        echo "$(date): Found .claudewatch file" >> "$LOG_FILE"
        # Check if it's JSON format
        if grep -q "config_path" .claudewatch 2>/dev/null; then
            # Extract config_path from JSON
            export CLAUDE_WATCH_CONFIG="$(python3 -c "import json; print(json.load(open('.claudewatch'))['config_path'])" 2>/dev/null)"
            echo "$(date): Using config from .claudewatch JSON: $CLAUDE_WATCH_CONFIG" >> "$LOG_FILE"
        else
            # Legacy format - direct path
            export CLAUDE_WATCH_CONFIG="$(cat .claudewatch)"
            echo "$(date): Using config from .claudewatch legacy: $CLAUDE_WATCH_CONFIG" >> "$LOG_FILE"
        fi
    else
        # Default config - uses 30-feature diverse coaching model
        export CLAUDE_WATCH_CONFIG="$PROJECT_ROOT/configs/diverse_coaching.json"
        echo "$(date): No .claudewatch found, using default: $CLAUDE_WATCH_CONFIG" >> "$LOG_FILE"
    fi
else
    echo "$(date): Using pre-set config: $CLAUDE_WATCH_CONFIG" >> "$LOG_FILE"
fi

# Validate config file exists
if [ ! -f "$CLAUDE_WATCH_CONFIG" ]; then
    echo "$(date): Error: Config file not found: $CLAUDE_WATCH_CONFIG" >> "$LOG_FILE"
    echo "Error: Config file not found: $CLAUDE_WATCH_CONFIG" >&2
    exit 1
fi

echo "$(date): Using config file: $CLAUDE_WATCH_CONFIG" >> "$LOG_FILE"

# Run the hook (add project root to PYTHONPATH for proper imports)
echo "$(date): Changing to PROJECT_ROOT and running hook" >> "$LOG_FILE"
echo "$(date): Command: cd $PROJECT_ROOT && PYTHONPATH=$PROJECT_ROOT python3 src/hooks/claude_watch_hook.py $@" >> "$LOG_FILE"

cd "$PROJECT_ROOT" && PYTHONPATH="$PROJECT_ROOT" python3 src/hooks/claude_watch_hook.py "$@"
EXIT_CODE=$?

echo "$(date): Hook completed with exit code: $EXIT_CODE" >> "$LOG_FILE"
exit $EXIT_CODE
