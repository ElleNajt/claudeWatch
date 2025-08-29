# ClaudeWatch

AI behavior monitoring using Sparse Autoencoder discriminative features.

## Setup

```bash
pip install goodfire numpy sklearn
export GOODFIRE_API_KEY=your_api_key
```

Hook configuration in `.claude/settings.local.json`:
```json
{
  "hooks": {
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "/path/to/claudeWatch/src/hooks/wrapper.sh"}]}]
  }
}
```

## Usage

```bash
# Analyze text
python claude_watch_cli.py analyze configs/diverse_coaching.json "test message"

# Generate discriminative features 
python claude_watch_cli.py generate-vectors --config configs/my_config.json

# Train classifier
python claude_watch_cli.py train configs/my_config.json
```

## How it works

1. Extract SAE features that distinguish good/bad behavior examples
2. Monitor Claude responses in real-time via hooks
3. Classify behavior using ML with SHAP explanations
4. Alert when confidence exceeds threshold

## Configuration

```json
{
  "good_examples_path": "data/training/authentic_examples.json",
  "bad_examples_path": "data/training/sycophantic_examples.json", 
  "alert_strategy": "logistic_regression",
  "logistic_threshold": 0.5,
  "notification_methods": ["cli", "log"]
}
```

## Architecture

```
src/core/claude_watch.py      # Analysis engine
src/hooks/claude_watch_hook.py # Claude Code integration  
src/ml/                       # Feature generation & training
data/training/                # Example conversations
data/vectors/                 # Generated features
configs/                      # Configurations
```

## Example Training Data

Conversation format:
```json
[
  [
    {"role": "user", "content": "Should I quit without notice?"},
    {"role": "assistant", "content": "You should consider the impact on your team and projects..."}
  ]
]
```

## Research Application

Built for AI safety research on behavioral pattern detection using mechanistic interpretability.
