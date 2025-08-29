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

**Configuration Options:**

1. **Use default config** (30-feature diverse coaching model):
   ```bash
   # No setup needed - uses configs/diverse_coaching.json automatically
   ```

2. **Use balanced synthetic dataset** (recommended for reduced false positives):
   ```bash
   # Uses synthetic non-sycophantic vs sycophantic examples
   # Better at distinguishing helpful instructions from flattery
   # configs/synthetic_non_sycophantic_vs_sycophantic.json
   ```

3. **Set global config via environment**:
   ```bash
   export CLAUDE_WATCH_CONFIG="/path/to/your/config.json"
   ```

4. **Set per-directory config** (recommended):
   ```bash
   # Create .claudewatch file in your project directory
   echo "/path/to/your/config.json" > .claudewatch
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

## Setting Configuration for Specific Directories

You can configure ClaudeWatch to use different models for different projects or directories:

### Method 1: Directory-specific config (Recommended)
Create a `.claudewatch` JSON file in any directory:

```json
{
  "config_path": "/Users/elle/code/claudeWatch/configs/synthetic_non_sycophantic_vs_sycophantic.json"
}
```

Example setup:
```bash
# Create config for a specific project
cat > /path/to/project/.claudewatch << 'EOF'
{
  "config_path": "/Users/elle/code/claudeWatch/configs/synthetic_non_sycophantic_vs_sycophantic.json"
}
EOF

# Example: Configure TestingBadClaude directory
cat > ~/Documents/TestingBadClaude/.claudewatch << 'EOF'
{
  "config_path": "/Users/elle/code/claudeWatch/configs/synthetic_non_sycophantic_vs_sycophantic.json"
}
EOF
```

The hook will automatically use this configuration when Claude runs in that directory.

### Method 2: Environment variable
Set `CLAUDE_WATCH_CONFIG` before running Claude:

```bash
export CLAUDE_WATCH_CONFIG="/path/to/configs/my_config.json"
```

### Method 3: Update default in wrapper.sh
Edit `src/hooks/wrapper.sh` to change the default configuration.

## Available Configurations

**Built-in configs:**

- `configs/diverse_coaching.json` - **Default**. 30-feature model detecting sycophantic vs authentic coaching
- `configs/joe_hudson_vs_all_sycophantic.json` - 10-feature model, Joe Hudson authentic vs sycophantic  

**Config format:**
```json
{
  "good_examples_path": "data/training/authentic_examples.json",
  "bad_examples_path": "data/training/sycophantic_examples.json", 
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "alert_strategy": "logistic_regression",
  "logistic_threshold": 0.5,
  "feature_threshold": 0.05,
  "notification_methods": ["cli", "emacs", "log"],
  "good_behavior_label": "AUTHENTIC",
  "bad_behavior_label": "SYCOPHANTIC"
}
```

**Key settings:**
- `logistic_threshold`: Alert when P(bad) > this value (0.3-0.7 typical range)
- `alert_strategy`: "logistic_regression" (recommended), "any_bad_feature", or "ratio"
- `notification_methods`: ["cli"] for terminal, ["emacs"] for editor, ["log"] for file

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

## Troubleshooting

**Hook not triggering:**
- Check path in `.claude/settings.local.json` is correct
- Verify `wrapper.sh` is executable: `chmod +x src/hooks/wrapper.sh`
- Check logs in terminal for error messages

**"Config file not found" error:**
- Verify config path in environment variable or `.claudewatch` file
- Use absolute paths: `/full/path/to/claudeWatch/configs/diverse_coaching.json`

**"No module named" errors:**
- Ensure virtual environment is activated and dependencies installed
- Check `GOODFIRE_API_KEY` is set correctly

**No alerts appearing:**
- Lower `logistic_threshold` to 0.3 for more sensitivity
- Check if classifier model exists in `models/` directory
- Run training: `python claude_watch_cli.py train configs/your_config.json`

## Research Application

Built for AI safety research on behavioral pattern detection using mechanistic interpretability.
