# SAE-based Detection Experiments

**Note:** The SAE (Sparse Autoencoder) approach was experimental and doesn't work as well as the claude_prompt strategy in the main README. Use claude_prompt for reliable results.

## Setup

If you want to experiment with SAE detection:

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

## Configuration Options

1. **Default config**: Uses configs/diverse_coaching.json automatically

2. **Balanced synthetic dataset**: configs/synthetic_non_sycophantic_vs_sycophantic.json

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

## Available SAE Configurations

**Experimental (SAE-based in configs/sae/):**
- `configs/sae/diverse_coaching.json` - Default SAE config. 30-feature model
- `configs/sae/joe_hudson_vs_all_sycophantic.json` - 10-feature model  
- `configs/sae/comprehensive_sycophancy_detection.json` - 17-feature expanded model
- Various other experimental SAE configurations in `configs/sae/`  

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
- `logistic_threshold`: Alert when P(bad) > this value (0.3-0.7 range)  
- `alert_strategy`: "logistic_regression", "any_bad_feature", "ratio"
- `notification_methods`: ["cli"] for terminal, ["emacs"] for editor, ["log"] for file

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