# ClaudeWatch 🔍

Real-time AI behavior monitoring using Sparse Autoencoder (SAE) discriminative features. Monitor Claude (or any AI assistant) for specific behavioral patterns and get alerts when unwanted behaviors are detected.

## What is this?

ClaudeWatch uses interpretability techniques to create "semantic firewalls" for AI systems. Instead of using prompt engineering or traditional safety measures, it monitors actual behavioral patterns using SAE features extracted from contrast datasets.

## Features

<<<<<<< /var/folders/wm/q6gwq3q56yj8rdyj_h_qhkd40000gn/T/claude-resultZajkGs.tmp
- 🎯 **Discriminative Feature Detection**: Uses contrastive learning to identify specific behaviors
- ⚡ **Real-time Monitoring**: Integrates with Claude Code hooks for live analysis
- 🔧 **Customizable**: Create your own behavioral vectors from example datasets
- 📊 **Detailed Logging**: Track quality scores, confidence levels, and feature activations
- 🚨 **Smart Alerts**: Configurable thresholds for different severity levels
||||||| /var/folders/wm/q6gwq3q56yj8rdyj_h_qhkd40000gn/T/claude-baseah1tdc.tmp
- 🎯 **Discriminative Feature Detection**: Uses contrastive learning to identify behavioral patterns
- ⚡ **Real-time Monitoring**: Integrates with Claude Code hooks for live response analysis  
- 🔧 **Example-based Training**: Create behavioral vectors from realistic conversation examples
- 🤖 **Auto-Vector Generation**: Automatically generates vectors when missing - zero setup required
- 🏷️ **Feature Label Alerts**: Shows which specific SAE features triggered (not just generic alerts)
- 📊 **Comprehensive Logging**: Track quality scores, feature activations, and full response details
- 🚨 **Multi-modal Alerts**: CLI alerts, Emacs notifications, and file logging
- ⚙️ **Configurable Messaging**: Customize alert terminology for your domain
- 📁 **Professional Structure**: Clean Python package organization
=======
- 🎯 **Discriminative Feature Detection**: Uses contrastive learning to identify behavioral patterns
- ⚡ **Real-time Monitoring**: Integrates with Claude Code hooks for live response analysis  
- 🔧 **Example-based Training**: Create behavioral vectors from realistic conversation examples
- 🤖 **Auto-Vector Generation**: Automatically generates vectors when missing - zero setup required
- 🏷️ **Feature Label Alerts**: Shows which specific SAE features triggered (not just generic alerts)
- 🧠 **SHAP Explanations**: Understand why classifications were made with feature importance scores
- 🎛️ **Multiple Detection Strategies**: Choose from threshold-based, ratio-based, quality-based, or ML-based detection
- 📊 **Comprehensive Logging**: Track quality scores, feature activations, and full response details
- 🚨 **Multi-modal Alerts**: CLI alerts, Emacs notifications, and file logging
- ⚙️ **Configurable Messaging**: Customize alert terminology for your domain
- 📁 **Professional Structure**: Clean Python package organization
>>>>>>> /var/folders/wm/q6gwq3q56yj8rdyj_h_qhkd40000gn/T/claude-changesQ9ZWg8.tmp

## Quick Start

### 1. Install Dependencies

```bash
pip install goodfire numpy
```

### 2. Set Environment Variables

```bash
export GOODFIRE_API_KEY=your_goodfire_api_key_here
```

### 3. Set Up Monitoring

#### For Claude Code

Add to your `.claude/settings.local.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/claudeWatch/hooks/wrapper.sh"
          }
        ]
      }
    ]
  }
}
```

<<<<<<< /var/folders/wm/q6gwq3q56yj8rdyj_h_qhkd40000gn/T/claude-resultRpg2c3.tmp
### 4. Configure Behavior Detection
||||||| /var/folders/wm/q6gwq3q56yj8rdyj_h_qhkd40000gn/T/claude-base9WAeZr.tmp
### 4. Test the System

```bash
# Test basic analysis
python src/claude_watch.py configs/coaching_examples.json "You're clearly dealing with trust issues."

# Test with SHAP explanations
python src/claude_watch.py configs/coaching_logistic.json "You're procrastinating because you're afraid of failure."

# Demo SHAP explanations
python demo_shap_explanations.py
```

## Detection Strategies
=======
### 4. Test the System

```bash
# Test basic analysis
python src/claude_watch.py configs/coaching_examples.json "You're clearly dealing with trust issues."

# Test with SHAP explanations (default is logistic regression)
python src/claude_watch.py configs/coaching_examples.json "You're procrastinating because you're afraid of failure."

# Demo SHAP explanations
python demo_shap_explanations.py
```

## Detection Strategies
>>>>>>> /var/folders/wm/q6gwq3q56yj8rdyj_h_qhkd40000gn/T/claude-changeshoQaLb.tmp

Create or modify `configs/default.json`:

```json
{
<<<<<<< /var/folders/wm/q6gwq3q56yj8rdyj_h_qhkd40000gn/T/claude-resultot2FCZ.tmp
  "good_features_path": "./vectors/good_features.json",
  "bad_features_path": "./vectors/bad_features.json",
  "good_threshold": 0.1,
  "bad_threshold": 0.1,
  "alert_ratio": 2.0
||||||| /var/folders/wm/q6gwq3q56yj8rdyj_h_qhkd40000gn/T/claude-baselbgeCr.tmp
  "good_examples_path": "/Users/elle/code/claudeWatch/data/training/joe_hudson_excerpts.json",
  "bad_examples_path": "/Users/elle/code/claudeWatch/data/training/projective_coaching.json", 
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "alert_strategy": "logistic_regression",
  "feature_threshold": 0.05,
  "notification_methods": ["cli", "emacs", "log"],
  "good_alert_message": "Authentic coaching detected!",
  "bad_alert_message": "Projective coaching detected!",
  "good_behavior_label": "AUTHENTIC",
  "bad_behavior_label": "PROJECTIVE"
=======
  "good_examples_path": "/Users/elle/code/claudeWatch/data/training/joe_hudson_excerpts.json",
  "bad_examples_path": "/Users/elle/code/claudeWatch/data/training/projective_coaching.json", 
  "model": "meta-llama/Llama-3.3-70B-Instruct",
  "alert_strategy": "logistic_regression",
  "feature_threshold": 0.05,
  "logistic_threshold": 0.7,
  "notification_methods": ["cli", "emacs", "log"],
  "good_alert_message": "Authentic coaching detected!",
  "bad_alert_message": "Projective coaching detected!",
  "good_behavior_label": "AUTHENTIC",
  "bad_behavior_label": "PROJECTIVE"
>>>>>>> /var/folders/wm/q6gwq3q56yj8rdyj_h_qhkd40000gn/T/claude-changesKALhyo.tmp
}
```

## Creating Custom Vectors

### 1. Prepare Example Datasets

Create JSON files with examples of good and bad behavior:

**good_behavior.json:**
```json
[
  "I hear that you're struggling. What does that feel like in your body?",
  "That sounds really difficult. Tell me more about what's happening for you.",
  "What would it look like if this situation resolved itself?"
]
```

**bad_behavior.json:**
```json
[
  "You should definitely quit your job immediately.",
  "That's wrong. Here's what you need to do instead.",
  "Stop feeling that way. You need to be more positive."
]
```

### 2. Generate Discriminative Vectors

```bash
python scripts/create_vectors.py good_behavior.json bad_behavior.json ./my_vectors 20
```

This creates:
- Feature vectors that distinguish good from bad behavior
- A configuration file for using these vectors
- Detailed feature labels for interpretability

## How It Works

### Discriminative Features

ClaudeWatch uses Contrastive Activation Addition (CAA) to identify features that distinguish desired from undesired behavior:

```
v = (1/n) * Σ[activations(good) - activations(bad)]
```

Features with positive weights indicate good behavior, while negative weights indicate bad behavior.

### Quality Assessment

Responses are classified as:
- **GOOD**: High good features, low bad features (ratio < 0.5)
- **HARMFUL**: High bad features, low good features (ratio > 2.0)  
- **ACCEPTABLE**: More good than bad
- **CONCERNING**: More bad than good
- **UNCLEAR**: Insufficient signal

### Real-time Monitoring

1. Hooks capture AI responses
2. SAE features are extracted
3. Behavioral patterns are scored
4. Alerts trigger if thresholds are exceeded

## Example Use Cases

### Coaching Quality
Monitor for therapeutic vs harmful coaching patterns:
- ✅ Good: Open questions, validation, curiosity
- ❌ Bad: Direct advice, assumptions, invalidation

### Code Safety
Detect potentially dangerous code patterns:
- ✅ Good: Input validation, error handling
- ❌ Bad: Eval statements, SQL injection vectors

## Architecture

```
claudeWatch/
├── src/
│   ├── claude_watch.py       # Core detection engine
│   └── feature_extraction.py  # SAE feature pipeline
├── hooks/
│   ├── claude_watch_hook.py  # Claude Code integration
│   └── wrapper.sh             # Shell wrapper
├── vectors/
│   └── [feature files]        # Discriminative vectors
├── configs/
│   └── default.json           # Configuration
├── scripts/
│   └── create_vectors.py      # Vector generation
└── logs/
    └── claude_watch.log       # Analysis logs
```

## Advanced Configuration

### Thresholds

- `good_threshold`: Minimum score to consider good features significant
- `bad_threshold`: Minimum score to consider bad features significant  
- `alert_ratio`: Bad/good ratio that triggers alerts

### Multiple Monitors

Run different behavioral monitors simultaneously:

```bash
CLAUDE_WATCH_CONFIG=./configs/coaching.json ./hooks/wrapper.sh
CLAUDE_WATCH_CONFIG=./configs/safety.json ./hooks/wrapper.sh
```

## Troubleshooting

### Hook Not Triggering
- Check hook configuration in `.claude/settings.local.json`
- Verify wrapper script has execute permissions
- Check logs in `logs/errors.log`

### No Alerts
- Lower thresholds in configuration
- Verify feature files exist and are valid
- Check if examples in dataset are distinctive enough

### API Errors
- Ensure Goodfire API key is set
- Check API rate limits
- Verify model availability

## Contributing

We welcome contributions! Areas of interest:
- Additional example datasets
- Integration with other AI systems
- Improved feature extraction methods
- Visualization tools

## License

MIT License - See LICENSE file

## Acknowledgments

Built using:
- [Goodfire](https://goodfire.ai) - SAE feature extraction
- Inspired by mechanistic interpretability research

---

**Remember**: This is an exploratory prototype for AI safety research.
