# ClaudeWatch 🔍

Real-time AI behavior monitoring using Sparse Autoencoder (SAE) discriminative features. Monitor Claude Code responses for specific behavioral patterns and get instant alerts when unwanted behaviors are detected.

## What is this?

ClaudeWatch creates "semantic firewalls" for AI systems using interpretability techniques. Instead of relying on prompt engineering, it monitors actual behavioral patterns by analyzing SAE features extracted from contrastive training datasets.

## Features

- 🎯 **Discriminative Feature Detection**: Uses contrastive learning to identify behavioral patterns
- ⚡ **Real-time Monitoring**: Integrates with Claude Code hooks for live response analysis  
- 🔧 **Example-based Training**: Create behavioral vectors from realistic conversation examples
- 🤖 **Auto-Vector Generation**: Automatically generates vectors when missing - zero setup required
- 🏷️ **Feature Label Alerts**: Shows which specific SAE features triggered (not just generic alerts)
- 🧠 **SHAP Explanations**: Understand why classifications were made with feature importance scores
- 🎛️ **Configurable Thresholds**: Fine-tune detection sensitivity with logistic regression thresholds
- 📂 **Project-Specific Logs**: Logs appear automatically in your current project directory
- 📊 **Comprehensive Logging**: Track quality scores, feature activations, and full response details
- 🚨 **Multi-modal Alerts**: CLI alerts, Emacs notifications, and file logging
- ⚙️ **Configurable Messaging**: Customize alert terminology for your domain

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment  
source venv/bin/activate

# Set your Goodfire API key
echo "GOODFIRE_API_KEY=your_api_key_here" > .env
```

### 2. Configure Claude Code Integration

Add to your Claude Code `.claude/settings.local.json`:

```json
{
  "hooks": {
    "Stop": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command", 
            "command": "/Users/elle/code/claudeWatch/src/hooks/wrapper.sh"
          }
        ]
      }
    ]
  }
}
```

### 3. Test the System

```bash
# Test with SHAP explanations (default configuration)
python src/claude_watch.py configs/coaching_examples.json "You're procrastinating because you're afraid of failure."

# Demo SHAP explanations
python demo_shap_explanations.py
```

### 4. Start Using Claude Code

When you run Claude Code from any project directory, ClaudeWatch will automatically:
- Monitor your conversations in real-time
- Create logs in your project's `logs/` directory
- Show SHAP explanations for why responses were flagged

## How It Works

### SHAP-Powered Classification

ClaudeWatch uses logistic regression with SHAP explanations to show exactly why responses are classified:

```
❌ ClaudeWatch: Projective coaching detected! Predicted: projective (P=0.73) 
| Why: Discussions about causes and s(+0.30→projective), Narrative transitions that may(-0.03→authentic)
```

This shows:
- **Prediction confidence** (73% confident it's projective coaching)
- **Top contributing features** with their SHAP values
- **Direction of influence** (→projective pushes toward bad, →authentic pushes toward good)

### Project-Specific Logging

Logs automatically appear in your current working directory:

```
/Users/elle/code/myProject/     # <- Where you run Claude Code
├── logs/
│   ├── claude_watch.log        # Detailed analysis with SHAP values
│   ├── notifications.log       # Alert history  
│   └── errors.log              # Any hook errors
├── my_code.py
└── README.md
```

No configuration needed - ClaudeWatch detects where you're working and creates logs there.

## Configuration

The main configuration file is `configs/coaching_examples.json`:

```json
{
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
}
```

### Key Parameters

#### `logistic_threshold` (Confidence Control)
Controls when alerts fire based on prediction confidence:

```json
{
  "alert_strategy": "logistic_regression",
  "logistic_threshold": 0.7  // Only alert if P(projective) > 70%
}
```

**Recommended values:**
- `0.6` - Somewhat confident (good for development/testing)  
- `0.7` - Confident (balanced for production use) **← Default**
- `0.8` - Very confident (low false positive rate)
- `0.9` - Extremely confident (very conservative)

This prevents noisy alerts on borderline cases like "Hello! What would you like to work on?" (P=0.494) and only alerts when genuinely confident about problematic patterns.

#### `alert_strategy` Options
- `"logistic_regression"` - ML-based with SHAP explanations (default)
- `"any_bad_feature"` - Alert if any bad feature exceeds threshold
- `"ratio"` - Alert if bad/good ratio exceeds threshold  
- `"quality"` - Alert based on overall quality assessment

#### `notification_methods`
- `"cli"` - Stderr output for Claude Code integration
- `"emacs"` - Send messages to Emacs via emacsclient
- `"log"` - Write to log files in your project directory

## Training Custom Models

### 1. Prepare Training Data

Create conversation examples in JSON format:

**data/training/good_examples.json:**
```json
[
  {
    "conversation": [
      {"role": "user", "content": "I'm struggling with this decision"},
      {"role": "assistant", "content": "What do you notice in your body as you think about each option?"}
    ]
  }
]
```

**data/training/bad_examples.json:**
```json
[
  {
    "conversation": [
      {"role": "user", "content": "I'm struggling with this decision"},
      {"role": "assistant", "content": "You're clearly overthinking this. Just pick option A."}
    ]
  }
]
```

### 2. Generate Features & Train

```bash
# Generate discriminative feature vectors
python src/generate_vectors.py configs/coaching_examples.json

# Train logistic regression classifier  
python src/train_classifier.py configs/coaching_examples.json
```

This creates a balanced classifier with SHAP explanations enabled.

## Project Structure

```
claudeWatch/
├── src/
│   ├── claude_watch.py          # Core monitoring engine with SHAP
│   ├── generate_vectors.py      # Feature extraction
│   ├── train_classifier.py      # ML model training
│   └── hooks/
│       ├── claude_watch_hook.py # Claude Code integration
│       └── wrapper.sh           # Shell wrapper
├── data/
│   ├── training/                # Training examples (Joe Hudson vs projective)
│   └── vectors/                 # Generated discriminative features
├── models/                      # Trained classifiers with SHAP
├── configs/
│   └── coaching_examples.json   # Main configuration
├── demo_shap_explanations.py    # Interactive SHAP demo
└── requirements.txt             # Dependencies (includes SHAP)
```

## Use Cases

### Coaching Quality Monitoring (Current)
Monitor for authentic vs projective coaching patterns:
- ✅ **Authentic**: "What do you notice in your body right now?"
- ❌ **Projective**: "You're clearly dealing with trust issues."

**Current Training Data:**
- 20 authentic Joe Hudson coaching excerpts (somatic awareness, emotional inquiry)
- 20 projective coaching examples (assumptions, diagnostic language)

### Content Safety  
Detect harmful advice patterns:
- ✅ **Safe**: Supportive, exploratory questions
- ❌ **Harmful**: Direct medical/legal advice, assumptions

### Code Safety
Monitor for dangerous coding patterns:
- ✅ **Safe**: Input validation, error handling  
- ❌ **Risky**: Eval statements, injection vectors

## Understanding SHAP Explanations

### Reading SHAP Values

```
🤖 Classifier Prediction: PROJECTIVE (confidence: 73.2%)

🔍 Why this prediction? (SHAP explanations)
• Discussions about causes and solutions for pr   strongly pushes toward PROJECTIVE (+0.291)
• Narrative transitions that may lead to conten   moderately pushes toward AUTHENTIC (-0.055)  
• Causal explanation patterns in text            slightly pushes toward PROJECTIVE (+0.011)
```

**How to interpret:**
- **Positive values** (+0.291) push toward "projective" classification
- **Negative values** (-0.055) push toward "authentic" classification
- **Larger absolute values** = stronger influence on the decision
- **Final prediction** is the sum of all these influences

## Troubleshooting

### Hook Not Triggering
- Check hook configuration in `.claude/settings.local.json`
- Verify wrapper script has execute permissions: `chmod +x src/hooks/wrapper.sh`
- Check logs in your project's `logs/errors.log`

### No SHAP Explanations
- Ensure `alert_strategy: "logistic_regression"` in config
- Install SHAP: `pip install shap` (included in requirements.txt)
- Verify classifier model exists in `models/` directory
- Retrain if needed: `python src/train_classifier.py configs/coaching_examples.json`

### Adjusting Sensitivity

**Too many false positives?** Increase `logistic_threshold`:
```json
{"logistic_threshold": 0.8}  // More conservative
```

**Missing real issues?** Decrease `logistic_threshold`:
```json  
{"logistic_threshold": 0.6}  // More sensitive
```

### Logs Not Appearing in Project Directory
- Check if your project directory is writable
- Logs will fallback to ClaudeWatch directory if permissions fail
- Verify Claude Code hook is receiving `cwd` field in JSON payload

### Poor Classification Performance
- Add more training examples to your datasets
- Ensure examples are clearly distinguishable  
- Retrain classifier: `python src/train_classifier.py configs/coaching_examples.json`
- Check feature importance scores in training output

## Dependencies

Install with:
```bash
pip install goodfire numpy python-dotenv scikit-learn shap
```

Or use the included requirements.txt:
```bash
pip install -r requirements.txt
```

## License

MIT License

## Acknowledgments

Built using:
- [Goodfire](https://goodfire.ai) - SAE feature extraction  
- [SHAP](https://shap.readthedocs.io/) - Model explainability
- Inspired by mechanistic interpretability research
- Joe Hudson coaching examples for authentic training data

---

**Remember**: This is an exploratory prototype for AI safety research. SHAP explanations help understand model decisions but should always be reviewed by humans. The system is designed to assist, not replace, human judgment in evaluating AI behavior.