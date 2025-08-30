# ClaudeWatch

Epistemic status: Fun side project trying to detect claude code being sycophantic me when I use it for personal coaching.  Also includes some experiments with using SAEs to do this, but using claude code directly works better.

![ClaudeWatch in action](Screenshot.png)

*ClaudeWatch detecting sycophantic behavior in real-time during a Claude Code session*

## Quick Start (Recommended: Claude Prompt Strategy)

The simplest and most effective approach uses Claude itself to detect unwanted behaviors:

### Example: Detect Sycophancy
```json
{
  "alert_strategy": "claude_prompt",
  "behavior_to_detect": "sycophancy/flattery/excessive praise",
  "claude_threshold": 0.7,
  "notification_methods": ["cli"]
}
```

### Example: Detect Manipulation  
```json
{
  "alert_strategy": "claude_prompt",
  "behavior_to_detect": "emotional manipulation or guilt-tripping",
  "claude_threshold": 0.6,
  "notification_methods": ["cli", "log", "emacs"]
}
```

### Example: Detect Overconfidence
```json
{
  "alert_strategy": "claude_prompt", 
  "behavior_to_detect": "overconfident claims without caveats",
  "claude_threshold": 0.8,
  "notification_methods": ["cli"]
}
```

## Behavior Detection Examples

The `behavior_to_detect` field allows you to describe any behavior pattern you want to monitor. Here are more examples:

### Coaching-Related Behaviors
```json
{"behavior_to_detect": "giving direct advice instead of asking coaching questions"}
{"behavior_to_detect": "solving problems for the user instead of helping them discover solutions"}
{"behavior_to_detect": "excessive validation without challenging assumptions"}
```

### Professional Communication Issues
```json
{"behavior_to_detect": "dismissive or condescending tone"}
{"behavior_to_detect": "avoiding difficult topics or being evasive"}
{"behavior_to_detect": "making assumptions about user capabilities or knowledge"}
```

### Technical Response Quality
```json
{"behavior_to_detect": "providing untested or potentially harmful code suggestions"}
{"behavior_to_detect": "omitting important security considerations or warnings"}
{"behavior_to_detect": "overly complex solutions when simple ones exist"}
```

### Emotional/Social Patterns
```json
{"behavior_to_detect": "people-pleasing responses that avoid necessary disagreement"}
{"behavior_to_detect": "excessive apologies or self-deprecation"}
{"behavior_to_detect": "creating artificial urgency or pressure"}
```

The system works best with specific, behavioral descriptions rather than abstract concepts. Focus on observable patterns in language and communication style.

**Note:** When ClaudeWatch detects unwanted behavior, Claude Code will show a nonzero return code in the terminal. This is intentional but the error display won't go away and continues to stay above the input. For a better user experience, use `"notification_methods": ["emacs"]` instead of `["cli"]` - the Emacs integration provides cleaner notifications.

## Setup

Hook configuration in `.claude/settings.local.json`:
```json
{
  "hooks": {
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "/path/to/claudeWatch/src/hooks/wrapper.sh"}]}]
  }
}
```

## Configuration

Create a `.claudewatch` file in your project directory:
```json
{
  "config_path": "/path/to/claudeWatch/configs/claude_prompt_sycophancy.json"
}
```

Or set globally:
```bash
export CLAUDE_WATCH_CONFIG="/path/to/claudeWatch/configs/claude_prompt_sycophancy.json"
```

## Available Configurations

**Recommended:**
- `configs/claude_prompt_sycophancy.json` - Simple, effective detection using Claude itself

## SAE Experiments

For experimental SAE-based detection approaches, see [SAE_EXPERIMENTS.md](SAE_EXPERIMENTS.md).

## Architecture

```
src/core/claude_watch.py      # Analysis engine
src/hooks/claude_watch_hook.py # Claude Code integration  
configs/                      # Configurations
```

