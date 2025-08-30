# ClaudeWatch

Epistemic status: Fun side project trying to detect claude code being sycophantic me when I use it for personal coaching.  Also includes some experiments with using SAEs to do this, but using claude code directly works better.

![ClaudeWatch in action](Screenshot.png)

*ClaudeWatch detecting extremely subtle sycophantic behavior in real-time during a Claude Code session*

![How it works](protecme.jpg)

*This is how it works*

## Quick Start 

ClaudeWatch monitors your Claude Code sessions and alerts you when Claude exhibits unwanted behaviors.

### 1. Install the Hook

Add this to your `.claude/settings.local.json`:

```json
{
  "hooks": {
    "Stop": [{"matcher": "", "hooks": [{"type": "command", "command": "/path/to/claudeWatch/src/hooks/wrapper.sh"}]}]
  }
}
```

### 2. Configure Behavior Detection

Create a config file or use the default. Here are some examples:

**Detect Sycophancy:**
```json
{
  "alert_strategy": "claude_prompt",
  "behavior_to_detect": "sycophancy/flattery/excessive praise",
  "claude_threshold": 0.7,
  "notification_methods": ["emacs"]
}
```

**Detect Manipulation:**
```json
{
  "alert_strategy": "claude_prompt",
  "behavior_to_detect": "emotional manipulation or guilt-tripping",
  "claude_threshold": 0.6,
  "notification_methods": ["emacs"]
}
```

**Detect Overconfidence:**
```json
{
  "alert_strategy": "claude_prompt", 
  "behavior_to_detect": "overconfident claims without caveats",
  "claude_threshold": 0.8,
  "notification_methods": ["emacs"]
}
```

### 3. Set Your Configuration

Create `.claudewatch` in your project:
```json
{
  "config_path": "/path/to/your/config.json"
}
```

### 4. Start Using Claude Code

ClaudeWatch will now monitor your sessions:
- **Emacs notification**: "🚨 ClaudeWatch: Sycophantic behavior detected!"
- **CLI Notification**: Shows nonzero exit code when unwanted behavior is found, prints in claude code directly. Unfortunately, claude code doesn't clear these out on the next message, leading to cruft. Use emacs instead, or customize it to use libnotify or somethign else.

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

