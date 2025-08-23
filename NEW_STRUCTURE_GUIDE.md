# ClaudeWatch New Structure Guide

## 🎯 Overview

ClaudeWatch has been reorganized into a clean, modular structure that separates concerns and improves maintainability. This guide explains the new organization and how to use it.

## 📁 New Directory Structure

```
claudeWatch/
├── claude_watch_cli.py              # Unified CLI entry point
├── src/
│   ├── __init__.py                   # Package-level imports
│   ├── core/                         # Core functionality
│   │   ├── __init__.py
│   │   ├── claude_watch.py          # Main monitoring engine
│   │   ├── config.py                # Configuration management
│   │   └── notifications.py         # Notification system
│   ├── ml/                          # Machine learning components
│   │   ├── __init__.py
│   │   ├── feature_extraction.py    # SAE feature extraction
│   │   ├── train_classifier.py      # Enhanced classifier training
│   │   ├── shap_explainer.py        # SHAP explanations
│   │   └── generate_vectors.py      # Legacy vector generation
│   ├── data_pipeline/               # YouTube data pipeline
│   │   ├── __init__.py
│   │   ├── discovery.py             # YouTube video discovery
│   │   ├── processing.py            # Video metadata processing
│   │   ├── transcription.py         # Audio transcription
│   │   ├── conversation_formatter.py # Chat format conversion
│   │   └── pipeline.py              # Complete pipeline orchestration
│   ├── hooks/                       # Claude Code integration
│   │   ├── __init__.py
│   │   ├── claude_watch_hook.py     # Hook implementation
│   │   └── wrapper.sh               # Shell wrapper
│   └── utils/                       # Shared utilities
│       ├── __init__.py
│       ├── data_loaders.py          # Data loading functions
│       ├── file_utils.py            # File operations
│       └── logging.py               # Logging utilities
├── data/                            # Data directories
├── models/                          # Trained models
├── configs/                         # Configuration files
└── logs/                           # Log files
```

## 🚀 Migration from Old Structure

### Old vs New Commands

| Old Command | New Command |
|-------------|-------------|
| `python src/claude_watch.py config.json "text"` | `python claude_watch_cli.py analyze config.json "text"` |
| `python src/generate_vectors.py config.json` | `python claude_watch_cli.py generate-vectors --config config.json` |
| `python src/train_classifier.py config.json` | `python claude_watch_cli.py train config.json` |
| `python src/generate_coaching_examples.py` | `python claude_watch_cli.py generate-examples` |
| `python src/youtube_discovery.py --diverse` | `python claude_watch_cli.py discover --diverse` |

### Import Changes

| Old Import | New Import |
|------------|------------|
| `from claude_watch import ClaudeWatch, WatchConfig` | `from src.core import ClaudeWatch, WatchConfig` |
| `from train_classifier import train_enhanced_classifier` | `from src.ml import train_enhanced_classifier` |
| `from youtube_discovery import YouTubeCoachDiscovery` | `from src.data_pipeline import YouTubeCoachDiscovery` |

## 🛠️ Using the New CLI

### Core Functionality

```bash
# Analyze text (main use case)
python claude_watch_cli.py analyze configs/coaching_examples.json "You clearly have trust issues."

# Generate feature vectors
python claude_watch_cli.py generate-vectors --config configs/coaching_examples.json

# Train classifier with enhanced features
python claude_watch_cli.py train configs/coaching_examples.json --generated-data
```

### Data Pipeline

```bash
# Generate complete training dataset
python claude_watch_cli.py generate-examples --max-videos-per-style 5

# Discovery only (no transcription)
python claude_watch_cli.py generate-examples --discovery-only

# Discover specific coach videos
python claude_watch_cli.py discover --coach "Brené Brown" --max-results 10

# Discover diverse coaching styles
python claude_watch_cli.py discover --diverse --output discovery_results.json
```

### Help and Options

```bash
# Show all available commands
python claude_watch_cli.py --help

# Show help for specific command
python claude_watch_cli.py analyze --help
python claude_watch_cli.py generate-examples --help
```

## 📦 Package-Level Usage

You can now import ClaudeWatch components at the package level:

```python
# Core functionality
from src import ClaudeWatch, WatchConfig, NotificationManager

# ML components  
from src import FeatureExtractor, train_enhanced_classifier, SHAPExplainer

# Data pipeline (if available)
from src import YouTubeCoachDiscovery, CoachingExamplesGenerator

# Utilities
from src import load_conversation_data, setup_logging
```

## 🔧 Configuration Updates

### Hook Configuration

Update your `.claude/settings.local.json` hook path:

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

### Environment Variables

No changes needed - the same environment variables work:

- `GOODFIRE_API_KEY` - Required for all functionality
- `ASSEMBLYAI_API_KEY` - Required for transcription pipeline
- `CLAUDE_PROJECT_DIR` - Automatically set by hooks

## 🎯 Module Responsibilities

### Core (`src/core/`)
- **claude_watch.py**: Main monitoring engine, feature analysis, alert logic
- **config.py**: Configuration loading, validation, JSON handling  
- **notifications.py**: Multi-modal notifications (CLI, Emacs, logging)

### ML (`src/ml/`)
- **feature_extraction.py**: SAE feature extraction using Goodfire API
- **train_classifier.py**: Enhanced logistic regression training with SHAP
- **shap_explainer.py**: Model interpretability and explanation utilities
- **generate_vectors.py**: Legacy vector generation (maintained for compatibility)

### Data Pipeline (`src/data_pipeline/`)
- **discovery.py**: YouTube video discovery using Claude + Playwright MCP
- **processing.py**: Video filtering, quality assessment, categorization
- **transcription.py**: AssemblyAI integration with speaker diarization
- **conversation_formatter.py**: Convert transcripts to training conversations
- **pipeline.py**: Complete pipeline orchestration from discovery to training data

### Utilities (`src/utils/`)
- **data_loaders.py**: Load conversation data in various formats
- **file_utils.py**: Safe file operations, JSON handling, directory management
- **logging.py**: Centralized logging, progress tracking, performance metrics

## 🧪 Testing the New Structure

### Basic Functionality Test

```bash
# 1. Test configuration loading
python -c "from src.core.config import WatchConfig; print(WatchConfig.from_json('configs/coaching_examples.json'))"

# 2. Test CLI analyze
python claude_watch_cli.py analyze configs/coaching_examples.json "What do you notice in your body right now?"

# 3. Test data pipeline discovery (if dependencies available)
python claude_watch_cli.py discover --coach "test" --max-results 1
```

### Import Tests

```python
# Test core imports
from src.core import ClaudeWatch, WatchConfig, NotificationManager

# Test ML imports
from src.ml import FeatureExtractor, SHAPExplainer

# Test utilities
from src.utils import load_conversation_data, setup_logging

# Test data pipeline (optional)
try:
    from src.data_pipeline import YouTubeCoachDiscovery
    print("Data pipeline available")
except ImportError:
    print("Data pipeline requires: pip install assemblyai yt-dlp claude-cli")
```

## 🚨 Breaking Changes

### 1. Direct Script Execution
- Old: `python src/claude_watch.py`
- New: `python claude_watch_cli.py analyze`

### 2. Import Paths  
- Old: `from claude_watch import ClaudeWatch`
- New: `from src.core import ClaudeWatch`

### 3. File Locations
- Scripts moved from `src/` root to organized subdirectories
- Use CLI for all functionality instead of direct script calls

## ✅ Benefits of New Structure

### **Organization**
- Clear separation of concerns (core, ML, data pipeline, utilities)
- Logical grouping of related functionality
- Easier to understand and navigate

### **Maintainability**  
- Modular components can be developed independently
- Cleaner import dependencies
- Better code reusability

### **Extensibility**
- Easy to add new pipeline components
- Clear place for new utilities and ML models
- Better separation of optional dependencies

### **Developer Experience**
- Unified CLI interface for all functionality
- Package-level imports for easy integration
- Comprehensive help and documentation

## 🔄 Backward Compatibility

The new structure maintains backward compatibility where possible:

- ✅ Configuration files work unchanged
- ✅ Environment variables unchanged  
- ✅ Hook functionality preserved
- ✅ All existing features available via new CLI
- ⚠️ Direct script execution requires CLI migration
- ⚠️ Import paths need updating for programmatic use

## 📚 Next Steps

1. **Update your workflows** to use the new CLI commands
2. **Test the new structure** with your existing configurations
3. **Update any custom scripts** to use new import paths
4. **Explore the enhanced data pipeline** for automated training data generation

The reorganized ClaudeWatch provides a solid foundation for future development while maintaining all existing functionality in a more maintainable structure.