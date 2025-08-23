# YouTube Pipeline Integration Guide

## Overview

This guide explains how to use the integrated YouTube scraping pipeline to generate diverse coaching training examples for ClaudeWatch. The pipeline extracts essential components from buddhaMindVector and adapts them for automated training data generation.

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Install required packages
pip install assemblyai yt-dlp claude-cli

# Set environment variables
echo "ASSEMBLYAI_API_KEY=your_key_here" >> .env
echo "GOODFIRE_API_KEY=your_key_here" >> .env

# Ensure Claude CLI is configured
claude --help
```

### 2. Generate Training Data

```bash
# Complete pipeline: YouTube discovery → transcription → training data
python claude_watch_cli.py generate-examples --output-dir data/generated_examples

# Discovery only (no transcription)
python claude_watch_cli.py generate-examples --discovery-only
```

### 3. Train Enhanced Classifier

```bash
# Train with both manual and generated examples
python claude_watch_cli.py train configs/diverse_coaching.json --generated-data

# Train with generated examples only
python claude_watch_cli.py train configs/diverse_coaching.json
```

### 4. Test the System

```bash
# Test with new diverse model
python claude_watch_cli.py analyze configs/diverse_coaching.json "You clearly have abandonment issues stemming from your relationship with your father."
```

## 📁 Pipeline Components

### 1. YouTube Discovery (`src/data_pipeline/discovery.py`)

Discovers coaching videos using Claude + Playwright MCP:

```bash
# Discover specific coach
python claude_watch_cli.py discover --coach "Brené Brown" --max-results 5

# Discover diverse coaching styles
python claude_watch_cli.py discover --diverse --max-results 3

# Search by coaching approach
python claude_watch_cli.py discover --style "trauma coaching" --max-results 5
```

**Features:**
- Automated Google/YouTube search using Claude subprocess
- Pre-filtering for authentic 1-on-1 coaching sessions
- Avoids lectures, seminars, and promotional content
- Structured JSON output with video metadata

### 2. Video Processing (`src/video_processing.py`)

Processes and filters discovered videos:

```bash
# Process discovery results
python src/video_processing.py discovery_results.json --filter-coaching --training-data

# Generate training data splits
python src/video_processing.py processed_videos.json --training-data
```

**Features:**
- Enhanced coaching content filtering
- Quality assessment scoring
- Categorization by coaching approach
- Training data candidate selection

### 3. Transcription (`src/transcription.py`)

Transcribes videos with speaker diarization:

```bash
# Transcribe single video
python src/transcription.py --video-url "https://youtube.com/watch?v=..." 

# Batch transcribe from metadata
python src/transcription.py --videos-json processed_videos.json

# Transcribe local file
python src/transcription.py --video-file audio.mp3
```

**Features:**
- AssemblyAI integration with speaker diarization
- Automatic speaker role detection
- Batch processing capabilities
- Error handling and progress tracking

### 4. Conversation Formatting (`src/conversation_formatter.py`)

Converts transcripts to training conversations:

```bash
# Process single transcript
python src/conversation_formatter.py --transcript transcript.json

# Batch process directory
python src/conversation_formatter.py --transcript-dir data/transcripts/
```

**Features:**
- Claude-powered coach/client identification  
- Conversation excerpt generation
- Quality assessment for training suitability
- ClaudeWatch-compatible format output

### 5. Complete Pipeline (`src/generate_coaching_examples.py`)

Orchestrates the entire process:

```bash
# Full pipeline
python src/generate_coaching_examples.py

# Custom parameters
python src/generate_coaching_examples.py \
  --max-videos-per-style 5 \
  --max-transcriptions 3 \
  --output-dir custom_examples

# Process existing discovery
python src/generate_coaching_examples.py --process-existing discovery.json
```

## 🎯 Training Data Strategy

### Positive Examples (Authentic Coaching)
- **Somatic coaching**: Body-based, nervous system awareness
- **Therapeutic coaching**: Trauma-informed, emotional depth
- **Sources**: Joe Hudson, Peter Levine, Gabor Maté style coaches
- **Characteristics**: Open inquiry, somatic awareness, non-directive

### Negative Examples (Projective Coaching)  
- **Directive coaching**: Solution-focused, advice-giving
- **Business coaching**: Goal-oriented, performance-focused
- **Sources**: Corporate coaches, self-help personalities
- **Characteristics**: Assumptions, diagnostic language, prescriptive advice

### Quality Metrics
- **Authentic indicators**: "What do you notice?", "How does that feel?", "What comes up?"
- **Projective indicators**: "You need to", "This sounds like", "You probably"
- **Depth assessment**: Emotional language, somatic references, process orientation

## ⚙️ Configuration Options

### Enhanced Configuration (`configs/diverse_coaching.json`)

```json
{
  "good_examples_path": "data/generated_examples/authentic_coaching_examples.json",
  "bad_examples_path": "data/generated_examples/projective_coaching_examples.json",
  "alert_strategy": "logistic_regression",
  "logistic_threshold": 0.7,
  "notification_methods": ["cli", "emacs", "log"]
}
```

### Discovery Configuration

Modify `src/youtube_discovery.py` to customize:
- Search terms for different coaching styles
- Video filtering criteria
- Maximum results per category
- Quality assessment thresholds

## 🔧 Troubleshooting

### Common Issues

**1. No videos discovered**
```bash
# Check Claude CLI configuration
claude --version

# Verify Playwright MCP is available
claude -p "test" --allowedTools mcp__playwright__browser_navigate
```

**2. Transcription failures**
```bash
# Check AssemblyAI API key
echo $ASSEMBLYAI_API_KEY

# Test with single video first
python src/transcription.py --video-url "https://youtube.com/watch?v=short_video"
```

**3. Poor training data quality**
```bash
# Review quality assessments
python src/conversation_formatter.py --transcript-dir data/transcripts/ --context "therapeutic coaching"

# Adjust filtering thresholds in video_processing.py
```

### Performance Optimization

**For large-scale processing:**
- Use `--discovery-only` for initial exploration
- Process in batches with `--max-transcriptions` limits
- Cache discovery results for repeated processing
- Use train/test splits for model validation

## 📊 Expected Results

### Training Data Volume
- **Discovery**: 15-30 videos across 6 coaching styles
- **Transcription**: 5-10 high-quality sessions per category
- **Conversations**: 50-100 training excerpts total
- **Training time**: 2-5 minutes with enhanced classifier

### Performance Improvements
- **Robustness**: Better handling of diverse coaching approaches
- **Accuracy**: Improved discrimination between authentic vs projective patterns
- **Coverage**: Reduced false positives on edge cases
- **Adaptability**: Self-updating through automated discovery

## 🔄 Maintenance

### Regular Updates
```bash
# Monthly discovery refresh
python src/generate_coaching_examples.py --max-videos-per-style 2

# Retrain with accumulated data
python src/train_classifier.py configs/diverse_coaching.json --generated-data

# Validate performance
python demo_shap_explanations.py
```

### Data Quality Monitoring
- Review quality assessment scores
- Check feature importance changes
- Monitor alert frequency and accuracy
- Update filtering criteria as needed

This pipeline transforms ClaudeWatch from manually curated examples to an automatically scaling system that learns from diverse, real-world coaching content.
