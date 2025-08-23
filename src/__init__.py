"""
ClaudeWatch - AI Behavior Monitoring System

A comprehensive system for monitoring AI behavior using SAE features,
with automated training data generation from YouTube coaching content.
"""

# Core functionality
from .core import ClaudeWatch, WatchConfig, NotificationManager

# Machine learning components
from .ml import (
    FeatureExtractor, 
    generate_discriminative_features,
    train_enhanced_classifier,
    SHAPExplainer
)

# Data pipeline (optional, requires additional dependencies)
try:
    from .data_pipeline import (
        YouTubeCoachDiscovery,
        VideoProcessor,
        TranscriptionService,
        ConversationFormatter,
        CoachingExamplesGenerator
    )
    DATA_PIPELINE_AVAILABLE = True
except ImportError:
    DATA_PIPELINE_AVAILABLE = False

# Utilities
from .utils import (
    load_conversation_data,
    load_diverse_examples,
    ensure_directory,
    safe_json_dump,
    safe_json_load,
    setup_logging,
    get_logger
)

__version__ = "1.0.0"

__all__ = [
    # Core
    'ClaudeWatch',
    'WatchConfig', 
    'NotificationManager',
    
    # ML
    'FeatureExtractor',
    'generate_discriminative_features',
    'train_enhanced_classifier',
    'SHAPExplainer',
    
    # Utilities
    'load_conversation_data',
    'load_diverse_examples',
    'ensure_directory',
    'safe_json_dump',
    'safe_json_load',
    'setup_logging',
    'get_logger',
    
    # Constants
    'DATA_PIPELINE_AVAILABLE'
]

# Add data pipeline components if available
if DATA_PIPELINE_AVAILABLE:
    __all__.extend([
        'YouTubeCoachDiscovery',
        'VideoProcessor', 
        'TranscriptionService',
        'ConversationFormatter',
        'CoachingExamplesGenerator'
    ])