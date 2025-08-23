"""
ClaudeWatch Data Pipeline
YouTube discovery, transcription, and training data generation
"""

from .discovery import YouTubeCoachDiscovery
from .processing import VideoProcessor
from .transcription import TranscriptionService, YouTubeTranscriber
from .conversation_formatter import ConversationFormatter
from .pipeline import CoachingExamplesGenerator

__all__ = [
    'YouTubeCoachDiscovery',
    'VideoProcessor', 
    'TranscriptionService',
    'YouTubeTranscriber',
    'ConversationFormatter',
    'CoachingExamplesGenerator'
]