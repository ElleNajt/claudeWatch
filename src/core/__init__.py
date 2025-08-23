"""
ClaudeWatch Core Components
Core functionality for AI behavior monitoring
"""

from .claude_watch import ClaudeWatch
from .config import WatchConfig
from .notifications import NotificationManager

__all__ = [
    'ClaudeWatch',
    'WatchConfig', 
    'NotificationManager'
]