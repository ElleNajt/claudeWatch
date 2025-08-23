#!/usr/bin/env python3
"""
ClaudeWatch Notification Management
Handles different notification methods (CLI, Emacs, logging)
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List


class NotificationManager:
    """Handles different notification methods"""

    def __init__(self, methods: List[str]):
        self.methods = methods

    def send(self, message: str, alert_level: str = "info"):
        """Send notification through configured methods"""
        for method in self.methods:
            try:
                if method == "cli":
                    self._send_cli(message, alert_level)
                elif method == "emacs":
                    self._send_emacs(message, alert_level)
                elif method == "log":
                    self._send_log(message, alert_level)
                else:
                    print(f"Unknown notification method: {method}")
            except Exception as e:
                print(f"Failed to send {method} notification: {e}")

    def _send_cli(self, message: str, alert_level: str):
        """Send to stderr for Claude Code CLI integration"""
        print(f"❌ ClaudeWatch: {message}", file=sys.stderr)

    def _send_emacs(self, message: str, alert_level: str):
        """Send notification to Emacs via emacsclient"""
        try:
            # Try to send message to Emacs
            subprocess.run(
                ["emacsclient", "-e", f'(message "ClaudeWatch: {message}")'],
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to CLI if emacs not available
            self._send_cli(message, alert_level)

    def _send_log(self, message: str, alert_level: str):
        """Log notification to file"""
        # Determine log directory - use project directory if available
        if os.environ.get("CLAUDE_PROJECT_DIR"):
            project_dir = Path(os.environ["CLAUDE_PROJECT_DIR"])
        else:
            # Fallback to current working directory
            project_dir = Path.cwd()

        # Use project directory for logs
        try:
            log_dir = project_dir / "logs"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "notifications.log"
        except (PermissionError, OSError):
            # Fallback to ClaudeWatch project directory if project dir isn't writable
            project_root = Path(__file__).parent.parent.parent
            log_file = project_root / "logs" / "notifications.log"
            log_file.parent.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp} [{alert_level.upper()}] {message}\n")


def send_notification(message: str, methods: List[str], alert_level: str = "info"):
    """Convenience function for sending notifications"""
    manager = NotificationManager(methods)
    manager.send(message, alert_level)