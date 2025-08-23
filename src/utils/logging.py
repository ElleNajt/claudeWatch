#!/usr/bin/env python3
"""
Logging Utilities for ClaudeWatch
Centralized logging configuration and progress tracking
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 log_format: Optional[str] = None) -> logging.Logger:
    """
    Setup centralized logging for ClaudeWatch
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        log_format: Custom log format string
        
    Returns:
        Configured logger instance
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create logger
    logger = logging.getLogger('claudewatch')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (uses claudewatch if None)
        
    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger('claudewatch')
    else:
        return logging.getLogger(f'claudewatch.{name}')


class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total_items: int, description: str = "Processing",
                 logger: Optional[logging.Logger] = None):
        self.total_items = total_items
        self.description = description
        self.logger = logger or get_logger('progress')
        self.completed = 0
        self.start_time = datetime.now()
        self.last_update = self.start_time
        
    def update(self, increment: int = 1, message: str = None):
        """Update progress"""
        self.completed += increment
        now = datetime.now()
        
        # Update every 10% or every 30 seconds
        progress_pct = (self.completed / self.total_items) * 100
        should_update = (
            progress_pct % 10 < (progress_pct - increment/self.total_items*100) % 10 or
            (now - self.last_update).total_seconds() > 30
        )
        
        if should_update or self.completed >= self.total_items:
            elapsed = now - self.start_time
            
            if self.completed > 0:
                rate = self.completed / elapsed.total_seconds()
                eta_seconds = (self.total_items - self.completed) / rate if rate > 0 else 0
                eta = f"{int(eta_seconds//60)}m {int(eta_seconds%60)}s"
            else:
                eta = "unknown"
            
            status_msg = f"{self.description}: {self.completed}/{self.total_items} ({progress_pct:.1f}%) - ETA: {eta}"
            
            if message:
                status_msg += f" - {message}"
            
            self.logger.info(status_msg)
            self.last_update = now
    
    def finish(self, message: str = None):
        """Mark as finished"""
        elapsed = datetime.now() - self.start_time
        final_msg = f"{self.description} completed: {self.completed}/{self.total_items} in {elapsed}"
        
        if message:
            final_msg += f" - {message}"
        
        self.logger.info(final_msg)


class TimedOperation:
    """Context manager for timing operations"""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger('timer')
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation_name} in {elapsed}")
        else:
            self.logger.error(f"Failed {self.operation_name} after {elapsed}: {exc_val}")


def log_function_call(logger: Optional[logging.Logger] = None):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger('function_calls')
            func_name = func.__name__
            
            # Log function entry
            func_logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                func_logger.debug(f"Completed {func_name}")
                return result
            except Exception as e:
                func_logger.error(f"Error in {func_name}: {e}")
                raise
        
        return wrapper
    return decorator


def log_performance_metrics(operation: str, metrics: Dict[str, Any], 
                          logger: Optional[logging.Logger] = None):
    """Log performance metrics"""
    perf_logger = logger or get_logger('performance')
    
    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    perf_logger.info(f"Performance [{operation}]: {metrics_str}")


def create_session_log_file(base_dir: str = "logs", 
                          prefix: str = "claudewatch") -> str:
    """Create a unique log file for this session"""
    log_dir = Path(base_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{prefix}_{timestamp}.log"
    
    return str(log_file)


class StepLogger:
    """Logger for multi-step processes"""
    
    def __init__(self, process_name: str, total_steps: int,
                 logger: Optional[logging.Logger] = None):
        self.process_name = process_name
        self.total_steps = total_steps
        self.logger = logger or get_logger('steps')
        self.current_step = 0
        self.start_time = datetime.now()
    
    def step(self, step_name: str, details: str = None):
        """Log a step completion"""
        self.current_step += 1
        
        elapsed = datetime.now() - self.start_time
        
        step_msg = f"[{self.current_step}/{self.total_steps}] {self.process_name}: {step_name}"
        
        if details:
            step_msg += f" - {details}"
        
        step_msg += f" (elapsed: {elapsed})"
        
        self.logger.info(step_msg)
    
    def finish(self, summary: str = None):
        """Log process completion"""
        elapsed = datetime.now() - self.start_time
        
        final_msg = f"✅ {self.process_name} completed in {elapsed}"
        
        if summary:
            final_msg += f" - {summary}"
        
        self.logger.info(final_msg)
    
    def error(self, error_msg: str):
        """Log process error"""
        elapsed = datetime.now() - self.start_time
        
        error_log = f"❌ {self.process_name} failed after {elapsed}: {error_msg}"
        self.logger.error(error_log)