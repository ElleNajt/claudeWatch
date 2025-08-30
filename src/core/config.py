#!/usr/bin/env python3
"""
ClaudeWatch Configuration Management
Handles configuration loading and validation
"""

import json
from dataclasses import dataclass
from typing import List, Union, Dict


@dataclass
class WatchConfig:
    """Configuration for behavior monitoring"""

    good_examples_path: Union[str, List[str]] = None
    bad_examples_path: str = None
    alert_threshold: float = 2.0  # Alert if bad/good ratio > this
    feature_threshold: float = 0.02  # Show features with activation > this
    alert_strategy: str = "any_bad_feature"  # "any_bad_feature", "ratio", "quality", "logistic_regression"
    logistic_threshold: float = (
        0.7  # Alert if P(projective) > this for logistic regression
    )
    notification_methods: List[str] = None  # ['cli', 'emacs', 'log']
    model: str = "meta-llama/Llama-3.3-70B-Instruct"  # Model to use for analysis

    # Configurable alert messages
    good_alert_message: str = "Good behavior detected!"
    bad_alert_message: str = "Bad behavior detected!"
    good_behavior_label: str = "GOOD"
    bad_behavior_label: str = "BAD"
    
    # Vector configuration
    _vector_source: str = None  # Optional custom vector file to use instead of auto-generated
    direct_vectors: Dict = None  # Optional direct vector specification with 'good' and 'bad' lists
    model_path: str = None  # Optional path to pre-trained classifier model

    def __post_init__(self):
        if self.notification_methods is None:
            self.notification_methods = ["cli"]

    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Filter out metadata keys that start with _, except vector configuration
        config_data = {k: v for k, v in data.items() if not k.startswith('_') or k == '_vector_source'}
        
        return cls(**config_data)

    def to_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            # Convert dataclass to dict, excluding None values
            config_dict = {k: v for k, v in self.__dict__.items() if v is not None}
            json.dump(config_dict, f, indent=2)

    def validate(self):
        """Validate configuration parameters"""
        errors = []
        
        # Check that either example paths OR direct_vectors are provided
        has_example_paths = self.good_examples_path and self.bad_examples_path
        has_direct_vectors = self.direct_vectors is not None
        
        if not has_example_paths and not has_direct_vectors:
            errors.append("Either (good_examples_path AND bad_examples_path) OR direct_vectors must be provided")
        
        # If using example paths, validate them
        if has_example_paths:
            if isinstance(self.good_examples_path, list) and len(self.good_examples_path) == 0:
                errors.append("good_examples_path list cannot be empty")
        
        # If using direct vectors, validate them
        if has_direct_vectors:
            if not isinstance(self.direct_vectors, dict):
                errors.append("direct_vectors must be a dictionary")
            elif 'good' not in self.direct_vectors or 'bad' not in self.direct_vectors:
                errors.append("direct_vectors must have 'good' and 'bad' keys")
            elif not isinstance(self.direct_vectors['good'], list) or not isinstance(self.direct_vectors['bad'], list):
                errors.append("direct_vectors 'good' and 'bad' must be lists")
        
        # Check thresholds
        if self.alert_threshold <= 0:
            errors.append("alert_threshold must be positive")
        if not 0 <= self.feature_threshold <= 1:
            errors.append("feature_threshold must be between 0 and 1")
        if not 0 <= self.logistic_threshold <= 1:
            errors.append("logistic_threshold must be between 0 and 1")
        
        # Check alert strategy
        valid_strategies = ["any_bad_feature", "ratio", "quality", "logistic_regression"]
        if self.alert_strategy not in valid_strategies:
            errors.append(f"alert_strategy must be one of: {valid_strategies}")
        
        # Check notification methods
        valid_methods = ["cli", "emacs", "log"]
        for method in self.notification_methods:
            if method not in valid_methods:
                errors.append(f"Unknown notification method: {method}. Valid: {valid_methods}")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        return True