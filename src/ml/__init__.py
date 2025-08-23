"""
ClaudeWatch Machine Learning Components
Feature extraction, model training, and explainability
"""

from .feature_extraction import FeatureExtractor, generate_discriminative_features
from .train_classifier import train_enhanced_classifier, load_diverse_examples
from .shap_explainer import SHAPExplainer

__all__ = [
    'FeatureExtractor',
    'generate_discriminative_features',
    'train_enhanced_classifier',
    'load_diverse_examples',
    'SHAPExplainer'
]