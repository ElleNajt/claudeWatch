#!/usr/bin/env python3
"""
SHAP Explainer for ClaudeWatch
Model interpretability and feature importance analysis
"""

import numpy as np
from typing import List, Dict, Optional, Tuple

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """SHAP-based model explainer for ClaudeWatch classifiers"""
    
    def __init__(self, model, feature_names: List[str], training_data: Optional[np.ndarray] = None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained sklearn classifier
            feature_names: List of feature names/labels
            training_data: Training data for explainer initialization (optional)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        
        # Initialize explainer based on model type
        if hasattr(model, 'coef_'):
            # Linear model - use LinearExplainer
            if training_data is not None:
                self.explainer = shap.LinearExplainer(model, training_data)
            else:
                # Use zero baseline for linear models
                baseline = np.zeros((1, len(feature_names)))
                self.explainer = shap.LinearExplainer(model, baseline)
        else:
            # Tree-based or other models - use TreeExplainer or general Explainer
            if training_data is not None:
                self.explainer = shap.Explainer(model, training_data)
            else:
                self.explainer = shap.Explainer(model)
    
    def explain_prediction(self, feature_vector: np.ndarray, 
                          top_k: int = 5) -> Dict:
        """
        Explain a single prediction with SHAP values
        
        Args:
            feature_vector: Feature vector to explain (1D array)
            top_k: Number of top features to return
            
        Returns:
            Dictionary with explanation details
        """
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(feature_vector)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # Multi-class output - take positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        if shap_values.ndim > 1:
            shap_values = shap_values[0]  # Take first sample
        
        # Get model prediction
        prediction = self.model.predict(feature_vector)[0]
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(feature_vector)[0]
        else:
            probabilities = None
        
        # Create feature importance ranking
        feature_importance = list(zip(self.feature_names, shap_values))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Get top contributing features
        top_features = feature_importance[:top_k]
        
        # Format explanation
        explanation = {
            'prediction': int(prediction),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'shap_values': shap_values.tolist(),
            'feature_names': self.feature_names,
            'top_features': [
                {
                    'name': name,
                    'shap_value': float(value),
                    'importance': abs(float(value)),
                    'direction': 'positive' if value > 0 else 'negative'
                }
                for name, value in top_features
            ],
            'explanation_text': self._format_explanation_text(top_features, prediction)
        }
        
        return explanation
    
    def _format_explanation_text(self, top_features: List[Tuple[str, float]], 
                                prediction: int) -> str:
        """Format human-readable explanation text"""
        if not top_features:
            return "No significant features identified"
        
        pred_label = "projective" if prediction == 1 else "authentic"
        
        explanations = []
        for name, shap_value in top_features[:3]:  # Top 3 features
            direction = "projective" if shap_value > 0 else "authentic"
            strength = "strongly" if abs(shap_value) > 0.1 else "moderately" if abs(shap_value) > 0.05 else "slightly"
            
            # Truncate long feature names for readability
            short_name = name[:50] + "..." if len(name) > 50 else name
            explanations.append(f"{short_name} {strength} pushes toward {direction} ({shap_value:+.3f})")
        
        return f"Predicted: {pred_label}. " + "; ".join(explanations)
    
    def batch_explain(self, feature_vectors: np.ndarray, 
                     top_k: int = 5) -> List[Dict]:
        """
        Explain multiple predictions
        
        Args:
            feature_vectors: Array of feature vectors (2D)
            top_k: Number of top features per explanation
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for i in range(len(feature_vectors)):
            explanation = self.explain_prediction(feature_vectors[i], top_k)
            explanations.append(explanation)
        
        return explanations
    
    def get_global_feature_importance(self, feature_vectors: np.ndarray) -> Dict:
        """
        Get global feature importance across a dataset
        
        Args:
            feature_vectors: Array of feature vectors (2D)
            
        Returns:
            Dictionary with global importance metrics
        """
        # Get SHAP values for all samples
        shap_values = self.explainer.shap_values(feature_vectors)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Calculate mean absolute SHAP values (global importance)
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Calculate mean SHAP values (directional importance)
        mean_shap = np.mean(shap_values, axis=0)
        
        # Create feature importance ranking
        importance_data = []
        for i, (name, abs_importance, direction_importance) in enumerate(
            zip(self.feature_names, mean_abs_shap, mean_shap)
        ):
            importance_data.append({
                'feature_index': i,
                'name': name,
                'global_importance': float(abs_importance),
                'directional_importance': float(direction_importance),
                'primary_direction': 'projective' if direction_importance > 0 else 'authentic'
            })
        
        # Sort by global importance
        importance_data.sort(key=lambda x: x['global_importance'], reverse=True)
        
        return {
            'global_feature_importance': importance_data,
            'most_important_features': importance_data[:10],
            'total_features': len(self.feature_names)
        }
    
    def create_summary_plot_data(self, feature_vectors: np.ndarray, 
                                max_features: int = 20) -> Dict:
        """
        Create data for SHAP summary plots
        
        Args:
            feature_vectors: Array of feature vectors (2D)
            max_features: Maximum number of features to include
            
        Returns:
            Dictionary with plot data
        """
        # Get SHAP values
        shap_values = self.explainer.shap_values(feature_vectors)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Calculate feature importance for ordering
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(feature_importance)[-max_features:][::-1]
        
        return {
            'shap_values': shap_values[:, top_indices].tolist(),
            'feature_values': feature_vectors[:, top_indices].tolist(),
            'feature_names': [self.feature_names[i] for i in top_indices],
            'feature_indices': top_indices.tolist()
        }


def create_explainer_from_model_file(model_path: str, 
                                   training_data: Optional[np.ndarray] = None) -> Optional[SHAPExplainer]:
    """
    Create SHAP explainer from saved model file
    
    Args:
        model_path: Path to pickled model file
        training_data: Optional training data for explainer
        
    Returns:
        SHAPExplainer instance or None if not available
    """
    if not SHAP_AVAILABLE:
        return None
    
    import pickle
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        features = model_data['features']
        feature_names = [f['label'] for f in features]
        
        # Check if explainer is already saved
        if 'explainer' in model_data and model_data['explainer']:
            # Use pre-trained explainer
            return model_data['explainer']
        else:
            # Create new explainer
            return SHAPExplainer(model, feature_names, training_data)
            
    except Exception as e:
        print(f"Failed to create SHAP explainer: {e}")
        return None