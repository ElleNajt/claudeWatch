#!/usr/bin/env python3
"""
ClaudeWatch Core Monitoring Engine
Main behavior monitoring functionality using SAE features
"""

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import goodfire
    from goodfire import Client
except ImportError:
    print("Error: Goodfire not installed. Run: pip install goodfire")
    exit(1)

try:
    from dotenv import load_dotenv
    # Load .env file if it exists
    load_dotenv()
except ImportError:
    # dotenv is optional
    pass

from .config import WatchConfig
from .notifications import NotificationManager


class ClaudeWatch:
    """Simple behavior monitor using discriminative SAE features"""

    def __init__(self, config: WatchConfig):
        """Initialize with configuration"""
        # Validate configuration
        config.validate()
        
        self.config = config
        
        # Check for API key
        api_key = os.environ.get("GOODFIRE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOODFIRE_API_KEY not found. Please either:\n"
                "  1. Create a .env file with: GOODFIRE_API_KEY=your_api_key\n"
                "  2. Set environment variable: export GOODFIRE_API_KEY=your_api_key"
            )
        
        self.client = Client(api_key=api_key)
        self.features = None
        self.notifier = NotificationManager(config.notification_methods)
        self.classifier_model = None
        self.shap_explainer = None
        
        # Load cached vectors and classifier
        self._load_cached_vectors()
        self._load_classifier_if_needed()

    def _load_cached_vectors(self):
        """Load cached vectors (only cached vectors are supported)"""
        cache_path = self._get_cache_path()

        print(f"Loading cached vectors from: {cache_path}")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cached vectors not found at {cache_path}. "
                "Please generate vectors first using generate_vectors.py"
            )

        print("Loading cached discriminative vectors...")
        with open(cache_path, "r") as f:
            vector_data = json.load(f)
        self.features = vector_data["features"]

        # Reconstruct Feature objects from cached data
        self.good_features = []
        self.bad_features = []

        for feat_data in self.features:
            feature_obj = goodfire.Feature(
                uuid=feat_data["uuid"],
                label=feat_data["label"],
                index_in_sae=feat_data["index_in_sae"],
            )
            if feat_data["type"] == "good":
                self.good_features.append(feature_obj)
            else:
                self.bad_features.append(feature_obj)

        print(
            f"Loaded {len(self.good_features)} good and {len(self.bad_features)} bad features"
        )

    def _load_classifier_if_needed(self):
        """Load logistic regression classifier if using that strategy"""
        if self.config.alert_strategy != "logistic_regression":
            return

        try:
            import shap
            SHAP_AVAILABLE = True
        except ImportError:
            print("Warning: SHAP not installed. Run: pip install shap")
            print("SHAP explanations will be disabled.")
            SHAP_AVAILABLE = False

        # Build classifier path
        good_name = Path(self.config.good_examples_path).stem
        bad_name = Path(self.config.bad_examples_path).stem
        model_name = self.config.model.split("/")[-1].replace("-", "_")
        script_dir = Path(__file__).parent.parent.parent
        
        # Try enhanced classifier first, fallback to original
        enhanced_model_path = script_dir / f"models/enhanced_classifier_{good_name}_vs_{bad_name}_{model_name}_*.pkl"
        model_paths = list(script_dir.glob(f"models/enhanced_classifier_{good_name}_vs_{bad_name}_{model_name}_*.pkl"))
        
        if model_paths:
            # Use most recent enhanced model
            model_path = max(model_paths, key=lambda p: p.stat().st_mtime)
        else:
            # Fallback to original model
            model_path = script_dir / f"models/projective_classifier_{good_name}_vs_{bad_name}_{model_name}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Classifier model not found at {model_path}. "
                "Please train classifier first using train_classifier.py"
            )

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.classifier_model = model_data["model"]

        # Load SHAP explainer if available
        if SHAP_AVAILABLE and "explainer" in model_data and model_data["explainer"]:
            self.shap_explainer = model_data["explainer"]
            print("Loaded classifier with SHAP explanations enabled")
        elif SHAP_AVAILABLE:
            # Create new SHAP explainer
            try:
                self.shap_explainer = shap.LinearExplainer(
                    self.classifier_model,
                    shap.maskers.Independent(data=np.zeros((1, len(self.features)))),
                )
                print("Created new SHAP explainer for classifier")
            except Exception as e:
                print(f"Failed to create SHAP explainer: {e}")
                self.shap_explainer = None
        else:
            self.shap_explainer = None
            print("Loaded classifier (SHAP explanations disabled)")

    def _get_cache_path(self) -> str:
        """Get path to cached vectors"""
        good_name = Path(self.config.good_examples_path).stem
        bad_name = Path(self.config.bad_examples_path).stem
        model_name = self.config.model.split("/")[-1].replace("-", "_")
        script_dir = Path(__file__).parent.parent.parent
        return str(script_dir / f"data/vectors/discriminative_{good_name}_vs_{bad_name}_{model_name}.json")

    def analyze(self, text: str) -> Dict:
        """Analyze text for behavioral patterns"""
        print(f"Analyzing: {text[:100]}...")

        # Get feature activations
        messages = [{"role": "assistant", "content": text}]
        all_activations = self.client.features.activations(
            messages=messages, model=self.config.model
        )
        mean_activations = all_activations.mean(axis=0)

        # Extract activations for our features
        good_activations = []
        bad_activations = []

        for feature in self.good_features:
            activation = mean_activations[feature.index_in_sae]
            good_activations.append(float(activation))

        for feature in self.bad_features:
            activation = mean_activations[feature.index_in_sae]
            bad_activations.append(float(activation))

        # Show activated features
        activated_features = []
        for i, feature in enumerate(self.good_features):
            if good_activations[i] > self.config.feature_threshold:
                activated_features.append({
                    "type": "good",
                    "label": feature.label,
                    "activation": good_activations[i]
                })

        for i, feature in enumerate(self.bad_features):
            if bad_activations[i] > self.config.feature_threshold:
                activated_features.append({
                    "type": "bad", 
                    "label": feature.label,
                    "activation": bad_activations[i]
                })

        # Determine if we should alert
        alert, explanation = self._should_alert(
            activated_features, good_activations, bad_activations
        )

        return {
            "text": text,
            "alert": alert,
            "activated_features": activated_features,
            "explanation": explanation,
            "good_activations": good_activations,
            "bad_activations": bad_activations
        }

    def _should_alert(self, activated_features: List[Dict], 
                      good_activations: List[float], bad_activations: List[float]) -> tuple:
        """Determine if we should alert based on strategy"""
        
        if self.config.alert_strategy == "logistic_regression":
            return self._logistic_alert(good_activations, bad_activations)
        elif self.config.alert_strategy == "any_bad_feature":
            return self._any_bad_feature_alert(activated_features)
        elif self.config.alert_strategy == "ratio":
            return self._ratio_alert(good_activations, bad_activations)
        elif self.config.alert_strategy == "quality":
            return self._quality_alert(good_activations, bad_activations)
        else:
            raise ValueError(f"Unknown alert strategy: {self.config.alert_strategy}")

    def _logistic_alert(self, good_activations: List[float], bad_activations: List[float]) -> tuple:
        """Use logistic regression classifier for alert decision"""
        if not self.classifier_model:
            return False, {"error": "Classifier not loaded"}
        
        # Create feature vector
        feature_vector = np.array(good_activations + bad_activations).reshape(1, -1)
        
        # Get prediction
        prediction = self.classifier_model.predict(feature_vector)[0]
        proba = self.classifier_model.predict_proba(feature_vector)[0, 1]  # P(projective)
        
        # Get SHAP explanation if available
        explanation = {
            "prediction": "projective" if prediction == 1 else "authentic",
            "probability": float(proba),
            "shap_values": None
        }
        
        if self.shap_explainer:
            try:
                shap_values = self.shap_explainer.shap_values(feature_vector)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Get positive class SHAP values
                explanation["shap_values"] = shap_values[0].tolist()
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
        
        # Alert if prediction is projective and confidence exceeds threshold
        should_alert = prediction == 1 and proba > self.config.logistic_threshold
        
        return should_alert, explanation

    def _any_bad_feature_alert(self, activated_features: List[Dict]) -> tuple:
        """Alert if any bad feature is activated"""
        bad_features = [f for f in activated_features if f["type"] == "bad"]
        should_alert = len(bad_features) > 0
        
        explanation = {
            "strategy": "any_bad_feature",
            "bad_features_count": len(bad_features),
            "activated_bad_features": bad_features
        }
        
        return should_alert, explanation

    def _ratio_alert(self, good_activations: List[float], bad_activations: List[float]) -> tuple:
        """Alert if bad/good ratio exceeds threshold"""
        total_good = sum(good_activations)
        total_bad = sum(bad_activations)
        
        if total_good == 0:
            ratio = float('inf') if total_bad > 0 else 0
        else:
            ratio = total_bad / total_good
        
        should_alert = ratio > self.config.alert_threshold
        
        explanation = {
            "strategy": "ratio",
            "ratio": ratio,
            "threshold": self.config.alert_threshold,
            "total_good": total_good,
            "total_bad": total_bad
        }
        
        return should_alert, explanation

    def _quality_alert(self, good_activations: List[float], bad_activations: List[float]) -> tuple:
        """Alert based on overall quality assessment"""
        total_good = sum(good_activations)
        total_bad = sum(bad_activations)
        
        if total_good + total_bad == 0:
            quality = "neutral"
        elif total_bad > total_good * self.config.alert_threshold:
            quality = "bad"
        else:
            quality = "good"
        
        should_alert = quality == "bad"
        
        explanation = {
            "strategy": "quality", 
            "quality": quality,
            "total_good": total_good,
            "total_bad": total_bad
        }
        
        return should_alert, explanation

    def send_notification(self, result: Dict, text: str):
        """Send notification based on result"""
        if result["alert"]:
            message = f"{self.config.bad_alert_message}"
            
            # Add explanation details for logistic regression
            if self.config.alert_strategy == "logistic_regression" and "explanation" in result:
                exp = result["explanation"]
                prob = exp.get("probability", 0)
                message += f" Predicted: {exp.get('prediction', 'unknown')} (P={prob:.3f})"
                
                # Add SHAP explanation if available
                if exp.get("shap_values") and self.features:
                    # Get top contributing features
                    shap_values = exp["shap_values"]
                    feature_names = [f["label"] for f in self.features]
                    
                    # Get top 2 most influential features
                    feature_importance = list(zip(feature_names, shap_values))
                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    if feature_importance:
                        top_features = []
                        for name, value in feature_importance[:2]:
                            direction = "projective" if value > 0 else "authentic"
                            # Truncate long feature names
                            short_name = name[:40] + "..." if len(name) > 40 else name
                            top_features.append(f"{short_name}({value:+.3f}→{direction})")
                        
                        message += f" | Why: {', '.join(top_features)}"
            
            self.notifier.send(message, alert_level="warning")
        else:
            # Only send good notifications if explicitly requested
            if "good" in self.config.notification_methods:
                message = f"{self.config.good_alert_message}"
                self.notifier.send(message, alert_level="info")


def main():
    """CLI entry point"""
    if len(sys.argv) < 3:
        print("Usage: python claude_watch.py <config.json> <text_to_analyze>")
        sys.exit(1)

    config_path = sys.argv[1]
    text = sys.argv[2]

    try:
        config = WatchConfig.from_json(config_path)
        watch = ClaudeWatch(config)
        result = watch.analyze(text)
        
        # Print analysis results
        print(f"\n🔍 Analysis Results:")
        print(f"Alert: {result['alert']}")
        
        if result.get('explanation'):
            exp = result['explanation']
            if 'prediction' in exp:
                print(f"Prediction: {exp['prediction']} (confidence: {exp.get('probability', 0):.3f})")
        
        print(f"\nActivated features:")
        for feature in result['activated_features']:
            print(f"  {feature['type']}: {feature['label']} ({feature['activation']:.3f})")
        
        # Send notifications
        watch.send_notification(result, text)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()