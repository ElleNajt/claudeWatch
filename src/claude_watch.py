#!/usr/bin/env python3
"""
ClaudeWatch - Simple AI behavior monitoring
Generate discriminative vectors from examples, then monitor in real-time
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
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

# Check for API key
GOODFIRE_API_KEY = os.environ.get("GOODFIRE_API_KEY")
if not GOODFIRE_API_KEY:
    print("Error: GOODFIRE_API_KEY not found")
    print("Please either:")
    print("  1. Create a .env file with: GOODFIRE_API_KEY=your_api_key")
    print("  2. Set environment variable: export GOODFIRE_API_KEY=your_api_key")
    exit(1)


class NotificationManager:
    """Handles different notification methods"""

    def __init__(self, methods: List[str]):
        self.methods = methods

    def send(self, message: str, alert_level: str = "info"):
        """Send notification via configured methods"""
        for method in self.methods:
            try:
                if method == "cli":
                    self._send_cli(message, alert_level)
                elif method == "emacs":
                    self._send_emacs(message, alert_level)
                elif method == "log":
                    self._send_log(message, alert_level)
            except Exception as e:
                print(f"Failed to send {method} notification: {e}")

    def _send_cli(self, message: str, alert_level: str):
        """Send to CLI (stderr for hooks)"""
        prefix = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}.get(
            alert_level, "🤖"
        )

        print(f"{prefix} ClaudeWatch: {message}", file=sys.stderr, flush=True)

    def _send_emacs(self, message: str, alert_level: str):
        """Send to Emacs via emacsclient"""
        # Escape message for emacs
        escaped_message = (
            message.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\n", "\\n")
            .replace("\t", "\\t")
        )

        prefix = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "success": "✅"}.get(
            alert_level, "🤖"
        )

        # Send to Emacs
        cmd = [
            "emacsclient",
            "--eval",
            f'(progn (message "{prefix} ClaudeWatch: {escaped_message[:100]}...") (sit-for 3))',
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

    def _send_log(self, message: str, alert_level: str):
        """Send to log file"""
        # Use absolute path relative to project root
        project_root = Path(__file__).parent.parent
        log_file = project_root / "logs" / "notifications.log"
        log_file.parent.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"{timestamp} [{alert_level.upper()}] {message}\n")


@dataclass
class WatchConfig:
    """Configuration for behavior monitoring"""

    good_examples_path: str
    bad_examples_path: str
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

    def __post_init__(self):
        if self.notification_methods is None:
            self.notification_methods = ["cli"]

    @classmethod
    def from_json(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


class ClaudeWatch:
    """Simple behavior monitor using discriminative SAE features"""

    def __init__(self, config: WatchConfig):
        """Initialize with configuration"""
        self.config = config
        self.client = Client(api_key=GOODFIRE_API_KEY)
        self.features = None
        self.notifier = NotificationManager(config.notification_methods)
        self.classifier_model = None
        self.shap_explainer = None
        self._load_cached_vectors()
        self._load_classifier_if_needed()

    def _load_cached_vectors(self):
        """Load cached vectors (only cached vectors are supported)"""
        cache_path = self._get_cache_path()

        print(f"Loading cached vectors from: {cache_path}")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cached vectors not found at {cache_path}. "
                "Please generate vectors first using a separate script."
            )

        print("Loading cached discriminative vectors...")
        with open(cache_path, "r") as f:
            vector_data = json.load(f)
        self.features = vector_data["features"]

        # Reconstruct Feature objects from cached data (like buddhaMindVector does)
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
        if self.config.alert_strategy == "logistic_regression":
            import pickle

            try:
                import shap
            except ImportError:
                print("Warning: SHAP not installed. Run: pip install shap")
                print("SHAP explanations will be disabled.")

            # Build classifier path
            good_name = Path(self.config.good_examples_path).stem
            bad_name = Path(self.config.bad_examples_path).stem
            model_name = self.config.model.split("/")[-1].replace("-", "_")
            script_dir = Path(__file__).parent.parent
            model_path = (
                script_dir
                / f"models/projective_classifier_{good_name}_vs_{bad_name}_{model_name}.pkl"
            )

            if not model_path.exists():
                raise FileNotFoundError(
                    f"Classifier model not found at {model_path}. "
                    "Please train classifier first using train_classifier.py"
                )

            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            self.classifier_model = model_data["model"]

            # Create SHAP explainer if available
            try:
                import shap

                # Use linear explainer for logistic regression
                self.shap_explainer = shap.LinearExplainer(
                    self.classifier_model,
                    shap.maskers.Independent(data=np.zeros((1, len(self.features)))),
                )
                print("Loaded classifier with SHAP explanations enabled")
            except:
                self.shap_explainer = None
                print("Loaded classifier (SHAP explanations disabled)")

    def _get_cache_path(self) -> str:
        """Get path for cached vectors"""
        good_name = Path(self.config.good_examples_path).stem
        bad_name = Path(self.config.bad_examples_path).stem
        # Include model in cache path for uniqueness
        model_name = self.config.model.split("/")[-1].replace("-", "_")
        # Use absolute path to avoid working directory issues
        script_dir = Path(__file__).parent.parent
        return str(
            script_dir
            / f"data/vectors/discriminative_{good_name}_vs_{bad_name}_{model_name}.json"
        )

    def send_notification(self, result: Dict, text: str = ""):
        """Send notifications based on analysis result"""
        quality = result.get("quality", "UNKNOWN")
        activated_features = result.get("activated_features", [])
        classifier_explanation = result.get("classifier_explanation")

        # Format feature labels for display
        def format_features(features, feature_type):
            filtered = [f for f in features if f["type"] == feature_type]
            if not filtered:
                return "None"
            return ", ".join(
                [f"{f['label']} ({f['activation']:.2f})" for f in filtered[:3]]
            )

        # Format SHAP explanation if available
        def format_shap_explanation(explanation):
            if not explanation or explanation.get("shap_values") is None:
                return ""

            shap_values = explanation["shap_values"]
            # Get top 3 features by absolute SHAP value
            feature_names = [f["label"] for f in self.features]
            shap_pairs = list(zip(feature_names, shap_values))
            shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

            top_features = []
            for name, value in shap_pairs[:3]:
                if abs(value) > 0.01:  # Only show meaningful contributions
                    direction = "→projective" if value > 0 else "→authentic"
                    top_features.append(f"{name[:30]}({value:+.2f}{direction})")

            if top_features:
                return f" | Why: {', '.join(top_features)}"
            return ""

        # Handle classifier-based alerts
        if classifier_explanation:
            prediction = classifier_explanation["prediction"]
            probability = classifier_explanation["probability"]
            shap_info = format_shap_explanation(classifier_explanation)

            if result.get("alert"):
                message = f"{self.config.bad_alert_message} Predicted: {prediction} (P={probability:.2f}){shap_info}"
                self.notifier.send(message, "error")
            else:
                message = f"Predicted: {prediction} (P={probability:.2f}){shap_info}"
                level = "success" if prediction == "authentic" else "warning"
                self.notifier.send(message, level)

        # Handle other alert strategies
        elif quality == self.config.bad_behavior_label:
            bad_features = format_features(activated_features, "bad")
            message = f"{self.config.bad_alert_message} Quality: {quality} | Violated: {bad_features}"
            self.notifier.send(message, "error")
        elif quality == self.config.good_behavior_label:
            good_features = format_features(activated_features, "good")
            message = f"{self.config.good_alert_message} Quality: {quality} | Detected: {good_features}"
            self.notifier.send(message, "success")
        elif quality in ["NEUTRAL", "UNCLEAR"]:
            all_features = (
                format_features(activated_features, "bad")
                if activated_features
                else "Low activation"
            )
            message = f"Analysis: {quality}, Good: {result.get('good_score', 0):.2f}, Bad: {result.get('bad_score', 0):.2f} | Features: {all_features}"
            self.notifier.send(message, "info")

    def analyze(self, text: str) -> Dict:
        """Analyze text for behavioral patterns"""
        if not text or not text.strip():
            return {
                "quality": "EMPTY",
                "good_score": 0.0,
                "bad_score": 0.0,
                "ratio": 0.0,
                "alert": False,
                "activated_features": [],
            }

        try:
            # Convert text to messages format (like buddhaMindVector)
            messages = [{"role": "assistant", "content": text}]

            # Get all feature activations using the buddhaMindVector approach
            all_activations = self.client.features.activations(
                messages=messages, model=self.config.model
            )

            # Get feature indices for our specific features
            good_feature_indices = [feat.index_in_sae for feat in self.good_features]
            bad_feature_indices = [feat.index_in_sae for feat in self.bad_features]

            # Calculate mean activation across tokens for each feature (like buddhaMindVector)
            import numpy as np

            mean_activations = all_activations.mean(axis=0)  # Average across tokens

            # Get scores for good and bad features with labels
            good_activations = []
            bad_activations = []
            activated_features = []
            all_activations_list = []  # For building feature vector

            for i, feat in enumerate(self.good_features):
                activation = mean_activations[feat.index_in_sae]
                good_activations.append(activation)
                all_activations_list.append(float(activation))
                if activation > self.config.feature_threshold:
                    activated_features.append(
                        {
                            "label": feat.label,
                            "type": "good",
                            "activation": float(activation),
                        }
                    )

            for i, feat in enumerate(self.bad_features):
                activation = mean_activations[feat.index_in_sae]
                bad_activations.append(activation)
                all_activations_list.append(float(activation))
                if activation > self.config.feature_threshold:
                    activated_features.append(
                        {
                            "label": feat.label,
                            "type": "bad",
                            "activation": float(activation),
                        }
                    )

            # Create feature vector for classifier
            feature_vector = np.array(all_activations_list)

            # Calculate scores
            good_score = sum(good_activations)
            bad_score = sum(bad_activations)

            # Determine quality assessment using buddhaMindVector logic
            total_score = good_score + bad_score
            if total_score < 0.1:  # Very low overall activation
                quality = "UNCLEAR"
            else:
                bad_ratio = bad_score / total_score
                good_ratio = good_score / total_score

                if bad_ratio > 0.65:  # Bad features dominate
                    quality = self.config.bad_behavior_label
                elif good_ratio > 0.65:  # Good features dominate
                    quality = self.config.good_behavior_label
                else:
                    quality = "UNCLEAR"

            # Calculate ratio for compatibility with alert threshold
            ratio = (
                bad_score / (good_score + 0.01)
                if good_score > 0
                else float("inf")
                if bad_score > 0.01
                else 0
            )

            # Determine alert based on configured strategy
            alert, explanation = self._should_alert(
                quality, ratio, activated_features, feature_vector
            )

            result_dict = {
                "quality": quality,
                "good_score": good_score,
                "bad_score": bad_score,
                "ratio": ratio,
                "alert": alert,
                "activated_features": activated_features,
            }

            # Add classifier explanation if available
            if explanation:
                result_dict["classifier_explanation"] = explanation

            return result_dict

        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                "quality": "ERROR",
                "good_score": 0.0,
                "bad_score": 0.0,
                "ratio": 0.0,
                "alert": False,
                "error": str(e),
                "activated_features": [],
            }

    def _should_alert(
        self,
        quality: str,
        ratio: float,
        activated_features: List[Dict],
        feature_vector: Optional[np.ndarray] = None,
    ) -> tuple[bool, Optional[Dict]]:
        """Determine if we should alert based on configured strategy

        Returns: (should_alert, explanation_dict)
        """
        if self.config.alert_strategy == "logistic_regression":
            if self.classifier_model is None or feature_vector is None:
                return False, None

            # Get prediction and probability
            prediction = self.classifier_model.predict(feature_vector.reshape(1, -1))[0]
            proba = self.classifier_model.predict_proba(feature_vector.reshape(1, -1))[
                0, 1
            ]

            # Get SHAP values if available
            shap_values = None
            if self.shap_explainer is not None:
                try:
                    import shap

                    shap_values = self.shap_explainer.shap_values(
                        feature_vector.reshape(1, -1)
                    )
                    if isinstance(shap_values, list):
                        shap_values = shap_values[0]
                    shap_values = shap_values.flatten()
                except:
                    pass

            explanation = {
                "prediction": "projective" if prediction == 1 else "authentic",
                "probability": float(proba),
                "shap_values": shap_values,
            }

            # Alert if predicted as projective (bad) with confidence above threshold
            return (
                prediction == 1 and proba > self.config.logistic_threshold,
                explanation,
            )

        elif self.config.alert_strategy == "any_bad_feature":
            # Alert if any bad feature exceeds threshold
            bad_features_violated = [
                feat
                for feat in activated_features
                if feat["type"] == "bad"
                and feat["activation"] > self.config.feature_threshold
            ]
            return len(bad_features_violated) > 0, None

        elif self.config.alert_strategy == "ratio":
            # Alert if bad/good ratio exceeds threshold
            return ratio > self.config.alert_threshold, None

        elif self.config.alert_strategy == "quality":
            # Alert if quality matches bad behavior label
            return quality == self.config.bad_behavior_label, None

        else:
            # Default to any_bad_feature
            bad_features_violated = [
                feat
                for feat in activated_features
                if feat["type"] == "bad"
                and feat["activation"] > self.config.feature_threshold
            ]
            return len(bad_features_violated) > 0, None


def main():
    """Example usage"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python claude_watch.py <config.json> [text]")
        sys.exit(1)

    config = WatchConfig.from_json(sys.argv[1])
    watch = ClaudeWatch(config)

    # Test text
    text = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "You should definitely quit your job right now."
    )
    result = watch.analyze(text)

    print(f"\nAnalyzing: {text[:100]}...")
    print(f"Quality: {result['quality']}")
    print(f"Good score: {result['good_score']:.3f}")
    print(f"Bad score: {result['bad_score']:.3f}")
    print(f"Ratio: {result['ratio']:.2f}")
    print(f"Alert: {result['alert']}")


if __name__ == "__main__":
    main()
