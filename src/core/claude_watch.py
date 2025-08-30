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
    GOODFIRE_AVAILABLE = True
except ImportError:
    print("Warning: Goodfire not installed. Claude prompt strategy will still work.")
    goodfire = None
    Client = None
    GOODFIRE_AVAILABLE = False

try:
    from dotenv import load_dotenv
    # Load .env file if it exists
    load_dotenv()
except ImportError:
    # dotenv is optional
    pass

try:
    from .config import WatchConfig
    from .notifications import NotificationManager
except ImportError:
    # For direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from config import WatchConfig
    from notifications import NotificationManager


class ClaudeWatch:
    """Simple behavior monitor using discriminative SAE features"""

    def __init__(self, config: WatchConfig):
        """Initialize with configuration"""
        # Validate configuration
        config.validate()
        
        self.config = config
        self.notifier = NotificationManager(config.notification_methods)
        self.features = None
        self.classifier_model = None
        self.shap_explainer = None
        self.client = None
        
        # Skip Goodfire initialization for claude_prompt strategy
        if config.alert_strategy == "claude_prompt":
            print("Using claude_prompt strategy - skipping Goodfire initialization")
            return
        
        # Check Goodfire availability for SAE-based strategies
        if not GOODFIRE_AVAILABLE:
            raise ValueError(
                "Goodfire not available but required for SAE-based strategies. "
                "Please run: pip install goodfire"
            )
        
        # Check for API key for SAE-based strategies
        api_key = os.environ.get("GOODFIRE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOODFIRE_API_KEY not found. Please either:\n"
                "  1. Create a .env file with: GOODFIRE_API_KEY=your_api_key\n"
                "  2. Set environment variable: export GOODFIRE_API_KEY=your_api_key"
            )
        
        self.client = Client(api_key=api_key)
        
        # Load vectors (cached, direct, or auto-generated) and classifier
        self._load_vectors()
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

    def _load_vectors(self):
        """Load vectors using priority: direct_vectors > cached vectors > error"""
        if self.config.direct_vectors:
            self._load_direct_vectors()
        else:
            self._load_cached_vectors()
    
    def _load_direct_vectors(self):
        """Load vectors directly specified in config"""
        print("Loading directly specified vectors from config...")
        
        direct_vectors = self.config.direct_vectors
        if not isinstance(direct_vectors, dict) or 'good' not in direct_vectors or 'bad' not in direct_vectors:
            raise ValueError("direct_vectors must be a dict with 'good' and 'bad' keys")
        
        good_specs = direct_vectors['good']
        bad_specs = direct_vectors['bad']
        
        if not isinstance(good_specs, list) or not isinstance(bad_specs, list):
            raise ValueError("direct_vectors 'good' and 'bad' must be lists")
        
        # Initialize feature lists
        self.good_features = []
        self.bad_features = []
        self.features = []
        
        # Process good vectors
        for spec in good_specs:
            if not isinstance(spec, dict) or 'uuid' not in spec:
                raise ValueError("Each vector spec must be a dict with 'uuid' key")
            
            # Create Feature object
            feature_obj = goodfire.Feature(
                uuid=spec['uuid'],
                label=spec.get('label', f"Good feature {spec['uuid'][:8]}"),
                index_in_sae=spec.get('index_in_sae', 0),  # Will be populated when needed
            )
            self.good_features.append(feature_obj)
            
            # Add to features list for compatibility
            self.features.append({
                'uuid': spec['uuid'],
                'label': spec.get('label', f"Good feature {spec['uuid'][:8]}"),
                'index_in_sae': spec.get('index_in_sae', 0),
                'type': 'good'
            })
        
        # Process bad vectors
        for spec in bad_specs:
            if not isinstance(spec, dict) or 'uuid' not in spec:
                raise ValueError("Each vector spec must be a dict with 'uuid' key")
            
            # Create Feature object
            feature_obj = goodfire.Feature(
                uuid=spec['uuid'],
                label=spec.get('label', f"Bad feature {spec['uuid'][:8]}"),
                index_in_sae=spec.get('index_in_sae', 0),  # Will be populated when needed
            )
            self.bad_features.append(feature_obj)
            
            # Add to features list for compatibility
            self.features.append({
                'uuid': spec['uuid'],
                'label': spec.get('label', f"Bad feature {spec['uuid'][:8]}"),
                'index_in_sae': spec.get('index_in_sae', 0),
                'type': 'bad'
            })
        
        print(f"Loaded {len(self.good_features)} good and {len(self.bad_features)} bad features directly from config")

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
        if hasattr(self.config, 'model_path') and self.config.model_path:
            # Use specified model path
            model_path = Path(self.config.model_path)
        else:
            # Auto-generate path from example names (legacy behavior)
            if isinstance(self.config.good_examples_path, list):
                # Create name from multiple files
                good_names = [Path(p).stem for p in self.config.good_examples_path]
                good_name = "_plus_".join(good_names)
            else:
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
        # Check if config specifies a custom vector source
        if hasattr(self.config, '_vector_source') and self.config._vector_source:
            script_dir = Path(__file__).parent.parent.parent
            return str(script_dir / f"data/vectors/{self.config._vector_source}")
        
        # Default behavior for auto-generated vectors
        if isinstance(self.config.good_examples_path, list):
            # Create name from multiple files
            good_names = [Path(p).stem for p in self.config.good_examples_path]
            good_name = "_plus_".join(good_names)
        else:
            good_name = Path(self.config.good_examples_path).stem
        bad_name = Path(self.config.bad_examples_path).stem
        model_name = self.config.model.split("/")[-1].replace("-", "_")
        script_dir = Path(__file__).parent.parent.parent
        return str(script_dir / f"data/vectors/discriminative_{good_name}_vs_{bad_name}_{model_name}.json")

    def analyze(self, input_data) -> Dict:
        """Analyze text or conversation for behavioral patterns"""
        
        # Handle both text and conversation formats
        if isinstance(input_data, str):
            # Legacy text format
            text = input_data
            messages = [{"role": "assistant", "content": text}]
            print(f"Analyzing: {text[:100]}...")
            analysis_text = text
        elif isinstance(input_data, list):
            # Conversation format
            messages = input_data
            # Create a summary for logging
            last_assistant = None
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    last_assistant = msg.get('content', '')[:100]
                    break
            print(f"Analyzing conversation (last response): {last_assistant}...")
            
            # Extract text for claude_prompt analysis
            analysis_text = ""
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    analysis_text = msg.get('content', '')
                    break
        else:
            raise ValueError("Input must be either string or list of message objects")

        # For claude_prompt strategy, skip SAE analysis
        if self.config.alert_strategy == "claude_prompt":
            should_alert, explanation = self._claude_prompt_alert(analysis_text)
            
            result = {
                "alert": should_alert,
                "explanation": explanation,
                "activated_features": [],
                "analysis_text": analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text
            }
            
            return result

        # SAE-based analysis for other strategies
        # Get feature activations using full conversation context
        all_activations = self.client.features.activations(
            messages=messages, model=self.config.model
        )
        mean_activations = all_activations.mean(axis=0)

        # Extract activations for our features
        good_activations = []
        bad_activations = []
        
        # Also store activations in original feature order for logistic regression
        self.all_activations = []
        for feature in self.features:
            activation = mean_activations[feature['index_in_sae']]
            self.all_activations.append(float(activation))

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

        # Get text for potential claude_prompt analysis
        if isinstance(input_data, str):
            analysis_text = input_data
        else:
            # For conversations, get the last assistant response
            analysis_text = ""
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    analysis_text = msg.get('content', '')
                    break

        # Determine if we should alert
        alert, explanation = self._should_alert(
            activated_features, good_activations, bad_activations, analysis_text
        )

        # Get text summary for response
        if isinstance(input_data, str):
            analyzed_text = input_data
        else:
            # For conversations, use the last assistant response as summary
            analyzed_text = last_assistant or "Conversation analysis"

        return {
            "text": analyzed_text,
            "alert": alert,
            "activated_features": activated_features,
            "explanation": explanation,
            "good_activations": good_activations,
            "bad_activations": bad_activations
        }

    def _should_alert(self, activated_features: List[Dict], 
                      good_activations: List[float], bad_activations: List[float], 
                      analysis_text: str = "") -> tuple:
        """Determine if we should alert based on strategy"""
        
        if self.config.alert_strategy == "logistic_regression":
            return self._logistic_alert(good_activations, bad_activations)
        elif self.config.alert_strategy == "any_bad_feature":
            return self._any_bad_feature_alert(activated_features)
        elif self.config.alert_strategy == "ratio":
            return self._ratio_alert(good_activations, bad_activations)
        elif self.config.alert_strategy == "quality":
            return self._quality_alert(good_activations, bad_activations)
        elif self.config.alert_strategy == "claude_prompt":
            return self._claude_prompt_alert(analysis_text)
        else:
            raise ValueError(f"Unknown alert strategy: {self.config.alert_strategy}")

    def _logistic_alert(self, good_activations: List[float], bad_activations: List[float]) -> tuple:
        """Use logistic regression classifier for alert decision"""
        if not self.classifier_model:
            return False, {"error": "Classifier not loaded"}
        
        # Create feature vector in the same order as training
        # The training script uses self.features which maintains original JSON order
        if hasattr(self, 'all_activations'):
            # Use pre-computed activations in original order
            feature_vector = np.array(self.all_activations).reshape(1, -1)
        else:
            # Fallback: combine good and bad (may not match training order)
            feature_vector = np.array(good_activations + bad_activations).reshape(1, -1)
        
        # Get prediction
        prediction = self.classifier_model.predict(feature_vector)[0]
        proba = self.classifier_model.predict_proba(feature_vector)[0, 1]  # P(projective)
        
        # Get SHAP explanation if available
        explanation = {
            "prediction": self.config.bad_behavior_label.lower() if prediction == 1 else self.config.good_behavior_label.lower(),
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

    def _claude_prompt_alert(self, analysis_text: str) -> tuple:
        """Alert based on Claude's assessment of the text"""
        import subprocess
        import os
        from datetime import datetime
        
        # Debug logging to file
        debug_log_path = "/tmp/claude_prompt_debug.log"
        
        # Get behavior to detect - support both old claude_prompt and new behavior_to_detect
        if hasattr(self.config, 'behavior_to_detect') and self.config.behavior_to_detect:
            behavior = self.config.behavior_to_detect
            # Build the prompt automatically
            prompt_template = f"Rate the following text for: {behavior}. Output only JSON, no other text: {{\"score\": 0.0-1.0}}"
        elif hasattr(self.config, 'claude_prompt') and self.config.claude_prompt:
            # Backwards compatibility - use claude_prompt if provided
            prompt_template = self.config.claude_prompt
        else:
            should_alert = False
            explanation = {
                "strategy": "claude_prompt",
                "error": "No behavior_to_detect or claude_prompt configured"
            }
            return should_alert, explanation
        
        try:
            # Prepare the full prompt for Claude
            full_prompt = f"{prompt_template}\n\nText to analyze:\n{analysis_text}"
            
            # Log debug info
            with open(debug_log_path, "a") as f:
                f.write(f"\n=== Claude Prompt Debug {datetime.now()} ===\n")
                f.write(f"Analysis text: {analysis_text[:200]}...\n")
                f.write(f"Prompt template: {prompt_template}\n")
                f.write(f"Full prompt length: {len(full_prompt)}\n")
            
            # Call claude -p with the prompt (no timeout)
            result = subprocess.run(
                ["claude", "-p", full_prompt],
                capture_output=True,
                text=True
            )
            
            # Log results
            with open(debug_log_path, "a") as f:
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"Stdout length: {len(result.stdout)}\n")
                f.write(f"Stdout: {result.stdout[:500]}\n")
                f.write(f"Stderr: {result.stderr[:200] if result.stderr else 'None'}\n")
            
            if result.returncode != 0:
                should_alert = False
                explanation = {
                    "strategy": "claude_prompt",
                    "error": f"Claude command failed: {result.stderr}"
                }
                return should_alert, explanation
            
            claude_response = result.stdout.strip()
            
            # Parse JSON response from Claude
            try:
                import json
                import re
                
                # Try to extract JSON from response (in case there's extra text)
                json_match = re.search(r'\{[^}]*"score"[^}]*\}', claude_response)
                if json_match:
                    json_str = json_match.group(0)
                    response_data = json.loads(json_str)
                else:
                    response_data = json.loads(claude_response)
                
                # Support both "score" and "sycophancy_score" for backwards compatibility
                score = response_data.get("score", response_data.get("sycophancy_score", 0.0))
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                # If parsing fails, default to no alert
                with open(debug_log_path, "a") as f:
                    f.write(f"JSON parse error: {e}\n")
                    f.write(f"Raw response: {claude_response}\n")
                score = 0.0
            
            should_alert = score > self.config.claude_threshold
            
            # Log final decision
            with open(debug_log_path, "a") as f:
                f.write(f"Parsed score: {score}\n")
                f.write(f"Threshold: {self.config.claude_threshold}\n")
                f.write(f"Should alert: {should_alert}\n")
            
            explanation = {
                "strategy": "claude_prompt",
                "score": score,
                "threshold": self.config.claude_threshold,
                "claude_response": claude_response[:200]  # Truncate for logging
            }
            
            return should_alert, explanation
            
        except Exception as e:
            should_alert = False
            explanation = {
                "strategy": "claude_prompt", 
                "error": f"Error calling Claude: {str(e)}"
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
                            direction = self.config.bad_behavior_label.lower() if value > 0 else self.config.good_behavior_label.lower()
                            # Show full feature names without truncation
                            top_features.append(f"{name}({value:+.3f}→{direction})")
                        
                        message += f" | Why: {', '.join(top_features)}"
            
            self.notifier.send(message, alert_level="alert")
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