#!/usr/bin/env python3
"""
Train a logistic regression classifier from diverse training examples.
Enhanced to handle multiple data formats and sources, including generated examples.
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
import sys
import glob
import time

try:
    import goodfire
    from goodfire import Client
except ImportError:
    print("Error: Goodfire not installed. Run: pip install goodfire")
    exit(1)

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Run: pip install shap for model explainability")

from ..core.config import WatchConfig

def load_diverse_examples(examples_path):
    """Load examples from various formats and sources"""
    examples_path = Path(examples_path)
    examples = []
    
    if examples_path.is_file():
        # Single file
        with open(examples_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            examples = data
        elif isinstance(data, dict) and 'conversations' in data:
            examples = data['conversations']
        elif isinstance(data, dict) and 'excerpts' in data:
            examples = data['excerpts']
        else:
            examples = [data]
    
    elif examples_path.is_dir():
        # Directory of files
        for file_path in examples_path.glob("*.json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                examples.extend(data)
            elif isinstance(data, dict) and 'conversation' in data:
                examples.append(data)
    
    else:
        # Try glob pattern
        for file_path in glob.glob(str(examples_path)):
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                examples.extend(data)
            elif isinstance(data, dict) and 'conversation' in data:
                examples.append(data)
    
    print(f"  Loaded {len(examples)} examples from {examples_path}")
    return examples

def extract_features_from_examples(client, examples, features, model):
    """Extract feature activations from example conversations"""
    feature_vectors = []
    
    for item in examples:
        # Handle different formats
        if isinstance(item, dict) and "conversation" in item:
            conversation = item["conversation"]
        elif isinstance(item, list):
            conversation = item
        else:
            continue
            
        # Convert conversation to text (take assistant responses)
        assistant_responses = [msg["content"] for msg in conversation if msg.get("role") == "assistant"]
        if not assistant_responses:
            continue
            
        # Use the last assistant response (or combine multiple for richer context)
        if len(assistant_responses) == 1:
            text = assistant_responses[0]
        else:
            # For multi-turn conversations, combine last 2 assistant responses
            text = " ".join(assistant_responses[-2:])
        
        # Skip very short responses
        if len(text.strip()) < 10:
            continue
        
        try:
            # Get feature activations
            messages = [{"role": "assistant", "content": text}]
            all_activations = client.features.activations(messages=messages, model=model)
            mean_activations = all_activations.mean(axis=0)
            
            # Extract activations for our specific features
            feature_vector = []
            for feat_data in features:
                activation = mean_activations[feat_data["index_in_sae"]]
                feature_vector.append(float(activation))
            
            feature_vectors.append(feature_vector)
            
        except Exception as e:
            print(f"    Warning: Failed to extract features from text: {str(e)[:100]}...")
            continue
    
    return feature_vectors

def train_enhanced_classifier(good_examples, bad_examples, features, model, client):
    """Train classifier with enhanced evaluation and SHAP support"""
    
    print("🔍 Extracting features from good examples...")
    good_vectors = extract_features_from_examples(client, good_examples, features, model)
    
    print("🔍 Extracting features from bad examples...")
    bad_vectors = extract_features_from_examples(client, bad_examples, features, model)
    
    if not good_vectors or not bad_vectors:
        raise ValueError("No feature vectors extracted. Check your examples format.")
    
    # Create training data
    X = np.array(good_vectors + bad_vectors)
    y = np.array([0] * len(good_vectors) + [1] * len(bad_vectors))  # 0=good, 1=bad
    
    print(f"📊 Training data: {len(good_vectors)} good, {len(bad_vectors)} bad examples")
    print(f"📊 Feature vector size: {X.shape[1]}")
    print(f"📊 Class balance: {len(good_vectors)/(len(good_vectors)+len(bad_vectors)):.1%} good, {len(bad_vectors)/(len(good_vectors)+len(bad_vectors)):.1%} bad")
    
    # Split for validation if we have enough data
    if len(X) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"📊 Using train/test split: {len(X_train)} train, {len(X_test)} test")
    else:
        X_train, X_test, y_train, y_test = X, X, y, y
        print("📊 Using all data for training (too few examples for split)")
    
    # Train logistic regression with balanced class weights
    classifier = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    classifier.fit(X_train, y_train)
    
    # Cross-validation if enough data
    if len(X_train) >= 10:
        cv_scores = cross_val_score(classifier, X_train, y_train, cv=min(5, len(X_train)//2))
        print(f"📊 Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Feature importance analysis
    feature_names = [f["label"] for f in features]
    coefficients = classifier.coef_[0]
    
    print("\n📈 Feature Importance (positive = promotes projective classification):")
    for name, coef in sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True):
        direction = "projective" if coef > 0 else "authentic"
        print(f"  {coef:+.3f} → {direction:10s} - {name}")
    
    # Evaluation
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Authentic", "Projective"]))
    
    # Probability distribution analysis
    good_probas = y_proba[:len([i for i in range(len(y_test)) if y_test[i] == 0])]
    bad_probas = y_proba[len(good_probas):]
    
    print(f"\n📊 Probability Distribution:")
    print(f"  Authentic examples - mean P(projective): {good_probas.mean():.3f} (std: {good_probas.std():.3f})")
    print(f"  Projective examples - mean P(projective): {bad_probas.mean():.3f} (std: {bad_probas.std():.3f})")
    
    # Threshold recommendations
    print(f"\n🎯 Threshold Recommendations:")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_at_threshold = (y_proba >= threshold).astype(int)
        if len(pred_at_threshold[pred_at_threshold == 1]) > 0:
            precision = y_test[pred_at_threshold == 1].mean()
            recall = (pred_at_threshold[y_test == 1] == 1).mean()
            print(f"  Threshold {threshold}: Precision={precision:.3f}, Recall={recall:.3f}")
    
    # SHAP analysis if available
    if SHAP_AVAILABLE and len(X_train) >= 5:
        print("\n🧠 Initializing SHAP explainer...")
        try:
            explainer = shap.LinearExplainer(classifier, X_train)
            print("✅ SHAP explainer ready for model interpretability")
        except Exception as e:
            print(f"⚠️ SHAP explainer initialization failed: {e}")
            explainer = None
    else:
        explainer = None
    
    return classifier, explainer, features

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_classifier.py <config.json> [--generated-data]")
        print("Options:")
        print("  --generated-data: Also load generated training examples")
        sys.exit(1)
    
    config_path = sys.argv[1]
    use_generated = "--generated-data" in sys.argv
    
    config = WatchConfig.from_json(config_path)
    
    # Load examples using enhanced loader
    print("📁 Loading training examples...")
    good_examples = load_diverse_examples(config.good_examples_path)
    bad_examples = load_diverse_examples(config.bad_examples_path)
    
    # Optionally load generated examples
    if use_generated:
        print("📁 Loading generated training examples...")
        
        # Look for generated examples in standard locations
        script_dir = Path(__file__).parent.parent
        generated_dir = script_dir / "data" / "generated_examples"
        
        if generated_dir.exists():
            generated_good = generated_dir / "authentic_coaching_examples.json"
            generated_bad = generated_dir / "projective_coaching_examples.json"
            
            if generated_good.exists():
                extra_good = load_diverse_examples(generated_good)
                good_examples.extend(extra_good)
                print(f"  Added {len(extra_good)} generated authentic examples")
            
            if generated_bad.exists():
                extra_bad = load_diverse_examples(generated_bad)
                bad_examples.extend(extra_bad)
                print(f"  Added {len(extra_bad)} generated projective examples")
    
    if not good_examples or not bad_examples:
        print("❌ No training examples found. Check your paths in the config.")
        sys.exit(1)
    
    # Load features - check for custom vector source first
    script_dir = Path(__file__).parent.parent.parent  # Go up to project root
    
    if hasattr(config, '_vector_source') and config._vector_source:
        # Use hand-curated or custom features
        vector_path = script_dir / f"data/vectors/{config._vector_source}"
        print(f"📊 Using custom vector source: {config._vector_source}")
    else:
        # Use auto-generated discriminative features
        good_name = Path(config.good_examples_path).stem
        bad_name = Path(config.bad_examples_path).stem
        model_name = config.model.split('/')[-1].replace('-', '_')
        vector_path = script_dir / f"data/vectors/discriminative_{good_name}_vs_{bad_name}_{model_name}.json"
        print(f"📊 Using auto-generated discriminative features")
    
    if not vector_path.exists():
        print(f"❌ Vector file not found: {vector_path}")
        if hasattr(config, '_vector_source') and config._vector_source:
            print("Please ensure the custom vector file exists in data/vectors/")
        else:
            print("Please generate vectors first using generate_vectors.py")
        sys.exit(1)
    
    with open(vector_path, 'r') as f:
        vector_data = json.load(f)
    
    features = vector_data["features"]
    feature_source = "hand-curated" if hasattr(config, '_vector_source') and config._vector_source else "auto-generated"
    print(f"📊 Using {len(features)} {feature_source} features")
    
    # Initialize Goodfire client
    GOODFIRE_API_KEY = os.environ.get("GOODFIRE_API_KEY")
    if not GOODFIRE_API_KEY:
        print("❌ GOODFIRE_API_KEY not found")
        sys.exit(1)
    
    client = Client(api_key=GOODFIRE_API_KEY)
    
    # Train enhanced classifier
    print("\n🚀 Training enhanced classifier...")
    classifier, explainer, features = train_enhanced_classifier(
        good_examples, bad_examples, features, config.model, client
    )
    
    # Save model with enhanced metadata
    model_dir = script_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    # Create model filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if hasattr(config, '_vector_source') and config._vector_source:
        # Use vector source name for custom features
        vector_name = Path(config._vector_source).stem
        model_path = model_dir / f"enhanced_classifier_{vector_name}_{timestamp}.pkl"
    else:
        # Use example names for auto-generated features
        good_name = Path(config.good_examples_path).stem
        bad_name = Path(config.bad_examples_path).stem
        model_name = config.model.split('/')[-1].replace('-', '_')
        model_path = model_dir / f"enhanced_classifier_{good_name}_vs_{bad_name}_{model_name}_{timestamp}.pkl"
    
    model_data = {
        "model": classifier,
        "explainer": explainer,
        "features": features,
        "config": {
            "good_examples_path": config.good_examples_path,
            "bad_examples_path": config.bad_examples_path,
            "model": config.model,
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "good_examples_count": len(good_examples),
            "bad_examples_count": len(bad_examples),
            "shap_available": explainer is not None
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Enhanced model saved to: {model_path}")
    print(f"✅ Ready to use with alert_strategy: 'logistic_regression'")
    print(f"✅ SHAP explanations: {'Available' if explainer else 'Not available'}")
    
    # Update symlink to latest model
    if hasattr(config, '_vector_source') and config._vector_source:
        vector_name = Path(config._vector_source).stem
        latest_model_path = model_dir / f"latest_classifier_{vector_name}.pkl"
    else:
        good_name = Path(config.good_examples_path).stem
        bad_name = Path(config.bad_examples_path).stem
        model_name = config.model.split('/')[-1].replace('-', '_')
        latest_model_path = model_dir / f"latest_classifier_{good_name}_vs_{bad_name}_{model_name}.pkl"
    
    if latest_model_path.exists():
        latest_model_path.unlink()
    
    try:
        latest_model_path.symlink_to(model_path.name)
        print(f"✅ Latest model symlink updated: {latest_model_path}")
    except OSError:
        # Fallback for systems without symlink support
        import shutil
        shutil.copy2(model_path, latest_model_path)
        print(f"✅ Latest model copied: {latest_model_path}")
    
    print(f"\n🎉 Training complete! Use this model path in your ClaudeWatch config.")
    print(f"💡 Recommended logistic_threshold: 0.7 (adjust based on precision/recall needs)")

if __name__ == "__main__":
    main()