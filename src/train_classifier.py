#!/usr/bin/env python3
"""
Train a logistic regression classifier from the existing training examples.
"""

import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
import sys

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

from claude_watch import WatchConfig

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
        assistant_responses = [msg["content"] for msg in conversation if msg["role"] == "assistant"]
        if not assistant_responses:
            continue
            
        # Use the last assistant response
        text = assistant_responses[-1]
        
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
    
    return feature_vectors

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_classifier.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    config = WatchConfig.from_json(config_path)
    
    # Load examples
    with open(config.good_examples_path, 'r') as f:
        good_examples = json.load(f)
    with open(config.bad_examples_path, 'r') as f:
        bad_examples = json.load(f)
    
    # Load existing features
    script_dir = Path(__file__).parent.parent
    good_name = Path(config.good_examples_path).stem
    bad_name = Path(config.bad_examples_path).stem
    model_name = config.model.split('/')[-1].replace('-', '_')
    vector_path = script_dir / f"data/vectors/discriminative_{good_name}_vs_{bad_name}_{model_name}.json"
    
    if not vector_path.exists():
        print(f"Vector file not found: {vector_path}")
        print("Please generate vectors first using generate_vectors.py")
        sys.exit(1)
    
    with open(vector_path, 'r') as f:
        vector_data = json.load(f)
    
    features = vector_data["features"]
    
    # Initialize Goodfire client
    GOODFIRE_API_KEY = os.environ.get("GOODFIRE_API_KEY")
    if not GOODFIRE_API_KEY:
        print("Error: GOODFIRE_API_KEY not found")
        sys.exit(1)
    
    client = Client(api_key=GOODFIRE_API_KEY)
    
    print("Extracting features from good examples...")
    good_vectors = extract_features_from_examples(client, good_examples, features, config.model)
    
    print("Extracting features from bad examples...")
    bad_vectors = extract_features_from_examples(client, bad_examples, features, config.model)
    
    # Create training data
    X = np.array(good_vectors + bad_vectors)
    y = np.array([0] * len(good_vectors) + [1] * len(bad_vectors))  # 0=good, 1=bad
    
    print(f"Training data: {len(good_vectors)} good, {len(bad_vectors)} bad examples")
    print(f"Feature vector size: {X.shape[1]}")
    
    # Train logistic regression with balanced class weights
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    model.fit(X, y)
    
    # Print feature importance
    feature_names = [f["label"] for f in features]
    coefficients = model.coef_[0]
    
    print("\nFeature Importance (positive = promotes bad classification):")
    for name, coef in sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {coef:+.3f} - {name}")
    
    # Evaluate on training data (for now)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Authentic", "Projective"]))
    
    # Show probability distribution
    print(f"\nProbability distribution:")
    print(f"Good examples - mean P(projective): {y_proba[:len(good_vectors)].mean():.3f}")
    print(f"Bad examples - mean P(projective): {y_proba[len(good_vectors):].mean():.3f}")
    
    # Save model
    model_dir = script_dir / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"projective_classifier_{good_name}_vs_{bad_name}_{model_name}.pkl"
    
    model_data = {
        "model": model,
        "features": features,
        "config": {
            "good_examples_path": config.good_examples_path,
            "bad_examples_path": config.bad_examples_path,
            "model": config.model
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Ready to use with alert_strategy: 'logistic_regression'")
    
    # Suggest threshold
    fpr, tpr, thresholds = roc_curve(y, y_proba)
    # Find threshold for 95% precision on bad class
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_at_threshold = (y_proba >= threshold).astype(int)
        if len(pred_at_threshold[pred_at_threshold == 1]) > 0:
            precision = y[pred_at_threshold == 1].mean()
            recall = (pred_at_threshold[y == 1] == 1).mean()
            print(f"Threshold {threshold}: Precision={precision:.3f}, Recall={recall:.3f}")

if __name__ == "__main__":
    main()