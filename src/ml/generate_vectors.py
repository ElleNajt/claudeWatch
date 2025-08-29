#!/usr/bin/env python3
"""
Generate discriminative vectors for ClaudeWatch.
This script creates the initial cached vectors that ClaudeWatch uses.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

try:
    import goodfire
    from goodfire import Client
except ImportError:
    print("Error: Goodfire not installed. Run: pip install goodfire")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check for API key
GOODFIRE_API_KEY = os.environ.get("GOODFIRE_API_KEY")
if not GOODFIRE_API_KEY:
    print("Error: GOODFIRE_API_KEY not found")
    print("Please either:")
    print("  1. Create a .env file with: GOODFIRE_API_KEY=your_api_key")
    print("  2. Set environment variable: export GOODFIRE_API_KEY=your_api_key")
    exit(1)

def load_examples(path: str):
    """Load examples from JSON file (expects conversation format)"""
    with open(path, "r") as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        # Check if it's a list of conversations or list of objects with conversations
        if data and isinstance(data[0], dict):
            if "conversation" in data[0]:
                # Extract conversations from objects with metadata
                return [item["conversation"] for item in data]
            else:
                # Already in correct format (list of conversations)
                return data
        elif data and isinstance(data[0], list):
            # Already a list of conversations
            return data
    elif isinstance(data, dict):
        # Fallback for other formats
        return data.get("examples", [])
    
    return data

def main():
    """Generate discriminative features from example configurations"""
    
    if len(sys.argv) < 2:
        print("Usage: python generate_vectors.py <config.json>")
        print("Example: python generate_vectors.py configs/coaching_examples.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    # Load configuration
    with open(config_path, "r") as f:
        config_data = json.load(f)
    
    good_examples_path = config_data["good_examples_path"]
    bad_examples_path = config_data["bad_examples_path"]
    model = config_data.get("model", "meta-llama/Llama-3.3-70B-Instruct")
    
    # Load examples
    print(f"Loading examples from:")
    print(f"  Good: {good_examples_path}")
    print(f"  Bad: {bad_examples_path}")
    
    good_examples = load_examples(good_examples_path)
    bad_examples = load_examples(bad_examples_path)
    
    print(f"Loaded {len(good_examples)} good examples")
    print(f"Loaded {len(bad_examples)} bad examples")
    
    # Balance datasets if needed (take min length)
    min_length = min(len(good_examples), len(bad_examples))
    good_examples = good_examples[:min_length]
    bad_examples = bad_examples[:min_length]
    
    print(f"Balanced to {min_length} examples each")
    
    # Initialize Goodfire client
    client = Client(api_key=GOODFIRE_API_KEY)
    
    print("Generating discriminative features using Goodfire...")
    
    # Generate discriminative features using Goodfire
    # contrast() returns a tuple: (features_toward_dataset1, features_toward_dataset2)
    # Use top_k=15 to get more features for better behavioral discrimination
    good_features, bad_features = client.features.contrast(
        dataset_1=good_examples,
        dataset_2=bad_examples,
        model=model,
        top_k=15,
    )
    
    print(f"Generated {len(good_features)} good and {len(bad_features)} bad features")
    
    # Display feature labels for review
    print("\n" + "="*60)
    print("GOOD FEATURES (promoting good behavior):")
    print("="*60)
    for i, feat in enumerate(good_features):
        print(f"  {i+1:2d}. {feat.label}")
        
    print("\n" + "="*60)
    print("BAD FEATURES (detecting harmful behavior):")
    print("="*60)
    for i, feat in enumerate(bad_features):
        print(f"  {i+1:2d}. {feat.label}")
    
    # Create metadata for caching
    features = []
    for feature in good_features:
        features.append({
            "uuid": feature.uuid,
            "index_in_sae": feature.index_in_sae,
            "label": feature.label,
            "type": "good",
        })
    for feature in bad_features:
        features.append({
            "uuid": feature.uuid,
            "index_in_sae": feature.index_in_sae,
            "label": feature.label,
            "type": "bad",
        })
    
    # Determine cache path with model name
    good_name = Path(good_examples_path).stem
    bad_name = Path(bad_examples_path).stem
    model_name = model.split('/')[-1].replace('-', '_')
    cache_path = f"./data/vectors/discriminative_{good_name}_vs_{bad_name}_{model_name}.json"
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    cache_data = {
        "generated": datetime.now().isoformat(),
        "good_examples_path": good_examples_path,
        "bad_examples_path": bad_examples_path,
        "model": model,
        "features": features,
    }
    
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)
    
    print(f"\n✅ Cached vectors to {cache_path}")
    print(f"✅ Ready to use with ClaudeWatch!")
    print(f"\nTo test: python claude_watch_cli.py analyze {config_path} \"test message\"")

if __name__ == "__main__":
    main()