#!/usr/bin/env python3
"""
Filter out structural/syntactic features from discriminative vectors
"""

import json
import sys
from pathlib import Path

# Patterns that indicate structural rather than behavioral features
STRUCTURAL_PATTERNS = [
    "start of a new conversation",
    "beginning of new conversation", 
    "conversation reset",
    "start of new conversation",
    "new conversation segment",
    "conversation segment marker",
    "topic reset",
    "topic switch",
    "system header",
    "chat format",
    "punctuation",
    "grammatical",
    "conjunctions",
    "connectives",
    "hyphens",
    "discourse markers",
    "natural text flow",
    "linguistic patterns"
]

def is_structural_feature(feature_label):
    """Check if a feature is likely structural/syntactic rather than behavioral"""
    label_lower = feature_label.lower()
    return any(pattern in label_lower for pattern in STRUCTURAL_PATTERNS)

def filter_vectors(input_path, output_path):
    """Filter out structural features from vector file"""
    with open(input_path) as f:
        data = json.load(f)
    
    original_count = len(data["features"])
    
    # Filter out structural features
    filtered_features = [
        feature for feature in data["features"] 
        if not is_structural_feature(feature["label"])
    ]
    
    data["features"] = filtered_features
    data["_filtered"] = True
    data["_original_count"] = original_count
    data["_filtered_count"] = len(filtered_features)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Filtered {original_count - len(filtered_features)} structural features")
    print(f"Kept {len(filtered_features)} behavioral features")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_structural_features.py <input_vectors.json> <output_vectors.json>")
        sys.exit(1)
    
    filter_vectors(sys.argv[1], sys.argv[2])