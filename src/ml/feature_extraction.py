#!/usr/bin/env python3
"""
Feature Extraction for ClaudeWatch
Generate discriminative SAE features from conversation examples
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

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


class FeatureExtractor:
    """Extract discriminative SAE features from conversation examples"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GOODFIRE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GOODFIRE_API_KEY not found. Please either:\n"
                "  1. Create a .env file with: GOODFIRE_API_KEY=your_api_key\n"
                "  2. Set environment variable: export GOODFIRE_API_KEY=your_api_key"
            )
        
        self.client = Client(api_key=self.api_key)
    
    def load_examples(self, path: str) -> List[List[Dict]]:
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
                # List of conversations (each conversation is a list of messages)
                return data
        
        print(f"Warning: Unexpected format in {path}, treating as single conversation")
        return [data] if isinstance(data, list) else []

    def conversation_to_text(self, conversation: List[Dict]) -> str:
        """Convert conversation to text for analysis"""
        # Extract assistant responses for analysis
        assistant_responses = [
            msg["content"] for msg in conversation 
            if msg.get("role") == "assistant"
        ]
        
        if not assistant_responses:
            return ""
        
        # Use the last assistant response or combine multiple
        if len(assistant_responses) == 1:
            return assistant_responses[0]
        else:
            # For multi-turn, combine last 2 responses for richer context
            return " ".join(assistant_responses[-2:])

    def extract_contrasts(self, good_examples: List[List[Dict]], 
                         bad_examples: List[List[Dict]], 
                         model: str = "meta-llama/Llama-3.3-70B-Instruct") -> List[goodfire.Feature]:
        """Extract contrasting features between good and bad examples"""
        
        print("🔍 Converting examples to text...")
        
        # Convert conversations to texts
        good_texts = []
        for conv in good_examples:
            text = self.conversation_to_text(conv)
            if text and len(text.strip()) > 10:  # Skip very short responses
                good_texts.append(text)
        
        bad_texts = []
        for conv in bad_examples:
            text = self.conversation_to_text(conv)
            if text and len(text.strip()) > 10:
                bad_texts.append(text)
        
        print(f"📊 Processed {len(good_texts)} good and {len(bad_texts)} bad examples")
        
        if not good_texts or not bad_texts:
            raise ValueError("Need at least one good and one bad example")
        
        # Prepare contrast messages
        good_messages = [[{"role": "assistant", "content": text}] for text in good_texts]
        bad_messages = [[{"role": "assistant", "content": text}] for text in bad_texts]
        
        print("🧠 Extracting contrasting features using Goodfire...")
        print("This may take a few minutes...")
        
        # Get contrasting features
        try:
            features = self.client.features.contrasts(
                dataset_1=good_messages,
                dataset_2=bad_messages,
                model=model
            )
            
            print(f"✅ Extracted {len(features)} discriminative features")
            return features
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            raise

    def save_features(self, features: List[goodfire.Feature], 
                     good_examples_path: str, bad_examples_path: str, 
                     model: str, output_dir: str = "data/vectors") -> str:
        """Save features to JSON file for caching"""
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filename
        good_name = Path(good_examples_path).stem
        bad_name = Path(bad_examples_path).stem
        model_name = model.split('/')[-1].replace('-', '_')
        output_file = output_dir / f"discriminative_{good_name}_vs_{bad_name}_{model_name}.json"
        
        # Convert features to serializable format
        features_data = []
        
        for i, feature in enumerate(features):
            # Determine if this is a "good" or "bad" feature based on contrasts
            # For contrasts, first half typically represent dataset_1 (good), second half dataset_2 (bad)
            feature_type = "good" if i < len(features) // 2 else "bad"
            
            features_data.append({
                "uuid": feature.uuid,
                "label": feature.label,
                "index_in_sae": feature.index_in_sae,
                "type": feature_type
            })
        
        # Create metadata
        vector_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "good_examples_path": good_examples_path,
                "bad_examples_path": bad_examples_path,
                "model": model,
                "feature_count": len(features),
                "good_features": len([f for f in features_data if f["type"] == "good"]),
                "bad_features": len([f for f in features_data if f["type"] == "bad"])
            },
            "features": features_data
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(vector_data, f, indent=2)
        
        print(f"💾 Features saved to: {output_file}")
        print(f"   Good features: {vector_data['metadata']['good_features']}")
        print(f"   Bad features: {vector_data['metadata']['bad_features']}")
        
        return str(output_file)


def generate_discriminative_features(good_examples_path: str, bad_examples_path: str, 
                                   model: str = "meta-llama/Llama-3.3-70B-Instruct",
                                   output_dir: str = "data/vectors") -> str:
    """
    Generate discriminative features from examples
    
    Args:
        good_examples_path: Path to good examples JSON
        bad_examples_path: Path to bad examples JSON  
        model: Model to use for feature extraction
        output_dir: Directory to save feature vectors
        
    Returns:
        Path to saved feature vectors
    """
    extractor = FeatureExtractor()
    
    # Load examples
    print(f"📁 Loading examples...")
    good_examples = extractor.load_examples(good_examples_path)
    bad_examples = extractor.load_examples(bad_examples_path)
    
    print(f"   Good examples: {len(good_examples)} conversations")
    print(f"   Bad examples: {len(bad_examples)} conversations")
    
    # Extract features
    features = extractor.extract_contrasts(good_examples, bad_examples, model)
    
    # Save features
    output_path = extractor.save_features(
        features, good_examples_path, bad_examples_path, model, output_dir
    )
    
    return output_path


def main():
    """CLI entry point"""
    import sys
    from ..core.config import WatchConfig
    
    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <config.json>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    config = WatchConfig.from_json(config_path)
    
    try:
        output_path = generate_discriminative_features(
            config.good_examples_path,
            config.bad_examples_path, 
            config.model
        )
        
        print(f"\n✅ Feature extraction complete!")
        print(f"   Vectors saved to: {output_path}")
        print(f"   Ready for use with ClaudeWatch")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()