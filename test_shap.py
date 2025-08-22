#!/usr/bin/env python3
"""
Test SHAP explanations with different types of coaching text
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from claude_watch import ClaudeWatch, WatchConfig

def test_text(config_path: str, text: str, label: str):
    """Test a text and show results"""
    print(f"\n=== Testing: {label} ===")
    print(f"Text: {text}")
    
    config = WatchConfig.from_json(config_path)
    watch = ClaudeWatch(config)
    result = watch.analyze(text)
    
    print(f"\nResults:")
    print(f"  Quality: {result['quality']}")
    print(f"  Alert: {result['alert']}")
    
    # Show classifier explanation if available
    if 'classifier_explanation' in result:
        exp = result['classifier_explanation']
        print(f"  Classifier: {exp['prediction']} (P={exp['probability']:.3f})")
        
        if exp['shap_values'] is not None:
            print(f"  SHAP Explanations:")
            shap_values = exp['shap_values']
            feature_names = [f["label"] for f in watch.features]
            
            # Sort by absolute SHAP value
            shap_pairs = list(zip(feature_names, shap_values))
            shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for i, (name, value) in enumerate(shap_pairs[:5]):
                direction = "→projective" if value > 0 else "→authentic"
                print(f"    {i+1}. {name[:50]}: {value:+.3f} {direction}")
    
    # Show notifications
    watch.send_notification(result, text)
    print("-" * 80)

def main():
    config_path = "configs/coaching_examples.json"
    
    # Test projective coaching examples
    test_text(config_path, 
             "You seem to be avoiding something important. This sounds like you're afraid of confrontation.",
             "Projective Example 1")
    
    test_text(config_path,
             "It's clear that you have trust issues stemming from childhood experiences.",
             "Projective Example 2")
    
    test_text(config_path,
             "You're procrastinating because you're afraid of failure. This pattern shows up everywhere in your life.",
             "Projective Example 3")
    
    # Test authentic coaching examples  
    test_text(config_path,
             "What do you notice happening in your body right now as you describe this?",
             "Authentic Example 1")
    
    test_text(config_path,
             "I'm curious about what you're experiencing emotionally as we talk about this.",
             "Authentic Example 2")
    
    test_text(config_path,
             "What would it be like to approach this situation with more compassion for yourself?",
             "Authentic Example 3")

if __name__ == "__main__":
    main()