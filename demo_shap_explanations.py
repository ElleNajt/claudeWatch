#!/usr/bin/env python3
"""
Demo of SHAP explanations in ClaudeWatch
Shows why specific coaching responses are classified as projective vs authentic
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from claude_watch import ClaudeWatch, WatchConfig

def demo_explanation(text: str, label: str):
    """Demo SHAP explanation for a text"""
    print(f"\n{'='*60}")
    print(f"🎯 {label}")
    print(f"📝 Text: {text}")
    
    config = WatchConfig.from_json("configs/coaching_examples.json")
    watch = ClaudeWatch(config)
    result = watch.analyze(text)
    
    if 'classifier_explanation' in result:
        exp = result['classifier_explanation']
        prediction = exp['prediction']
        probability = exp['probability']
        
        print(f"\n🤖 Classifier Prediction: {prediction.upper()} (confidence: {probability:.1%})")
        
        if exp['shap_values'] is not None:
            print(f"\n🔍 Why this prediction? (SHAP explanations)")
            shap_values = exp['shap_values']
            feature_names = [f["label"] for f in watch.features]
            
            # Sort by absolute SHAP value and show top contributors
            shap_pairs = list(zip(feature_names, shap_values))
            shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for i, (name, value) in enumerate(shap_pairs):
                if abs(value) > 0.001:  # Only show meaningful contributions
                    direction = "pushes toward PROJECTIVE" if value > 0 else "pushes toward AUTHENTIC"
                    magnitude = "strongly" if abs(value) > 0.1 else "moderately" if abs(value) > 0.03 else "slightly"
                    print(f"   • {name[:45]:<45} {magnitude:>10} {direction} ({value:+.3f})")
    
    print("-" * 60)

def main():
    print("🎭 ClaudeWatch SHAP Explanations Demo")
    print("=====================================")
    print("This shows how AI models explain their coaching style classifications using SHAP values.")
    
    # Example that strongly triggers projective features
    demo_explanation(
        "You're clearly procrastinating because you're afraid of failure, and this same avoidance pattern shows up in all your relationships and work situations.",
        "Strong Projective Example"
    )
    
    # Example that triggers both but projective wins
    demo_explanation(
        "It sounds like you might be avoiding something. What do you think could be underneath that?",
        "Mixed Example (leans projective)"
    )
    
    # Example that strongly triggers authentic features
    demo_explanation(
        "I notice my body getting tense when I think about that conversation. What are you noticing in your body right now?",
        "Strong Authentic Example"
    )
    
    print("\n📊 Understanding SHAP Values:")
    print("• Positive values push the prediction toward PROJECTIVE coaching")
    print("• Negative values push the prediction toward AUTHENTIC coaching") 
    print("• Larger absolute values = stronger influence on the decision")
    print("• The final prediction is the sum of all these influences")

if __name__ == "__main__":
    main()