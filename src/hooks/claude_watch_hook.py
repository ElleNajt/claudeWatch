#!/usr/bin/env python3
"""
ClaudeWatch Hook - Monitors Claude Code responses in real-time
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Enhanced logging for debugging
HOOK_LOG_FILE = "/tmp/claudewatch_hook.log"

def log_message(msg):
    """Log message with timestamp"""
    with open(HOOK_LOG_FILE, "a") as f:
        f.write(f"{datetime.now()}: {msg}\n")

log_message(f"ClaudeWatch hook started with args: {sys.argv}")
log_message(f"Current working directory: {os.getcwd()}")
log_message(f"Python path: {sys.path[:3]}")  # First 3 entries

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    log_message("Importing ClaudeWatch modules...")
    from core.claude_watch import ClaudeWatch
    from core.config import WatchConfig
    log_message("Successfully imported ClaudeWatch modules")
except Exception as e:
    log_message(f"Failed to import ClaudeWatch modules: {e}")
    import traceback
    log_message(f"Full traceback: {traceback.format_exc()}")
    sys.exit(1)


def make_json_serializable(obj):
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        # For custom objects, convert to dict
        return make_json_serializable(obj.__dict__)
    else:
        return obj


def format_readable_log_entry(log_entry, features=None, config=None):
    """Format log entry for human readability"""
    readable_entry = {
        "timestamp": log_entry["timestamp"],
        "alert": log_entry.get("alert", False),
        "response_preview": log_entry["response"],
        "full_response": log_entry["full_response"],
        "analysis": {
            "mode": log_entry.get("analysis_mode", "unknown"),
            "conversation_length": log_entry.get("conversation_length", 0),
        }
    }
    
    # Add human-readable activated features
    if "activated_features" in log_entry and log_entry["activated_features"]:
        readable_entry["activated_features"] = [
            {
                "type": f["type"],
                "label": f["label"], 
                "activation": f["activation"]
            }
            for f in log_entry["activated_features"]
        ]
    
    # Add readable explanation if available
    if "explanation" in log_entry:
        explanation = log_entry["explanation"]
        readable_explanation = {
            "prediction": explanation.get("prediction", "unknown"),
            "probability": explanation.get("probability", 0.0)
        }
        
        # Add feature contributions with names instead of indices
        if "shap_values" in explanation and features:
            shap_values = explanation["shap_values"]
            if len(shap_values) == len(features):
                feature_contributions = []
                for i, (shap_val, feature) in enumerate(zip(shap_values, features)):
                    if abs(shap_val) > 0.001:  # Only show meaningful contributions
                        feature_contributions.append({
                            "feature": feature["label"],
                            "contribution": shap_val,
                            "direction": (config.bad_behavior_label.lower() if shap_val > 0 else config.good_behavior_label.lower()) if config else ("class_1" if shap_val > 0 else "class_0")
                        })
                
                # Sort by absolute contribution
                feature_contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
                readable_explanation["feature_contributions"] = feature_contributions[:10]  # Top 10
        
        readable_entry["explanation"] = readable_explanation
    
    # Add detailed feature activations
    if "good_activations" in log_entry and features:
        good_activations = log_entry["good_activations"]
        bad_activations = log_entry.get("bad_activations", [])
        
        # Get active features with their activation levels
        active_good_features = []
        active_bad_features = []
        
        # Process good features
        for i, activation in enumerate(good_activations):
            if activation > 0.001 and i < len(features):  # Lowered threshold to see more activations
                feature = features[i]
                if hasattr(feature, 'label'):
                    active_good_features.append({
                        "label": feature.label,
                        "activation": round(activation, 3),
                        "uuid": str(getattr(feature, 'uuid', 'unknown'))[:8]
                    })
        
        # Process bad features (assuming they come after good features in the features list)
        good_count = len(good_activations)
        for i, activation in enumerate(bad_activations):
            if activation > 0.0001 and (good_count + i) < len(features):  # Even lower threshold
                feature = features[good_count + i]
                if hasattr(feature, 'label'):
                    active_bad_features.append({
                        "label": feature.label,
                        "activation": round(activation, 3),
                        "uuid": str(getattr(feature, 'uuid', 'unknown'))[:8]
                    })
        
        # Sort by activation level
        active_good_features.sort(key=lambda x: x["activation"], reverse=True)
        active_bad_features.sort(key=lambda x: x["activation"], reverse=True)
        
        readable_entry["feature_activations"] = {
            "good_features": active_good_features,
            "bad_features": active_bad_features,
            "summary": {
                "good_features_active": len(active_good_features),
                "bad_features_active": len(active_bad_features),
                "total_good_activation": round(sum(good_activations), 3),
                "total_bad_activation": round(sum(bad_activations), 3)
            }
        }
    elif "good_activations" in log_entry:
        # Fallback for when features aren't available
        good_activations = log_entry["good_activations"]
        bad_activations = log_entry.get("bad_activations", [])
        readable_entry["activations_summary"] = {
            "good_features_active": len([x for x in good_activations if x > 0.001]),
            "bad_features_active": len([x for x in bad_activations if x > 0.001]),
            "total_good_activation": round(sum(good_activations), 3),
            "total_bad_activation": round(sum(bad_activations), 3)
        }
    
    return readable_entry


def extract_conversation(transcript_path: str) -> list:
    """Extract conversation from Claude Code JSONL transcript"""
    if not os.path.exists(transcript_path):
        return []
    
    conversation = []
    with open(transcript_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get('type') in ['user', 'assistant']:
                    message = entry.get('message', {})
                    role = message.get('role')
                    content = ''
                    
                    # Handle different content formats
                    if isinstance(message.get('content'), str):
                        content = message.get('content', '')
                    elif isinstance(message.get('content'), list):
                        for item in message.get('content', []):
                            if isinstance(item, dict) and item.get('type') == 'text':
                                content += item.get('text', '')
                    
                    if role and content.strip():
                        conversation.append({
                            'role': role,
                            'content': content.strip()
                        })
            except json.JSONDecodeError:
                continue
    
    return conversation


def generate_vectors_if_needed(config_path: str):
    """Generate vectors if they don't exist"""
    try:
        # Load config to check if vectors exist
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Skip vector generation for claude_prompt strategy (doesn't use vectors)
        if config_data.get("alert_strategy") == "claude_prompt":
            log_message("Using claude_prompt strategy, skipping vector check")
            return True
        
        # Handle both list and string formats for examples paths
        good_examples_path = config_data.get("good_examples_path")
        bad_examples_path = config_data.get("bad_examples_path")
        
        # Skip vector generation if using direct vectors or curated vectors
        if config_data.get("direct_vectors") or config_data.get("_vector_source"):
            return True
        
        if isinstance(good_examples_path, list):
            good_name = Path(good_examples_path[0]).stem if good_examples_path else "unknown"
        else:
            good_name = Path(good_examples_path).stem if good_examples_path else "unknown"
            
        if isinstance(bad_examples_path, list):
            bad_name = Path(bad_examples_path[0]).stem if bad_examples_path else "unknown"  
        else:
            bad_name = Path(bad_examples_path).stem if bad_examples_path else "unknown"
        model = config_data.get("model", "meta-llama/Llama-3.3-70B-Instruct")
        model_name = model.split('/')[-1].replace('-', '_')
        project_root = Path(__file__).parent.parent.parent
        vector_path = project_root / f"data/vectors/discriminative_{good_name}_vs_{bad_name}_{model_name}.json"
        
        # If vectors don't exist, generate them
        if not vector_path.exists():
            print(f"Vectors not found at {vector_path}, generating...", file=sys.stderr)
            
            # Run vector generation script via CLI
            generate_script = project_root / "claude_watch_cli.py"
            result = subprocess.run([
                sys.executable, str(generate_script), "generate-vectors", "--config", str(config_path)
            ], capture_output=True, text=True, cwd=project_root)
            
            if result.returncode != 0:
                print(f"Vector generation failed: {result.stderr}", file=sys.stderr)
                return False
            
            print("✅ Vectors generated successfully!", file=sys.stderr)
            return True
        
        return True
        
    except Exception as e:
        print(f"Error checking/generating vectors: {e}", file=sys.stderr)
        return False


def main():
    """Main hook function"""
    
    log_message("=== ClaudeWatch Hook Main Function Started ===")
    
    # Configuration  
    CONFIG_PATH = os.environ.get('CLAUDE_WATCH_CONFIG', 
                                 str(Path(__file__).parent.parent.parent / 'configs' / 'claude_prompt_sycophancy.json'))
    log_message(f"Config path: {CONFIG_PATH}")
    
    # Read hook event from stdin first to get cwd and transcript path
    try:
        log_message("Reading hook event from stdin...")
        input_data = sys.stdin.read().strip()
        log_message(f"Raw input data length: {len(input_data)}")
        if not input_data:
            log_message("No input data received, exiting")
            sys.exit(0)
        event_data = json.loads(input_data)
        log_message(f"Parsed event data keys: {list(event_data.keys())}")
        log_message(f"Hook event name: {event_data.get('hook_event_name')}")
    except json.JSONDecodeError as e:
        log_message(f"JSON decode error: {e}")
        sys.exit(0)
    except Exception as e:
        log_message(f"Error reading input: {e}")
        sys.exit(0)
    
    # Only process Stop events
    if event_data.get('hook_event_name') != 'Stop':
        log_message(f"Not a Stop event, exiting. Event name: {event_data.get('hook_event_name')}")
        sys.exit(0)
    
    log_message("Processing Stop event...")
    
    # Get project directory from cwd field in hook payload  
    project_dir_str = event_data.get('cwd')
    if project_dir_str:
        project_dir = Path(project_dir_str)
        log_message(f"Using project directory from cwd: {project_dir}")
    else:
        # Fallback to current directory if cwd not provided
        project_dir = Path.cwd()
        log_message(f"Using current directory as project dir: {project_dir}")
    
    # Get transcript path for conversation extraction
    transcript_path = event_data.get('transcript_path', '')
    log_message(f"Transcript path: {transcript_path}")
    
    # Use project directory for logs
    LOG_DIR = project_dir / 'logs'
    try:
        LOG_DIR.mkdir(exist_ok=True)
    except (PermissionError, OSError) as e:
        # Fallback to ClaudeWatch project directory if project dir isn't writable
        LOG_DIR = Path(__file__).parent.parent.parent / 'logs'
        LOG_DIR.mkdir(exist_ok=True)
    
    # Extract conversation
    log_message(f"Extracting conversation from: {transcript_path}")
    conversation = extract_conversation(transcript_path)
    log_message(f"Extracted {len(conversation)} messages from conversation")
    
    if len(conversation) < 2:
        log_message("Conversation too short (< 2 messages), exiting")
        sys.exit(0)
    
    # Get latest assistant response
    assistant_responses = [msg for msg in conversation if msg['role'] == 'assistant']
    log_message(f"Found {len(assistant_responses)} assistant responses")
    if not assistant_responses:
        log_message("No assistant responses found, exiting")
        sys.exit(0)
    
    # For subtle pattern detection, analyze conversation context
    # Environment variable to control analysis mode
    analysis_mode = os.environ.get('CLAUDE_WATCH_ANALYSIS_MODE', 'conversation_context')
    
    if analysis_mode == 'single_response':
        # Original behavior - just latest response as text
        analysis_input = assistant_responses[-1]['content']
    elif analysis_mode == 'text_concat':
        # Concatenated text approach (for backward compatibility)
        if len(assistant_responses) >= 3:
            recent_responses = assistant_responses[-3:]
            analysis_input = " ".join([resp['content'] for resp in recent_responses])
        elif len(assistant_responses) >= 2:
            recent_responses = assistant_responses[-2:]
            analysis_input = " ".join([resp['content'] for resp in recent_responses])
        else:
            analysis_input = assistant_responses[-1]['content']
    else:  # 'conversation_context' - pass full conversation format to SAE
        # Get recent conversation turns for better pattern detection
        # Use last 6 messages (3 user-assistant pairs) for context
        if len(conversation) >= 6:
            analysis_input = conversation[-6:]
        else:
            # Use all available conversation if less than 6 messages
            analysis_input = conversation
    
    # Generate vectors if needed before analysis
    log_message(f"Checking if vectors need to be generated for config: {CONFIG_PATH}")
    if not generate_vectors_if_needed(CONFIG_PATH):
        log_message("Vector generation check failed, exiting")
        sys.exit(0)
    log_message("Vector generation check passed")
    
    # Set environment variable so notification system knows project directory  
    os.environ['CLAUDE_PROJECT_DIR'] = str(project_dir)
    
    # Load configuration and analyze
    try:
        log_message(f"Loading config from: {CONFIG_PATH}")
        config = WatchConfig.from_json(CONFIG_PATH)
        log_message(f"Config loaded successfully. Alert strategy: {config.alert_strategy}")
        
        log_message("Initializing ClaudeWatch...")
        watch = ClaudeWatch(config)
        log_message("ClaudeWatch initialized successfully")
        
        log_message(f"Analyzing input (type: {type(analysis_input).__name__})")
        result = watch.analyze(analysis_input)
        log_message(f"Analysis complete. Alert: {result.get('alert', False)}")
        
        # For logging, get a readable summary of what was analyzed
        if isinstance(analysis_input, list):
            # Conversation format - get last assistant response for logging
            last_assistant_response = None
            for msg in reversed(analysis_input):
                if msg.get('role') == 'assistant':
                    last_assistant_response = msg.get('content', '')
                    break
            display_response = last_assistant_response or "No assistant response found"
        else:
            # Text format
            display_response = analysis_input
        
        # Log result with all details including activated features
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'response': display_response[:200],
            'full_response': display_response,  # Keep full response for analysis
            'analysis_mode': analysis_mode,
            'conversation_length': len(conversation),
            **result
        }
        
        # Make the log entry JSON-serializable and readable
        serializable_log_entry = make_json_serializable(log_entry)
        readable_log_entry = format_readable_log_entry(serializable_log_entry, watch.features, config)
        
        log_file = LOG_DIR / 'claude_watch.log'
        with open(log_file, 'a') as f:
            f.write(json.dumps(readable_log_entry, indent=2) + '\n')
        
        # Send notifications using the integrated notification system
        watch.send_notification(result, display_response)
        
        # Exit with code 1 if there's an alert to show stderr in Claude Code CLI
        if result.get('alert', False):
            sys.exit(1)
        
    except Exception as e:
        log_message(f"Error during analysis: {e}")
        import traceback
        log_message(f"Full traceback: {traceback.format_exc()}")
        
        # Log error but don't interfere with Claude
        with open(LOG_DIR / 'errors.log', 'a') as f:
            f.write(f"{datetime.now()}: {e}\n")
            f.write(f"Traceback: {traceback.format_exc()}\n")
    
    sys.exit(0)


if __name__ == "__main__":
    main()