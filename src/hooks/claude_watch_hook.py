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

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from claude_watch import ClaudeWatch, WatchConfig


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
        
        good_examples_path = config_data["good_examples_path"]
        bad_examples_path = config_data["bad_examples_path"]
        
        # Determine expected vector path with model name
        good_name = Path(good_examples_path).stem
        bad_name = Path(bad_examples_path).stem
        model = config_data.get("model", "meta-llama/Llama-3.3-70B-Instruct")
        model_name = model.split('/')[-1].replace('-', '_')
        project_root = Path(__file__).parent.parent.parent
        vector_path = project_root / f"data/vectors/discriminative_{good_name}_vs_{bad_name}_{model_name}.json"
        
        # If vectors don't exist, generate them
        if not vector_path.exists():
            print(f"Vectors not found at {vector_path}, generating...", file=sys.stderr)
            
            # Run vector generation script
            generate_script = project_root / "src" / "generate_vectors.py"
            result = subprocess.run([
                sys.executable, str(generate_script), str(config_path)
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
    
    # Configuration  
    CONFIG_PATH = os.environ.get('CLAUDE_WATCH_CONFIG', 
                                 str(Path(__file__).parent.parent.parent / 'configs' / 'coaching_examples.json'))
    
    # Read hook event from stdin first to get cwd and transcript path
    try:
        input_data = sys.stdin.read().strip()
        if not input_data:
            sys.exit(0)
        event_data = json.loads(input_data)
    except json.JSONDecodeError:
        sys.exit(0)
    
    # Only process Stop events
    if event_data.get('hook_event_name') != 'Stop':
        sys.exit(0)
    
    # Get project directory from cwd field in hook payload  
    project_dir_str = event_data.get('cwd')
    if project_dir_str:
        project_dir = Path(project_dir_str)
    else:
        # Fallback to current directory if cwd not provided
        project_dir = Path.cwd()
    
    # Get transcript path for conversation extraction
    transcript_path = event_data.get('transcript_path', '')
    
    # Use project directory for logs
    LOG_DIR = project_dir / 'logs'
    try:
        LOG_DIR.mkdir(exist_ok=True)
    except (PermissionError, OSError) as e:
        # Fallback to ClaudeWatch project directory if project dir isn't writable
        LOG_DIR = Path(__file__).parent.parent.parent / 'logs'
        LOG_DIR.mkdir(exist_ok=True)
    
    # Extract conversation
    conversation = extract_conversation(transcript_path)
    
    if len(conversation) < 2:
        sys.exit(0)
    
    # Get latest assistant response
    assistant_responses = [msg for msg in conversation if msg['role'] == 'assistant']
    if not assistant_responses:
        sys.exit(0)
    
    latest_response = assistant_responses[-1]['content']
    
    # Generate vectors if needed before analysis
    if not generate_vectors_if_needed(CONFIG_PATH):
        sys.exit(0)
    
    # Set environment variable so notification system knows project directory  
    os.environ['CLAUDE_PROJECT_DIR'] = str(project_dir)
    
    # Load configuration and analyze
    try:
        config = WatchConfig.from_json(CONFIG_PATH)
        watch = ClaudeWatch(config)
        result = watch.analyze(latest_response)
        
        # Log result with all details including activated features
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'response': latest_response[:200],
            'full_response': latest_response,  # Keep full response for analysis
            **result
        }
        
        # Make the log entry JSON-serializable
        serializable_log_entry = make_json_serializable(log_entry)
        
        log_file = LOG_DIR / 'claude_watch.log'
        with open(log_file, 'a') as f:
            f.write(json.dumps(serializable_log_entry) + '\n')
        
        # Send notifications using the integrated notification system
        watch.send_notification(result, latest_response)
        
        # Exit with code 1 if there's an alert to show stderr in Claude Code CLI
        if result.get('alert', False):
            sys.exit(1)
        
    except Exception as e:
        # Log error but don't interfere with Claude
        with open(LOG_DIR / 'errors.log', 'a') as f:
            f.write(f"{datetime.now()}: {e}\n")
    
    sys.exit(0)


if __name__ == "__main__":
    main()