#!/usr/bin/env python3
"""
Data Loading Utilities for ClaudeWatch
Common functions for loading conversation data in various formats
"""

import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Union


def load_conversation_data(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load conversation data from a single file
    Handles various conversation formats used across ClaudeWatch
    
    Args:
        file_path: Path to JSON file containing conversation data
        
    Returns:
        List of conversation dictionaries
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different data formats
    conversations = []
    
    if isinstance(data, list):
        # List of conversations or conversation objects
        for item in data:
            if isinstance(item, dict):
                if "conversation" in item:
                    # Object with conversation field (and metadata)
                    conversations.append(item["conversation"])
                elif "role" in item:
                    # Single message - collect into conversation
                    if not conversations:
                        conversations.append([])
                    conversations[-1].append(item)
                else:
                    # Assume it's a conversation list
                    conversations.append(item)
            elif isinstance(item, list):
                # Direct conversation (list of messages)
                conversations.append(item)
    
    elif isinstance(data, dict):
        # Single object
        if "conversation" in data:
            conversations.append(data["conversation"])
        elif "conversations" in data:
            # Multiple conversations in a container
            conversations.extend(data["conversations"])
        elif "role" in data:
            # Single message
            conversations.append([data])
        else:
            # Treat as metadata container and extract conversation-like data
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict) and "role" in value[0]:
                    conversations.append(value)
    
    return conversations


def load_diverse_examples(examples_path: Union[str, Path]) -> List[Dict]:
    """
    Load examples from various formats and sources
    Enhanced version that handles files, directories, and glob patterns
    
    Args:
        examples_path: Path to file, directory, or glob pattern
        
    Returns:
        List of conversation objects (may include metadata)
    """
    examples_path = Path(examples_path)
    examples = []
    
    if examples_path.is_file():
        # Single file
        conversations = load_conversation_data(examples_path)
        
        # Convert to objects with conversation field if needed
        for conv in conversations:
            if isinstance(conv, list):
                examples.append({"conversation": conv})
            else:
                examples.append(conv)
    
    elif examples_path.is_dir():
        # Directory of files
        for file_path in examples_path.glob("*.json"):
            try:
                file_conversations = load_conversation_data(file_path)
                for conv in file_conversations:
                    if isinstance(conv, list):
                        examples.append({
                            "conversation": conv,
                            "source_file": str(file_path)
                        })
                    else:
                        # Add source file metadata if not present
                        if isinstance(conv, dict) and "source_file" not in conv:
                            conv["source_file"] = str(file_path)
                        examples.append(conv)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
    
    else:
        # Try glob pattern
        matching_files = glob.glob(str(examples_path))
        
        if not matching_files:
            raise FileNotFoundError(f"No files found matching: {examples_path}")
        
        for file_path in matching_files:
            try:
                file_conversations = load_conversation_data(file_path)
                for conv in file_conversations:
                    if isinstance(conv, list):
                        examples.append({
                            "conversation": conv,
                            "source_file": file_path
                        })
                    else:
                        if isinstance(conv, dict) and "source_file" not in conv:
                            conv["source_file"] = file_path
                        examples.append(conv)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
    
    return examples


def extract_assistant_responses(conversation: List[Dict]) -> List[str]:
    """
    Extract assistant responses from a conversation
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content'
        
    Returns:
        List of assistant response texts
    """
    return [
        msg["content"] 
        for msg in conversation 
        if msg.get("role") == "assistant" and msg.get("content")
    ]


def extract_user_messages(conversation: List[Dict]) -> List[str]:
    """
    Extract user messages from a conversation
    
    Args:
        conversation: List of message dictionaries with 'role' and 'content'
        
    Returns:
        List of user message texts
    """
    return [
        msg["content"] 
        for msg in conversation 
        if msg.get("role") == "user" and msg.get("content")
    ]


def filter_conversations_by_length(conversations: List[List[Dict]], 
                                  min_turns: int = 2, 
                                  max_turns: int = None) -> List[List[Dict]]:
    """
    Filter conversations by turn count
    
    Args:
        conversations: List of conversations
        min_turns: Minimum number of turns (messages)
        max_turns: Maximum number of turns (None for no limit)
        
    Returns:
        Filtered list of conversations
    """
    filtered = []
    
    for conv in conversations:
        turn_count = len(conv)
        
        if turn_count >= min_turns:
            if max_turns is None or turn_count <= max_turns:
                filtered.append(conv)
    
    return filtered


def validate_conversation_format(conversation: List[Dict]) -> bool:
    """
    Validate that a conversation has the expected format
    
    Args:
        conversation: List of message dictionaries
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(conversation, list):
        return False
    
    for msg in conversation:
        if not isinstance(msg, dict):
            return False
        
        if "role" not in msg or "content" not in msg:
            return False
        
        if msg["role"] not in ["user", "assistant", "system", "other"]:
            return False
        
        if not isinstance(msg["content"], str):
            return False
    
    return True


def merge_conversation_datasets(*datasets: List[Dict]) -> List[Dict]:
    """
    Merge multiple conversation datasets, removing duplicates
    
    Args:
        *datasets: Variable number of conversation datasets
        
    Returns:
        Merged dataset with duplicates removed
    """
    merged = []
    seen_conversations = set()
    
    for dataset in datasets:
        for item in dataset:
            # Create a hash of the conversation content for deduplication
            if "conversation" in item:
                conv_text = json.dumps(item["conversation"], sort_keys=True)
            else:
                conv_text = json.dumps(item, sort_keys=True)
            
            conv_hash = hash(conv_text)
            
            if conv_hash not in seen_conversations:
                seen_conversations.add(conv_hash)
                merged.append(item)
    
    return merged


def split_dataset(dataset: List[Dict], train_ratio: float = 0.8, 
                  random_seed: int = 42) -> tuple[List[Dict], List[Dict]]:
    """
    Split dataset into train and test sets
    
    Args:
        dataset: List of conversation examples
        train_ratio: Fraction for training set
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_set, test_set)
    """
    import random
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Shuffle dataset
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    
    # Split
    split_index = int(len(shuffled) * train_ratio)
    train_set = shuffled[:split_index]
    test_set = shuffled[split_index:]
    
    return train_set, test_set