#!/usr/bin/env python3
"""
File Utilities for ClaudeWatch
Common file operations, path handling, and directory management
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime


def ensure_directory(path: Union[str, Path], parents: bool = True) -> Path:
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        path: Path to directory
        parents: Create parent directories if needed
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=True)
    return path


def safe_json_dump(data: Any, file_path: Union[str, Path], 
                   indent: int = 2, backup: bool = True) -> bool:
    """
    Safely write JSON data to file with optional backup
    
    Args:
        data: Data to write to JSON
        file_path: Path to output file
        indent: JSON indentation
        backup: Create backup of existing file
        
    Returns:
        True if successful, False otherwise
    """
    file_path = Path(file_path)
    
    try:
        # Create directory if needed
        ensure_directory(file_path.parent)
        
        # Create backup if file exists and backup requested
        if backup and file_path.exists():
            backup_path = file_path.with_suffix(f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            shutil.copy2(file_path, backup_path)
        
        # Write to temporary file first, then move (atomic operation)
        temp_path = file_path.with_suffix('.tmp')
        
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        # Move temp file to final location
        temp_path.replace(file_path)
        
        return True
        
    except Exception as e:
        print(f"Error writing JSON to {file_path}: {e}")
        
        # Clean up temp file if it exists
        temp_path = file_path.with_suffix('.tmp')
        if temp_path.exists():
            temp_path.unlink()
        
        return False


def safe_json_load(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Safely load JSON data from file
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or can't be loaded
        
    Returns:
        Loaded data or default value
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return default
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return default


def get_unique_filename(base_path: Union[str, Path], 
                       extension: str = None) -> Path:
    """
    Get a unique filename by adding numbers if file exists
    
    Args:
        base_path: Base file path
        extension: File extension (if not in base_path)
        
    Returns:
        Unique file path
    """
    base_path = Path(base_path)
    
    if extension and not base_path.suffix:
        base_path = base_path.with_suffix(extension)
    
    if not base_path.exists():
        return base_path
    
    # Extract parts
    directory = base_path.parent
    stem = base_path.stem
    suffix = base_path.suffix
    
    # Find unique filename
    counter = 1
    while True:
        new_path = directory / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def copy_with_metadata(src: Union[str, Path], dst: Union[str, Path], 
                      metadata: Dict = None) -> bool:
    """
    Copy file and optionally add metadata
    
    Args:
        src: Source file path
        dst: Destination file path
        metadata: Optional metadata to add/update
        
    Returns:
        True if successful, False otherwise
    """
    try:
        src_path = Path(src)
        dst_path = Path(dst)
        
        # Ensure destination directory exists
        ensure_directory(dst_path.parent)
        
        # Copy file
        shutil.copy2(src_path, dst_path)
        
        # Add metadata if provided and file is JSON
        if metadata and dst_path.suffix.lower() == '.json':
            try:
                # Load existing data
                with open(dst_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Add metadata
                if isinstance(data, dict):
                    if '_metadata' not in data:
                        data['_metadata'] = {}
                    data['_metadata'].update(metadata)
                    data['_metadata']['copied_at'] = datetime.now().isoformat()
                    data['_metadata']['original_file'] = str(src_path)
                    
                    # Write back
                    with open(dst_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        
            except Exception as e:
                print(f"Warning: Could not add metadata to {dst_path}: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error copying {src} to {dst}: {e}")
        return False


def find_files_by_pattern(directory: Union[str, Path], 
                         pattern: str, recursive: bool = True) -> list[Path]:
    """
    Find files matching a pattern in directory
    
    Args:
        directory: Directory to search
        pattern: Glob pattern to match
        recursive: Search subdirectories
        
    Returns:
        List of matching file paths
    """
    directory = Path(directory)
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def get_file_info(file_path: Union[str, Path]) -> Dict:
    """
    Get comprehensive file information
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"exists": False}
    
    stat = file_path.stat()
    
    info = {
        "exists": True,
        "path": str(file_path),
        "name": file_path.name,
        "stem": file_path.stem,
        "suffix": file_path.suffix,
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir()
    }
    
    # Add JSON-specific info
    if file_path.suffix.lower() == '.json':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            info["json_valid"] = True
            info["json_type"] = type(data).__name__
            
            if isinstance(data, (list, dict)):
                info["json_length"] = len(data)
            
            # Check for conversation data
            if isinstance(data, list) and data and isinstance(data[0], dict):
                if "role" in data[0]:
                    info["content_type"] = "conversation"
                elif "conversation" in data[0]:
                    info["content_type"] = "conversation_objects"
                    
        except Exception:
            info["json_valid"] = False
    
    return info


def cleanup_temp_files(directory: Union[str, Path], 
                      patterns: list[str] = None) -> int:
    """
    Clean up temporary files in directory
    
    Args:
        directory: Directory to clean
        patterns: List of patterns to match (default: common temp patterns)
        
    Returns:
        Number of files removed
    """
    if patterns is None:
        patterns = ['*.tmp', '*.temp', '*~', '.DS_Store', '*.pyc', '__pycache__']
    
    directory = Path(directory)
    removed_count = 0
    
    for pattern in patterns:
        for file_path in directory.rglob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    removed_count += 1
                elif file_path.is_dir() and not any(file_path.iterdir()):
                    # Remove empty directories
                    file_path.rmdir()
                    removed_count += 1
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")
    
    return removed_count


def archive_old_files(directory: Union[str, Path], 
                     archive_dir: Union[str, Path],
                     days_old: int = 30,
                     pattern: str = "*") -> int:
    """
    Archive files older than specified days
    
    Args:
        directory: Directory to check
        archive_dir: Directory to move old files to
        days_old: Files older than this many days will be archived
        pattern: File pattern to match
        
    Returns:
        Number of files archived
    """
    from datetime import timedelta
    
    directory = Path(directory)
    archive_dir = Path(archive_dir)
    
    # Ensure archive directory exists
    ensure_directory(archive_dir)
    
    cutoff_time = datetime.now() - timedelta(days=days_old)
    archived_count = 0
    
    for file_path in directory.glob(pattern):
        if not file_path.is_file():
            continue
        
        # Check file modification time
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        if file_mtime < cutoff_time:
            try:
                # Create archive path with date prefix
                date_prefix = file_mtime.strftime("%Y%m%d_")
                archive_path = archive_dir / f"{date_prefix}{file_path.name}"
                
                # Make unique if needed
                archive_path = get_unique_filename(archive_path)
                
                # Move file
                shutil.move(str(file_path), str(archive_path))
                archived_count += 1
                
            except Exception as e:
                print(f"Could not archive {file_path}: {e}")
    
    return archived_count