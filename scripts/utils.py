import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

def initLogger(log_file: Optional[Path] = None):
    handlers: list = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )

def getPath(path_str, must_exist=True):
    """
    Convert string to resolved Path object with optional existence validation.
    
    Args:
        path_str: String or Path-like object representing a file/directory path
        must_exist: If True, raises error if path doesn't exist
    
    Returns:
        Path: Resolved absolute Path object with expanded user directory (~)
    
    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist
    """
    path = Path(path_str).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def loadJson(path):
    """
    Load and parse JSON data from a file.
    
    Args:
        path: File path to JSON file (string or Path object)
    
    Returns:
        Parsed JSON data (dict, list, or other JSON-serializable type)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def saveJson(data, path):
    """
    Save data to a JSON file with with 2-space indentation and UTF-8 encoding, preserving non-ASCII characters.
    
    Args:
        data: Python object to serialize (dict, list, etc.)
        path: File path where JSON will be saved (string or Path object)
    
    Raises:
        TypeError: If data is not JSON serializable
        IOError: If unable to write to file
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def loadMarkdown(path):
    """
    Load lines from a Markdown file.
    
    Args:
        path: File path to Markdown file (string or Path object)
    
    Returns:
        List of lines including newline characters
    
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
