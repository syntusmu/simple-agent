"""
Common utility functions for the simple-agent application.

This module provides utility functions for:
- Configuration loading from INI files
- Type conversion and validation
- Environment variable handling
- File path utilities
"""

import configparser
import os
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)


def load_config(
    config_path: str = "config.ini",
    section: Optional[str] = None,
    key: Optional[str] = None,
    default: Any = None,
    convert_types: bool = True
) -> Union[Dict[str, Any], Any]:
    """
    Load configuration from INI file by section and key-value pairs.
    
    Args:
        config_path: Path to the configuration file
        section: Specific section to load (if None, loads all sections)
        key: Specific key to load from section (requires section to be specified)
        default: Default value to return if key/section not found
        convert_types: Whether to automatically convert string values to appropriate types
        
    Returns:
        - If section and key specified: Value of the key
        - If only section specified: Dictionary of key-value pairs in that section
        - If neither specified: Dictionary of all sections with their key-value pairs
        
    Examples:
        # Load entire config
        config = load_config()
        
        # Load specific section
        deepseek_config = load_config(section='deepseek')
        
        # Load specific key
        api_key = load_config(section='deepseek', key='api_key')
        
        # With default value
        temperature = load_config(section='deepseek', key='temperature', default=0.7)
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Configuration file {config_path} not found")
        if key is not None:
            return default
        return {} if section is None else {}
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Return specific key value
        if section is not None and key is not None:
            try:
                value = config.get(section, key)
                return _convert_value(value) if convert_types else value
            except (configparser.NoSectionError, configparser.NoOptionError):
                logger.debug(f"Key '{key}' not found in section '{section}', returning default: {default}")
                return default
        
        # Return specific section
        elif section is not None:
            try:
                section_dict = {}
                for key, value in config.items(section):
                    section_dict[key] = _convert_value(value) if convert_types else value
                return section_dict
            except configparser.NoSectionError:
                logger.debug(f"Section '{section}' not found, returning empty dict")
                return {}
        
        # Return all sections
        else:
            all_config = {}
            for section_name in config.sections():
                section_dict = {}
                for key, value in config.items(section_name):
                    section_dict[key] = _convert_value(value) if convert_types else value
                all_config[section_name] = section_dict
            return all_config
            
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        if key is not None:
            return default
        return {} if section is None else {}


def _convert_value(value: str) -> Union[str, int, float, bool]:
    """
    Convert string value to appropriate Python type.
    
    Args:
        value: String value from config file
        
    Returns:
        Converted value (bool, int, float, or original string)
    """
    if not isinstance(value, str):
        return value
    
    # Convert boolean values
    if value.lower() in ('true', 'yes', '1', 'on'):
        return True
    elif value.lower() in ('false', 'no', '0', 'off'):
        return False
    
    # Convert numeric values
    try:
        # Try integer first
        if '.' not in value and 'e' not in value.lower():
            return int(value)
        else:
            return float(value)
    except ValueError:
        pass
    
    # Return as string if no conversion possible
    return value


def get_config_value(
    section: str,
    key: str,
    config_path: str = "config.ini",
    default: Any = None,
    env_var: Optional[str] = None,
    convert_type: bool = True
) -> Any:
    """
    Get configuration value with fallback to environment variable.
    
    Args:
        section: Configuration section name
        key: Configuration key name
        config_path: Path to configuration file
        default: Default value if not found
        env_var: Environment variable name to check as fallback
        convert_type: Whether to convert string values to appropriate types
        
    Returns:
        Configuration value, environment variable value, or default
        
    Examples:
        # Get API key with environment fallback
        api_key = get_config_value('deepseek', 'api_key', env_var='DEEPSEEK_API_KEY')
        
        # Get temperature with default
        temp = get_config_value('deepseek', 'temperature', default=0.7)
    """
    # First try config file
    value = load_config(config_path, section, key, default=None, convert_types=convert_type)
    
    # Skip dummy values
    if value is not None and isinstance(value, str) and not value.startswith(('sk-dummy', 'dummy-', 'AIzaSy-dummy')):
        return value
    
    # Fallback to environment variable
    if env_var:
        env_value = os.getenv(env_var)
        if env_value is not None:
            return _convert_value(env_value) if convert_type else env_value
    
    # Return default
    return default


def load_section_config(
    section: str,
    config_path: str = "config.ini",
    required_keys: Optional[List[str]] = None,
    env_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Load entire section configuration with environment variable fallbacks.
    
    Args:
        section: Section name to load
        config_path: Path to configuration file
        required_keys: List of keys that must be present
        env_mapping: Mapping of config keys to environment variable names
        
    Returns:
        Dictionary of configuration values
        
    Raises:
        ValueError: If required keys are missing
        
    Examples:
        # Load DeepSeek config with environment fallbacks
        config = load_section_config(
            'deepseek',
            required_keys=['api_key'],
            env_mapping={'api_key': 'DEEPSEEK_API_KEY'}
        )
    """
    config_dict = load_config(config_path, section=section)
    
    # Apply environment variable fallbacks
    if env_mapping:
        for config_key, env_var in env_mapping.items():
            if config_key in config_dict:
                # Skip dummy values and use environment fallback
                value = config_dict[config_key]
                if isinstance(value, str) and value.startswith(('sk-dummy', 'dummy-', 'AIzaSy-dummy')):
                    env_value = os.getenv(env_var)
                    if env_value:
                        config_dict[config_key] = _convert_value(env_value)
            else:
                # Key not in config, try environment
                env_value = os.getenv(env_var)
                if env_value:
                    config_dict[config_key] = _convert_value(env_value)
    
    # Check required keys
    if required_keys:
        missing_keys = []
        for key in required_keys:
            if key not in config_dict or config_dict[key] is None:
                missing_keys.append(key)
            elif isinstance(config_dict[key], str) and config_dict[key].startswith(('sk-dummy', 'dummy-', 'AIzaSy-dummy')):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys in section '{section}': {missing_keys}")
    
    return config_dict


def validate_config_section(
    section: str,
    config_path: str = "config.ini",
    schema: Optional[Dict[str, Dict[str, Any]]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate configuration section against a schema.
    
    Args:
        section: Section name to validate
        config_path: Path to configuration file
        schema: Validation schema with key requirements
            Format: {
                'key_name': {
                    'required': bool,
                    'type': type,
                    'choices': list (optional),
                    'min_value': number (optional),
                    'max_value': number (optional)
                }
            }
            
    Returns:
        Tuple of (is_valid, list_of_errors)
        
    Examples:
        schema = {
            'api_key': {'required': True, 'type': str},
            'temperature': {'required': False, 'type': float, 'min_value': 0.0, 'max_value': 2.0},
            'model': {'required': True, 'type': str, 'choices': ['deepseek-chat', 'deepseek-coder']}
        }
        is_valid, errors = validate_config_section('deepseek', schema=schema)
    """
    if schema is None:
        return True, []
    
    config_dict = load_config(config_path, section=section)
    errors = []
    
    for key, requirements in schema.items():
        value = config_dict.get(key)
        
        # Check required keys
        if requirements.get('required', False):
            if value is None:
                errors.append(f"Required key '{key}' is missing in section '{section}'")
                continue
            elif isinstance(value, str) and value.startswith(('sk-dummy', 'dummy-', 'AIzaSy-dummy')):
                errors.append(f"Required key '{key}' has dummy value in section '{section}'")
                continue
        
        # Skip validation if value is None and not required
        if value is None:
            continue
        
        # Check type
        expected_type = requirements.get('type')
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Key '{key}' in section '{section}' should be {expected_type.__name__}, got {type(value).__name__}")
            continue
        
        # Check choices
        choices = requirements.get('choices')
        if choices and value not in choices:
            errors.append(f"Key '{key}' in section '{section}' should be one of {choices}, got '{value}'")
        
        # Check numeric ranges
        if isinstance(value, (int, float)):
            min_value = requirements.get('min_value')
            max_value = requirements.get('max_value')
            
            if min_value is not None and value < min_value:
                errors.append(f"Key '{key}' in section '{section}' should be >= {min_value}, got {value}")
            
            if max_value is not None and value > max_value:
                errors.append(f"Key '{key}' in section '{section}' should be <= {max_value}, got {value}")
    
    return len(errors) == 0, errors


def get_config_sections(config_path: str = "config.ini") -> List[str]:
    """
    Get list of all sections in configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of section names
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Configuration file {config_path} not found")
        return []
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        return config.sections()
    except Exception as e:
        logger.error(f"Error reading configuration file {config_path}: {e}")
        return []


def config_exists(config_path: str = "config.ini") -> bool:
    """
    Check if configuration file exists.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(config_path).exists()


def _test_config_utilities():
    """Test function for configuration utilities."""
    print("Testing Configuration Utilities...")
    
    try:
        # Test 1: Load entire config
        print("\n1. Loading entire configuration:")
        all_config = load_config()
        print(f"Found {len(all_config)} sections: {list(all_config.keys())}")
        
        # Test 2: Load specific section
        print("\n2. Loading DeepSeek section:")
        deepseek_config = load_config(section='deepseek')
        print(f"DeepSeek config keys: {list(deepseek_config.keys())}")
        
        # Test 3: Load specific key
        print("\n3. Loading specific key:")
        api_key = load_config(section='deepseek', key='api_key')
        temperature = load_config(section='deepseek', key='temperature', default=0.7)
        print(f"API key: {api_key[:20]}..." if api_key else "API key: None")
        print(f"Temperature: {temperature}")
        
        # Test 4: Get config value with environment fallback
        print("\n4. Testing environment fallback:")
        api_key_with_fallback = get_config_value(
            'deepseek', 'api_key', 
            env_var='DEEPSEEK_API_KEY',
            default='no-key-found'
        )
        print(f"API key with fallback: {api_key_with_fallback}")
        
        # Test 5: Load section with validation
        print("\n5. Loading section with validation:")
        try:
            config = load_section_config(
                'deepseek',
                env_mapping={'api_key': 'DEEPSEEK_API_KEY'}
            )
            print(f"Loaded config: {list(config.keys())}")
        except ValueError as e:
            print(f"Validation error: {e}")
        
        # Test 6: Get available sections
        print("\n6. Available sections:")
        sections = get_config_sections()
        print(f"Sections: {sections}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")


if __name__ == "__main__":
    _test_config_utilities()
