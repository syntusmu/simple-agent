"""
Common utility functions for configuration and document processing services.

This module provides shared functionality including:
- Configuration loading from INI files
- SSL configuration for downloading models
- Document processing initialization (online/offline modes)
- Model cache management
"""

import configparser
import logging
import os
import ssl
import urllib.request
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
from functools import lru_cache

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONFIG_PATH = "config.ini"
DEFAULT_CACHE_DIR = "model_artifacts"

# Model name to configuration section mapping
MODEL_TO_SECTION_MAP = {
    # OpenAI models
    'gpt-4': 'openai',
    'gpt-4-turbo': 'openai',
    'gpt-4o': 'openai',
    'gpt-3.5-turbo': 'openai',
    'gpt-3.5-turbo-16k': 'openai',
    
    # DeepSeek models
    'deepseek-chat': 'deepseek',
    'deepseek-coder': 'deepseek',
    'deepseek-v3': 'deepseek',
    'deepseek-v2.5': 'deepseek',
    
    # Qwen models
    'qwen-max': 'qwen',
    'qwen-plus': 'qwen',
    'qwen-turbo': 'qwen',
    'qwen3-32b': 'qwen',
    'qwen2.5-72b': 'qwen',
    'qwen2-72b': 'qwen',
    'qwen-vl-max': 'qwen',
    'qwen-vl-plus': 'qwen',
}

# Configuration constants for LLM sections
SUPPORTED_CONFIG_SECTIONS = ['llm', 'deepseek', 'qwen', 'openai', 'embedding', 'postgresql']


# =============================================================================
# Configuration Loading Functions
# =============================================================================

@lru_cache(maxsize=1)
def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Dict[str, str]]:
    """
    Load all sections and key-value pairs from config.ini file with caching.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary with sections as keys and their key-value pairs as values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        configparser.Error: If config file is malformed
        
    Example:
        config = load_config()
        # Returns: {'llm': {'api_key': 'sk-...', 'model': 'deepseek-chat'}, ...}
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        config = configparser.ConfigParser()
        config.read(config_path, encoding='utf-8')
        
        result = {}
        for section_name in config.sections():
            result[section_name] = dict(config.items(section_name))
        
        logger.debug(f"Loaded configuration from {config_path} with {len(result)} sections")
        return result
        
    except configparser.Error as e:
        logger.error(f"Configuration file parsing error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        return {}


def load_section_config(section: str, config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, str]:
    """
    Load key-value pairs from a specific section using cached config.
    
    Args:
        section: Section name to load
        config_path: Path to the configuration file
        
    Returns:
        Dictionary of key-value pairs from the specified section
        
    Raises:
        ValueError: If section name is empty
        
    Example:
        llm_config = load_section_config('llm')
        # Returns: {'api_key': 'sk-...', 'model': 'deepseek-chat', ...}
    """
    if not section or not section.strip():
        raise ValueError("Section name cannot be empty")
    
    section = section.strip()
    config = load_config(config_path)
    
    if section in config:
        logger.debug(f"Loaded section '{section}' with {len(config[section])} keys")
        return config[section]
    else:
        logger.warning(f"Section '{section}' not found in configuration")
        return {}


def get_config_value(
    section: str, 
    key: str, 
    config_path: str = DEFAULT_CONFIG_PATH, 
    default: Optional[str] = None
) -> Optional[str]:
    """
    Get a specific configuration value using cached config.
    
    Args:
        section: Section name
        key: Key name
        config_path: Path to the configuration file
        default: Default value if key not found
        
    Returns:
        Configuration value or default
        
    Raises:
        ValueError: If section or key name is empty
        
    Example:
        api_key = get_config_value('llm', 'api_key')
    """
    if not section or not section.strip():
        raise ValueError("Section name cannot be empty")
    if not key or not key.strip():
        raise ValueError("Key name cannot be empty")
    
    section = section.strip()
    key = key.strip()
    
    section_config = load_section_config(section, config_path)
    value = section_config.get(key, default)
    
    logger.debug(f"Retrieved config value [{section}].{key} = {value if value != default else f'default({default})'}'")
    return value


def get_section_by_model_name(model_name: str) -> str:
    """
    Get the configuration section name for a given model name.
    
    Args:
        model_name: Name of the model (e.g., 'gpt-4', 'deepseek-chat', 'qwen-max')
        
    Returns:
        Configuration section name ('openai', 'deepseek', 'qwen', or 'llm')
        
    Raises:
        ValueError: If model name is empty
        
    Example:
        section = get_section_by_model_name('gpt-4')
        # Returns: 'openai'
    """
    if not model_name or not model_name.strip():
        raise ValueError("Model name cannot be empty")
    
    model_name = model_name.strip().lower()
    
    # Direct lookup in mapping
    if model_name in MODEL_TO_SECTION_MAP:
        section = MODEL_TO_SECTION_MAP[model_name]
        logger.debug(f"Model '{model_name}' mapped to section '{section}'")
        return section
    
    # Fallback: try to infer from model name patterns
    if 'gpt' in model_name or 'openai' in model_name:
        logger.debug(f"Model '{model_name}' inferred as OpenAI model")
        return 'openai'
    elif 'deepseek' in model_name:
        logger.debug(f"Model '{model_name}' inferred as DeepSeek model")
        return 'deepseek'
    elif 'qwen' in model_name or 'tongyi' in model_name:
        logger.debug(f"Model '{model_name}' inferred as Qwen model")
        return 'qwen'
    else:
        # Default to 'llm' section for unknown models
        logger.warning(f"Unknown model '{model_name}', defaulting to 'llm' section")
        return 'llm'


def get_available_config_sections(config_path: str = DEFAULT_CONFIG_PATH) -> List[str]:
    """
    Get list of available configuration sections.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        List of section names found in the configuration file
        
    Example:
        sections = get_available_config_sections()
        # Returns: ['llm', 'deepseek', 'qwen', 'embedding', 'postgresql']
    """
    config = load_config(config_path)
    available_sections = list(config.keys())
    logger.debug(f"Available config sections: {available_sections}")
    return available_sections


def clear_config_cache() -> None:
    """
    Clear the configuration cache to force reload on next access.
    
    This is useful when the configuration file has been modified
    and you want to reload the changes.
    """
    load_config.cache_clear()
    logger.info("Configuration cache cleared")


# =============================================================================
# Document Processing Configuration Functions
# =============================================================================

def configure_ssl():
    """Configure SSL context to handle certificate verification issues.
    
    This is needed when downloading models (like EasyOCR) that may encounter
    SSL certificate verification failures in certain environments.
    """
    try:
        # Create unverified SSL context for downloading models
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Install the unverified context as default
        urllib.request.install_opener(
            urllib.request.build_opener(
                urllib.request.HTTPSHandler(context=ssl_context)
            )
        )
        logger.info("SSL context configured to bypass certificate verification")
        return True
    except Exception as e:
        logger.warning(f"Failed to configure SSL context: {e}")
        return False


def setup_model_cache_dir(cache_dir: str = DEFAULT_CACHE_DIR):
    """Setup model cache directory for document processing libraries.
    
    Args:
        cache_dir: Base directory name for caching models
    """
    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different model types
    subdirs = {
        "docling": cache_path / "docling",
        "transformers": cache_path / "transformers", 
        "huggingface": cache_path / "huggingface",
        "easyocr": cache_path / "easyocr",
        "tokenizers": cache_path / "tokenizers",
        "local_models": cache_path / "local_models"
    }
    
    for name, path in subdirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created model directory: {path}")
    
    # Set environment variables for different libraries
    os.environ["DOCLING_CACHE_DIR"] = str(subdirs["docling"])
    os.environ["TRANSFORMERS_CACHE"] = str(subdirs["transformers"])
    os.environ["HF_HOME"] = str(subdirs["huggingface"])
    os.environ["EASYOCR_MODULE_PATH"] = str(subdirs["easyocr"])
    os.environ["TOKENIZERS_CACHE"] = str(subdirs["tokenizers"])
    
    logger.info(f"Model cache directory configured: {cache_path}")
    return cache_path, subdirs


def check_docling_models_available():
    """Check if local docling models are available with actual model files."""
    try:
        cache_path = Path(DEFAULT_CACHE_DIR).resolve()
        docling_cache = cache_path / "docling"
        
        # Check for layout model in the root docling directory
        layout_model = docling_cache / "model.safetensors"
        layout_config = docling_cache / "config.json"
        layout_available = layout_model.exists() and layout_config.exists()
        
        # Check for TableFormer models in accurate and fast subdirectories
        accurate_model = docling_cache / "accurate" / "tableformer_accurate.safetensors"
        accurate_config = docling_cache / "accurate" / "tm_config.json"
        fast_model = docling_cache / "fast" / "tableformer_fast.safetensors"
        fast_config = docling_cache / "fast" / "tm_config.json"
        
        tableformer_available = (accurate_model.exists() and accurate_config.exists()) or \
                               (fast_model.exists() and fast_config.exists())
        
        logger.info(f"Local model availability - Layout: {layout_available} ({layout_model.exists()}, {layout_config.exists()}), TableFormer: {tableformer_available} (Accurate: {accurate_model.exists() and accurate_config.exists()}, Fast: {fast_model.exists() and fast_config.exists()})")
        return layout_available and tableformer_available  # Need both for full functionality
    except Exception as e:
        logger.warning(f"Error checking docling models: {e}")
        return False


def configure_huggingface_online():
    """Configure Hugging Face Hub for online mode with local caching and offline fallback."""
    try:
        # Set cache directory to model_artifacts/docling
        cache_path = Path(DEFAULT_CACHE_DIR).resolve()
        docling_cache = cache_path / "docling"
        
        # Ensure cache directory exists
        docling_cache.mkdir(parents=True, exist_ok=True)
        
        # Configure Hugging Face cache paths
        os.environ["HF_HOME"] = str(docling_cache / "huggingface")
        os.environ["TRANSFORMERS_CACHE"] = str(docling_cache / "transformers")
        os.environ["TOKENIZERS_CACHE"] = str(docling_cache / "tokenizers")
        
        # Check if we have local models available
        models_available = check_docling_models_available()
        
        if models_available:
            # If local models exist, start in offline mode to avoid network issues
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            logger.info(f"Hugging Face Hub configured for offline mode (local models available) with cache: {docling_cache}")
        else:
            # Try online mode first, but allow fallback to offline
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            logger.info(f"Hugging Face Hub configured for online mode with caching to: {docling_cache}")
        
        # Performance and privacy settings
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"  # Disable telemetry
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # Disable progress bars
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer warnings
        
        return True
    except Exception as e:
        logger.warning(f"Failed to configure Hugging Face Hub: {e}")
        return False


def configure_docling_online():
    """Configure Docling for online mode with local caching to model_artifacts/docling."""
    try:
        # Set up cache directory in model_artifacts/docling
        cache_path = Path(DEFAULT_CACHE_DIR).resolve()
        docling_cache = cache_path / "docling"
        
        # Ensure cache directory exists
        docling_cache.mkdir(parents=True, exist_ok=True)
        
        # Configure Docling cache directory
        os.environ["DOCLING_CACHE_DIR"] = str(docling_cache)
        
        # Set up subdirectories for different model types
        tableformer_dir = docling_cache / "tableformer"
        layout_dir = docling_cache / "layout"
        
        tableformer_dir.mkdir(exist_ok=True)
        layout_dir.mkdir(exist_ok=True)
        
        # Configure specific model cache paths
        os.environ["TABLEFORMER_CACHE_DIR"] = str(tableformer_dir)
        os.environ["LAYOUT_CACHE_DIR"] = str(layout_dir)
        
        logger.info(f"Docling configured for online mode with caching to: {docling_cache}")
        return True
            
    except Exception as e:
        logger.warning(f"Failed to configure Docling online mode: {e}")
        return False


def get_local_model_paths() -> Dict[str, Path]:
    """Get standardized local model paths for different components."""
    cache_path = Path(DEFAULT_CACHE_DIR).resolve()
    
    return {
        "easyocr_detection": cache_path / "easyocr" / "craft_mlt_25k.pth",
        "easyocr_recognition": cache_path / "easyocr" / "latin_g2.pth", 
        "transformers_base": cache_path / "docling" / "transformers",
        "docling_models": cache_path / "docling",  # Use model_artifacts/docling for docling models
        "docling_layout": cache_path / "docling" / "layout",
        "docling_tableformer": cache_path / "docling" / "tableformer",
        "tokenizers": cache_path / "docling" / "tokenizers"
    }


def download_and_cache_models():
    """Download required models to local cache for offline usage."""
    try:
        model_paths = get_local_model_paths()
        
        # Create directories
        for path in model_paths.values():
            if isinstance(path, Path):
                path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("Model download directories prepared")
        
        # Note: Actual model downloads would happen here when online
        # For now, we ensure the directory structure exists
        
        return True
    except Exception as e:
        logger.error(f"Failed to prepare model download directories: {e}")
        return False


def configure_huggingface_offline():
    """Configure Hugging Face Hub for offline mode using local model artifacts."""
    try:
        # Set cache directory to model_artifacts/docling
        cache_path = Path(DEFAULT_CACHE_DIR).resolve()
        docling_cache = cache_path / "docling"
        
        # Ensure cache directory exists
        docling_cache.mkdir(parents=True, exist_ok=True)
        
        # Configure Hugging Face cache paths
        os.environ["HF_HOME"] = str(docling_cache / "huggingface")
        os.environ["TRANSFORMERS_CACHE"] = str(docling_cache / "transformers")
        os.environ["TOKENIZERS_CACHE"] = str(docling_cache / "tokenizers")
        
        # Force offline mode to use local models only
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        # Performance and privacy settings
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        logger.info(f"Hugging Face Hub configured for offline mode with cache: {docling_cache}")
        return True
    except Exception as e:
        logger.warning(f"Failed to configure Hugging Face Hub offline mode: {e}")
        return False


def configure_docling_offline():
    """Configure Docling for offline mode using local model artifacts."""
    try:
        # Set up cache directory in model_artifacts/docling
        cache_path = Path(DEFAULT_CACHE_DIR).resolve()
        docling_cache = cache_path / "docling"
        
        # Ensure cache directory exists
        docling_cache.mkdir(parents=True, exist_ok=True)
        
        # Configure Docling cache directory
        os.environ["DOCLING_CACHE_DIR"] = str(docling_cache)
        
        # Set up specific model paths for offline mode
        os.environ["TABLEFORMER_CACHE_DIR"] = str(docling_cache)
        os.environ["LAYOUT_CACHE_DIR"] = str(docling_cache)
        
        # Configure specific model file paths
        accurate_model = docling_cache / "accurate" / "tableformer_accurate.safetensors"
        fast_model = docling_cache / "fast" / "tableformer_fast.safetensors"
        layout_model = docling_cache / "model.safetensors"
        
        if accurate_model.exists():
            os.environ["TABLEFORMER_ACCURATE_MODEL_PATH"] = str(accurate_model)
        if fast_model.exists():
            os.environ["TABLEFORMER_FAST_MODEL_PATH"] = str(fast_model)
        if layout_model.exists():
            os.environ["LAYOUT_MODEL_PATH"] = str(layout_model)
        
        logger.info(f"Docling configured for offline mode using models in: {docling_cache}")
        return True
            
    except Exception as e:
        logger.warning(f"Failed to configure Docling offline mode: {e}")
        return False


def initialize_document_processing():
    """Initialize common configurations for document processing.
    
    This should be called before using any document processing libraries
    to ensure proper SSL and caching configuration. Uses online mode first
    with caching to model_artifacts/docling.
    """
    # Setup model cache directory
    cache_path, subdirs = setup_model_cache_dir()
    
    # Configure SSL for model downloads
    ssl_configured = configure_ssl()
    
    # Configure Hugging Face for online mode with local caching
    hf_configured = configure_huggingface_online()
    
    # Configure Docling for online mode with local caching
    docling_configured = configure_docling_online()
    
    # Prepare model download directories
    models_prepared = download_and_cache_models()
    
    logger.info(f"Document processing initialized (SSL: {'OK' if ssl_configured else 'FAILED'}, HF_ONLINE: {'OK' if hf_configured else 'FAILED'}, DOCLING_ONLINE: {'OK' if docling_configured else 'FAILED'}, MODELS: {'OK' if models_prepared else 'FAILED'})")
    return ssl_configured and hf_configured and docling_configured and models_prepared


def initialize_offline_document_processing():
    """Initialize configurations for offline document processing using local model artifacts.
    
    This should be called when you want to use only local models without any network access.
    Uses models from model_artifacts/docling directory.
    """
    # Setup model cache directory
    cache_path, subdirs = setup_model_cache_dir()
    
    # Configure Hugging Face for offline mode
    hf_configured = configure_huggingface_offline()
    
    # Configure Docling for offline mode
    docling_configured = configure_docling_offline()
    
    # Verify local models are available
    models_available = check_docling_models_available()
    
    logger.info(f"Offline document processing initialized (HF_OFFLINE: {'OK' if hf_configured else 'FAILED'}, DOCLING_OFFLINE: {'OK' if docling_configured else 'FAILED'}, LOCAL_MODELS: {'OK' if models_available else 'FAILED'})")
    return hf_configured and docling_configured and models_available


# =============================================================================
# Testing and Utility Functions
# =============================================================================

def _test_config_functions() -> None:
    """Test the core configuration loading functions."""
    print("Testing Core Configuration Functions...")
    
    try:
        # Test basic config loading
        print("\n📋 Testing basic config loading:")
        config = load_config()
        print(f"   Total sections loaded: {len(config)}")
        
        # Test section loading
        print("\n📋 Testing section loading:")
        available_sections = get_available_config_sections()
        print(f"   Available sections: {available_sections}")
        
        for section in ['llm', 'deepseek', 'qwen']:
            if section in available_sections:
                section_config = load_section_config(section)
                print(f"   [{section}] keys: {list(section_config.keys())}")
            else:
                print(f"   [{section}] not found in config")
        
        # Test model name mapping
        print("\n📋 Testing model name mapping:")
        test_models = ['gpt-4', 'deepseek-chat', 'qwen-max', 'unknown-model']
        for model in test_models:
            try:
                section = get_section_by_model_name(model)
                print(f"   {model} → [{section}]")
            except Exception as e:
                print(f"   {model} → Error: {e}")
        
        # Test specific value retrieval
        print("\n📋 Testing value retrieval:")
        api_key = get_config_value('llm', 'api_key')
        print(f"   LLM API key configured: {bool(api_key)}")
        
        # Test cache clearing
        print("\n📋 Testing cache management:")
        clear_config_cache()
        print("   Cache cleared successfully")
        
    except Exception as e:
        print(f"   ❌ Error in config tests: {e}")
    
    print("\n✅ Core configuration function tests completed!")


if __name__ == "__main__":
    _test_config_functions()
else:
    # Auto-initialize when module is imported (default to online mode)
    initialize_document_processing()
