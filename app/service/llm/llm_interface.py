"""
Simple LLM Interface for creating LLM clients with invoke() method.

This module provides a simple factory function to create LLM clients
that can be invoked with llm.invoke().
"""

import logging
from typing import Optional
from pathlib import Path

# Import config utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
try:
    from app.utils.common import get_config_value, load_section_config
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(Path(__file__).parent.parent.parent / "app"))
    from utils.common import get_config_value, load_section_config

# LangChain imports
from langchain.chat_models.base import BaseChatModel

# Optional imports for different providers
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_deepseek import ChatDeepSeek
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False


# Configure logging
logger = logging.getLogger(__name__)

# Default models for each provider (fallback only)
DEFAULT_MODELS = {
    'deepseek': 'deepseek-chat',  # Fallback if config.ini doesn't have model
    'openai': 'gpt-3.5-turbo'
}


def _create_llm(provider: str, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Create LLM client based on provider."""
    # Get model name from config or use provided/default
    if not model_name:
        # First try to get model from provider's own config section
        provider_config = load_section_config(provider)
        model_name = provider_config.get('model')
        
        # If not found, try llm_service config
        if not model_name:
            llm_config = load_section_config('llm_service')
            if llm_config.get('default_provider') == provider:
                model_name = llm_config.get('default_model', DEFAULT_MODELS.get(provider))
            else:
                model_name = DEFAULT_MODELS.get(provider)
    
    # Load provider configuration
    config = load_section_config(provider)
    config.update(kwargs)
    
    try:
        if provider == 'deepseek':
            api_key = get_config_value(provider, 'api_key', env_var='DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError("DeepSeek API key required. Set in config.ini or DEEPSEEK_API_KEY environment variable.")
            
            # Check if using custom base URL (like DashScope)
            base_url = get_config_value(provider, 'base_url', default=None)
            if base_url:
                # Use OpenAI-compatible interface for DashScope
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI not available. Install with: pip install langchain-openai")
                return ChatOpenAI(
                    model=model_name,
                    openai_api_key=api_key,
                    base_url=base_url,
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens', 2048),
                    timeout=config.get('timeout', 30),
                    max_retries=config.get('max_retries', 3)
                )
            else:
                # Use native DeepSeek client for official DeepSeek API
                if not DEEPSEEK_AVAILABLE:
                    raise ImportError("ChatDeepSeek not available. Install with: pip install langchain-deepseek")
                return ChatDeepSeek(
                    model=model_name,
                    deepseek_api_key=api_key,
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens', 2048),
                    timeout=config.get('timeout', 30),
                    max_retries=config.get('max_retries', 3)
                )
        
        elif provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not available. Install with: pip install langchain-openai")
            
            api_key = get_config_value(provider, 'api_key', env_var='OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required. Set in config.ini or OPENAI_API_KEY environment variable.")
            
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                base_url=get_config_value(provider, 'base_url', default="https://api.openai.com/v1"),
                temperature=config.get('temperature', 0.7),
                max_tokens=config.get('max_tokens', 2048),
                timeout=config.get('timeout', 30),
                max_retries=config.get('max_retries', 3)
            )
        
        
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {list(DEFAULT_MODELS.keys())}")
            
    except Exception as e:
        logger.error(f"Failed to create {provider} LLM: {e}")
        raise


def create_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    Factory function to create an LLM client that can be invoked with llm.invoke().
    
    Args:
        provider: LLM provider ('deepseek', 'openai')
        model_name: Model name (uses default if None)
        **kwargs: Additional configuration
        
    Returns:
        LLM client with invoke() method
        
    Examples:
        # Use defaults from config.ini (DeepSeek)
        llm = create_llm()
        response = llm.invoke("Hello!")
        
        # Specific provider and model
        llm = create_llm(provider='deepseek', model_name='deepseek-chat')
        
        # OpenAI
        llm = create_llm(provider='openai', model_name='gpt-4')
    """
    # Use ChatDeepSeek as default provider
    if provider is None:
        provider = 'deepseek'
    
    return _create_llm(provider=provider, model_name=model_name, **kwargs)


def _test_llm():
    """Test function for the LLM client."""
    print("Testing LLM Client...")
    
    try:
        # Test LLM creation
        print("\n1. Creating LLM client...")
        llm = create_llm()
        print(f"LLM client created successfully")
        
        # Test basic invoke
        print("\n2. Testing basic invoke...")
        response = llm.invoke("Hello! Say 'Hi' back.")
        print(f"Response: {response.content}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Test failed: {e}")


if __name__ == "__main__":
    _test_llm()