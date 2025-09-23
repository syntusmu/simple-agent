"""LLM Interface for creating LLM clients with invoke() and streaming methods.

This module provides factory functions to create LLM clients with both
regular invoke() and streaming capabilities, including callback handlers
for real-time token streaming.

Supported Providers:
    - OpenAI: GPT models (gpt-4, gpt-3.5-turbo, etc.)
    - DeepSeek: DeepSeek models (deepseek-chat, deepseek-coder, etc.) - Native LangChain support
    - Qwen: Qwen models (qwen-max, qwen-plus, qwen3-32b, etc.)

Configuration Sections:
    [llm]      # Default section - OpenAI-compatible interface
    [deepseek] # Native DeepSeek integration (langchain_deepseek)
    [qwen]     # Native Qwen/Tongyi integration (langchain_tongyi)
    [openai]   # Native OpenAI integration (langchain_openai)
    
Section-to-Provider Mapping:
    [llm] → OpenAI-compatible interface (default)
    [deepseek] → Native DeepSeek (if available) or OpenAI-compatible fallback
    [qwen] → Native Tongyi (if available) or OpenAI-compatible fallback
    [openai] → Native OpenAI integration
    
Configuration Format:
    [section_name]
    api_key = your_api_key
    base_url = your_base_url  # optional
    model = your_model        # optional
"""

import logging
import queue
import threading
import time
from typing import Optional, Generator, Any, Dict, List
from pathlib import Path

# Configure logging first
logger = logging.getLogger(__name__)

# Import config utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.utils.common import (
    get_config_value, 
    load_section_config,
    get_available_config_sections,
    get_section_by_model_name,
    MODEL_TO_SECTION_MAP
)

# LangChain imports
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage

# Import all available LangChain integrations
try:
    from langchain_deepseek import ChatDeepSeek
    DEEPSEEK_AVAILABLE = True
except ImportError:
    ChatDeepSeek = None
    DEEPSEEK_AVAILABLE = False

try:
    from langchain_tongyi import ChatTongyi
    TONGYI_AVAILABLE = True
except ImportError:
    ChatTongyi = None
    TONGYI_AVAILABLE = False

# Configuration section to LangChain package mapping
SECTION_TO_PROVIDER = {
    'llm': 'openai',      # Default section uses OpenAI-compatible interface
    'deepseek': 'deepseek',  # DeepSeek section uses native DeepSeek
    'qwen': 'qwen',       # Qwen section uses Tongyi
    'openai': 'openai'    # OpenAI section uses OpenAI
}

# Default models for each section
DEFAULT_MODELS = {
    'llm': 'qwen-plus',       # Default section configured for Qwen
    'deepseek': 'deepseek-v3',
    'qwen': 'qwen3-32b',
    'openai': 'gpt-3.5-turbo'
}

# Default base URLs for providers
DEFAULT_BASE_URLS = {
    'llm': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'deepseek': 'https://api.deepseek.com/v1',
    'qwen': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'openai': 'https://api.openai.com/v1'
}

# Default configuration values
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3

# Streaming constants
DEFAULT_STREAM_TIMEOUT = 1.0
DEFAULT_MAX_TIMEOUTS = 100
DEFAULT_CHUNK_SIZE = 2
DEFAULT_CHUNK_DELAY = 0.05

# Supported sections
SUPPORTED_SECTIONS = list(SECTION_TO_PROVIDER.keys())


# =============================================================================
# CORE LLM CREATION FUNCTIONS
# =============================================================================

def _create_llm(section: str, model_name: Optional[str] = None, **kwargs) -> BaseChatModel:
    """
    Create LLM client based on configuration section.
    
    Args:
        section: Configuration section name ('llm', 'deepseek', 'qwen', 'openai')
        model_name: Model name (uses default if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured LLM client
        
    Raises:
        ValueError: If section is unsupported or API key is missing
    """
    _validate_section(section)
    
    model_name = model_name or _get_model_name(section)
    config = _load_section_config_with_kwargs(section, **kwargs)
    common_params = _get_common_params(config)
    
    # Create client based on section-to-provider mapping
    provider = SECTION_TO_PROVIDER[section]
    
    if provider == 'openai' or section == 'llm':
        return _create_openai_compatible_llm(section, model_name, common_params)
    elif provider == 'deepseek' and section == 'deepseek':
        return _create_native_deepseek_llm(section, model_name, common_params)
    elif provider == 'qwen' and section == 'qwen':
        return _create_native_qwen_llm(section, model_name, common_params)
    else:
        # Fallback to OpenAI-compatible interface
        return _create_openai_compatible_llm(section, model_name, common_params)


def _validate_section(section: str) -> None:
    """Validate that the configuration section is supported."""
    if section not in SUPPORTED_SECTIONS:
        raise ValueError(f"Unsupported section: {section}. Supported: {SUPPORTED_SECTIONS}")


def _get_model_name(section: str) -> str:
    """Get model name from config or use default."""
    section_config = load_section_config(section)
    return section_config.get('model', DEFAULT_MODELS.get(section))


def _load_section_config_with_kwargs(section: str, **kwargs) -> Dict[str, Any]:
    """Load section configuration and merge with kwargs."""
    config = load_section_config(section)
    config.update(kwargs)
    return config


def _get_common_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get common parameters for LLM clients."""
    return {
        'streaming': True,  # Enable streaming by default
        'temperature': float(config.get('temperature', str(DEFAULT_TEMPERATURE))),
        'max_tokens': int(config.get('max_tokens', str(DEFAULT_MAX_TOKENS))),
        'timeout': int(config.get('timeout', str(DEFAULT_TIMEOUT))),
        'max_retries': int(config.get('max_retries', str(DEFAULT_MAX_RETRIES)))
    }


def _create_openai_compatible_llm(section: str, model_name: str, common_params: Dict[str, Any]) -> ChatOpenAI:
    """Create OpenAI-compatible LLM client for any section."""
    api_key = get_config_value(section, 'api_key')
    if not api_key:
        raise ValueError(f"API key is required. Please set 'api_key' in the [{section}] section of config.ini")
    
    base_url = get_config_value(section, 'base_url', default=DEFAULT_BASE_URLS.get(section))
    
    logger.info(f"Creating OpenAI-compatible client for [{section}] section")
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        base_url=base_url,
        **common_params
    )


def _create_native_deepseek_llm(section: str, model_name: str, common_params: Dict[str, Any]) -> BaseChatModel:
    """Create native DeepSeek LLM client."""
    api_key = get_config_value(section, 'api_key')
    if not api_key:
        raise ValueError(f"DeepSeek API key is required. Please set 'api_key' in the [{section}] section of config.ini")
    
    base_url = get_config_value(section, 'base_url')
    
    if base_url:
        # Use custom base URL with OpenAI-compatible interface
        logger.info(f"Using custom base URL for DeepSeek: {base_url}")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            base_url=base_url,
            **common_params
        )
    elif DEEPSEEK_AVAILABLE:
        # Use native DeepSeek client (preferred)
        logger.info("Using native DeepSeek LangChain integration")
        return ChatDeepSeek(
            model=model_name,
            deepseek_api_key=api_key,
            **common_params
        )
    else:
        # Fallback to OpenAI-compatible interface
        logger.info("Using OpenAI-compatible interface for DeepSeek (langchain_deepseek not available)")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            base_url=DEFAULT_BASE_URLS['deepseek'],
            **common_params
        )


def _create_native_qwen_llm(section: str, model_name: str, common_params: Dict[str, Any]) -> BaseChatModel:
    """Create native Qwen/Tongyi LLM client."""
    api_key = get_config_value(section, 'api_key')
    if not api_key:
        raise ValueError(f"Qwen API key is required. Please set 'api_key' in the [{section}] section of config.ini")
    
    base_url = get_config_value(section, 'base_url')
    
    if base_url:
        # Use custom base URL with OpenAI-compatible interface
        logger.info(f"Using custom base URL for Qwen: {base_url}")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            base_url=base_url,
            **common_params
        )
    elif TONGYI_AVAILABLE:
        # Use native Tongyi client (preferred)
        logger.info("Using native Tongyi LangChain integration")
        return ChatTongyi(
            model=model_name,
            dashscope_api_key=api_key,
            **common_params
        )
    else:
        # Fallback to OpenAI-compatible interface
        logger.info("Using OpenAI-compatible interface for Qwen (langchain_tongyi not available)")
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            base_url=DEFAULT_BASE_URLS['qwen'],
            **common_params
        )


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def create_llm(
    section: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    Factory function to create an LLM client based on configuration section or model name.
    
    Args:
        section: Configuration section ('llm', 'deepseek', 'qwen', 'openai'). 
                If None and model_name provided, auto-detects section from model name.
                Defaults to 'llm' if both are None.
        model_name: Model name (uses default if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        LLM client with invoke() method
        
    Raises:
        ValueError: If section is unsupported or required configuration is missing
        
    Examples:
        # Use default [llm] section (OpenAI-compatible)
        llm = create_llm()
        
        # Auto-detect section from model name
        llm = create_llm(model_name='deepseek-chat')  # Auto-detects [deepseek] section
        llm = create_llm(model_name='gpt-4')          # Auto-detects [openai] section
        llm = create_llm(model_name='qwen-max')       # Auto-detects [qwen] section
        
        # Explicit section specification
        llm = create_llm(section='deepseek', model_name='deepseek-chat')
        llm = create_llm(section='qwen', model_name='qwen-max')
        llm = create_llm(section='openai', model_name='gpt-4')
        
    Configuration Auto-Detection:
        - If model_name provided but no section: auto-detects section from model name
        - If section provided: uses specified section
        - If neither provided: defaults to [llm] section
        
    Configuration Mapping:
        [llm] → OpenAI-compatible interface (default)
        [deepseek] → Native langchain_deepseek (if available)
        [qwen] → Native langchain_tongyi (if available)
        [openai] → Native langchain_openai
    """
    # Auto-detect section from model name if section not provided
    if section is None and model_name:
        section = get_section_by_model_name(model_name)
        logger.info(f"Auto-detected section '{section}' for model '{model_name}'")
    elif section is None:
        # Default to 'llm' section if neither section nor model_name provided
        section = 'llm'
    
    return _create_llm(section=section, model_name=model_name, **kwargs)


def create_llm_by_model_name(
    model_name: str,
    **kwargs
) -> BaseChatModel:
    """
    Create an LLM client by model name with automatic section detection.
    
    This is a convenience function that automatically determines the correct
    configuration section based on the model name.
    
    Args:
        model_name: Model name (e.g., 'gpt-4', 'deepseek-chat', 'qwen-max')
        **kwargs: Additional configuration parameters
        
    Returns:
        LLM client with invoke() method
        
    Examples:
        # Automatically detects sections
        llm = create_llm_by_model_name('gpt-4')          # Uses [openai] section
        llm = create_llm_by_model_name('deepseek-chat')  # Uses [deepseek] section
        llm = create_llm_by_model_name('qwen-max')       # Uses [qwen] section
    """
    section = get_section_by_model_name(model_name)
    logger.info(f"Creating LLM for model '{model_name}' using section '{section}'")
    return create_llm(section=section, model_name=model_name, **kwargs)


# =============================================================================
# STREAMING FUNCTIONALITY
# =============================================================================

class EnhancedStreamingCallbackHandler(BaseCallbackHandler):
    """
    Enhanced callback handler for streaming LLM responses.
    
    This handler captures tokens from LLM responses and provides them
    through a queue-based interface for real-time streaming.
    """
    
    def __init__(self):
        """Initialize the streaming callback handler."""
        self.token_queue = queue.Queue()
        self.current_response = ""
        self.finished = False
        self.agent_finished = False
        self.error = None
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Called when LLM generates a new token."""
        self.current_response += token
        # Only queue non-empty tokens to reduce noise
        if token and token.strip():
            self.token_queue.put(token)
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes."""
        # Don't mark as finished yet - agent might continue with tools
        pass
    
    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when agent finishes completely."""
        self.agent_finished = True
        self.finished = True
        self.token_queue.put(None)  # Signal end
    
    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        """Called when tool execution starts."""
        # Don't stream tool execution details to keep output clean
        pass
    
    def on_tool_end(self, output, **kwargs) -> None:
        """Called when tool execution ends."""
        # Don't stream tool output details to keep output clean
        pass
    
    def on_llm_error(self, error, **kwargs) -> None:
        """Called when LLM encounters an error."""
        self.error = error
        self.finished = True
        self.token_queue.put(None)
    
    def get_response(self) -> str:
        """Get the complete response collected so far."""
        return self.current_response
    
    def get_tokens(self, timeout: float = DEFAULT_STREAM_TIMEOUT) -> Generator[str, None, None]:
        """
        Generator that yields tokens as they become available.
        
        Args:
            timeout: Timeout for waiting for tokens
            
        Yields:
            Streaming tokens from the LLM
        """
        timeout_count = 0
        max_timeouts = DEFAULT_MAX_TIMEOUTS
        
        while True:
            try:
                token = self.token_queue.get(timeout=timeout)
                timeout_count = 0  # Reset timeout counter
                
                if token is None:  # End signal
                    break
                
                if token and token.strip():
                    yield token
                    
            except queue.Empty:
                timeout_count += 1
                if self.finished or timeout_count >= max_timeouts:
                    break
                continue


def create_streaming_llm(
    section: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    Create an LLM client optimized for streaming with auto-detection support.
    
    Args:
        section: Configuration section ('llm', 'deepseek', 'qwen', 'openai').
                If None and model_name provided, auto-detects section from model name.
                Defaults to 'llm' if both are None.
        model_name: Model name (uses default if None)
        **kwargs: Additional configuration parameters
        
    Returns:
        LLM client with streaming enabled
        
    Examples:
        # Default [llm] section streaming
        llm = create_streaming_llm()
        
        # Auto-detect section from model name
        llm = create_streaming_llm(model_name='deepseek-chat')  # Auto-detects [deepseek]
        llm = create_streaming_llm(model_name='gpt-4')          # Auto-detects [openai]
        llm = create_streaming_llm(model_name='qwen-max')       # Auto-detects [qwen]
        
        # Explicit section specification
        llm = create_streaming_llm(section='deepseek', model_name='deepseek-chat')
        llm = create_streaming_llm(section='qwen', model_name='qwen-max')
        llm = create_streaming_llm(section='openai', model_name='gpt-4')
    """
    # Ensure streaming is enabled
    kwargs['streaming'] = True
    return create_llm(section=section, model_name=model_name, **kwargs)


def create_streaming_llm_by_model_name(
    model_name: str,
    **kwargs
) -> BaseChatModel:
    """
    Create a streaming LLM client by model name with automatic section detection.
    
    Args:
        model_name: Model name (e.g., 'gpt-4', 'deepseek-chat', 'qwen-max')
        **kwargs: Additional configuration parameters
        
    Returns:
        LLM client with streaming enabled
        
    Examples:
        llm = create_streaming_llm_by_model_name('deepseek-chat')
        llm = create_streaming_llm_by_model_name('qwen-max')
    """
    kwargs['streaming'] = True
    return create_llm_by_model_name(model_name, **kwargs)


def stream_llm_response(
    llm: BaseChatModel,
    messages: list,
    callback_handler: Optional[EnhancedStreamingCallbackHandler] = None,
    timeout: float = 10.0
) -> Generator[str, None, None]:
    """
    Stream response from LLM with callback handler.
    
    Args:
        llm: LLM instance
        messages: List of messages to send
        callback_handler: Optional callback handler (creates one if None)
        timeout: Timeout for thread cleanup in seconds
        
    Yields:
        Streaming tokens from the LLM response
        
    Raises:
        Exception: If LLM encounters an error during processing
    """
    if callback_handler is None:
        callback_handler = EnhancedStreamingCallbackHandler()
    
    def run_llm():
        """Run LLM in separate thread."""
        try:
            llm.invoke(messages, config={"callbacks": [callback_handler]})
            if not callback_handler.finished:
                callback_handler.finished = True
                callback_handler.token_queue.put(None)
        except Exception as e:
            callback_handler.error = e
            callback_handler.finished = True
            callback_handler.token_queue.put(None)
    
    # Start LLM execution in background thread
    llm_thread = threading.Thread(target=run_llm)
    llm_thread.start()
    
    # Stream tokens as they become available
    try:
        yield from callback_handler.get_tokens()
    finally:
        # Ensure thread cleanup
        llm_thread.join(timeout=timeout)
        if callback_handler.error:
            raise callback_handler.error


def create_fallback_stream(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    delay: float = DEFAULT_CHUNK_DELAY
) -> Generator[str, None, None]:
    """
    Create a fallback stream by chunking text.
    
    Args:
        text: Text to stream
        chunk_size: Number of words per chunk
        delay: Delay between chunks
        
    Yields:
        Text chunks for streaming
    """
    if not text:
        return
    
    words = text.split()
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if i + chunk_size < len(words):
            chunk += " "
        yield chunk
        if delay > 0:
            time.sleep(delay)


# =============================================================================
# UTILITY AND INFORMATION FUNCTIONS
# =============================================================================

def get_section_info() -> Dict[str, Dict[str, str]]:
    """
    Get information about all supported configuration sections.
    
    Returns:
        Dictionary with section information including models, URLs, and providers
    """
    return {
        section: {
            'provider': SECTION_TO_PROVIDER[section],
            'default_model': DEFAULT_MODELS[section],
            'base_url': DEFAULT_BASE_URLS[section],
            'native_available': _check_native_availability(section)
        }
        for section in SUPPORTED_SECTIONS
    }


def _check_native_availability(section: str) -> str:
    """Check if native LangChain package is available for the section."""
    provider = SECTION_TO_PROVIDER[section]
    if provider == 'deepseek':
        return 'Yes' if DEEPSEEK_AVAILABLE else 'No'
    elif provider == 'qwen':
        return 'Yes' if TONGYI_AVAILABLE else 'No'
    elif provider == 'openai':
        return 'Yes'  # langchain_openai is always available
    else:
        return 'OpenAI-compatible'


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def _test_llm() -> None:
    """Test function for the LLM client."""
    print("Testing LLM Client...")
    
    test_cases = [
        ("Testing section support", _test_section_support),
        ("Testing model name detection", _test_model_name_detection),
        ("Testing config sections", _test_config_sections),
        ("Creating LLM client", _test_llm_creation),
        ("Testing basic invoke", _test_basic_invoke),
        ("Testing streaming", _test_streaming),
        ("Testing fallback streaming", _test_fallback_streaming)
    ]
    
    for i, (description, test_func) in enumerate(test_cases, 1):
        try:
            print(f"\n{i}. {description}...")
            test_func()
            print(f"✅ {description} completed successfully!")
        except Exception as e:
            logger.error(f"{description} failed: {e}")
            print(f"❌ {description} failed: {e}")
            return
    
    print("\n🎉 All tests completed successfully!")


def _test_section_support() -> None:
    """Test that all configuration sections are supported."""
    expected_sections = ['llm', 'deepseek', 'qwen', 'openai']
    
    for section in expected_sections:
        assert section in SUPPORTED_SECTIONS, f"{section} should be supported"
        assert section in DEFAULT_MODELS, f"{section} should have a default model"
        assert section in DEFAULT_BASE_URLS, f"{section} should have a default base URL"
        
        # Test section validation
        _validate_section(section)
        
        provider = SECTION_TO_PROVIDER[section]
        print(f"   ✅ [{section}] → {provider}: {DEFAULT_MODELS[section]} @ {DEFAULT_BASE_URLS[section]}")
    
    print(f"   📋 All {len(expected_sections)} sections supported")


def _test_model_name_detection() -> None:
    """Test model name to section detection."""
    test_models = [
        ('gpt-4', 'openai'),
        ('deepseek-chat', 'deepseek'), 
        ('qwen-max', 'qwen'),
        ('unknown-model', 'llm')
    ]
    
    for model, expected_section in test_models:
        try:
            detected_section = get_section_by_model_name(model)
            assert detected_section == expected_section, f"Expected {expected_section}, got {detected_section}"
            print(f"   ✅ {model} → [{detected_section}]")
        except Exception as e:
            print(f"   ❌ {model} → Error: {e}")
    
    print(f"   📋 Model name detection tests completed")


def _test_config_sections() -> None:
    """Test available configuration sections."""
    try:
        available_sections = get_available_config_sections()
        print(f"   Available config sections: {available_sections}")
        
        for section in SUPPORTED_SECTIONS:
            if section in available_sections:
                print(f"   ✅ [{section}] section available")
            else:
                print(f"   ⚠️ [{section}] section not found in config")
                
    except Exception as e:
        print(f"   ❌ Error checking config sections: {e}")
    
    print(f"   📋 Config section tests completed")


def _test_llm_creation() -> None:
    """Test LLM client creation."""
    llm = create_llm()
    assert llm is not None, "LLM client should not be None"


def _test_basic_invoke() -> None:
    """Test basic LLM invoke functionality."""
    llm = create_llm()
    response = llm.invoke("Hello! Say 'Hi' back.")
    print(f"Response: {response.content}")
    assert response.content, "Response should not be empty"


def _test_streaming() -> None:
    """Test streaming functionality."""
    streaming_llm = create_streaming_llm()
    messages = [BaseMessage(content="Tell me a short joke", type="human")]
    
    print("Streaming response: ", end="")
    token_count = 0
    for token in stream_llm_response(streaming_llm, messages):
        print(token, end="", flush=True)
        token_count += 1
    print()  # New line after streaming
    
    assert token_count > 0, "Should receive at least one token"


def _test_fallback_streaming() -> None:
    """Test fallback streaming functionality."""
    test_text = "This is a test of fallback streaming functionality."
    print("Fallback stream: ", end="")
    
    chunk_count = 0
    for chunk in create_fallback_stream(test_text):
        print(chunk, end="", flush=True)
        chunk_count += 1
    print()  # New line after streaming
    
    assert chunk_count > 0, "Should receive at least one chunk"


if __name__ == "__main__":
    _test_llm()