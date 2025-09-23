"""Embedding Service with Qwen and OpenAI support.

Provides a unified embedding service that:
- Supports Qwen (DashScope) and OpenAI embedding providers
- Uses centralized configuration management from [embedding] section
- Compatible with LangChain Embeddings interface
- Includes batch processing and metadata handling

Configuration:
    The service loads configuration from config.ini [embedding] section:
    - api_key: API key for the embedding provider
    - base_url: Base URL for the API (optional)
    - model: Model name to use for embeddings
    - provider: Provider type ('qwen' or 'openai')
    - batch_size: Batch size for processing (optional, default: 10)
    - dimensions: Embedding dimensions (optional, default: 1024)

Classes:
    CustomEmbeddingService: Main embedding service class

Functions:
    create_embedding_service: Factory function for creating embedding service
    get_embedding_config_info: Get configuration information
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI

# Import config utilities
from app.utils.common import get_config_value, load_section_config

# LangChain imports
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_SECTION = "embedding"
DEFAULT_PROVIDER = "qwen"
DEFAULT_MODELS = {
    "qwen": "text-embedding-v3",
    "openai": "text-embedding-3-small"
}
DEFAULT_BASE_URLS = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1"
}
DEFAULT_BATCH_SIZE = 10
DEFAULT_DIMENSIONS = 1024
SUPPORTED_PROVIDERS = ["qwen", "openai"]



# =============================================================================
# MAIN EMBEDDING SERVICE CLASS
# =============================================================================

class CustomEmbeddingService(Embeddings):
    """Unified embedding service compatible with LangChain.
    
    Loads configuration from config.ini [embedding] section and supports
    multiple providers (Qwen, OpenAI) with automatic fallback.
    
    Attributes:
        config: Configuration dictionary from [embedding] section
        provider: Provider name ('qwen' or 'openai')
        model_name: Model name for embeddings
    """
    
    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize embedding service.
        
        Args:
            provider: Provider ('openai', 'qwen'). Auto-detected from config if None.
            model_name: Model name (uses config or provider default if None).
        """
        # Load embedding configuration
        self.config = self._load_embedding_config()
        
        # Set provider and model
        self.provider = self._get_provider(provider)
        self.model_name = self._get_model_name(model_name)
        
        # Initialize provider-specific client
        self._initialize_provider()
        
        logger.info(f"Embedding service initialized: {self.provider}/{self.model_name} from [{DEFAULT_EMBEDDING_SECTION}] section")
    
    def _load_embedding_config(self) -> Dict[str, str]:
        """Load embedding configuration from [embedding] section.
        
        Returns:
            Dictionary containing embedding configuration
        """
        config = load_section_config(DEFAULT_EMBEDDING_SECTION)
        if not config:
            logger.warning(f"No [{DEFAULT_EMBEDDING_SECTION}] section found in config.ini")
            return {}
        
        logger.debug(f"Loaded embedding config from [{DEFAULT_EMBEDDING_SECTION}] section: {list(config.keys())}")
        return config
    
    def _get_provider(self, provider: Optional[str]) -> str:
        """Get provider from parameter, config, or default.
        
        Args:
            provider: Explicit provider override
            
        Returns:
            Provider name ('qwen' or 'openai')
        """
        if provider:
            return provider.lower()
        
        # Get from config
        config_provider = self.config.get('provider', '').lower()
        if config_provider in ['qwen', 'openai']:
            return config_provider
        
        # Default fallback
        logger.info(f"No provider specified, using default: {DEFAULT_PROVIDER}")
        return DEFAULT_PROVIDER
    
    def _get_model_name(self, model_name: Optional[str]) -> str:
        """Get model name from parameter, config, or default.
        
        Args:
            model_name: Explicit model name override
            
        Returns:
            Model name for the provider
        """
        if model_name:
            return model_name
        
        # Get from config
        config_model = self.config.get('model')
        if config_model:
            return config_model
        
        # Default fallback
        default_model = DEFAULT_MODELS.get(self.provider, DEFAULT_MODELS['qwen'])
        logger.info(f"No model specified for {self.provider}, using default: {default_model}")
        return default_model
    
    def _validate_provider(self, provider: str) -> None:
        """Validate provider is supported.
        
        Args:
            provider: Provider name to validate
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported: {SUPPORTED_PROVIDERS}")
    
    def _initialize_provider(self) -> None:
        """Initialize provider-specific clients."""
        try:
            self._validate_provider(self.provider)
            
            if self.provider == 'openai':
                self._init_openai()
            elif self.provider == 'qwen':
                self._init_qwen()
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider}: {e}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI client from [embedding] section."""
        api_key = self.config.get('api_key')
        if not api_key:
            raise ValueError(f"OpenAI API key required in [{DEFAULT_EMBEDDING_SECTION}] section of config.ini")
        
        # Optional base URL override
        base_url = self.config.get('base_url')
        
        client_params = {
            'model': self.model_name,
            'openai_api_key': api_key
        }
        
        if base_url:
            client_params['openai_api_base'] = base_url
            logger.info(f"Using custom OpenAI base URL: {base_url}")
        
        self._openai_client = OpenAIEmbeddings(**client_params)
    
    def _init_qwen(self):
        """Initialize Qwen/DashScope client from [embedding] section."""
        api_key = self.config.get('api_key')
        if not api_key:
            raise ValueError(f"Qwen API key required in [{DEFAULT_EMBEDDING_SECTION}] section of config.ini")
        
        base_url = self.config.get('base_url', DEFAULT_BASE_URLS['qwen'])
        
        logger.info(f"Initializing Qwen embedding client with base URL: {base_url}")
        
        # Use OpenAI client for Qwen API calls
        # Note: While Qwen API is OpenAI-compatible, it has different parameter expectations
        # (e.g., 'contents' vs 'input') so we use the raw OpenAI client with custom embedding calls
        self._qwen_client = OpenAI(api_key=api_key, base_url=base_url)
    
    def _call_qwen_api(self, texts: List[str]) -> List[List[float]]:
        """Call Qwen API with batch processing using OpenAI client.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If API call fails
            
        Note:
            Qwen API is OpenAI-compatible but has different parameter expectations.
            We use the OpenAI client directly for embedding calls to handle these differences.
        """
        try:
            batch_size = int(self.config.get('batch_size', DEFAULT_BATCH_SIZE))
            dimensions = int(self.config.get('dimensions', DEFAULT_DIMENSIONS))
            all_embeddings = []
            
            logger.debug(f"Processing {len(texts)} texts in batches of {batch_size} using Qwen API")
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Use OpenAI client with embedding-specific API call
                completion = self._qwen_client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                    dimensions=dimensions,
                    encoding_format="float"
                )
                
                batch_embeddings = [item.embedding for item in completion.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error calling Qwen embedding API: {e}")
            raise
    
    # =============================================================================
    # LANGCHAIN INTERFACE METHODS
    # =============================================================================
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents (LangChain interface).
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embed_texts(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query (LangChain interface).
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        return self.embed_text(text)
    
    # =============================================================================
    # CORE EMBEDDING METHODS
    # =============================================================================
    def embed_text(self, text: str) -> List[float]:
        """Embed single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            Exception: If embedding fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
        
        try:
            text = text.strip()
            if self.provider == 'openai':
                return self._openai_client.embed_query(text)
            else:  # qwen
                # Use custom Qwen API call for single text
                embeddings = self._call_qwen_api([text])
                return embeddings[0] if embeddings else []
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding fails
        """
        if not texts:
            logger.warning("Empty text list provided for embedding")
            return []
        
        try:
            # Filter empty texts
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                logger.warning("No valid texts found after filtering")
                return []
            
            logger.info(f"Embedding {len(valid_texts)} texts using {self.provider}")
            
            if self.provider == 'openai':
                return self._openai_client.embed_documents(valid_texts)
            else:  # qwen
                # Use custom Qwen API call for batch texts
                return self._call_qwen_api(valid_texts)
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise
    
    def embed_documents_with_metadata(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Embed documents with metadata preservation."""
        if not documents:
            logger.warning("Empty document list provided")
            return []
        
        try:
            texts = [doc.page_content for doc in documents]
            embeddings = self.embed_texts(texts)
            
            logger.info(f"Successfully embedded {len(documents)} documents with metadata")
            
            return [
                {
                    'embedding': embedding,
                    'text': doc.page_content,
                    'index': i,
                    'metadata': doc.metadata or {}
                }
                for i, (doc, embedding) in enumerate(zip(documents, embeddings))
            ]
        except Exception as e:
            logger.error(f"Error embedding documents with metadata: {e}")
            raise



# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_embedding_service(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> CustomEmbeddingService:
    """Create embedding service instance using [embedding] section configuration.
    
    Args:
        provider: Provider ('openai', 'qwen'). Auto-detected from config if None.
        model_name: Model name (uses config or default if None).
        **kwargs: Additional parameters (ignored for compatibility).
    
    Returns:
        Configured embedding service loaded from [embedding] section.
        
    Raises:
        ValueError: If configuration is invalid or missing required fields.
    """
    try:
        logger.info(f"Creating embedding service from [{DEFAULT_EMBEDDING_SECTION}] section")
        return CustomEmbeddingService(provider=provider, model_name=model_name)
    except Exception as e:
        logger.error(f"Error creating embedding service: {e}")
        raise


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_embedding_config_info() -> Dict[str, Any]:
    """Get embedding configuration information.
    
    Returns:
        Dictionary with embedding configuration details including:
        - section: Configuration section name
        - config_available: Whether config is available
        - provider: Configured or default provider
        - model: Configured model name
        - api_key_configured: Whether API key is configured
        - base_url: Configured base URL
        - available_keys: List of available config keys
        - default_models: Default models for each provider
        - default_base_urls: Default base URLs for each provider
    """
    config = load_section_config(DEFAULT_EMBEDDING_SECTION)
    
    return {
        'section': DEFAULT_EMBEDDING_SECTION,
        'config_available': bool(config),
        'provider': config.get('provider', DEFAULT_PROVIDER) if config else DEFAULT_PROVIDER,
        'model': config.get('model') if config else None,
        'api_key_configured': bool(config.get('api_key')) if config else False,
        'base_url': config.get('base_url') if config else None,
        'available_keys': list(config.keys()) if config else [],
        'default_models': DEFAULT_MODELS,
        'default_base_urls': DEFAULT_BASE_URLS,
        'supported_providers': SUPPORTED_PROVIDERS,
        'default_batch_size': DEFAULT_BATCH_SIZE,
        'default_dimensions': DEFAULT_DIMENSIONS
    }


def validate_embedding_config(config: Dict[str, str]) -> List[str]:
    """Validate embedding configuration.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required fields
    if not config.get('api_key'):
        errors.append("Missing required 'api_key' in configuration")
    
    # Check provider
    provider = config.get('provider', DEFAULT_PROVIDER)
    if provider not in SUPPORTED_PROVIDERS:
        errors.append(f"Unsupported provider '{provider}'. Supported: {SUPPORTED_PROVIDERS}")
    
    # Check batch_size if provided
    batch_size = config.get('batch_size')
    if batch_size:
        try:
            batch_size_int = int(batch_size)
            if batch_size_int <= 0:
                errors.append("batch_size must be a positive integer")
        except ValueError:
            errors.append("batch_size must be a valid integer")
    
    # Check dimensions if provided
    dimensions = config.get('dimensions')
    if dimensions:
        try:
            dimensions_int = int(dimensions)
            if dimensions_int <= 0:
                errors.append("dimensions must be a positive integer")
        except ValueError:
            errors.append("dimensions must be a valid integer")
    
    return errors



if __name__ == "__main__":
    _test_embedding_service()
