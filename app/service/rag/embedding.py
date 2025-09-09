"""Embedding Service with Qwen and OpenAI support.

Provides a unified embedding service that:
- Supports Qwen (DashScope) and OpenAI embedding providers
- Uses centralized configuration management
- Compatible with LangChain Embeddings interface
- Includes batch processing and metadata handling
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from openai import OpenAI

# Import config utilities
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.utils.common import get_config_value, load_section_config

# LangChain imports
from langchain.embeddings.base import Embeddings
from langchain.schema import Document

try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class CustomEmbeddingService(Embeddings):
    """Unified embedding service compatible with LangChain.
    
    Supports multiple providers (Qwen, OpenAI) with automatic configuration
    from config.ini and environment variables.
    """
    
    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize embedding service.
        
        Args:
            provider: Provider ('openai', 'qwen', 'embedding'). Defaults to 'qwen'.
            model_name: Model name (uses provider default if None).
        """
        self.provider = (provider or 'qwen').lower()
        self.model_name = model_name or self._get_model_name()
        self._initialize_provider()
        logger.info(f"Embedding service initialized: {self.provider}/{self.model_name}")
    
    def _get_model_name(self) -> str:
        """Get model name from config or use defaults."""
        if self.provider == 'openai':
            return get_config_value('openai', 'model') or 'text-embedding-3-small'
        else:  # qwen or embedding
            return (get_config_value('qwen', 'model') or 
                   get_config_value('embedding', 'model') or 
                   'text-embedding-v4')
    
    def _initialize_provider(self):
        """Initialize provider-specific clients."""
        try:
            if self.provider == 'openai':
                self._init_openai()
            elif self.provider in ['qwen', 'embedding']:
                self._init_qwen()
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider}: {e}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("Install langchain-openai: pip install langchain-openai")
        
        api_key = get_config_value('openai', 'api_key', env_var='OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required in config.ini or OPENAI_API_KEY")
        
        self._openai_client = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=api_key
        )
    
    def _init_qwen(self):
        """Initialize Qwen/DashScope client."""
        api_key = (get_config_value('qwen', 'api_key', env_var='QWEN_API_KEY') or 
                  get_config_value('embedding', 'api_key', env_var='DASHSCOPE_API_KEY'))
        if not api_key:
            raise ValueError("Qwen API key required in config.ini or DASHSCOPE_API_KEY")
        
        base_url = (get_config_value('qwen', 'base_url') or 
                   get_config_value('embedding', 'base_url') or
                   "https://dashscope.aliyuncs.com/compatible-mode/v1")
        
        self._qwen_client = OpenAI(api_key=api_key, base_url=base_url)
    
    def _call_qwen_api(self, texts: List[str]) -> List[List[float]]:
        """Call Qwen API with batch processing."""
        batch_size = 10  # Qwen API limit
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            completion = self._qwen_client.embeddings.create(
                model=self.model_name,
                input=batch_texts,
                dimensions=1024,
                encoding_format="float"
            )
            
            batch_embeddings = [item.embedding for item in completion.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    # LangChain interface
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents (LangChain interface)."""
        return self.embed_texts(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query (LangChain interface)."""
        return self.embed_text(text)
    
    def embed_text(self, text: str) -> List[float]:
        """Embed single text."""
        if not text or not text.strip():
            return []
        
        text = text.strip()
        if self.provider == 'openai':
            return self._openai_client.embed_query(text)
        else:
            embeddings = self._call_qwen_api([text])
            return embeddings[0] if embeddings else []
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        if not texts:
            return []
        
        # Filter empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            return []
        
        if self.provider == 'openai':
            return self._openai_client.embed_documents(valid_texts)
        else:
            return self._call_qwen_api(valid_texts)
    
    def embed_documents_with_metadata(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Embed documents with metadata preservation."""
        if not documents:
            return []
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.embed_texts(texts)
        
        return [
            {
                'embedding': embedding,
                'text': doc.page_content,
                'index': i,
                'metadata': doc.metadata or {}
            }
            for i, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]


def create_embedding_service(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> CustomEmbeddingService:
    """Create embedding service instance.
    
    Args:
        provider: Provider ('openai', 'qwen', 'embedding'). Auto-detected if None.
        model_name: Model name (uses default if None).
        **kwargs: Ignored for compatibility.
    
    Returns:
        Configured embedding service.
    """
    if provider is None:
        # Auto-detect provider from config
        embedding_config = load_section_config('embedding')
        if embedding_config and embedding_config.get('model'):
            provider = 'embedding'
        else:
            service_config = load_section_config('embedding_service')
            provider = service_config.get('default_provider', 'qwen')
    
    return CustomEmbeddingService(provider=provider, model_name=model_name)


def _test_embedding_service():
    """Test embedding service functionality."""
    print("Testing Embedding Service...")
    
    try:
        service = create_embedding_service()
        
        # Single text test
        text = "This is a test sentence for embedding."
        embedding = service.embed_text(text)
        print(f"Single embedding: {len(embedding)} dimensions")
        
        # Batch test
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing deals with text analysis.",
            "Vector databases store high-dimensional embeddings."
        ]
        
        embeddings = service.embed_texts(texts)
        print(f"Batch embeddings: {len(embeddings)} x {len(embeddings[0])}")
        
        # LangChain interface test
        lc_embedding = service.embed_query(text)
        lc_embeddings = service.embed_documents(texts)
        print(f"LangChain compatibility: {len(lc_embedding)}, {len(lc_embeddings)}")
        
        # Document metadata test
        documents = [
            Document(page_content=text, metadata={"source": f"doc_{i}"})
            for i, text in enumerate(texts)
        ]
        
        doc_results = service.embed_documents_with_metadata(documents)
        print(f"Documents with metadata: {len(doc_results)} processed")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    _test_embedding_service()
