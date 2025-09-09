"""
RAG Retriever Service with Reciprocal Rank Fusion (RRF) using LangChain EnsembleRetriever.

This module provides a comprehensive retrieval service that combines:
- Vector similarity search results
- BM25 keyword search results
- LangChain EnsembleRetriever for RRF ranking
- Configurable retrieval parameters
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever, Document

from ..vector.chromadb import ChromaDBVectorService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomBM25Retriever(BaseRetriever):
    """
    Custom BM25 Retriever that integrates with ChromaDB vector service.
    """
    
    vector_service: ChromaDBVectorService
    
    def __init__(self, vector_service: ChromaDBVectorService) -> None:
        """Initialize CustomBM25Retriever.
        
        Args:
            vector_service: ChromaDB vector service instance
        """
        super().__init__(vector_service=vector_service)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using BM25 search.
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant Document objects
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to BM25 retriever")
            return []
            
        try:
            results = self.vector_service.bm25_search(query, n_results=20)
            documents = [
                Document(
                    page_content=result['document'],
                    metadata=result['metadata']
                )
                for result in results
            ]
            return documents
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return []


class RRFRetriever:
    """
    RAG Retriever using LangChain EnsembleRetriever for RRF to combine vector and BM25 search results.
    
    Features:
    - LangChain EnsembleRetriever with RRF algorithm
    - Configurable retrieval parameters (weights, result counts)
    - Integration with ChromaDB vector service
    - Support for metadata filtering
    - Automatic fallback strategies
    """
    
    def __init__(
        self,
        vector_service: ChromaDBVectorService,
        default_top_k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> None:
        """
        Initialize RRF Retriever using LangChain EnsembleRetriever.
        
        Args:
            vector_service: ChromaDB vector service instance
            default_top_k: Default number of results to return (default: 10)
            vector_weight: Weight for vector search results in ensemble (default: 0.5)
            bm25_weight: Weight for BM25 search results in ensemble (default: 0.5)
            
        Raises:
            ValueError: If weights are invalid or default_top_k is not positive
        """
        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive")
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("Weights must be non-negative")
        if vector_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one weight must be positive")
        self.vector_service = vector_service
        self.default_top_k = default_top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Create individual retrievers
        self.vector_retriever = vector_service.vectorstore.as_retriever(
            search_kwargs={"k": default_top_k * 2}
        )
        
        # Create BM25 retriever if BM25 is enabled
        if vector_service.enable_bm25:
            self.bm25_retriever = CustomBM25Retriever(vector_service)
            
            # Create ensemble retriever with RRF
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.vector_retriever, self.bm25_retriever],
                weights=[vector_weight, bm25_weight]
            )
        else:
            self.bm25_retriever = None
            self.ensemble_retriever = None
        
        logger.info(f"RRF Retriever initialized with EnsembleRetriever, weights: vector={vector_weight}, bm25={bm25_weight}")
    
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_vector: bool = True,
        use_bm25: bool = True
    ) -> List[Document]:
        """
        Retrieve documents using EnsembleRetriever with RRF.
        
        Args:
            query: Search query
            top_k: Final number of results to return (default: self.default_top_k)
            use_vector: Whether to use vector search
            use_bm25: Whether to use BM25 search
            
        Returns:
            List of retrieved LangChain Document objects
            
        Raises:
            ValueError: If query is empty or top_k is invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k is None:
            top_k = self.default_top_k
        
        query_preview = query[:50] + '...' if len(query) > 50 else query
        logger.info(f"Retrieving documents for query: '{query_preview}'")
        logger.info(f"Parameters: top_k={top_k}, use_vector={use_vector}, use_bm25={use_bm25}")
        
        try:
            # Use EnsembleRetriever if both methods are enabled
            if use_vector and use_bm25 and self.ensemble_retriever:
                documents = self.ensemble_retriever.get_relevant_documents(query)
                logger.info(f"EnsembleRetriever returned {len(documents)} results")
                return documents[:top_k]
            
            # Fallback to single method if only one is enabled
            elif use_vector:
                documents = self.vector_retriever.get_relevant_documents(query)
                logger.info(f"Vector search returned {len(documents)} results")
                return documents[:top_k]
            
            elif use_bm25 and self.bm25_retriever:
                documents = self.bm25_retriever.get_relevant_documents(query)
                logger.info(f"BM25 search returned {len(documents)} results")
                return documents[:top_k]
            
            else:
                logger.warning("No search methods enabled or available")
                return []
                
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Retrieve documents and return additional scoring information.
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional arguments passed to retrieve()
            
        Returns:
            Tuple of (documents, scoring_info)
        """
        documents = self.retrieve(query, top_k, **kwargs)
        
        # Calculate scoring statistics
        scoring_info = {
            'total_results': len(documents),
            'vector_weight': self.vector_weight,
            'bm25_weight': self.bm25_weight,
            'ensemble_enabled': self.ensemble_retriever is not None,
            'bm25_enabled': self.bm25_retriever is not None
        }
        
        return documents, scoring_info
    
    def retrieve_documents(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Retrieve documents as LangChain Document objects (alias for retrieve).
        
        Args:
            query: Search query
            top_k: Number of results to return
            **kwargs: Additional arguments passed to retrieve()
            
        Returns:
            List of LangChain Document objects
            
        Note:
            This is an alias method for retrieve() to maintain API compatibility.
        """
        return self.retrieve(query, top_k, **kwargs)
    


def _test_retriever() -> None:
    """Basic testing function for RRF Retriever.
    
    This function demonstrates:
    - Service initialization
    - Document retrieval with different methods
    - Statistics reporting
    """
    print("Testing RRF Retriever with EnsembleRetriever...")
    
    try:
        # Initialize vector service
        vector_service = ChromaDBVectorService()
        
        # Initialize RRF retriever
        retriever = RRFRetriever(vector_service=vector_service)
        
        # Test retrieval with a sample query
        print("\nTesting EnsembleRetriever with RRF...")
        try:
            documents = retriever.retrieve("machine learning", top_k=3)
            
            if documents:
                for i, doc in enumerate(documents, 1):
                    content_preview = doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
                    print(f"Result {i}: {content_preview}")
                    print(f"  Metadata: {doc.metadata}")
            else:
                print("No results found. Make sure documents are added to the vector service.")
        except ValueError as e:
            print(f"Validation error: {e}")
        except Exception as e:
            print(f"Retrieval error: {e}")
        
        # Show basic retriever info
        print("\nRetriever Configuration:")
        print(f"  Default top_k: {retriever.default_top_k}")
        print(f"  Vector weight: {retriever.vector_weight}")
        print(f"  BM25 weight: {retriever.bm25_weight}")
        print(f"  Ensemble enabled: {retriever.ensemble_retriever is not None}")
        print(f"  BM25 enabled: {retriever.bm25_retriever is not None}")
        
    except Exception as e:
        logger.error(f"Error testing RRF retriever: {e}")


if __name__ == "__main__":
    _test_retriever()