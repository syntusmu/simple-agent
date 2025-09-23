"""
RAG Retriever Service with Reciprocal Rank Fusion (RRF) using LangChain EnsembleRetriever.

This module provides a comprehensive retrieval service that combines:
- Vector similarity search results
- BM25 keyword search results
- LangChain EnsembleRetriever for RRF ranking
- Configurable retrieval parameters
- Automatic fallback strategies

Classes:
    CustomBM25Retriever: Custom BM25 retriever integrated with ChromaDB
    RRFRetriever: Main retriever class using EnsembleRetriever for RRF

Functions:
    create_rrf_retriever: Factory function for creating RRF retriever instances
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever, Document
from ..vector.chromadb import ChromaDBVectorService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TOP_K = 10
DEFAULT_VECTOR_WEIGHT = 0.5
DEFAULT_BM25_WEIGHT = 0.5
DEFAULT_SEARCH_MULTIPLIER = 2
MAX_QUERY_PREVIEW_LENGTH = 50



# =============================================================================
# CUSTOM BM25 RETRIEVER CLASS
# =============================================================================

class CustomBM25Retriever(BaseRetriever):
    """
    Custom BM25 Retriever that integrates with ChromaDB vector service.
    
    This retriever provides BM25 keyword-based search functionality
    that integrates seamlessly with ChromaDB's vector service.
    
    Attributes:
        vector_service: ChromaDB vector service instance
        n_results: Number of results to retrieve from BM25 search
    """
    
    vector_service: ChromaDBVectorService
    n_results: int
    
    def __init__(self, vector_service: ChromaDBVectorService, n_results: int = 20) -> None:
        """Initialize CustomBM25Retriever.
        
        Args:
            vector_service: ChromaDB vector service instance
            n_results: Number of results to retrieve from BM25 search
            
        Raises:
            ValueError: If n_results is not positive
        """
        if n_results <= 0:
            raise ValueError("n_results must be positive")
            
        super().__init__(vector_service=vector_service, n_results=n_results)
        logger.debug(f"CustomBM25Retriever initialized with n_results={n_results}")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using BM25 search.
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant Document objects
            
        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to BM25 retriever")
            return []
            
        try:
            logger.debug(f"Performing BM25 search for query: '{query[:50]}...'")
            results = self.vector_service.bm25_search(query, n_results=self.n_results)
            
            documents = [
                Document(
                    page_content=result['document'],
                    metadata=result['metadata']
                )
                for result in results
            ]
            
            logger.debug(f"BM25 search returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error in BM25 retrieval: {e}")
            return []



# =============================================================================
# MAIN RRF RETRIEVER CLASS
# =============================================================================

class RRFRetriever:
    """
    RAG Retriever using LangChain EnsembleRetriever for RRF to combine vector and BM25 search results.
    
    This class provides a comprehensive retrieval service that combines multiple
    search methods using Reciprocal Rank Fusion (RRF) for optimal results.
    
    Features:
    - LangChain EnsembleRetriever with RRF algorithm
    - Configurable retrieval parameters (weights, result counts)
    - Integration with ChromaDB vector service
    - Support for metadata filtering
    - Automatic fallback strategies
    - Comprehensive error handling and logging
    
    Attributes:
        vector_service: ChromaDB vector service instance
        default_top_k: Default number of results to return
        vector_weight: Weight for vector search results
        bm25_weight: Weight for BM25 search results
        vector_retriever: LangChain vector retriever
        bm25_retriever: Custom BM25 retriever
        ensemble_retriever: LangChain ensemble retriever with RRF
    """
    
    def __init__(
        self,
        vector_service: ChromaDBVectorService,
        default_top_k: int = DEFAULT_TOP_K,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        search_multiplier: int = DEFAULT_SEARCH_MULTIPLIER
    ) -> None:
        """
        Initialize RRF Retriever using LangChain EnsembleRetriever.
        
        Args:
            vector_service: ChromaDB vector service instance
            default_top_k: Default number of results to return
            vector_weight: Weight for vector search results in ensemble
            bm25_weight: Weight for BM25 search results in ensemble
            search_multiplier: Multiplier for initial search results before RRF
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        self._validate_init_params(default_top_k, vector_weight, bm25_weight, search_multiplier)
        
        # Store configuration
        self.vector_service = vector_service
        self.default_top_k = default_top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.search_multiplier = search_multiplier
        
        # Initialize retrievers
        self._initialize_retrievers()
        
        logger.info(
            f"RRF Retriever initialized: top_k={default_top_k}, "
            f"weights=[vector={vector_weight}, bm25={bm25_weight}], "
            f"search_multiplier={search_multiplier}"
        )
    
    def _validate_init_params(
        self, 
        default_top_k: int, 
        vector_weight: float, 
        bm25_weight: float, 
        search_multiplier: int
    ) -> None:
        """Validate initialization parameters.
        
        Args:
            default_top_k: Default number of results
            vector_weight: Vector search weight
            bm25_weight: BM25 search weight
            search_multiplier: Search multiplier
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if default_top_k <= 0:
            raise ValueError("default_top_k must be positive")
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("Weights must be non-negative")
        if vector_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one weight must be positive")
        if search_multiplier <= 0:
            raise ValueError("search_multiplier must be positive")
    
    def _initialize_retrievers(self) -> None:
        """Initialize individual and ensemble retrievers."""
        # Create vector retriever with expanded search
        search_k = self.default_top_k * self.search_multiplier
        self.vector_retriever = self.vector_service.vectorstore.as_retriever(
            search_kwargs={"k": search_k}
        )
        
        # Create BM25 retriever with expanded search
        self.bm25_retriever = CustomBM25Retriever(
            self.vector_service, 
            n_results=search_k
        )
        
        # Create ensemble retriever with RRF
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[self.vector_weight, self.bm25_weight]
        )
        
        logger.debug("Individual and ensemble retrievers initialized successfully")
    
    # =============================================================================
    # CORE RETRIEVAL METHODS
    # =============================================================================
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """        
        Retrieve documents using EnsembleRetriever with RRF.
        
        This method always uses both vector similarity search and BM25 keyword search
        combined with Reciprocal Rank Fusion (RRF) for optimal retrieval performance.
        
        Args:
            query: Search query
            top_k: Final number of results to return (default: self.default_top_k)
            
        Returns:
            List of retrieved LangChain Document objects
            
        Raises:
            ValueError: If query is empty or top_k is invalid
        """
        # Validate inputs
        self._validate_query(query)
        top_k = self._validate_and_set_top_k(top_k)
        
        # Log retrieval attempt
        query_preview = self._create_query_preview(query)
        logger.info(f"Retrieving documents for query: '{query_preview}'")
        logger.info(f"Parameters: top_k={top_k}, method=ensemble_rrf")
        
        try:
            # Always use ensemble retrieval with RRF (vector + BM25)
            documents = self._execute_retrieval_strategy(query)
            
            # Apply top_k limit and return
            final_documents = documents[:top_k]
            logger.info(f"Returning {len(final_documents)} documents (limited from {len(documents)})")
            return final_documents
                
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def _validate_query(self, query: str) -> None:
        """Validate query input.
        
        Args:
            query: Query string to validate
            
        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
    
    def _validate_and_set_top_k(self, top_k: Optional[int]) -> int:
        """Validate and set top_k value.
        
        Args:
            top_k: Number of results to return
            
        Returns:
            Validated top_k value
            
        Raises:
            ValueError: If top_k is invalid
        """
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        return top_k if top_k is not None else self.default_top_k
    
    def _create_query_preview(self, query: str) -> str:
        """Create a preview of the query for logging.
        
        Args:
            query: Full query string
            
        Returns:
            Truncated query string for logging
        """
        if len(query) <= MAX_QUERY_PREVIEW_LENGTH:
            return query
        return query[:MAX_QUERY_PREVIEW_LENGTH] + '...'
    
    def _execute_retrieval_strategy(self, query: str) -> List[Document]:
        """Execute RRF retrieval strategy using both vector and BM25 search.
        
        This method always uses the EnsembleRetriever which combines:
        - Vector similarity search for semantic matching
        - BM25 keyword search for lexical matching
        - Reciprocal Rank Fusion (RRF) for optimal result ranking
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved documents ranked by RRF
        """
        # Always use EnsembleRetriever with RRF (vector + BM25)
        documents = self.ensemble_retriever.get_relevant_documents(query)
        logger.info(f"EnsembleRetriever (RRF) returned {len(documents)} results")
        return documents
    
    # =============================================================================
    # ENHANCED RETRIEVAL METHODS
    # =============================================================================
    
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
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Tuple of (documents, scoring_info)
        """
        documents = self.retrieve(query, top_k)
        
        # Calculate comprehensive scoring statistics
        scoring_info = {
            'total_results': len(documents),
            'requested_top_k': top_k or self.default_top_k,
            'vector_weight': self.vector_weight,
            'bm25_weight': self.bm25_weight,
            'search_multiplier': self.search_multiplier,
            'ensemble_enabled': True,  # Always enabled
            'retrieval_method': 'ensemble_rrf',  # Always RRF
            'vector_search_enabled': True,  # Always enabled
            'bm25_search_enabled': True     # Always enabled
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
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            List of LangChain Document objects
            
        Note:
            This is an alias method for retrieve() to maintain API compatibility.
            Always uses ensemble RRF retrieval (vector + BM25).
        """
        return self.retrieve(query, top_k)
    
    # =============================================================================
    # UTILITY AND INFORMATION METHODS
    # =============================================================================
    
    def get_retriever_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the retriever configuration.
        
        Returns:
            Dictionary containing retriever configuration details
        """
        return {
            'default_top_k': self.default_top_k,
            'vector_weight': self.vector_weight,
            'bm25_weight': self.bm25_weight,
            'search_multiplier': self.search_multiplier,
            'retrieval_method': 'ensemble_rrf',
            'vector_search_enabled': True,
            'bm25_search_enabled': True,
            'ensemble_enabled': True,
            'retrievers': {
                'vector': {
                    'type': 'VectorStoreRetriever',
                    'search_k': self.default_top_k * self.search_multiplier,
                    'enabled': True
                },
                'bm25': {
                    'type': 'CustomBM25Retriever',
                    'n_results': self.default_top_k * self.search_multiplier,
                    'enabled': True
                },
                'ensemble': {
                    'type': 'EnsembleRetriever',
                    'algorithm': 'RRF',
                    'weights': [self.vector_weight, self.bm25_weight],
                    'enabled': True
                }
            }
        }
    
    def update_weights(self, vector_weight: float, bm25_weight: float) -> None:
        """Update the weights for vector and BM25 retrievers.
        
        Args:
            vector_weight: New weight for vector search
            bm25_weight: New weight for BM25 search
            
        Raises:
            ValueError: If weights are invalid
        """
        # Validate new weights
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("Weights must be non-negative")
        if vector_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one weight must be positive")
        
        # Update weights
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Recreate ensemble retriever with new weights
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[vector_weight, bm25_weight]
        )
        
        logger.info(f"Updated retriever weights: vector={vector_weight}, bm25={bm25_weight}")
    

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_rrf_retriever(
    vector_service: ChromaDBVectorService,
    **kwargs
) -> RRFRetriever:
    """Create an RRF retriever instance with optional configuration.
    
    Args:
        vector_service: ChromaDB vector service instance
        **kwargs: Additional configuration parameters for RRFRetriever
        
    Returns:
        Configured RRFRetriever instance
    """
    try:
        logger.info("Creating RRF retriever with factory function")
        return RRFRetriever(vector_service=vector_service, **kwargs)
    except Exception as e:
        logger.error(f"Error creating RRF retriever: {e}")
        raise


def get_default_retriever_config() -> Dict[str, Any]:
    """Get default configuration for RRF retriever.
    
    Returns:
        Dictionary with default configuration values
    """
    return {
        'default_top_k': DEFAULT_TOP_K,
        'vector_weight': DEFAULT_VECTOR_WEIGHT,
        'bm25_weight': DEFAULT_BM25_WEIGHT,
        'search_multiplier': DEFAULT_SEARCH_MULTIPLIER
    }


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def _test_retriever() -> None:
    """Comprehensive testing function for RRF Retriever.
    
    This function demonstrates:
    - Service initialization
    - Document retrieval with different methods
    - Statistics reporting
    - Configuration management
    """
    print("Testing RRF Retriever with EnsembleRetriever...")
    
    try:
        # Test factory function
        print("\n1. Testing factory function...")
        vector_service = ChromaDBVectorService()
        retriever = create_rrf_retriever(vector_service)
        print(f"   ✅ RRF retriever created successfully")
        
        # Test retriever info
        print("\n2. Testing retriever information...")
        info = retriever.get_retriever_info()
        print(f"   ✅ Retriever info: {info['ensemble_enabled']} ensemble, {info['default_top_k']} default top_k")
        
        # Test retrieval with different strategies
        print("\n3. Testing RRF ensemble retrieval...")
        test_query = "machine learning"
        
        # Test RRF ensemble retrieval
        try:
            documents, scores = retriever.retrieve_with_scores(test_query, top_k=3)
            print(f"   ✅ RRF ensemble retrieval: {len(documents)} documents, method: {scores['retrieval_method']}")
            print(f"      Vector enabled: {scores['vector_search_enabled']}, BM25 enabled: {scores['bm25_search_enabled']}")
            
            if documents:
                for i, doc in enumerate(documents, 1):
                    content_preview = doc.page_content[:100] + '...' if len(doc.page_content) > 100 else doc.page_content
                    print(f"      Result {i}: {content_preview}")
            else:
                print("      No results found. Make sure documents are added to the vector service.")
                
        except ValueError as e:
            print(f"   ⚠️ Validation error: {e}")
        except Exception as e:
            print(f"   ❌ Retrieval error: {e}")
        
        # Test weight updates
        print("\n4. Testing weight updates...")
        try:
            retriever.update_weights(0.7, 0.3)
            updated_info = retriever.get_retriever_info()
            print(f"   ✅ Weights updated: vector={updated_info['vector_weight']}, bm25={updated_info['bm25_weight']}")
        except Exception as e:
            print(f"   ❌ Weight update error: {e}")
        
        print("\n✅ All RRF retriever tests completed!")
        
    except Exception as e:
        logger.error(f"Error testing RRF retriever: {e}")
        print(f"❌ Test failed: {e}")



if __name__ == "__main__":
    _test_retriever()