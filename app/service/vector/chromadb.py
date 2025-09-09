"""
Simplified ChromaDB Vector Service using LangChain Chroma integration.

This module provides a streamlined interface for:
- Storing document chunks with vector embeddings
- Retrieving similar documents using semantic search
- Managing collections and metadata
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from ..rag.embedding import create_embedding_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDBVectorService:
    """
    Simplified ChromaDB-based vector service using LangChain Chroma.
    
    Features:
    - Automatic embedding generation using custom embedding service
    - Persistent storage with configurable database path
    - Collection management with metadata filtering
    - Simplified API using LangChain abstractions
    """
    
    DEFAULT_COLLECTION = "documents"
    
    def __init__(
        self,
        db_path: str = "./data/chroma_db",
        embedding_provider: str = 'qwen',
        embedding_model: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        enable_bm25: bool = False
    ):
        """
        Initialize ChromaDB vector service using custom embedding service.
        
        Args:
            db_path: Path to ChromaDB database directory
            embedding_provider: Embedding provider ('qwen' or 'openai')
            embedding_model: Embedding model name (uses provider default if None)
            collection_name: Name of the ChromaDB collection
            enable_bm25: Whether to enable BM25 search functionality
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.enable_bm25 = enable_bm25
        
        # Ensure database directory exists
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize custom embedding service
        logger.info(f"Loading embedding service: {embedding_provider}")
        self.embedding_service = create_embedding_service(
            provider=embedding_provider,
            model_name=embedding_model
        )
        
        # Initialize Chroma vector store
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_service,
            persist_directory=str(self.db_path)
        )
        
        logger.info(f"ChromaDB service initialized with {embedding_provider} embeddings, collection: {collection_name}")
    
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of document IDs that were added
        """
        if not documents:
            return []
        
        logger.info(f"Adding {len(documents)} documents to collection")
        
        try:
            ids = self.vectorstore.add_documents(documents)
            logger.info(f"Successfully added {len(documents)} documents")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Metadata filter conditions
            
        Returns:
            List of search results with documents and metadata
        """
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=n_results,
                filter=where
            )
            
            results = []
            for doc, score in docs_with_scores:
                result = {
                    'document': doc.page_content,
                    'metadata': doc.metadata,
                    'id': doc.metadata.get('id', ''),
                    'similarity': 1 - score
                }
                results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the collection."""
        try:
            all_docs = self.vectorstore.get()
            count = len(all_docs['ids']) if all_docs['ids'] else 0
            
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'db_path': str(self.db_path)
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not ids:
            return True
            
        try:
            self.vectorstore.delete(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            all_docs = self.vectorstore.get()
            if all_docs['ids']:
                self.vectorstore.delete(ids=all_docs['ids'])
            
            logger.info(f"Successfully cleared collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def bm25_search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 search for documents (fallback to similarity search).
        
        Note: This is a placeholder implementation that falls back to similarity search
        since ChromaDB doesn't have native BM25 support in this configuration.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Metadata filter conditions
            
        Returns:
            List of search results with documents and metadata
        """
        logger.warning("BM25 search not implemented, falling back to similarity search")
        return self.similarity_search(query, n_results, where)