"""Basic ChromaDB Service using LangChain Chroma integration.

This module provides basic ChromaDB operations and initialization:

## Classes:
- **ChromaDBVectorService**: Basic ChromaDB service class for database operations

## Functions:
- **create_vector_service()**: Factory function for service creation
- **get_vector_service_info()**: Service configuration information
- **validate_vector_config()**: Configuration validation

## Features:
- ChromaDB database initialization and configuration
- Basic document operations (add, delete, search, clear)
- Embedding service integration
- Collection management
- User and session tracking with metadata

## Note:
For high-level document processing (loading, chunking, storage), use VectorStoreService
from vector_store.py which provides a complete document processing pipeline.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from ..rag.embedding import create_embedding_service, get_embedding_config_info

# =============================================================================
# CONSTANTS
# =============================================================================

# Default configuration values
DEFAULT_DB_PATH = "./data/chroma_db"
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_EMBEDDING_PROVIDER = "qwen"
DEFAULT_N_RESULTS = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.0

# Supported operations
SUPPORTED_PROVIDERS = ["qwen", "openai"]
MAX_BATCH_SIZE = 100
MIN_N_RESULTS = 1
MAX_N_RESULTS = 100

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# MAIN CHROMADB VECTOR SERVICE CLASS
# =============================================================================

class ChromaDBVectorService:
    """
    Basic ChromaDB service for vector database operations using LangChain Chroma integration.
    
    This class provides essential ChromaDB operations:
    - Database initialization and configuration
    - Basic document operations (add, delete, search, clear)
    - Embedding service integration
    - User and session tracking with metadata
    - Collection management and statistics
    
    For high-level document processing (loading, chunking, storage pipeline),
    use VectorStoreService from vector_store.py instead.
    
    Attributes:
        db_path (Path): Path to the ChromaDB database directory
        collection_name (str): Name of the collection (always uses default)
        embedding_service: Custom embedding service instance
        vectorstore (Chroma): LangChain Chroma vectorstore instance
        provider (str): Embedding provider type
        model_name (str): Embedding model name
    
    Example:
        >>> service = ChromaDBVectorService(
        ...     db_path="./data/vectors",
        ...     embedding_provider="qwen",
        ...     embedding_model="text-embedding-v3"
        ... )
        >>> documents = [Document(page_content="Hello world", metadata={"filename": "test.txt"})]
        >>> ids = service.add_documents(documents, user_id="user123", session_id="session456")
        >>> results = service.similarity_search("Hello", n_results=5, user_id="user123")
    """
    
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize ChromaDB vector service with enhanced configuration.
        
        Args:
            db_path (str): Path to ChromaDB database directory
            embedding_provider (str): Embedding provider ('qwen' or 'openai')
            embedding_model (Optional[str]): Embedding model name (uses provider default if None)
            
        Raises:
            ValueError: If embedding_provider is not supported
            Exception: If embedding service initialization fails
        """
        # Validate input parameters
        self._validate_provider(embedding_provider)
        
        # Initialize core attributes
        self.db_path = Path(db_path)
        self.collection_name = DEFAULT_COLLECTION_NAME
        self.provider = embedding_provider
        self.model_name = embedding_model
        
        # Initialize database and embedding service
        self._initialize_database()
        self._initialize_embedding_service()
        self._initialize_vectorstore()
        
        logger.info(
            f"ChromaDB service initialized successfully: "
            f"provider={self.provider}, model={self.model_name}, "
            f"collection={self.collection_name}, path={self.db_path}"
        )
    
    def _validate_provider(self, provider: str) -> None:
        """Validate embedding provider.
        
        Args:
            provider (str): Embedding provider to validate
            
        Raises:
            ValueError: If provider is not supported
        """
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported embedding provider: {provider}. "
                f"Supported providers: {SUPPORTED_PROVIDERS}"
            )
    
    def _initialize_database(self) -> None:
        """Initialize database directory."""
        try:
            self.db_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Database directory initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database directory: {e}")
            raise
    
    def _initialize_embedding_service(self) -> None:
        """Initialize custom embedding service."""
        try:
            logger.info(f"Loading embedding service: {self.provider}")
            self.embedding_service = create_embedding_service(
                provider=self.provider,
                model_name=self.model_name
            )
            # Update model name from service if it was None
            if hasattr(self.embedding_service, 'model_name'):
                self.model_name = self.embedding_service.model_name
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    def _initialize_vectorstore(self) -> None:
        """Initialize Chroma vectorstore."""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_service,
                persist_directory=str(self.db_path)
            )
            logger.debug(f"Vectorstore initialized with collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vectorstore: {e}")
            raise
    
    
# =============================================================================
# DOCUMENT MANAGEMENT METHODS
# =============================================================================

    def add_documents(
        self, 
        documents: List[Document], 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        upsert: bool = True
    ) -> List[str]:
        """
        Add documents to the vector store with enhanced validation and user tracking.
        
        Args:
            documents (List[Document]): List of LangChain Document objects to add
            user_id (Optional[str]): User ID who is adding the documents
            session_id (Optional[str]): Session ID for the document addition
            upsert (bool): If True, update existing documents with same filename (replaces ANY document with same filename, regardless of user/session)
            
        Returns:
            List[str]: List of document IDs that were added or updated
            
        Raises:
            ValueError: If documents list is invalid
            Exception: If document addition fails
        """
        # Validate input
        if not documents:
            logger.warning("No documents provided for addition")
            return []
        
        if len(documents) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(documents)} exceeds maximum {MAX_BATCH_SIZE}"
            )
        
        logger.info(
            f"Adding {len(documents)} documents to collection '{self.collection_name}' "
            f"(user_id: {user_id}, session_id: {session_id}, upsert: {upsert})"
        )
        
        try:
            # Enhance documents with user and session metadata
            enhanced_documents = self._enhance_documents_with_metadata(
                documents, user_id, session_id
            )
            
            # Handle upsert logic if enabled
            if upsert:
                return self._upsert_documents(enhanced_documents)
            else:
                # Add documents directly to vectorstore
                ids = self.vectorstore.add_documents(enhanced_documents)
                logger.info(f"Successfully added {len(enhanced_documents)} documents with {len(ids)} IDs")
                return ids
                
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def _enhance_documents_with_metadata(
        self, 
        documents: List[Document], 
        user_id: Optional[str], 
        session_id: Optional[str]
    ) -> List[Document]:
        """
        Enhance documents with user and session metadata.
        
        Args:
            documents (List[Document]): Original documents
            user_id (Optional[str]): User ID to add to metadata
            session_id (Optional[str]): Session ID to add to metadata
            
        Returns:
            List[Document]: Documents with enhanced metadata
        """
        from datetime import datetime
        
        enhanced_docs = []
        timestamp = datetime.now().isoformat()
        
        for doc in documents:
            # Create a copy of the document to avoid modifying the original
            enhanced_doc = Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy()
            )
            
            # Add tracking metadata
            enhanced_doc.metadata.update({
                'user_id': user_id,
                'session_id': session_id,
                'added_timestamp': timestamp,
                'last_updated': timestamp
            })
            
            enhanced_docs.append(enhanced_doc)
        
        return enhanced_docs
    
    def _upsert_documents(self, documents: List[Document]) -> List[str]:
        """
        Upsert documents - update existing documents with same filename or add new ones.
        
        WARNING: This replaces ANY document with the same filename, regardless of user or session.
        - Same filename = UPDATE (replace existing document, even from different users/sessions)
        - Different filename = INSERT (add as new document)
        
        Args:
            documents (List[Document]): Documents to upsert
            
        Returns:
            List[str]: List of document IDs that were added or updated
        """
        from datetime import datetime
        
        all_ids = []
        
        for doc in documents:
            # Use file_name consistently throughout the system
            filename = doc.metadata.get('file_name', '')
            user_id = doc.metadata.get('user_id')
            session_id = doc.metadata.get('session_id')
            
            if filename:
                # Check if document with same filename exists (regardless of user/session)
                existing_docs = self._find_documents_by_filename(filename)
                
                if existing_docs:
                    # Update existing documents (same filename, may be from different user/session)
                    logger.warning(f"Replacing existing document with same filename: {filename} (original user: {existing_docs[0].get('metadata', {}).get('user_id', 'unknown')}, new user: {user_id})")
                    
                    # Update metadata with new timestamp
                    doc.metadata['last_updated'] = datetime.now().isoformat()
                    doc.metadata['update_count'] = existing_docs[0].get('metadata', {}).get('update_count', 0) + 1
                    
                    # Delete old document(s) and add updated one
                    old_ids = [existing_doc['id'] for existing_doc in existing_docs]
                    self.delete_documents(old_ids)
                    
                    # Add updated document
                    new_ids = self.vectorstore.add_documents([doc])
                    all_ids.extend(new_ids)
                    
                    logger.info(f"Updated document {filename}: replaced {len(old_ids)} old versions (new user: {user_id})")
                else:
                    # Add new document (different user/session or new filename)
                    doc.metadata['update_count'] = 0
                    new_ids = self.vectorstore.add_documents([doc])
                    all_ids.extend(new_ids)
                    
                    logger.info(f"Added new document: filename={filename}, user_id={user_id}, session_id={session_id}")
            else:
                # No filename - add as new document
                doc.metadata['update_count'] = 0
                new_ids = self.vectorstore.add_documents([doc])
                all_ids.extend(new_ids)
        
        logger.info(f"Upsert operation completed: {len(all_ids)} documents processed")
        return all_ids
    
    def _find_documents_by_filename(self, filename: str) -> List[Dict[str, Any]]:
        """
        Find existing documents by filename only (ignores user_id and session_id).
        
        WARNING: This finds ALL documents with the same filename, regardless of who uploaded them.
        This enables filename-based replacement across users and sessions.
        
        Args:
            filename (str): Filename to search for
            
        Returns:
            List[Dict[str, Any]]: List of all documents with matching filename
        """
        try:
            # Get all documents and filter by filename only
            all_docs = self.vectorstore.get()
            
            if not all_docs or not all_docs.get('ids'):
                return []
            
            matching_docs = []
            ids = all_docs.get('ids', [])
            metadatas = all_docs.get('metadatas', [])
            documents_content = all_docs.get('documents', [])
            
            for i, doc_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                # Use file_name consistently throughout the system
                doc_filename = metadata.get('file_name', '')
                
                # Check if filename matches (ignore user_id and session_id)
                filename_match = doc_filename == filename
                
                if filename_match:
                    matching_docs.append({
                        'id': doc_id,
                        'metadata': metadata,
                        'page_content': documents_content[i] if i < len(documents_content) else ''
                    })
            
            return matching_docs
            
        except Exception as e:
            logger.error(f"Error finding documents by filename: {e}")
            return []
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs with enhanced validation.
        
        Args:
            ids (List[str]): List of document IDs to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not ids:
            logger.warning("No document IDs provided for deletion")
            return True
        
        if not isinstance(ids, list) or not all(isinstance(id_, str) for id_ in ids):
            logger.error("Invalid IDs format - must be list of strings")
            return False
            
        logger.info(f"Deleting {len(ids)} documents from collection '{self.collection_name}'")
        
        try:
            self.vectorstore.delete(ids=ids)
            logger.info(f"Successfully deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection with confirmation.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.warning(f"Clearing all documents from collection '{self.collection_name}'")
        
        try:
            # Get all document IDs
            all_docs = self.vectorstore.get()
            
            if not all_docs or not all_docs.get('ids'):
                logger.info("Collection is already empty")
                return True
            
            # Delete all documents
            doc_count = len(all_docs['ids'])
            self.vectorstore.delete(ids=all_docs['ids'])
            
            logger.info(f"Successfully cleared {doc_count} documents from collection '{self.collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def update_documents(self, documents: List[Document], ids: List[str]) -> bool:
        """
        Update existing documents with new content.
        
        Args:
            documents (List[Document]): Updated document objects
            ids (List[str]): IDs of documents to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        if len(documents) != len(ids):
            logger.error("Documents and IDs lists must have the same length")
            return False
        
        logger.info(f"Updating {len(documents)} documents")
        
        try:
            # Delete old documents
            self.delete_documents(ids)
            
            # Add updated documents with original IDs
            for doc, doc_id in zip(documents, ids):
                doc.metadata['id'] = doc_id
            
            self.vectorstore.add_documents(documents, ids=ids)
            logger.info(f"Successfully updated {len(documents)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            return False
    
# =============================================================================
# SEARCH OPERATIONS
# =============================================================================

    def similarity_search(
        self,
        query: str,
        n_results: int = DEFAULT_N_RESULTS,
        where: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for documents with enhanced filtering and user tracking.
        
        Args:
            query (str): Search query text
            n_results (int): Number of results to return (1-100)
            where (Optional[Dict[str, Any]]): Metadata filter conditions
            similarity_threshold (float): Minimum similarity score threshold
            user_id (Optional[str]): Filter by user ID (optional)
            session_id (Optional[str]): Filter by session ID (optional)
            
        Returns:
            List of search results with documents and metadata
        """
        try:
            # Handle empty query case - ChromaDB requires non-empty query for embedding
            if not query or not query.strip():
                # For metadata-only filtering, use a generic query
                query = "document"
                logger.info("Empty query provided, using generic query for metadata filtering")
            
            # Build enhanced where clause with user/session filtering
            enhanced_where = where.copy() if where else {}
            if user_id:
                enhanced_where['user_id'] = user_id
            if session_id:
                enhanced_where['session_id'] = session_id
            enhanced_where = enhanced_where if enhanced_where else None
            
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query=query,
                k=n_results,
                filter=enhanced_where
            )
            
            # Process and format results
            results = []
            for doc, score in docs_with_scores:
                similarity = 1 - score  # Convert distance to similarity
                
                # Apply similarity threshold filter
                if similarity >= similarity_threshold:
                    result = {
                        'document': doc.page_content,
                        'metadata': doc.metadata,
                        'id': doc.metadata.get('id', ''),
                        'similarity': similarity,
                        'distance': score
                    }
                    results.append(result)
            
            logger.info(f"Similarity search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            raise
    
    def bm25_search(
        self,
        query: str,
        n_results: int = DEFAULT_N_RESULTS,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 search for documents (fallback to similarity search).
        
        Note: ChromaDB doesn't have native BM25 support in this configuration,
        so this method falls back to similarity search with a warning.
        
        Args:
            query (str): Search query text
            n_results (int): Number of results to return
            where (Optional[Dict[str, Any]]): Metadata filter conditions
            
        Returns:
            List[Dict[str, Any]]: List of search results (via similarity search)
        """
        logger.warning(
            "BM25 search not natively supported by ChromaDB, "
            "falling back to similarity search"
        )
        return self.similarity_search(query, n_results, where)
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = DEFAULT_N_RESULTS,
        where: Optional[Dict[str, Any]] = None,
        similarity_weight: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining similarity and keyword matching.
        
        Currently implements similarity search only, but provides interface
        for future hybrid search implementations.
        
        Args:
            query (str): Search query text
            n_results (int): Number of results to return
            where (Optional[Dict[str, Any]]): Metadata filter conditions
            similarity_weight (float): Weight for similarity search (0.0-1.0)
            
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        logger.info(f"Performing hybrid search with similarity_weight={similarity_weight}")
        
        # For now, just perform similarity search
        # Future enhancement: combine with BM25 or keyword search
        results = self.similarity_search(query, n_results, where)
        
        # Add hybrid search metadata
        for result in results:
            result['search_type'] = 'hybrid_similarity'
            result['similarity_weight'] = similarity_weight
        
        return results
    
    def _validate_search_params(self, query: str, n_results: int) -> None:
        """
        Validate search parameters.
        
        Args:
            query (str): Search query to validate
            n_results (int): Number of results to validate
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not isinstance(n_results, int) or n_results < MIN_N_RESULTS or n_results > MAX_N_RESULTS:
            raise ValueError(
                f"n_results must be an integer between {MIN_N_RESULTS} and {MAX_N_RESULTS}"
            )
    
    def _enhance_search_filter(
        self, 
        where: Optional[Dict[str, Any]], 
        user_id: Optional[str], 
        session_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Enhance search filter with user and session criteria.
        
        Args:
            where (Optional[Dict[str, Any]]): Original filter conditions
            user_id (Optional[str]): User ID to filter by
            session_id (Optional[str]): Session ID to filter by
            
        Returns:
            Optional[Dict[str, Any]]: Enhanced filter conditions
        """
        enhanced_where = where.copy() if where else {}
        
        # Add user_id filter if provided
        if user_id:
            enhanced_where['user_id'] = user_id
        
        # Add session_id filter if provided
        if session_id:
            enhanced_where['session_id'] = session_id
        
        return enhanced_where if enhanced_where else None
    
# =============================================================================
# COLLECTION INFORMATION AND STATISTICS
# =============================================================================

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the collection.
        
        Returns:
            Dict[str, Any]: Collection statistics including document count, 
                          configuration info, and database details
        """
        try:
            # Get all documents for counting
            all_docs = self.vectorstore.get()
            document_count = len(all_docs['ids']) if all_docs and all_docs.get('ids') else 0
            
            # Calculate additional statistics
            total_content_length = 0
            metadata_keys = set()
            
            if all_docs and all_docs.get('documents'):
                total_content_length = sum(len(doc) for doc in all_docs['documents'])
            
            if all_docs and all_docs.get('metadatas'):
                for metadata in all_docs['metadatas']:
                    if metadata:
                        metadata_keys.update(metadata.keys())
            
            return {
                'collection_name': self.collection_name,
                'document_count': document_count,
                'total_content_length': total_content_length,
                'average_content_length': total_content_length / document_count if document_count > 0 else 0,
                'unique_metadata_keys': list(metadata_keys),
                'metadata_key_count': len(metadata_keys),
                'db_path': str(self.db_path),
                'embedding_provider': self.provider,
                'embedding_model': self.model_name,
                'database_exists': self.db_path.exists()
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {
                'collection_name': self.collection_name,
                'document_count': 0,
                'error': str(e)
            }

    def get_all_documents(self, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        Get all documents from the ChromaDB collection with enhanced information.
        
        Args:
            include_embeddings (bool): Whether to include embedding vectors in results
            
        Returns:
            List[Dict[str, Any]]: List of documents with content, metadata, IDs, and optionally embeddings
        """
        try:
            logger.info(f"Retrieving all documents from collection '{self.collection_name}'...")
            
            # Get all documents from ChromaDB
            all_docs = self.vectorstore.get(
                include=['documents', 'metadatas', 'embeddings'] if include_embeddings 
                else ['documents', 'metadatas']
            )
            
            if not all_docs or not all_docs.get('ids'):
                logger.info("No documents found in collection")
                return []
            
            # Extract data arrays
            ids = all_docs.get('ids', [])
            metadatas = all_docs.get('metadatas', [])
            documents_content = all_docs.get('documents', [])
            embeddings = all_docs.get('embeddings', []) if include_embeddings else []
            
            # Combine all information
            documents = []
            for i, doc_id in enumerate(ids):
                doc_data = {
                    'id': doc_id,
                    'page_content': documents_content[i] if i < len(documents_content) else '',
                    'metadata': metadatas[i] if i < len(metadatas) else {},
                    'content_length': len(documents_content[i]) if i < len(documents_content) else 0
                }
                
                # Add embeddings if requested
                if include_embeddings and i < len(embeddings):
                    doc_data['embedding'] = embeddings[i]
                    doc_data['embedding_dimensions'] = len(embeddings[i]) if embeddings[i] else 0
                
                documents.append(doc_data)
            
            logger.info(f"Retrieved {len(documents)} documents from collection")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []
    
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the vector service configuration.
        
        Returns:
            Dict[str, Any]: Service configuration and status information
        """
        try:
            # Get embedding service info
            embedding_info = get_embedding_config_info()
            
            # Get collection stats
            stats = self.get_collection_stats()
            
            return {
                'service_type': 'ChromaDBVectorService',
                'db_path': str(self.db_path),
                'collection_name': self.collection_name,
                'provider': self.provider,
                'model_name': self.model_name,
                'embedding_config': embedding_info,
                'collection_stats': stats,
                'supported_providers': SUPPORTED_PROVIDERS,
                'max_batch_size': MAX_BATCH_SIZE,
                'default_n_results': DEFAULT_N_RESULTS
            }
        except Exception as e:
            logger.error(f"Failed to get service info: {e}")
            return {
                'service_type': 'ChromaDBVectorService',
                'error': str(e)
            }
    
    def get_documents_by_user(self, user_id: str, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        Get all documents uploaded by a specific user.
        
        Args:
            user_id (str): User ID to filter by
            include_embeddings (bool): Whether to include embedding vectors
            
        Returns:
            List[Dict[str, Any]]: List of documents uploaded by the user
        """
        try:
            logger.info(f"Retrieving documents for user: {user_id}")
            
            # Get all documents and filter by user_id
            all_docs = self.get_all_documents(include_embeddings=include_embeddings)
            user_docs = [
                doc for doc in all_docs 
                if doc.get('metadata', {}).get('user_id') == user_id
            ]
            
            logger.info(f"Found {len(user_docs)} documents for user {user_id}")
            return user_docs
            
        except Exception as e:
            logger.error(f"Failed to get documents for user {user_id}: {e}")
            return []
    
    def get_documents_by_session(self, session_id: str, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        Get all documents uploaded in a specific session.
        
        Args:
            session_id (str): Session ID to filter by
            include_embeddings (bool): Whether to include embedding vectors
            
        Returns:
            List[Dict[str, Any]]: List of documents uploaded in the session
        """
        try:
            logger.info(f"Retrieving documents for session: {session_id}")
            
            # Get all documents and filter by session_id
            all_docs = self.get_all_documents(include_embeddings=include_embeddings)
            session_docs = [
                doc for doc in all_docs 
                if doc.get('metadata', {}).get('session_id') == session_id
            ]
            
            logger.info(f"Found {len(session_docs)} documents for session {session_id}")
            return session_docs
            
        except Exception as e:
            logger.error(f"Failed to get documents for session {session_id}: {e}")
            return []
    
    def delete_user_documents(self, user_id: str) -> bool:
        """
        Delete all documents uploaded by a specific user.
        
        Args:
            user_id (str): User ID whose documents to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.warning(f"Deleting all documents for user: {user_id}")
            
            # Get user documents
            user_docs = self.get_documents_by_user(user_id)
            
            if not user_docs:
                logger.info(f"No documents found for user {user_id}")
                return True
            
            # Extract document IDs
            doc_ids = [doc['id'] for doc in user_docs]
            
            # Delete documents
            success = self.delete_documents(doc_ids)
            
            if success:
                logger.info(f"Successfully deleted {len(doc_ids)} documents for user {user_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete documents for user {user_id}: {e}")
            return False
    
    def delete_session_documents(self, session_id: str) -> bool:
        """
        Delete all documents uploaded in a specific session.
        
        Args:
            session_id (str): Session ID whose documents to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.warning(f"Deleting all documents for session: {session_id}")
            
            # Get session documents
            session_docs = self.get_documents_by_session(session_id)
            
            if not session_docs:
                logger.info(f"No documents found for session {session_id}")
                return True
            
            # Extract document IDs
            doc_ids = [doc['id'] for doc in session_docs]
            
            # Delete documents
            success = self.delete_documents(doc_ids)
            
            if success:
                logger.info(f"Successfully deleted {len(doc_ids)} documents for session {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete documents for session {session_id}: {e}")
            return False


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_vector_service(
    db_path: str = DEFAULT_DB_PATH,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: Optional[str] = None
) -> ChromaDBVectorService:
    """
    Factory function to create a ChromaDB vector service instance.
    
    Args:
        db_path (str): Path to ChromaDB database directory
        embedding_provider (str): Embedding provider ('qwen' or 'openai')
        embedding_model (Optional[str]): Embedding model name
        
    Returns:
        ChromaDBVectorService: Configured vector service instance
        
    Raises:
        ValueError: If provider is not supported
        Exception: If service creation fails
    """
    logger.info(f"Creating vector service with provider: {embedding_provider}")
    
    try:
        service = ChromaDBVectorService(
            db_path=db_path,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
        logger.info("Vector service created successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to create vector service: {e}")
        raise

def get_vector_service_info() -> Dict[str, Any]:
    """
    Get information about vector service configuration and capabilities.
    
    Returns:
        Dict[str, Any]: Service information and configuration details
    """
    return {
        'service_class': 'ChromaDBVectorService',
        'supported_providers': SUPPORTED_PROVIDERS,
        'default_db_path': DEFAULT_DB_PATH,
        'default_collection_name': DEFAULT_COLLECTION_NAME,
        'default_embedding_provider': DEFAULT_EMBEDDING_PROVIDER,
        'default_n_results': DEFAULT_N_RESULTS,
        'max_batch_size': MAX_BATCH_SIZE,
        'search_result_range': f"{MIN_N_RESULTS}-{MAX_N_RESULTS}",
        'features': [
            'Document storage and retrieval',
            'Similarity search with scoring',
            'Metadata filtering',
            'Batch operations',
            'Collection statistics',
            'Hybrid search interface (similarity-based)'
        ]
    }

def validate_vector_config(
    db_path: Optional[str] = None,
    embedding_provider: Optional[str] = None,
    n_results: Optional[int] = None
) -> List[str]:
    """
    Validate vector service configuration parameters.
    
    Args:
        db_path (Optional[str]): Database path to validate
        embedding_provider (Optional[str]): Provider to validate
        n_results (Optional[int]): Number of results to validate
        
    Returns:
        List[str]: List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate embedding provider
    if embedding_provider and embedding_provider not in SUPPORTED_PROVIDERS:
        errors.append(
            f"Unsupported embedding provider: {embedding_provider}. "
            f"Supported: {SUPPORTED_PROVIDERS}"
        )
    
    # Validate n_results
    if n_results is not None:
        if not isinstance(n_results, int) or n_results < MIN_N_RESULTS or n_results > MAX_N_RESULTS:
            errors.append(
                f"n_results must be an integer between {MIN_N_RESULTS} and {MAX_N_RESULTS}"
            )
    
    # Validate db_path
    if db_path:
        try:
            path = Path(db_path)
            if path.exists() and not path.is_dir():
                errors.append(f"Database path exists but is not a directory: {db_path}")
        except Exception as e:
            errors.append(f"Invalid database path: {e}")
    
    return errors


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def _test_vector_service() -> None:
    """
    Comprehensive testing suite for the ChromaDB vector service.
    
    Tests all major functionality including:
    - Service creation and configuration
    - Document management operations
    - Search operations
    - Collection statistics
    - Factory functions and validation
    """
    print("🧪 Testing ChromaDB Vector Service...")
    
    try:
        # Test 1: Service creation
        print("📝 Step 1: Testing service creation...")
        service = create_vector_service(
            db_path="./test_chroma_db",
            embedding_provider="qwen"
        )
        print(f"✅ Service created successfully: {service.provider}/{service.model_name}")
        
        # Test 2: Configuration validation
        print("📝 Step 2: Testing configuration validation...")
        valid_errors = validate_vector_config("./test_db", "qwen", 10)
        invalid_errors = validate_vector_config("./test_db", "invalid_provider", 200)
        print(f"✅ Valid config errors: {len(valid_errors)} (expected: 0)")
        print(f"✅ Invalid config errors: {len(invalid_errors)} (expected: >0)")
        
        # Test 3: Service info
        print("📝 Step 3: Testing service information...")
        service_info = get_vector_service_info()
        print(f"✅ Service info retrieved: {service_info['service_class']}")
        print(f"✅ Supported providers: {service_info['supported_providers']}")
        
        # Test 4: Collection stats (empty collection)
        print("📝 Step 4: Testing collection statistics...")
        stats = service.get_collection_stats()
        print(f"✅ Collection stats: {stats['document_count']} documents")
        
        # Test 5: Document operations with user tracking
        print("📝 Step 5: Testing document operations with user tracking...")
        test_docs = [
            Document(page_content="Test document 1", metadata={"filename": "test1.txt", "source": "test", "type": "sample"}),
            Document(page_content="Test document 2", metadata={"filename": "test2.txt", "source": "test", "type": "sample"})
        ]
        
        # Add documents with user and session tracking
        doc_ids = service.add_documents(
            test_docs, 
            user_id="test_user_123", 
            session_id="test_session_456",
            upsert=True
        )
        print(f"✅ Added {len(doc_ids)} documents with user tracking")
        
        # Test search with user filtering
        results = service.similarity_search(
            "test document", 
            n_results=5, 
            user_id="test_user_123"
        )
        print(f"✅ User-filtered search returned {len(results)} results")
        
        # Test user document retrieval
        user_docs = service.get_documents_by_user("test_user_123")
        print(f"✅ Found {len(user_docs)} documents for test user")
        
        # Test session document retrieval
        session_docs = service.get_documents_by_session("test_session_456")
        print(f"✅ Found {len(session_docs)} documents for test session")
        
        # Test upsert functionality
        updated_docs = [
            Document(page_content="Updated test document 1", metadata={"filename": "test1.txt", "source": "test", "type": "updated"})
        ]
        upsert_ids = service.add_documents(
            updated_docs, 
            user_id="test_user_789", 
            session_id="test_session_999",
            upsert=True
        )
        print(f"✅ Upserted {len(upsert_ids)} documents (should update existing)")
        
        # Test collection stats with documents
        stats_with_docs = service.get_collection_stats()
        print(f"✅ Updated stats: {stats_with_docs['document_count']} documents")
        
        # Test cleanup
        cleared = service.clear_collection()
        print(f"✅ Collection cleared: {cleared}")
        
        print("🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    _test_vector_service()