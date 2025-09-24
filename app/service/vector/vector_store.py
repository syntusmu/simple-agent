"""Vector Store Service for high-level document processing and storage.

This module provides a complete document processing pipeline:
1. Load documents using document_loader.py (handles all formats including docling preprocessing)
2. Split loaded documents into chunks using document_splitter.py
3. Generate embeddings and store in ChromaDB using chromadb.py

For basic ChromaDB operations, use ChromaDBVectorService from chromadb.py directly.
This service focuses on high-level document workflow management.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime

from langchain.schema import Document

# Import document processing services
from ..document.document_loader import MultiFileLoader
from ..document.document_splitter import DocumentSplitter
from .chromadb import ChromaDBVectorService

# Configure logging
logger = logging.getLogger(__name__)

# Supported file formats (delegated to document_loader)
SUPPORTED_FORMATS = {'.pdf', '.docx', '.doc', '.csv', '.txt', '.md', '.markdown', '.xlsx', '.xls', '.html', '.htm'}


class VectorStoreService:
    """
    High-level vector store service for complete document processing pipeline.
    
    This service orchestrates the complete document workflow:
    1. Document loading and validation
    2. Document chunking and preprocessing 
    3. Storage with user/session tracking via ChromaDB service
    4. Batch processing and analytics
    
    For basic ChromaDB operations, use ChromaDBVectorService directly.
    """
    
    def __init__(
        self,
        db_path: str = "./data/chroma_db",
        embedding_provider: str = "qwen",
        embedding_model: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the vector store service.
        
        Args:
            db_path: Path to ChromaDB database directory
            embedding_provider: Embedding provider ('qwen', 'openai', 'embedding')
            embedding_model: Embedding model name (uses provider default if None)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.collection_name = "documents"  # Always use default collection
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize document splitter
        self.document_splitter = DocumentSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize vector service (ChromaDB + embedding)
        self.vector_service = ChromaDBVectorService(
            db_path=db_path,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model
        )
        
        logger.info(f"VectorStoreService initialized: {embedding_provider} embeddings, collection: {self.collection_name}")
        logger.info(f"Chunk settings: size={chunk_size}, overlap={chunk_overlap}")

    # =============================================================================
    # DOCUMENT LOADING AND VALIDATION METHODS
    # =============================================================================

    def validate_file(self, file_path: str) -> bool:
        """
        Basic file validation - detailed validation is handled by document_loader.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file exists and has supported extension
        """
        try:
            if not os.path.exists(file_path) or not os.path.isfile(file_path):
                return False
            
            file_ext = Path(file_path).suffix.lower()
            return file_ext in SUPPORTED_FORMATS
            
        except Exception:
            return False

    def load_document(self, file_path: str, original_filename: str = None) -> List[Document]:
        """
        Load a document using document_loader (handles all formats and preprocessing).
        
        Args:
            file_path: Path to the document
            original_filename: Original filename to use in metadata (if different from file_path)
            original_filename: Original filename to use in metadata (if different from file_path)
            
        Returns:
            List of Document objects
        """
        try:
            # Basic validation
            if not self.validate_file(file_path):
                raise ValueError(f"Invalid file: {file_path}")
            
            # Use original_filename if provided, otherwise use file_path name
            file_name = original_filename if original_filename else Path(file_path).name
            logger.info(f"Loading document: {file_name}")
            
            # Use MultiFileLoader with docling preprocessing enabled
            loader = MultiFileLoader(
                path=file_path, 
                show_progress=False, 
                use_docling=True
            )
            documents = loader.load_all()
            
            # Add metadata to documents
            timestamp = datetime.now().isoformat()
            file_hash = self._calculate_file_hash(file_path)
            
            for doc in documents:
                doc.metadata.update({
                    'filename': file_name,  # Use 'filename' key for consistency with ChromaDB
                    'file_name': file_name,  # Keep both for backward compatibility
                    'file_path': file_path,
                    'file_type': Path(file_path).suffix.lower(),
                    'upload_timestamp': timestamp,
                    'file_hash': file_hash
                })
            
            logger.info(f"Successfully loaded {len(documents)} document(s) from {file_name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate MD5 hash of a file for deduplication.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 hash string
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return ""

    # =============================================================================
    # DOCUMENT PROCESSING METHODS
    # =============================================================================

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks using document_splitter."""
        try:
            logger.info(f"Chunking {len(documents)} documents...")
            chunked_docs = self.document_splitter.split(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunked_docs):
                chunk.metadata.update({
                    'chunk_index': i,
                    'chunk_size': len(chunk.page_content),
                    'original_source': chunk.metadata.get('source', 'unknown')
                })
            
            logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
            return chunked_docs
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise

    # =============================================================================
    # DOCUMENT STORAGE AND PIPELINE METHODS
    # =============================================================================

    def store_documents(
        self, 
        documents: List[Document], 
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        upsert: bool = True
    ) -> Dict[str, Any]:
        """
        Store documents in ChromaDB with embeddings and user tracking.
        
        Args:
            documents (List[Document]): Documents to store
            user_id (Optional[str]): User ID who is storing the documents
            session_id (Optional[str]): Session ID for the document storage
            upsert (bool): If True, update existing documents with same filename
            
        Returns:
            Dict[str, Any]: Storage result with success status and metadata
        """
        try:
            if not documents:
                return {"success": False, "message": "No documents to store"}
            
            logger.info(f"Storing {len(documents)} documents with user tracking...")
            doc_ids = self.vector_service.add_documents(
                documents, 
                user_id=user_id, 
                session_id=session_id, 
                upsert=upsert
            )
            
            return {
                "success": True,
                "documents_stored": len(documents),
                "document_ids": doc_ids,
                "collection_name": self.collection_name,
                "embedding_provider": self.embedding_provider,
                "user_id": user_id,
                "session_id": session_id,
                "upsert_enabled": upsert,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            return {"success": False, "message": f"Storage error: {str(e)}", "documents_stored": 0}

    def process_file(
        self, 
        file_path: str, 
        chunk_documents: bool = True, 
        original_filename: str = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        upsert: bool = True
    ) -> Dict[str, Any]:
        """
        Complete pipeline: load, chunk, and store a document with user tracking.
        
        Args:
            file_path: Path to the file to process
            chunk_documents: Whether to chunk documents before storage
            original_filename: Original filename to use in metadata (if different from file_path)
            user_id: Optional user ID for tracking document ownership
            session_id: Optional session ID for tracking document sessions
            upsert: Whether to update existing documents with same filename
            
        Returns:
            Dict[str, Any]: Processing result dictionary
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Load document
            documents = self.load_document(file_path, original_filename=original_filename)
            
            # Step 2: Chunk documents if requested
            if chunk_documents:
                documents = self.chunk_documents(documents)
            
            # Step 3: Store documents with embeddings and user tracking
            storage_result = self.store_documents(
                documents, 
                user_id=user_id, 
                session_id=session_id, 
                upsert=upsert
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Use original_filename if provided, otherwise use file_path name
            display_name = original_filename if original_filename else Path(file_path).name
            
            result = {
                "success": storage_result["success"],
                "file_path": file_path,
                "file_name": display_name,
                "documents_loaded": len(documents),
                "documents_stored": storage_result.get("documents_stored", 0),
                "chunks_stored": storage_result.get("documents_stored", 0),
                "collection_name": self.collection_name,
                "user_id": user_id,
                "session_id": session_id,
                "upsert_enabled": upsert,
                "processing_time_seconds": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            if not storage_result["success"]:
                result["error"] = storage_result.get("message", "Unknown error")
            
            logger.info(f"Processed file {display_name} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "success": False,
                "file_path": file_path,
                "file_name": original_filename if original_filename else Path(file_path).name,
                "file_name": original_filename if original_filename else Path(file_path).name,
                "error": str(e),
                "processing_time_seconds": 0,
                "timestamp": datetime.now().isoformat()
            }

    def batch_process_files(
        self, 
        file_paths: List[str], 
        chunk_documents: bool = True,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        upsert: bool = True
    ) -> Dict[str, Any]:
        """
        Process multiple files in batch with user tracking.
        
        Args:
            file_paths (List[str]): List of file paths to process
            chunk_documents (bool): Whether to chunk documents before storage
            user_id (Optional[str]): User ID for batch processing
            session_id (Optional[str]): Session ID for batch processing
            upsert (bool): If True, update existing documents with same filename
            
        Returns:
            Dict[str, Any]: Batch processing results
        """
        try:
            start_time = datetime.now()
            results = []
            successful = failed = 0
            
            logger.info(f"Starting batch processing of {len(file_paths)} files (user: {user_id}, session: {session_id})")
            
            for file_path in file_paths:
                result = self.process_file(
                    file_path, 
                    chunk_documents=chunk_documents,
                    user_id=user_id,
                    session_id=session_id,
                    upsert=upsert
                )
                results.append(result)
                if result["success"]:
                    successful += 1
                else:
                    failed += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": failed == 0,
                "total_files": len(file_paths),
                "successful": successful,
                "failed": failed,
                "results": results,
                "user_id": user_id,
                "session_id": session_id,
                "upsert_enabled": upsert,
                "processing_time_seconds": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {
                "success": False, 
                "error": str(e), 
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }

    # =============================================================================
    # SEARCH AND RETRIEVAL METHODS
    # =============================================================================

    def search_documents(
        self,
        query: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents (delegates to ChromaDB service)."""
        return self.vector_service.similarity_search(
            query, n_results, where, user_id=user_id, session_id=session_id
        )

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get enhanced collection statistics with VectorStore-specific info."""
        try:
            stats = self.vector_service.get_collection_stats()
            stats.update({
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'embedding_provider': self.embedding_provider,
                'supported_formats': sorted(SUPPORTED_FORMATS)
            })
            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}

    # =============================================================================
    # COLLECTION INFORMATION METHODS
    # =============================================================================

    def get_all_documents_info(self) -> Dict[str, Any]:
        """
        Get all document information from ChromaDB collection.
        
        Returns:
            Dictionary containing all documents with their metadata and statistics
        """
        try:
            logger.info("Retrieving all documents from ChromaDB...")
            
            # Get all documents from ChromaDB
            all_docs = self.vector_service.get_all_documents()
            
            # Process documents to extract useful information
            documents_info = []
            file_stats = {}
            total_chunks = 0
            
            for doc in all_docs:
                metadata = doc.get('metadata', {})
                file_name = metadata.get('file_name', 'Unknown')
                file_path = metadata.get('file_path', '')
                file_type = metadata.get('file_type', '')
                upload_timestamp = metadata.get('upload_timestamp', '')
                chunk_index = metadata.get('chunk_index', 0)
                chunk_size = metadata.get('chunk_size', len(doc.get('page_content', '')))
                
                # Track per-file statistics
                if file_name not in file_stats:
                    file_stats[file_name] = {
                        'file_name': file_name,
                        'file_path': file_path,
                        'file_type': file_type,
                        'upload_timestamp': upload_timestamp,
                        'chunk_count': 0,
                        'total_content_size': 0,
                        'first_chunk_preview': ''
                    }
                
                file_stats[file_name]['chunk_count'] += 1
                file_stats[file_name]['total_content_size'] += chunk_size
                total_chunks += 1
                
                # Store first chunk as preview (only for display, not for JSON cache)
                if chunk_index == 0 or not file_stats[file_name]['first_chunk_preview']:
                    content = doc.get('page_content', '')
                    file_stats[file_name]['first_chunk_preview'] = content[:200] + ('...' if len(content) > 200 else '')
                
                # Add individual chunk info (without content preview for JSON cache)
                documents_info.append({
                    'id': doc.get('id', ''),
                    'file_name': file_name,
                    'file_path': file_path,
                    'file_type': file_type,
                    'upload_timestamp': upload_timestamp,
                    'chunk_index': chunk_index,
                    'chunk_size': chunk_size,
                    'metadata': metadata
                })
            
            # Convert file_stats to list and remove preview for JSON cache
            files_summary = []
            for file_stat in file_stats.values():
                # Create a copy without preview for JSON cache
                file_summary = {
                    'file_name': file_stat['file_name'],
                    'file_path': file_stat['file_path'],
                    'file_type': file_stat['file_type'],
                    'upload_timestamp': file_stat['upload_timestamp'],
                    'chunk_count': file_stat['chunk_count'],
                    'total_content_size': file_stat['total_content_size']
                }
                files_summary.append(file_summary)
            
            result = {
                'success': True,
                'total_documents': len(all_docs),
                'total_chunks': total_chunks,
                'unique_files': len(files_summary),
                'files_summary': files_summary,
                'all_chunks': documents_info,
                'collection_name': self.collection_name,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Retrieved {len(all_docs)} documents from {len(files_summary)} unique files")
            return result
            
        except Exception as e:
            logger.error(f"Error getting all documents info: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'total_documents': 0,
                'total_chunks': 0,
                'unique_files': 0,
                'files_summary': [],
                'all_chunks': [],
                'timestamp': datetime.now().isoformat()
            }

    def clear_collection(self) -> bool:
        """Clear all documents from the collection (delegates to ChromaDB service)."""
        return self.vector_service.clear_collection()
    
    # =============================================================================
    # DELEGATED METHODS - Call ChromaDB service directly
    # =============================================================================
    
    def get_documents_by_user(self, user_id: str, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Get all documents uploaded by a specific user (delegates to ChromaDB service)."""
        return self.vector_service.get_documents_by_user(user_id, include_embeddings)
    
    def get_documents_by_session(self, session_id: str, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """Get all documents uploaded in a specific session (delegates to ChromaDB service)."""
        return self.vector_service.get_documents_by_session(session_id, include_embeddings)
    
    def delete_user_documents(self, user_id: str) -> bool:
        """Delete all documents uploaded by a specific user (delegates to ChromaDB service)."""
        return self.vector_service.delete_user_documents(user_id)
    
    def delete_session_documents(self, session_id: str) -> bool:
        """Delete all documents uploaded in a specific session (delegates to ChromaDB service)."""
        return self.vector_service.delete_session_documents(session_id)
    
    # =============================================================================
    # ENHANCED ANALYTICS METHODS
    # =============================================================================
    
    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get enhanced statistics for a specific user's documents.
        
        Args:
            user_id (str): User ID to get stats for
            
        Returns:
            Dict[str, Any]: User-specific statistics with VectorStore enhancements
        """
        try:
            user_docs = self.get_documents_by_user(user_id)
            
            total_docs = len(user_docs)
            total_content_length = sum(doc.get('content_length', 0) for doc in user_docs)
            
            # Get unique files and types
            unique_files = set()
            file_types = set()
            for doc in user_docs:
                metadata = doc.get('metadata', {})
                filename = metadata.get('filename') or metadata.get('file_name', '')
                if filename:
                    unique_files.add(filename)
                file_type = metadata.get('file_type', '')
                if file_type:
                    file_types.add(file_type)
            
            return {
                'user_id': user_id,
                'total_documents': total_docs,
                'unique_files': len(unique_files),
                'total_content_length': total_content_length,
                'average_content_length': total_content_length / total_docs if total_docs > 0 else 0,
                'file_types': list(file_types),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'collection_name': self.collection_name,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting stats for user {user_id}: {str(e)}")
            return {
                'user_id': user_id,
                'total_documents': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get enhanced statistics for a specific session's documents.
        
        Args:
            session_id (str): Session ID to get stats for
            
        Returns:
            Dict[str, Any]: Session-specific statistics with VectorStore enhancements
        """
        try:
            session_docs = self.get_documents_by_session(session_id)
            
            total_docs = len(session_docs)
            total_content_length = sum(doc.get('content_length', 0) for doc in session_docs)
            
            # Get unique files, users, and types
            unique_files = set()
            unique_users = set()
            file_types = set()
            for doc in session_docs:
                metadata = doc.get('metadata', {})
                filename = metadata.get('filename') or metadata.get('file_name', '')
                if filename:
                    unique_files.add(filename)
                user_id = metadata.get('user_id', '')
                if user_id:
                    unique_users.add(user_id)
                file_type = metadata.get('file_type', '')
                if file_type:
                    file_types.add(file_type)
            
            return {
                'session_id': session_id,
                'total_documents': total_docs,
                'unique_files': len(unique_files),
                'unique_users': len(unique_users),
                'total_content_length': total_content_length,
                'average_content_length': total_content_length / total_docs if total_docs > 0 else 0,
                'file_types': list(file_types),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'collection_name': self.collection_name,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting stats for session {session_id}: {str(e)}")
            return {
                'session_id': session_id,
                'total_documents': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# =============================================================================
# FACTORY AND CONVENIENCE FUNCTIONS
# =============================================================================

def create_vector_store(
    db_path: str = "./data/chroma_db",
    embedding_provider: str = "qwen",
    **kwargs
) -> VectorStoreService:
    """Factory function to create a VectorStoreService instance."""
    return VectorStoreService(
        db_path=db_path,
        embedding_provider=embedding_provider,
        **kwargs
    )

def process_file(file_path: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to process a single file."""
    vector_store = create_vector_store()
    return vector_store.process_file(file_path, **kwargs)

def is_supported_format(file_path: str) -> bool:
    """Check if a file format is supported."""
    return Path(file_path).suffix.lower() in SUPPORTED_FORMATS

def get_supported_formats() -> List[str]:
    """Get list of supported file formats."""
    return sorted(SUPPORTED_FORMATS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Simple test
    vector_store = create_vector_store()
    print(f"Supported formats: {get_supported_formats()}")
    print(f"Collection stats: {vector_store.get_collection_stats()}")
    
    # Test file validation
    is_valid = vector_store.validate_file("nonexistent.pdf")
    print(f"Validation test: {is_valid}")
    
    print(f"PDF supported: {is_supported_format('test.pdf')}")
    print(f"TXT supported: {is_supported_format('test.txt')}")
    print(f"Unknown supported: {is_supported_format('test.xyz')}")