"""
Vector Store Service for document processing and storage.

This module provides a streamlined document processing pipeline:
1. Load documents using document_loader.py (handles all formats including docling preprocessing)
2. Split loaded documents into chunks using document_splitter.py
3. Generate embeddings and store in ChromaDB using chromadb.py

Simple interface: provide file_path -> get documents stored in vector database
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
    Streamlined vector store service for document processing and storage.
    
    Pipeline:
    1. Load documents using document_loader (handles all formats and preprocessing)
    2. Split documents into chunks using document_splitter
    3. Store chunks with embeddings in ChromaDB
    """
    
    def __init__(
        self,
        db_path: str = "./data/chroma_db",
        embedding_provider: str = "qwen",
        embedding_model: Optional[str] = None,
        collection_name: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the vector store service.
        
        Args:
            db_path: Path to ChromaDB database directory
            embedding_provider: Embedding provider ('qwen', 'openai', 'embedding')
            embedding_model: Embedding model name (uses provider default if None)
            collection_name: Name of the ChromaDB collection
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.db_path = db_path
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.collection_name = collection_name
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
            embedding_model=embedding_model,
            collection_name=collection_name
        )
        
        logger.info(f"VectorStoreService initialized: {embedding_provider} embeddings, collection: {collection_name}")
        logger.info(f"Chunk settings: size={chunk_size}, overlap={chunk_overlap}")

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

    def load_document(self, file_path: str) -> List[Document]:
        """
        Load a document using document_loader (handles all formats and preprocessing).
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of Document objects
        """
        try:
            # Basic validation
            if not self.validate_file(file_path):
                raise ValueError(f"Invalid file: {file_path}")
            
            file_name = Path(file_path).name
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
                    'source_file': file_name,
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

    def store_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """Store documents in ChromaDB with embeddings."""
        try:
            if not documents:
                return {"success": False, "message": "No documents to store"}
            
            logger.info(f"Storing {len(documents)} documents...")
            doc_ids = self.vector_service.add_documents(documents)
            
            return {
                "success": True,
                "documents_stored": len(documents),
                "document_ids": doc_ids,
                "collection_name": self.collection_name,
                "embedding_provider": self.embedding_provider,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            return {"success": False, "message": f"Storage error: {str(e)}", "documents_stored": 0}

    def process_file(self, file_path: str, chunk_documents: bool = True) -> Dict[str, Any]:
        """
        Complete pipeline: load, chunk, and store a document.
        
        Args:
            file_path: Path to the file to process
            chunk_documents: Whether to chunk documents before storage
            
        Returns:
            Processing result dictionary
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Load document
            documents = self.load_document(file_path)
            
            # Step 2: Chunk documents if requested
            if chunk_documents:
                documents = self.chunk_documents(documents)
            
            # Step 3: Store documents with embeddings
            storage_result = self.store_documents(documents)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": storage_result["success"],
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "documents_loaded": len(documents),
                "documents_stored": storage_result.get("documents_stored", 0),
                "chunks_stored": storage_result.get("documents_stored", 0),  # Add chunks_stored field
                "collection_name": self.collection_name,
                "processing_time_seconds": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            if not storage_result["success"]:
                result["error"] = storage_result.get("message", "Unknown error")
            
            logger.info(f"Processed file {Path(file_path).name} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return {
                "success": False,
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def batch_process_files(self, file_paths: List[str], chunk_documents: bool = True) -> Dict[str, Any]:
        """Process multiple files in batch."""
        try:
            start_time = datetime.now()
            results = []
            successful = failed = 0
            
            for file_path in file_paths:
                result = self.process_file(file_path, chunk_documents)
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
                "processing_time_seconds": round(processing_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def search_documents(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Metadata filter conditions
            
        Returns:
            List of search results with documents and metadata
        """
        try:
            return self.vector_service.similarity_search(query, n_results, where)
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
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

    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            return self.vector_service.clear_collection()
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False


# Convenience functions
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