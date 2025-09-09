"""Simple store utilities for document storage and deletion.

This module provides a minimal interface to the vector store service:
- store_document(file_path): Store a document in the vector database
- delete_document(file_name): Delete documents by filename from the vector database

Example:
    from app.utils.store_utils import store_document, delete_document
    
    # Store a document
    result = store_document('path/to/document.pdf')
    if result['success']:
        print(f"Stored {result['file_name']} successfully")
    
    # Delete a document
    result = delete_document('document.pdf')
    if result['success']:
        print(f"Deleted {result['documents_deleted']} documents")
"""

import logging
from pathlib import Path
from typing import Any, Dict

from ..service.vector.vector_store import VectorStoreService

# Configure logging
logger = logging.getLogger(__name__)


def store_document(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    Store a document in the vector database.
    
    Args:
        file_path: Path to the file to store. Must be a valid file path.
        **kwargs: Additional arguments passed to VectorStoreService:
            - collection_name: Name of the collection to store in
            - chunk_size: Size of text chunks for splitting
            - chunk_overlap: Overlap between chunks
            - use_docling: Whether to use docling for PDF processing
        
    Returns:
        Dictionary with storage result information containing:
            - success: Boolean indicating if operation succeeded
            - file_path: Original file path
            - file_name: Name of the file
            - error: Error message if operation failed
            - chunks_stored: Number of chunks stored (if successful)
    
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If file_path is empty or invalid
    """
    # Input validation
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path must be a non-empty string")
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path_obj.is_file():
        raise ValueError(f"Path is not a file: {file_path}")
    
    try:
        # Create vector store service
        logger.info(f"Initializing vector store service for: {file_path_obj.name}")
        vector_store = VectorStoreService(**kwargs)
        
        # Process and store the file
        logger.info(f"Processing and storing document: {file_path}")
        result = vector_store.process_file(file_path)
        
        if result["success"]:
            chunks_count = result.get('chunks_stored', 'unknown')
            logger.info(f"Successfully stored document: {result['file_name']} ({chunks_count} chunks)")
        else:
            error_msg = result.get('error', 'Unknown error')
            logger.error(f"Failed to store document {result['file_name']}: {error_msg}")
            
        return result
        
    except Exception as e:
        error_msg = f"Error storing document {file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "file_path": file_path,
            "file_name": file_path_obj.name,
            "error": error_msg
        }


def delete_document(file_name: str, **kwargs) -> Dict[str, Any]:
    """
    Delete documents by filename from the vector database.
    
    Args:
        file_name: Name of the file to delete (searches by source_file metadata).
                  Must be a non-empty string.
        **kwargs: Additional arguments passed to VectorStoreService:
            - collection_name: Name of the collection to delete from
        
    Returns:
        Dictionary with deletion result information containing:
            - success: Boolean indicating if operation succeeded
            - file_name: Name of the file that was deleted
            - documents_deleted: Number of document chunks deleted
            - deleted_ids: List of deleted document IDs (if successful)
            - error: Error message if operation failed
    
    Raises:
        ValueError: If file_name is empty or invalid
    """
    # Input validation
    if not file_name or not isinstance(file_name, str):
        raise ValueError("file_name must be a non-empty string")
    
    file_name = file_name.strip()
    if not file_name:
        raise ValueError("file_name cannot be empty or whitespace only")
    
    try:
        # Create vector store service
        logger.info(f"Initializing vector store service for deletion of: {file_name}")
        vector_store = VectorStoreService(**kwargs)
        
        # Search for documents with matching filename
        logger.info(f"Searching for documents with filename: {file_name}")
        search_results = vector_store.search_documents(
            query="",  # Empty query to get all documents
            n_results=1000,  # Large number to get all matches
            where={"source_file": file_name}
        )
        
        if not search_results:
            logger.warning(f"No documents found with filename: {file_name}")
            return {
                "success": False,
                "file_name": file_name,
                "error": f"No documents found with filename: {file_name}",
                "documents_deleted": 0
            }
        
        # Extract document IDs
        doc_ids = [result.get('id') for result in search_results if result.get('id')]
        
        if not doc_ids:
            logger.error(f"Found {len(search_results)} documents but no valid IDs for deletion")
            return {
                "success": False,
                "file_name": file_name,
                "error": "Found documents but no valid IDs for deletion",
                "documents_deleted": 0
            }
        
        # Delete documents by IDs
        logger.info(f"Attempting to delete {len(doc_ids)} document chunks for: {file_name}")
        success = vector_store.vector_service.delete_documents(doc_ids)
        
        if success:
            logger.info(f"Successfully deleted {len(doc_ids)} documents for file: {file_name}")
            return {
                "success": True,
                "file_name": file_name,
                "documents_deleted": len(doc_ids),
                "deleted_ids": doc_ids
            }
        else:
            logger.error(f"Failed to delete documents from vector store for: {file_name}")
            return {
                "success": False,
                "file_name": file_name,
                "error": "Failed to delete documents from vector store",
                "documents_deleted": 0
            }
            
    except Exception as e:
        error_msg = f"Error deleting document {file_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "file_name": file_name,
            "error": error_msg,
            "documents_deleted": 0
        }


if __name__ == "__main__":
    # Simple test functionality
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "store" and len(sys.argv) > 2:
            result = store_document(sys.argv[2])
            print(f"Store result: {result}")
        elif sys.argv[1] == "delete" and len(sys.argv) > 2:
            result = delete_document(sys.argv[2])
            print(f"Delete result: {result}")
        else:
            print("Usage: python store_utils.py [store|delete] <file_path>")
    else:
        print("Usage: python store_utils.py [store|delete] <file_path>")