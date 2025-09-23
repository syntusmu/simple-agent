"""Document storage utilities for vector database operations.

This module provides a simplified interface for common document operations:

1. Document Storage:
   - store_document(): Store documents in vector database with chunking and embeddings
   - batch_store_documents(): Store multiple documents in batch

2. Document Deletion:
   - delete_document(): Delete documents by filename
   - clear_all_documents(): Clear entire vector database

3. Document Information:
   - get_document_info(): Get information about stored documents
   - list_stored_files(): List all stored filenames

Usage:
    from app.utils.store_utils import store_document, delete_document
    
    # Store a document
    result = store_document('path/to/document.pdf', user_id='user123')
    
    # Delete a document
    result = delete_document('document.pdf')
"""

import logging
import sys
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..service.vector.vector_store import VectorStoreService

# Configure logging
logger = logging.getLogger(__name__)


def store_document(file_path: str, original_filename: str = None, **kwargs) -> Dict[str, Any]:
    """
    Store a document in the vector database with chunking and embeddings.
    
    Args:
        file_path: Path to the file to store
        original_filename: Original filename to use in metadata (if different from file_path)
        **kwargs: Additional arguments for VectorStoreService
        
    Returns:
        Dict[str, Any]: Operation result with success status and metadata
        
    Raises:
        ValueError: If file_path is invalid
        FileNotFoundError: If file doesn't exist
        
    Example:
        >>> result = store_document('document.pdf', user_id='user123', upsert=True)
        >>> print(f"Stored {result['chunks_stored']} chunks")
    """
    # Validate input
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path must be a non-empty string")
    
    file_path_obj = Path(file_path)
    if not file_path_obj.exists() or not file_path_obj.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Use original filename if provided, otherwise extract from path
    display_name = original_filename if original_filename else file_path_obj.name
    
    try:
        display_name = original_filename if original_filename else file_path_obj.name
        logger.info(f"Storing document: {display_name}")
        vector_store = VectorStoreService()
        result = vector_store.process_file(file_path, original_filename=original_filename)
        
        # Log result
        if result["success"]:
            logger.info(
                f"Successfully stored {result['file_name']}: "
                f"{result.get('chunks_stored', 0)} chunks in {result.get('processing_time_seconds', 0):.2f}s"
            )
        else:
            logger.error(f"Failed to store {result['file_name']}: {result.get('error', 'Unknown error')}")
            
        return result
        
    except Exception as e:
        logger.error(f"Error storing document {file_path}: {e}")
        return {
            "success": False,
            "file_path": file_path,
            "file_name": original_filename if original_filename else file_path_obj.name,
            "error": str(e)
        }


def delete_document(file_name: str, **kwargs) -> Dict[str, Any]:
    """
    Delete all documents with the specified filename from the vector database.
    
    Args:
        file_name (str): Name of the file to delete (e.g., 'document.pdf')
        **kwargs: Additional arguments for VectorStoreService
        
    Returns:
        Dict[str, Any]: Operation result with success status and deletion count
        
    Raises:
        ValueError: If file_name is invalid
        
    Example:
        >>> result = delete_document('old_document.pdf')
        >>> print(f"Deleted {result['documents_deleted']} chunks")
    """
    # Validate input
    if not file_name or not isinstance(file_name, str) or not file_name.strip():
        raise ValueError("file_name must be a non-empty string")
    
    file_name = file_name.strip()
    
    try:
        logger.info(f"Deleting documents for filename: {file_name}")
        vector_store = VectorStoreService()
        
        # Get all documents and filter by filename
        all_docs = vector_store.vector_service.get_all_documents()
        search_results = [
            doc for doc in all_docs 
            if doc.get('metadata', {}).get('file_name') == file_name
        ]
        
        if not search_results:
            logger.warning(f"No documents found with filename: {file_name}")
            return _create_result(
                success=False, 
                operation="delete",
                file_name=file_name, 
                error=f"No documents found with filename: {file_name}"
            )
        
        # Extract document IDs
        doc_ids = [result.get('id') for result in search_results if result.get('id')]
        
        if not doc_ids:
            logger.error("Found documents but no valid IDs for deletion")
            return _create_result(
                success=False,
                operation="delete",
                file_name=file_name,
                error="No valid document IDs found"
            )
        
        # Delete documents
        logger.info(f"Deleting {len(doc_ids)} document chunks for {file_name}")
        success = vector_store.vector_service.delete_documents(doc_ids)
        
        if success:
            logger.info(f"Successfully deleted {len(doc_ids)} chunks for {file_name}")
            return _create_result(
                success=True,
                operation="delete",
                file_name=file_name,
                documents_deleted=len(doc_ids),
                deleted_ids=doc_ids
            )
        else:
            logger.error("Failed to delete documents from vector store")
            return _create_result(
                success=False,
                operation="delete",
                file_name=file_name,
                error="Vector store deletion failed"
            )
            
    except Exception as e:
        logger.error(f"Error deleting document {file_name}: {e}")
        return _create_result(
            success=False,
            operation="delete",
            file_name=file_name,
            error=str(e)
        )


def clear_all_documents(**kwargs) -> Dict[str, Any]:
    """
    Clear all documents from the vector database.
    
    Args:
        **kwargs: Additional arguments for VectorStoreService
        
    Returns:
        Dict[str, Any]: Operation result
        
    Warning:
        This operation cannot be undone. All documents will be permanently deleted.
        
    Example:
        >>> result = clear_all_documents()
        >>> print(f"Cleared database: {result['success']}")
    """
    try:
        logger.warning("Clearing ALL documents from vector database")
        vector_store = VectorStoreService()
        
        # Get current stats before clearing
        stats = vector_store.get_collection_stats()
        doc_count = stats.get('document_count', 0)
        
        if doc_count == 0:
            logger.info("No documents to clear")
            return _create_result(
                success=True,
                operation="clear_all",
                message="No documents to clear",
                documents_deleted=0
            )
        
        # Clear the collection
        success = vector_store.clear_collection()
        
        if success:
            logger.info(f"Successfully cleared {doc_count} documents from database")
            return _create_result(
                success=True,
                operation="clear_all",
                message=f"Cleared {doc_count} documents",
                documents_deleted=doc_count
            )
        else:
            logger.error("Failed to clear vector database")
            return _create_result(
                success=False,
                operation="clear_all",
                error="Failed to clear vector database"
            )
            
    except Exception as e:
        logger.error(f"Error clearing all documents: {e}")
        return _create_result(
            success=False,
            operation="clear_all",
            error=str(e)
        )


# =============================================================================
# DOCUMENT INFORMATION FUNCTIONS
# =============================================================================

def get_document_info(**kwargs) -> Dict[str, Any]:
    """
    Get information about all stored documents.
    
    Args:
        **kwargs: Additional arguments for VectorStoreService
        
    Returns:
        Dict[str, Any]: Document information including stats and file list
        
    Example:
        >>> info = get_document_info()
        >>> print(f"Total documents: {info['total_documents']}")
        >>> print(f"Unique files: {info['unique_files']}")
    """
    try:
        logger.info("Retrieving document information")
        vector_store = VectorStoreService()
        
        # Get comprehensive document info
        info = vector_store.get_all_documents_info()
        
        if info['success']:
            logger.info(
                f"Retrieved info for {info['total_documents']} documents "
                f"from {info['unique_files']} unique files"
            )
        else:
            logger.error(f"Failed to retrieve document info: {info.get('error', 'Unknown error')}")
            
        return info
        
    except Exception as e:
        logger.error(f"Error retrieving document info: {e}")
        return {
            "success": False,
            "error": str(e),
            "total_documents": 0,
            "unique_files": 0,
            "files_summary": []
        }


def list_stored_files(**kwargs) -> List[str]:
    """
    Get a list of all stored filenames.
    
    Args:
        **kwargs: Additional arguments for VectorStoreService
        
    Returns:
        List[str]: List of unique filenames stored in the database
        
    Example:
        >>> files = list_stored_files()
        >>> print(f"Stored files: {files}")
    """
    try:
        logger.info("Listing stored files")
        info = get_document_info(**kwargs)
        
        if info['success']:
            files = [file_info['file_name'] for file_info in info.get('files_summary', [])]
            logger.info(f"Found {len(files)} unique files")
            return files
        else:
            logger.error("Failed to retrieve file list")
            return []
            
    except Exception as e:
        logger.error(f"Error listing stored files: {e}")
        return []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _create_result(
    success: bool,
    operation: str,
    file_name: Optional[str] = None,
    error: Optional[str] = None,
    message: Optional[str] = None,
    documents_deleted: int = 0,
    deleted_ids: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Helper function to create consistent operation result dictionaries.
    
    Args:
        success (bool): Whether the operation succeeded
        operation (str): Type of operation performed
        file_name (Optional[str]): Name of file involved in operation
        error (Optional[str]): Error message if operation failed
        message (Optional[str]): Success message if operation succeeded
        documents_deleted (int): Number of documents deleted
        deleted_ids (Optional[List[str]]): List of deleted document IDs
        **kwargs: Additional fields to include in result
        
    Returns:
        Dict[str, Any]: Standardized result dictionary
    """
    from datetime import datetime
    
    result = {
        "success": success,
        "operation": operation,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add optional fields
    if file_name:
        result["file_name"] = file_name
    if documents_deleted > 0:
        result["documents_deleted"] = documents_deleted
    if success:
        if message:
            result["message"] = message
        if deleted_ids:
            result["deleted_ids"] = deleted_ids
    else:
        if error:
            result["error"] = error
    
    # Add any additional fields
    result.update(kwargs)
    
    return result


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """
    Command line interface for document storage utilities.
    
    Usage:
        python store_utils.py store <file_path> [user_id] [session_id]
        python store_utils.py batch <file1> <file2> ... [user_id] [session_id]
        python store_utils.py delete <filename>
        python store_utils.py clear
        python store_utils.py info
        python store_utils.py list
    """
    import json
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage: python store_utils.py [store|batch|delete|clear|info|list] [args...]")
        print("")
        print("Commands:")
        print("  store <file_path> [user_id] [session_id]  - Store a single document")
        print("  batch <file1> <file2> ... [user_id]       - Store multiple documents")
        print("  delete <filename>                         - Delete documents by filename")
        print("  clear                                     - Clear all documents")
        print("  info                                      - Show database information")
        print("  list                                      - List stored filenames")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    try:
        if action == "store":
            if len(sys.argv) < 3:
                print("Usage: python store_utils.py store <file_path> [user_id] [session_id]")
                sys.exit(1)
            
            file_path = sys.argv[2]
            user_id = sys.argv[3] if len(sys.argv) > 3 else None
            session_id = sys.argv[4] if len(sys.argv) > 4 else None
            
            result = store_document(file_path, user_id=user_id, session_id=session_id)
            
        elif action == "batch":
            if len(sys.argv) < 3:
                print("Usage: python store_utils.py batch <file1> <file2> ... [user_id]")
                sys.exit(1)
            
            # Extract files and optional user_id
            files = []
            user_id = None
            
            for arg in sys.argv[2:]:
                if Path(arg).exists():
                    files.append(arg)
                else:
                    user_id = arg  # Assume non-file argument is user_id
                    break
            
            if not files:
                print("No valid files provided")
                sys.exit(1)
            
            result = batch_store_documents(files, user_id=user_id)
            
        elif action == "delete":
            if len(sys.argv) < 3:
                print("Usage: python store_utils.py delete <filename>")
                sys.exit(1)
            
            filename = sys.argv[2]
            result = delete_document(filename)
            
        elif action == "clear":
            print("WARNING: This will delete ALL documents from the database!")
            confirm = input("Are you sure? (yes/no): ").lower().strip()
            
            if confirm in ['yes', 'y']:
                result = clear_all_documents()
            else:
                print("Operation cancelled")
                sys.exit(0)
                
        elif action == "info":
            result = get_document_info()
            
        elif action == "list":
            files = list_stored_files()
            result = {
                "success": True,
                "operation": "list",
                "files": files,
                "count": len(files)
            }
            
        else:
            print(f"Unknown action: {action}")
            print("Valid actions: store, batch, delete, clear, info, list")
            sys.exit(1)
        
        # Print result
        print(json.dumps(result, indent=2, default=str))
        
        # Exit with appropriate code
        sys.exit(0 if result.get('success', False) else 1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()