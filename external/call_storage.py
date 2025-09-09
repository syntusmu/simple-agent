"""
Simple interface to call storage functions from app.utils.store_utils directly.

This module provides a direct interface to the store_document function
from app/utils/store_utils.py for storing documents in ChromaDB.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils.store_utils import store_document, delete_document


def call_storage(file_path: str, **kwargs) -> dict:
    """
    Store a document using the store_document function from app.utils.store_utils.
    
    Args:
        file_path: Path to the document file (relative to project root or absolute)
        **kwargs: Additional arguments passed to store_document function
    
    Returns:
        Dictionary with storage result information
    """
    try:
        # Resolve file path (handle relative paths from project root)
        if not os.path.isabs(file_path):
            file_path = str(project_root / file_path)
        
        print(f"📁 Storing document: {Path(file_path).name}")
        print(f"📂 Full path: {file_path}")
        
        # Call store_document function directly
        result = store_document(file_path, **kwargs)
        
        # Display results
        if result["success"]:
            print(f"✅ Successfully stored {result['file_name']}")
            chunks_count = result.get('chunks_stored', 'unknown')
            print(f"📄 Chunks stored: {chunks_count}")
        else:
            print(f"❌ Failed to store {result.get('file_name', 'document')}")
            print(f"🚫 Error: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "file_path": file_path
        }


def main():
    """Main function to store the HR document."""
    # Store the specified HR document
    input_path = "data\各地产假规定一览表  to GE.xlsx"
    
    print("🚀 Starting document storage process...")
    print(f"📄 Target document: {input_path}")
    
    # Call the storage function
    result = call_storage(file_path=input_path)
    
    # Print final result
    print("\n" + "="*50)
    if result["success"]:
        print("🎉 Document storage completed successfully!")
    else:
        print("💥 Document storage failed!")
        print(f"Error: {result.get('error', 'Unknown error')}")
    print("="*50)
    
    # Exit with appropriate code
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()