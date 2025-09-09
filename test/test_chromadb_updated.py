#!/usr/bin/env python3
"""
Test script for updated ChromaDB service with custom embedding service.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.service.vector.chromadb import ChromaDBVectorService
from langchain.schema import Document

def test_chromadb_service():
    """Test the updated ChromaDB service with custom embedding service."""
    print("Testing Updated ChromaDB Service...")
    
    try:
        # Test ChromaDB service creation with Qwen embeddings (default)
        print("\n1. Creating ChromaDB service with Qwen embeddings...")
        vector_service = ChromaDBVectorService(
            db_path="./data/test_chroma_db",
            embedding_provider="qwen",
            collection_name="test_collection"
        )
        print("✅ ChromaDB service created successfully with Qwen embeddings")
        
        # Test adding documents
        print("\n2. Testing document addition...")
        test_docs = [
            Document(
                page_content="This is a test document about machine learning.",
                metadata={"source": "test1.txt", "id": "doc1"}
            ),
            Document(
                page_content="Another document discussing artificial intelligence.",
                metadata={"source": "test2.txt", "id": "doc2"}
            )
        ]
        
        vector_service.add_documents(test_docs)
        print("✅ Documents added successfully")
        
        # Test similarity search
        print("\n3. Testing similarity search...")
        results = vector_service.similarity_search("machine learning", n_results=2)
        print(f"✅ Found {len(results)} similar documents")
        for i, result in enumerate(results):
            print(f"   Result {i+1}: {result['content'][:50]}...")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_chromadb_service()
