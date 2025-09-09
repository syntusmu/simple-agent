#!/usr/bin/env python3
"""
Test script for the refactored RRF Retriever using LangChain EnsembleRetriever.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.service.vector.chromadb import ChromaDBVectorService
from app.service.rag.retriever import RRFRetriever
from langchain.schema import Document

def test_refactored_retriever():
    """Test the refactored RRF Retriever implementation."""
    print("Testing Refactored RRF Retriever with LangChain EnsembleRetriever...")
    
    try:
        # Initialize vector service
        print("\n1. Initializing ChromaDB Vector Service...")
        vector_service = ChromaDBVectorService(
            db_path="./test_chroma_db",
            enable_bm25=True
        )
        
        # Add some test documents
        print("2. Adding test documents...")
        test_docs = [
            Document(
                page_content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
                metadata={"source": "ml_intro.txt", "topic": "machine_learning"}
            ),
            Document(
                page_content="Deep learning uses neural networks with multiple layers to process data.",
                metadata={"source": "dl_basics.txt", "topic": "deep_learning"}
            ),
            Document(
                page_content="Natural language processing helps computers understand human language.",
                metadata={"source": "nlp_guide.txt", "topic": "nlp"}
            ),
            Document(
                page_content="Computer vision enables machines to interpret visual information from images.",
                metadata={"source": "cv_overview.txt", "topic": "computer_vision"}
            )
        ]
        
        ids = vector_service.add_documents(test_docs)
        print(f"Added {len(ids)} documents with IDs: {ids}")
        
        # Initialize RRF retriever
        print("\n3. Initializing RRF Retriever with EnsembleRetriever...")
        retriever = RRFRetriever(
            vector_service=vector_service,
            default_top_k=3,
            vector_weight=0.6,
            bm25_weight=0.4
        )
        
        # Test retrieval
        print("\n4. Testing retrieval with query 'machine learning algorithms'...")
        documents = retriever.retrieve("machine learning algorithms", top_k=3)
        
        if documents:
            print(f"Retrieved {len(documents)} documents:")
            for i, doc in enumerate(documents, 1):
                print(f"  Result {i}:")
                print(f"    Content: {doc.page_content[:80]}...")
                print(f"    Metadata: {doc.metadata}")
        else:
            print("No documents retrieved.")
        
        # Test with scores
        print("\n5. Testing retrieval with scores...")
        documents, scoring_info = retriever.retrieve_with_scores("neural networks", top_k=2)
        
        print(f"Scoring info: {scoring_info}")
        if documents:
            for i, doc in enumerate(documents, 1):
                print(f"  Result {i}: {doc.page_content[:60]}...")
        
        # Show retriever stats
        print("\n6. Retriever Statistics:")
        stats = retriever.get_retriever_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value}")
            else:
                print(f"  {key}: {value}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_refactored_retriever()
    sys.exit(0 if success else 1)
