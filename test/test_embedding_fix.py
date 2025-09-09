#!/usr/bin/env python3
"""
Test script to verify the embedding API fix.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app.service.rag.embedding import create_embedding_service
    
    print("🧪 Testing embedding service after URL fix...")
    
    # Create embedding service
    service = create_embedding_service(provider='embedding')
    print(f"✅ Embedding service created successfully")
    
    # Test single text embedding
    test_text = "This is a test sentence for embedding."
    print(f"📝 Testing text: '{test_text}'")
    
    embedding = service.embed_text(test_text)
    print(f"✅ Embedding generated successfully: {len(embedding)} dimensions")
    
    print("🎉 Embedding API is working correctly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
