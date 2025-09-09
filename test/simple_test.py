import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Starting simple test...")

try:
    # Test config loading
    from app.utils.common import get_config_value
    api_key = get_config_value('embedding', 'api_key')
    base_url = get_config_value('embedding', 'base_url')
    
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    print(f"Base URL: {base_url}")
    
    # Test embedding service creation
    from app.service.rag.embedding import create_embedding_service
    service = create_embedding_service(provider='embedding')
    print("Embedding service created successfully")
    
    # Test simple embedding
    result = service.embed_text("Hello world")
    print(f"Embedding successful: {len(result)} dimensions")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
