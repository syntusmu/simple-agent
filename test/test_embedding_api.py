#!/usr/bin/env python3
"""
Test script for embedding API key from config.ini
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.utils.common import get_config_value
import requests
import json

def test_embedding_api():
    """Test the embedding API key from config.ini"""
    print("Testing Embedding API Key...")
    
    try:
        # Get configuration from config.ini
        api_key = get_config_value('embedding', 'api_key')
        base_url = get_config_value('embedding', 'base_url')
        model = get_config_value('embedding', 'model')
        
        print(f"API Key: {api_key[:10]}...")
        print(f"Base URL: {base_url}")
        print(f"Model: {model}")
        
        # Test embedding request
        url = f"{base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "input": "Hello, this is a test message for embedding.",
            "encoding_format": "float"
        }
        
        print(f"\nSending request to: {url}")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'data' in result and len(result['data']) > 0:
                embedding = result['data'][0]['embedding']
                print(f"✅ SUCCESS! Embedding generated with {len(embedding)} dimensions")
                print(f"First 5 values: {embedding[:5]}")
                return True
            else:
                print("❌ FAILED: No embedding data in response")
                print(f"Response: {result}")
                return False
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    test_embedding_api()
