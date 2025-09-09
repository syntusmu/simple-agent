#!/usr/bin/env python3
"""
Test script for DeepSeek API key from config.ini
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.utils.common import get_config_value
import requests
import json

def test_deepseek_api():
    """Test the DeepSeek API key from config.ini"""
    print("Testing DeepSeek API Key...")
    
    try:
        # Get configuration from config.ini
        api_key = get_config_value('deepseek', 'api_key')
        base_url = get_config_value('deepseek', 'base_url')
        model = get_config_value('deepseek', 'model')
        temperature = float(get_config_value('deepseek', 'temperature', default='0'))
        max_tokens = int(get_config_value('deepseek', 'max_tokens', default='2048'))
        
        print(f"API Key: {api_key[:10]}...")
        print(f"Base URL: {base_url}")
        print(f"Model: {model}")
        print(f"Temperature: {temperature}")
        print(f"Max Tokens: {max_tokens}")
        
        # Test chat completion request
        url = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello! Please respond with 'API test successful' if you can see this message."}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        print(f"\nSending request to: {url}")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0]['message']['content']
                print(f"✅ SUCCESS! DeepSeek responded:")
                print(f"Response: {message}")
                
                # Check usage info
                if 'usage' in result:
                    usage = result['usage']
                    print(f"Token usage - Prompt: {usage.get('prompt_tokens', 'N/A')}, Completion: {usage.get('completion_tokens', 'N/A')}, Total: {usage.get('total_tokens', 'N/A')}")
                
                return True
            else:
                print("❌ FAILED: No response message in result")
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
    test_deepseek_api()
