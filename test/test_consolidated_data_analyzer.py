#!/usr/bin/env python3
"""
Test script for the consolidated data analyzer with DeepSeek LLM integration.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.tools.data_analyzer import ExcelPandasAnalyzer, SUPPORTED_FORMATS
from app.service.llm.llm_interface import create_llm

def test_llm_initialization():
    """Test LLM initialization with DeepSeek."""
    print("Testing LLM initialization...")
    try:
        llm = create_llm(provider='deepseek', temperature=0)
        if llm:
            print("✅ LLM initialized successfully with DeepSeek")
            print(f"LLM type: {type(llm)}")
            return True
        else:
            print("❌ LLM initialization failed")
            return False
    except Exception as e:
        print(f"❌ LLM initialization failed: {e}")
        return False

def test_analyzer_creation():
    """Test analyzer creation."""
    print("\nTesting analyzer creation...")
    try:
        analyzer = ExcelPandasAnalyzer()
        print("✅ ExcelPandasAnalyzer created successfully")
        print(f"LLM available: {analyzer.llm is not None}")
        return True
    except Exception as e:
        print(f"❌ Analyzer creation failed: {e}")
        return False

def test_supported_formats():
    """Test supported formats."""
    print(f"\nSupported formats: {SUPPORTED_FORMATS}")
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Consolidated Data Analyzer with DeepSeek LLM")
    print("=" * 60)
    
    tests = [
        test_llm_initialization,
        test_analyzer_creation,
        test_supported_formats
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check DeepSeek API configuration.")

if __name__ == "__main__":
    main()
