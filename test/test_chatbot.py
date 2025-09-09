"""
Simple test suite for chatbot.py - testing only simple_chat function.
"""

import unittest
import sys
import os
from unittest.mock import patch
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.chat.chatbot import simple_chat


class TestSimpleChat(unittest.TestCase):
    """Test cases for simple_chat functionality."""
    
    @patch('app.chat.chatbot.chat_with_agent')
    def test_simple_chat_basic(self, mock_chat_with_agent):
        """Test basic simple_chat function."""
        mock_chat_with_agent.return_value = "Hello! I'm ready to help."
        
        result = simple_chat("Hello!")
        
        self.assertEqual(result, "Hello! I'm ready to help.")
        mock_chat_with_agent.assert_called_once_with(
            "default_user", "default_session", "Hello!"
        )
    
    @patch('app.chat.chatbot.chat_with_agent')
    def test_simple_chat_with_custom_params(self, mock_chat_with_agent):
        """Test simple_chat with custom user_id and session."""
        mock_chat_with_agent.return_value = "Custom response"
        
        result = simple_chat("Test query", "custom_user", "custom_session")
        
        self.assertEqual(result, "Custom response")
        mock_chat_with_agent.assert_called_once_with(
            "custom_user", "custom_session", "Test query"
        )
    
    @patch('app.chat.chatbot.chat_with_agent')
    def test_simple_chat_error_handling(self, mock_chat_with_agent):
        """Test simple_chat error handling."""
        mock_chat_with_agent.return_value = "Error: Connection failed"
        
        result = simple_chat("Test query")
        
        self.assertEqual(result, "Error: Connection failed")
    
    @patch('app.chat.chatbot.chat_with_agent')
    def test_simple_chat_empty_query(self, mock_chat_with_agent):
        """Test simple_chat with empty query."""
        mock_chat_with_agent.return_value = "Please provide a valid query"
        
        result = simple_chat("")
        
        self.assertEqual(result, "Please provide a valid query")
        mock_chat_with_agent.assert_called_once_with(
            "default_user", "default_session", ""
        )


def run_simple_chat_tests():
    """Run simple chat tests."""
    print("=" * 50)
    print("RUNNING SIMPLE CHAT TESTS")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSimpleChat))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nResult: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success


if __name__ == "__main__":
    print("🤖 SIMPLE CHAT TEST SUITE")
    print("=" * 50)
    
    success = run_simple_chat_tests()
    
    if not success:
        sys.exit(1)
