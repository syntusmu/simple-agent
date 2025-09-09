"""
Simple Chatbot that calls chat_interface and returns only the agent response.

This module provides a simplified interface to get only the LLM response
from the chat interface without additional metadata.
"""

import logging
from typing import Optional, List, Dict

from .chat_interface import process_chat_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chat_with_agent(
    user_id: str,
    user_session: str,
    user_query: str,
    file_upload_list: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Simple chat function that returns only the agent response.
    
    Args:
        user_id: Unique user identifier
        user_session: Session identifier
        user_query: User's query/message
        file_upload_list: Optional list of uploaded files
        
    Returns:
        Agent response string only
    """
    try:
        # Call chat interface
        result = process_chat_request(
            user_id=user_id,
            user_session=user_session,
            user_query=user_query,
            file_upload_list=file_upload_list
        )
        
        # Extract only the agent response
        if result.get("success"):
            return result.get("response", "No response generated")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
            
    except Exception as e:
        error_msg = f"Chatbot error: {str(e)}"
        logger.error(error_msg)
        return error_msg


def simple_chat(user_query: str, user_id: str = "default_user", user_session: str = "default_session") -> str:
    """
    Even simpler chat function with default user and session.
    
    Args:
        user_query: User's query/message
        user_id: User identifier (default: "default_user")
        user_session: Session identifier (default: "default_session")
        
    Returns:
        Agent response string only
    """
    return chat_with_agent(user_id, user_session, user_query)


def _test_chatbot():
    """Test function for the chatbot."""
    print("Testing Simple Chatbot...")
    
    try:
        # Test simple chat
        response1 = simple_chat("Hello, what tools do you have?")
        print(f"Response 1: {response1}")
        
        # Test with file upload
        file_list = [
            {
                "file_name": "各地产假规定一览表  to GE.xlsx",
                "file_path": "data/各地产假规定一览表  to GE.xlsx"
            }
        ]
        
        response2 = chat_with_agent(
            user_id="test_user",
            user_session="test_session",
            user_query="what is shanghai policy in the file uploaded",
            file_upload_list=file_list
        )
        print(f"Response 2: {response2}")
        
    except Exception as e:
        logger.error(f"Error testing chatbot: {e}")


if __name__ == "__main__":
    _test_chatbot()