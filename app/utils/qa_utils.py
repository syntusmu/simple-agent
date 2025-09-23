"""Question-answering utilities for chat interactions.

Provides simplified interface for chat-based question answering:
- chat_with_agent(): Simple chat function returning agent responses
- chat_with_agent_stream(): Streaming chat function yielding response chunks

## Usage Examples

### Static Response
```python
response = chat_with_agent("user123", "session1", "Hello!")
print(response)
```

### Streaming Response
```python
for chunk in chat_with_agent_stream("user123", "session1", "Hello!"):
    print(chunk, end="", flush=True)
```

### Command Line Testing
```bash
# Static response
python3 -m app.utils.qa_utils user123 session1 "Hello!"

# Streaming response  
python3 -m app.utils.qa_utils user123 session1 "Hello!" --stream
```
"""

import logging
from typing import Optional, List, Dict

from ..chat.chat_interface import process_chat_request, process_chat_request_stream

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
        Agent response as string
    """
    try:
        # Log request info
        file_info = "No files"
        if file_upload_list:
            first_file = file_upload_list[0]
            file_name = first_file.get("file_name", "Unknown")
            file_info = f"File: {file_name}"
        
        logger.info(f"Chat request - User: {user_id}, Session: {user_session}, Query: {user_query}, {file_info}")
        
        # Process chat request
        result = process_chat_request(
            user_id=user_id,
            user_session=user_session,
            user_query=user_query,
            file_upload_list=file_upload_list
        )
        
        # Return agent response or error
        if result.get("success"):
            return result.get("response", "No response generated")
        else:
            return f"Error: {result.get('error', 'Unknown error occurred')}"
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return f"Error: {str(e)}"


def chat_with_agent_stream(
    user_id: str,
    user_session: str,
    user_query: str,
    file_upload_list: Optional[List[Dict[str, str]]] = None
):
    """
    Streaming chat function that yields agent response chunks.
    
    Args:
        user_id: Unique user identifier
        user_session: Session identifier
        user_query: User's query/message
        file_upload_list: Optional list of uploaded files
        
    Yields:
        Dict containing streaming response chunks or error information
    """
    try:
        # Log request info
        file_info = "No files"
        if file_upload_list:
            first_file = file_upload_list[0]
            file_name = first_file.get("file_name", "Unknown")
            file_info = f"File: {file_name}"
        
        logger.info(f"Streaming chat request - User: {user_id}, Session: {user_session}, Query: {user_query}, {file_info}")
        
        # Process streaming chat request
        for chunk in process_chat_request_stream(
            user_id=user_id,
            user_session=user_session,
            user_query=user_query,
            file_upload_list=file_upload_list
        ):
            if chunk.get("success"):
                if chunk.get("type") == "chunk":
                    # Yield content chunks
                    yield chunk.get("content", "")
                elif chunk.get("type") == "start":
                    # Initial metadata - could be used for setup
                    continue
                elif chunk.get("type") == "end":
                    # Completion signal
                    break
                elif chunk.get("type") == "error":
                    # Stream error
                    yield f"Error: {chunk.get('error', 'Unknown streaming error')}"
                    break
            else:
                # General error
                yield f"Error: {chunk.get('error', 'Unknown error occurred')}"
                break
                
    except Exception as e:
        logger.error(f"Streaming chat error: {e}")
        yield f"Error: {str(e)}"


def _test_streaming():
    """Test function for streaming chat."""
    print("Testing Streaming Chat...")
    
    try:
        user_id, session, query = "test_user", "test_session", "Hello, can you help me?"
        
        print(f"\nStreaming response for: {query}")
        print("Response: ", end="", flush=True)
        
        for chunk in chat_with_agent_stream(user_id, session, query):
            print(chunk, end="", flush=True)
        
        print("\n\nStreaming test completed!")
        
    except Exception as e:
        print(f"Streaming test failed: {e}")


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 4:
        print("Usage: python qa_utils.py <user_id> <session> <query> [--stream]")
        print("  --stream: Use streaming output instead of static response")
        sys.exit(1)
        
    user_id, session, query = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Check if streaming mode is requested
    if len(sys.argv) > 4 and sys.argv[4] == "--stream":
        print(f"Streaming response for: {query}")
        print("Response: ", end="", flush=True)
        for chunk in chat_with_agent_stream(user_id, session, query):
            print(chunk, end="", flush=True)
        print()  # New line at the end
    else:
        response = chat_with_agent(user_id, session, query)
        print(f"Response: {response}")