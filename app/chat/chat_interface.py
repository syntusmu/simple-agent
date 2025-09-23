"""Chat Interface for ReAct Agent with Session Management.

This module provides a comprehensive chat interface for interacting with the ReAct agent,
including session management, file upload handling, and both synchronous and streaming
chat capabilities.

Classes:
    ChatSession: Represents individual chat sessions with memory and agent instances
    ChatInterface: Main interface for handling chat interactions

Functions:
    process_chat_request: Convenience function for synchronous chat
    process_chat_request_stream: Convenience function for streaming chat
"""

import logging
from typing import Dict, Any, Optional, List, Generator
from datetime import datetime

from ..agent.agent import create_react_agent_instance
from ..agent.memory import create_memory_manager
from ..service.rag.retriever import RRFRetriever
from ..service.vector.chromadb import ChromaDBVectorService

logger = logging.getLogger(__name__)

# Constants
SESSION_KEY_SEPARATOR = "_"
DEFAULT_ERROR_MESSAGE = "An unexpected error occurred"



# =============================================================================
# SESSION MANAGEMENT CLASSES
# =============================================================================

class ChatSession:
    """Represents a chat session with memory and agent instance.
    
    Each session maintains its own agent instance, memory manager, and services
    to ensure proper isolation between different user sessions.
    """
    
    def __init__(self, user_id: str, session_id: str):
        """Initialize chat session with required services.
        
        Args:
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session
            
        Raises:
            Exception: If initialization of services fails
        """
        self.user_id = user_id
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Initialize services for this session
        try:
            self.vector_service = ChromaDBVectorService()
            self.retriever = RRFRetriever(vector_service=self.vector_service)
            self.memory_manager = create_memory_manager()
            self.agent = create_react_agent_instance(
                retriever=self.retriever,
                memory_manager=self.memory_manager
            )
            logger.info(f"Created chat session {session_id} for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to initialize chat session {session_id}: {e}")
            raise
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get session information.
        
        Returns:
            Dictionary containing session metadata
        """
        return {
            'user_id': self.user_id,
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }



# =============================================================================
# MAIN CHAT INTERFACE CLASS
# =============================================================================

class ChatInterface:
    """Main chat interface for handling user interactions with the ReAct agent.
    
    This class manages multiple chat sessions, handles file uploads, and provides
    both synchronous and streaming chat capabilities.
    """
    
    def __init__(self):
        """Initialize chat interface with empty session storage."""
        self.sessions: Dict[str, ChatSession] = {}
        logger.info("Chat interface initialized")
    
    def _get_session_key(self, user_id: str, user_session: str) -> str:
        """Generate session key from user ID and session ID.
        
        Args:
            user_id: User identifier
            user_session: Session identifier
            
        Returns:
            Combined session key
        """
        return f"{user_id}{SESSION_KEY_SEPARATOR}{user_session}"
    
    def _get_or_create_session(self, user_id: str, user_session: str) -> ChatSession:
        """Get existing session or create new one.
        
        Args:
            user_id: User identifier
            user_session: Session identifier
            
        Returns:
            ChatSession instance
            
        Raises:
            Exception: If session creation fails
        """
        if not user_id or not user_session:
            raise ValueError("User ID and session ID cannot be empty")
        
        session_key = self._get_session_key(user_id, user_session)
        
        if session_key not in self.sessions:
            self.sessions[session_key] = ChatSession(user_id, user_session)
        
        session = self.sessions[session_key]
        session.update_activity()
        return session
    
    def _validate_query(self, user_query: str) -> bool:
        """Validate user query.
        
        Args:
            user_query: User input query
            
        Returns:
            True if query is valid, False otherwise
        """
        return bool(user_query and user_query.strip())
    
    def _process_file_uploads(
        self,
        session: ChatSession,
        file_upload_list: Optional[List[Dict[str, str]]]
    ) -> List[str]:
        """Process uploaded files and pass file list to agent.
        
        Args:
            session: Chat session instance
            file_upload_list: List of file upload dictionaries
            
        Returns:
            List of processing result messages
        """
        if not file_upload_list:
            return []
        
        results = []
        try:
            session.agent.file_dict_list = file_upload_list
            success_msg = f"✅ File list passed to agent ({len(file_upload_list)} files)"
            results.append(success_msg)
            logger.info(f"Passed {len(file_upload_list)} files to agent for session {session.session_id}")
        except Exception as e:
            error_msg = f"❌ Error passing file list to agent: {str(e)}"
            results.append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def _create_error_response(
        self,
        error_msg: str,
        user_id: str,
        user_session: str,
        upload_results: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create standardized error response.
        
        Args:
            error_msg: Error message
            user_id: User identifier
            user_session: Session identifier
            upload_results: File upload results
            
        Returns:
            Error response dictionary
        """
        return {
            "success": False,
            "error": error_msg,
            "user_id": user_id,
            "session_id": user_session,
            "upload_results": upload_results or [],
            "timestamp": datetime.now().isoformat()
        }
    
    def chat(
        self,
        user_id: str,
        user_session: str,
        user_query: str,
        file_upload_list: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Process synchronous chat request and get agent response.
        
        Args:
            user_id: Unique identifier for the user
            user_session: Unique identifier for the session
            user_query: User's input query
            file_upload_list: Optional list of uploaded files
            
        Returns:
            Dictionary containing response data or error information
        """
        try:
            # Get or create session
            session = self._get_or_create_session(user_id, user_session)
            
            # Process file uploads if provided
            upload_results = self._process_file_uploads(session, file_upload_list)
            
            # Validate user query
            if not self._validate_query(user_query):
                return self._create_error_response(
                    "Empty query provided",
                    user_id,
                    user_session,
                    upload_results
                )
            
            # Get agent response
            agent_response = session.agent.run(user_query)
            
            # Create success response
            response = {
                "success": True,
                "user_id": user_id,
                "session_id": user_session,
                "query": user_query,
                "response": agent_response,
                "upload_results": upload_results,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Processed chat for user {user_id}, session {user_session}")
            return response
            
        except Exception as e:
            error_msg = f"Error processing chat request: {str(e)}"
            logger.error(error_msg)
            return self._create_error_response(error_msg, user_id, user_session)
    
    def chat_stream(
        self,
        user_id: str,
        user_session: str,
        user_query: str,
        file_upload_list: Optional[List[Dict[str, str]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Process streaming chat request and yield agent response chunks.
        
        Args:
            user_id: Unique identifier for the user
            user_session: Unique identifier for the session
            user_query: User's input query
            file_upload_list: Optional list of uploaded files
            
        Yields:
            Dictionary containing response chunks or error information
        """
        try:
            # Get or create session
            session = self._get_or_create_session(user_id, user_session)
            
            # Process file uploads if provided
            upload_results = self._process_file_uploads(session, file_upload_list)
            
            # Validate user query
            if not self._validate_query(user_query):
                yield {
                    "success": False,
                    "error": "Empty query provided",
                    "upload_results": upload_results,
                    "type": "error"
                }
                return
            
            # Yield initial response with metadata
            yield {
                "success": True,
                "user_id": user_id,
                "session_id": user_session,
                "query": user_query,
                "upload_results": upload_results,
                "timestamp": datetime.now().isoformat(),
                "type": "start"
            }
            
            # Stream agent response
            try:
                for chunk in session.agent.run_stream(user_query):
                    yield {
                        "success": True,
                        "type": "chunk",
                        "content": chunk
                    }
            except Exception as stream_error:
                logger.error(f"Streaming error for session {user_session}: {stream_error}")
                yield {
                    "success": False,
                    "type": "error",
                    "error": f"Streaming error: {str(stream_error)}"
                }
                return
            
            # Yield completion signal
            yield {
                "success": True,
                "type": "end",
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Processed streaming chat for user {user_id}, session {user_session}")
            
        except Exception as e:
            error_msg = f"Error processing streaming chat request: {str(e)}"
            logger.error(error_msg)
            yield {
                "success": False,
                "error": error_msg,
                "user_id": user_id,
                "session_id": user_session,
                "timestamp": datetime.now().isoformat(),
                "type": "error"
            }
    
    def get_session_count(self) -> int:
        """Get the number of active sessions.
        
        Returns:
            Number of active sessions
        """
        return len(self.sessions)
    
    def cleanup_inactive_sessions(self, max_inactive_hours: int = 24) -> int:
        """Clean up inactive sessions.
        
        Args:
            max_inactive_hours: Maximum hours of inactivity before cleanup
            
        Returns:
            Number of sessions cleaned up
        """
        current_time = datetime.now()
        inactive_sessions = []
        
        for session_key, session in self.sessions.items():
            hours_inactive = (current_time - session.last_activity).total_seconds() / 3600
            if hours_inactive > max_inactive_hours:
                inactive_sessions.append(session_key)
        
        for session_key in inactive_sessions:
            del self.sessions[session_key]
        
        if inactive_sessions:
            logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
        
        return len(inactive_sessions)


# =============================================================================
# GLOBAL INSTANCE AND CONVENIENCE FUNCTIONS
# =============================================================================

# Global chat interface instance
chat_interface = ChatInterface()


def process_chat_request(
    user_id: str,
    user_session: str,
    user_query: str,
    file_upload_list: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Convenience function to process synchronous chat requests.
    
    Args:
        user_id: Unique identifier for the user
        user_session: Unique identifier for the session
        user_query: User's input query
        file_upload_list: Optional list of uploaded files
        
    Returns:
        Dictionary containing response data or error information
    """
    return chat_interface.chat(user_id, user_session, user_query, file_upload_list)


def process_chat_request_stream(
    user_id: str,
    user_session: str,
    user_query: str,
    file_upload_list: Optional[List[Dict[str, str]]] = None
) -> Generator[Dict[str, Any], None, None]:
    """Convenience function to process streaming chat requests.
    
    Args:
        user_id: Unique identifier for the user
        user_session: Unique identifier for the session
        user_query: User's input query
        file_upload_list: Optional list of uploaded files
        
    Yields:
        Dictionary containing response chunks or error information
    """
    yield from chat_interface.chat_stream(user_id, user_session, user_query, file_upload_list)


def get_chat_interface_stats() -> Dict[str, Any]:
    """Get statistics about the chat interface.
    
    Returns:
        Dictionary containing interface statistics
    """
    return {
        "active_sessions": chat_interface.get_session_count(),
        "interface_initialized": True
    }


def cleanup_chat_sessions(max_inactive_hours: int = 24) -> int:
    """Clean up inactive chat sessions.
    
    Args:
        max_inactive_hours: Maximum hours of inactivity before cleanup
        
    Returns:
        Number of sessions cleaned up
    """
    return chat_interface.cleanup_inactive_sessions(max_inactive_hours)


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def _test_chat_interface() -> None:
    """Test function for the chat interface."""
    print("Testing Chat Interface...")
    
    try:
        # Test static response
        print("\n1. Testing synchronous response...")
        response = process_chat_request(
            user_id="test_user",
            user_session="test_session_1",
            user_query="Hello, how are you?"
        )
        
        if response.get("success"):
            print(f"✅ Static Response: {response.get('response', 'No response')}")
        else:
            print(f"❌ Static Response Error: {response.get('error')}")
        
        # Test streaming response
        print("\n2. Testing streaming response...")
        print("Streaming Response: ", end="", flush=True)
        
        chunk_count = 0
        for chunk in process_chat_request_stream(
            user_id="test_user",
            user_session="test_session_2",
            user_query="What can you help me with?"
        ):
            if chunk.get("success") and chunk.get("type") == "chunk":
                print(chunk.get("content", ""), end="", flush=True)
                chunk_count += 1
            elif chunk.get("type") == "error":
                print(f"\n❌ Streaming Error: {chunk.get('error')}")
                break
        
        print(f"\n✅ Received {chunk_count} chunks")
        
        # Test interface statistics
        print("\n3. Testing interface statistics...")
        stats = get_chat_interface_stats()
        print(f"Active sessions: {stats['active_sessions']}")
        
        # Test session cleanup
        print("\n4. Testing session cleanup...")
        cleaned = cleanup_chat_sessions(0)  # Clean all sessions for testing
        print(f"Cleaned up {cleaned} sessions")
        
        print("\n✅ Chat interface tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing chat interface: {e}")
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    _test_chat_interface()