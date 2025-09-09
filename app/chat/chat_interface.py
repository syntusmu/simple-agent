"""Chat Interface for ReAct Agent with Session Management."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..agent.agent import create_react_agent_instance
from ..agent.memory import create_memory_manager
from ..service.rag.retriever import RRFRetriever
from ..service.vector.chromadb import ChromaDBVectorService

logger = logging.getLogger(__name__)


class ChatSession:
    """Represents a chat session with memory and agent instance."""
    
    def __init__(self, user_id: str, session_id: str):
        """Initialize chat session."""
        self.user_id = user_id
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
        # Initialize services for this session
        self.vector_service = ChromaDBVectorService(
            collection_name=f"user_{user_id}_session_{session_id}"
        )
        self.retriever = RRFRetriever(vector_service=self.vector_service)
        self.memory_manager = create_memory_manager()
        self.agent = create_react_agent_instance(
            retriever=self.retriever,
            memory_manager=self.memory_manager
        )
        
        logger.info(f"Created chat session {session_id} for user {user_id}")
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()


class ChatInterface:
    """Chat Interface for handling user interactions with the ReAct agent."""
    
    def __init__(self):
        """Initialize chat interface."""
        self.sessions: Dict[str, ChatSession] = {}
        logger.info("Chat interface initialized")
    
    def _get_or_create_session(self, user_id: str, user_session: str) -> ChatSession:
        """Get existing session or create new one."""
        session_key = f"{user_id}_{user_session}"
        
        if session_key not in self.sessions:
            self.sessions[session_key] = ChatSession(user_id, user_session)
        
        session = self.sessions[session_key]
        session.update_activity()
        return session
    
    def _process_file_uploads(self, session: ChatSession, file_upload_list: List[Dict[str, str]]) -> List[str]:
        """Process uploaded files and pass file list to agent."""
        if not file_upload_list:
            return []
        
        results = []
        try:
            session.agent.file_dict_list = file_upload_list
            results.append(f"✅ File list passed to agent ({len(file_upload_list)} files)")
            logger.info(f"Passed {len(file_upload_list)} files to agent for session {session.session_id}")
        except Exception as e:
            error_msg = f"❌ Error passing file list to agent: {str(e)}"
            results.append(error_msg)
            logger.error(error_msg)
        
        return results
    
    def chat(self, user_id: str, user_session: str, user_query: str, 
             file_upload_list: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process chat request and get LLM response."""
        try:
            session = self._get_or_create_session(user_id, user_session)
            
            # Process file uploads if provided
            upload_results = []
            if file_upload_list:
                upload_results = self._process_file_uploads(session, file_upload_list)
            
            # Validate user query
            if not user_query or not user_query.strip():
                return {
                    "success": False,
                    "error": "Empty query provided",
                    "upload_results": upload_results
                }
            
            # Get agent response
            agent_response = session.agent.run(user_query)
            
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
            return {
                "success": False,
                "error": error_msg,
                "user_id": user_id,
                "session_id": user_session,
                "timestamp": datetime.now().isoformat()
            }


# Global chat interface instance
chat_interface = ChatInterface()


def process_chat_request(user_id: str, user_session: str, user_query: str,
                        file_upload_list: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Convenience function to process chat requests."""
    return chat_interface.chat(user_id, user_session, user_query, file_upload_list)


def _test_chat_interface():
    """Test function for the chat interface."""
    print("Testing Chat Interface...")
    
    try:
        response = process_chat_request(
            user_id="test_user",
            user_session="test_session_1",
            user_query="Hello, how are you?"
        )
        print(f"Response: {response['response']}")
        
    except Exception as e:
        logger.error(f"Error testing chat interface: {e}")


if __name__ == "__main__":
    _test_chat_interface()