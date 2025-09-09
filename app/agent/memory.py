"""
Simple Memory Management for Agent Chat History.

This module provides a streamlined memory management system using
LangChain's ConversationSummaryBufferMemory for chat history.
"""

import logging
from typing import Optional, Dict, Any, List

# LangChain memory imports
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI

# Import LLM interface for DeepSeek
from ..service.llm.llm_interface import create_llm

# Token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# Alternative tokenizer fallback
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)


class TokenCountingChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI subclass that provides accurate token counting using tiktoken.
    
    This class extends ChatOpenAI to add the get_num_tokens_from_messages method
    that's required by ConversationSummaryBufferMemory.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with tokenizer for accurate token counting."""
        super().__init__(*args, **kwargs)
        
        # Initialize tokenizer (store as private attribute to avoid Pydantic issues)
        self._tokenizer = None
        self._tokenizer_type = None
        
        if TIKTOKEN_AVAILABLE:
            # Use cl100k_base encoding (standard tokenizer used by GPT-3.5/GPT-4)
            # This provides good token counting for most models including DeepSeek
            self._tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
            self._tokenizer_type = "tiktoken"
            logger.debug("Initialized tiktoken with cl100k_base encoding for token counting")
        elif TRANSFORMERS_AVAILABLE:
            # Download and use a suitable tokenizer from transformers
            try:
                # Use GPT-2 tokenizer as a good general-purpose tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self._tokenizer_type = "transformers"
                logger.info("Downloaded and initialized GPT-2 tokenizer for token counting")
            except Exception as e:
                logger.error(f"Failed to download tokenizer: {e}")
                self._tokenizer = None
        
        if self._tokenizer is None and not TIKTOKEN_AVAILABLE:
            logger.warning("No tokenizer available - neither tiktoken nor transformers is installed")
    
    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """
        Count tokens in messages using available tokenizer.
        
        Args:
            messages: List of messages to count tokens for
            
        Returns:
            Token count
        """
        if self._tokenizer_type == "tiktoken":
            return self._count_tokens_tiktoken(messages)
        elif self._tokenizer_type == "transformers":
            return self._count_tokens_transformers(messages)
        else:
            raise RuntimeError("No tokenizer available. Please install tiktoken (pip install tiktoken) or transformers (pip install transformers)")
    
    def _count_tokens_tiktoken(self, messages: List[BaseMessage]) -> int:
        """Count tokens using tiktoken."""
        total_tokens = 0
        
        for message in messages:
            # Get message content and role
            if hasattr(message, 'content'):
                content = message.content or ""
                role = getattr(message, 'type', 'user')
            elif isinstance(message, dict):
                content = message.get('content', '')
                role = message.get('role', 'user')
            else:
                content = str(message)
                role = 'user'
            
            # Count tokens using tiktoken
            content_tokens = len(self._tiktoken_encoding.encode(content))
            role_tokens = len(self._tiktoken_encoding.encode(role))
            
            # Add message overhead
            message_tokens = content_tokens + role_tokens + 3
            total_tokens += message_tokens
        
        # Add conversation overhead
        total_tokens += 3
        
        logger.debug(f"Counted {total_tokens} tokens for {len(messages)} messages using tiktoken")
        return total_tokens
    
    def _count_tokens_transformers(self, messages: List[BaseMessage]) -> int:
        """Count tokens using transformers tokenizer."""
        total_tokens = 0
        
        for message in messages:
            # Get message content and role
            if hasattr(message, 'content'):
                content = message.content or ""
                role = getattr(message, 'type', 'user')
            elif isinstance(message, dict):
                content = message.get('content', '')
                role = message.get('role', 'user')
            else:
                content = str(message)
                role = 'user'
            
            # Count tokens using transformers tokenizer
            content_tokens = len(self._tokenizer.encode(content))
            role_tokens = len(self._tokenizer.encode(role))
            
            # Add message overhead
            message_tokens = content_tokens + role_tokens + 3
            total_tokens += message_tokens
        
        # Add conversation overhead
        total_tokens += 3
        
        logger.debug(f"Counted {total_tokens} tokens for {len(messages)} messages using transformers tokenizer")
        return total_tokens
    


class SimpleMemoryManager:
    """
    Simple memory management using ConversationSummaryBufferMemory.
    
    Features:
    - Conversation summary buffer for efficient memory usage
    - Automatic summarization of older conversations
    - Configurable token limits and message counts
    - Direct integration with LangChain memory system
    - Clean interface for chat history management
    """
    
    def __init__(self, max_token_limit: int = 2000, llm=None):
        """
        Initialize simple memory manager with ConversationSummaryBufferMemory.
        
        Args:
            max_token_limit: Maximum number of tokens to keep in buffer before summarizing
            llm: Language model for summarization (default: DeepSeek with tiktoken)
        """
        self.max_token_limit = max_token_limit
        
        if llm is None:
            # Create a TokenCountingChatOpenAI instance for DeepSeek
            from ..utils.common import get_config_value
            api_key = get_config_value('deepseek', 'api_key')
            base_url = get_config_value('deepseek', 'base_url')
            model = get_config_value('deepseek', 'model', default='deepseek-v3')
            
            self.llm = TokenCountingChatOpenAI(
                model=model,
                openai_api_key=api_key,
                base_url=base_url,
                temperature=0,
                max_tokens=2048
            )
        else:
            self.llm = llm
        
        # Initialize conversation summary buffer memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )
        
        logger.info(f"Memory manager initialized with {max_token_limit} token limit")
    
    def add_message(self, human_input: str, ai_output: str):
        """
        Add a conversation turn to memory.
        
        Args:
            human_input: User's message
            ai_output: AI's response
        """
        self.memory.save_context(
            {"input": human_input},
            {"output": ai_output}
        )
        
        logger.debug("Added conversation turn to memory")
    
    def get_chat_history(self) -> List[BaseMessage]:
        """
        Get current chat history messages.
        
        Returns:
            List of chat messages
        """
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear all memory content."""
        self.memory.clear()
        logger.info("Memory cleared")
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """
        Get memory variables for use in prompts.
        
        Returns:
            Dictionary with memory variables
        """
        return self.memory.load_memory_variables({})
    
    def get_history_string(self) -> str:
        """
        Get chat history as a formatted string.
        
        Returns:
            Formatted string of conversation history
        """
        memory_vars = self.get_memory_variables()
        return memory_vars.get('chat_history', '')


def create_memory_manager(max_token_limit: int = 2000, llm=None) -> SimpleMemoryManager:
    """
    Factory function to create a memory manager.
    
    Args:
        max_token_limit: Maximum number of tokens to keep in buffer
        llm: Language model for summarization (default: DeepSeek)
        
    Returns:
        Configured SimpleMemoryManager instance
    """
    return SimpleMemoryManager(max_token_limit=max_token_limit, llm=llm)


# Test function
def _test_memory():
    """Test the simple memory manager."""
    print("Testing Simple Memory Manager...")
    
    try:
        # Create memory manager
        memory_manager = create_memory_manager(max_token_limit=500)  # Small limit for testing
        
        # Add some conversation turns
        memory_manager.add_message("Hello, how are you?", "I'm doing well, thank you!")
        memory_manager.add_message("What's the weather like?", "I don't have access to current weather data.")
        memory_manager.add_message("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!")
        
        # Get chat history
        history = memory_manager.get_chat_history()
        print(f"Chat history has {len(history)} messages")
        
        # Get memory variables
        memory_vars = memory_manager.get_memory_variables()
        print(f"Memory variables: {list(memory_vars.keys())}")
        
        # Get history string
        history_str = memory_manager.get_history_string()
        print(f"History string length: {len(history_str)}")
        
        print("✅ Memory manager test completed successfully!")
        
    except Exception as e:
        print(f"❌ Memory manager test failed: {e}")


if __name__ == "__main__":
    _test_memory()