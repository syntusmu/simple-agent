"""Simple Memory Management for Agent Chat History.

This module provides a streamlined memory management system using
LangChain's ConversationSummaryBufferMemory for chat history.

Classes:
    TokenCountingChatOpenAI: ChatOpenAI subclass with accurate token counting
    SimpleMemoryManager: Main memory management class with conversation buffer

Functions:
    create_memory_manager: Factory function for creating memory managers
    _test_memory: Testing function for memory functionality

Constants:
    DEFAULT_ENCODING: Default tiktoken encoding
    FALLBACK_TOKENIZER: Fallback tokenizer name
    MESSAGE_OVERHEAD_TOKENS: Token overhead per message
    CONVERSATION_OVERHEAD_TOKENS: Token overhead per conversation
    DEFAULT_MAX_TOKEN_LIMIT: Default maximum token limit
    DEFAULT_TEMPERATURE: Default LLM temperature
    DEFAULT_MAX_TOKENS: Default maximum tokens for LLM
    DEFAULT_MODEL: Default model name
"""

import logging
from typing import Optional, Dict, Any, List, Union

# LangChain memory imports
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI

# Import LLM interface for DeepSeek
from ..service.llm.llm_interface import create_llm

# Token counting
import tiktoken
from transformers import AutoTokenizer

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_ENCODING = "cl100k_base"
FALLBACK_TOKENIZER = "gpt2"
MESSAGE_OVERHEAD_TOKENS = 3
CONVERSATION_OVERHEAD_TOKENS = 3
DEFAULT_MAX_TOKEN_LIMIT = 2000
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MODEL = 'deepseek-v3'


# =============================================================================
# TOKEN COUNTING CHAT OPENAI CLASS
# =============================================================================

class TokenCountingChatOpenAI(ChatOpenAI):
    """
    ChatOpenAI subclass that provides accurate token counting using tiktoken.
    
    This class extends ChatOpenAI to add the get_num_tokens_from_messages method
    that's required by ConversationSummaryBufferMemory.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with tokenizer for accurate token counting.
        
        Args:
            *args: Variable length argument list for ChatOpenAI
            **kwargs: Arbitrary keyword arguments for ChatOpenAI
        """
        super().__init__(*args, **kwargs)
        
        # Initialize tokenizer (store as private attribute to avoid Pydantic issues)
        self._tokenizer = None
        self._tokenizer_type = None
        self._tiktoken_encoding = None
        
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self) -> None:
        """Initialize the most appropriate tokenizer available.
        
        Tries tiktoken first (preferred for accuracy), falls back to transformers.
        
        Raises:
            No exceptions - handles all errors internally with fallbacks
        """
        try:
            # Use cl100k_base encoding (standard tokenizer used by GPT-3.5/GPT-4)
            # This provides good token counting for most models including DeepSeek
            self._tiktoken_encoding = tiktoken.get_encoding(DEFAULT_ENCODING)
            self._tokenizer_type = "tiktoken"
            logger.debug(f"Initialized tiktoken with {DEFAULT_ENCODING} encoding for token counting")
        except ImportError as e:
            logger.warning(f"tiktoken not available: {e}. Falling back to transformers.")
            self._initialize_fallback_tokenizer()
        except Exception as e:
            logger.error(f"Failed to initialize tiktoken: {e}. Falling back to transformers.")
            self._initialize_fallback_tokenizer()
    
    def _initialize_fallback_tokenizer(self) -> None:
        """Initialize fallback tokenizer using transformers.
        
        Uses GPT-2 tokenizer as a reasonable fallback when tiktoken is unavailable.
        
        Raises:
            No exceptions - handles all errors internally
        """
        try:
            # Fallback to GPT-2 tokenizer from transformers
            self._tokenizer = AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER)
            self._tokenizer_type = "transformers"
            logger.info(f"Initialized {FALLBACK_TOKENIZER} tokenizer for token counting")
        except Exception as e:
            logger.error(f"Failed to initialize any tokenizer: {e}")
            self._tokenizer = None
            self._tokenizer_type = None
    
    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """
        Count tokens in messages using available tokenizer.
        
        Args:
            messages: List of messages to count tokens for
            
        Returns:
            Token count
            
        Raises:
            RuntimeError: If no tokenizer is available
        """
        if not self._tokenizer_type:
            raise RuntimeError(
                "No tokenizer available. Please install tiktoken (pip install tiktoken) "
                "or transformers (pip install transformers)"
            )
        
        return self._count_tokens(messages)
    
    def _count_tokens(self, messages: List[BaseMessage]) -> int:
        """Count tokens in messages using the available tokenizer."""
        total_tokens = 0
        
        for message in messages:
            content, role = self._extract_message_content_and_role(message)
            message_tokens = self._count_message_tokens(content, role)
            total_tokens += message_tokens
        
        # Add conversation overhead
        total_tokens += CONVERSATION_OVERHEAD_TOKENS
        
        logger.debug(
            f"Counted {total_tokens} tokens for {len(messages)} messages "
            f"using {self._tokenizer_type}"
        )
        return total_tokens
    
    def _extract_message_content_and_role(self, message: Union[BaseMessage, dict, str]) -> tuple[str, str]:
        """Extract content and role from a message object.
        
        Args:
            message: Message object (BaseMessage, dict, or string)
            
        Returns:
            Tuple of (content, role) strings
        """
        if hasattr(message, 'content'):
            content = message.content or ""
            role = getattr(message, 'type', 'user')
        elif isinstance(message, dict):
            content = message.get('content', '')
            role = message.get('role', 'user')
        else:
            content = str(message)
            role = 'user'
        
        return content, role
    
    def _count_message_tokens(self, content: str, role: str) -> int:
        """Count tokens for a single message's content and role.
        
        Args:
            content: Message content string
            role: Message role string
            
        Returns:
            Total token count including overhead
            
        Raises:
            RuntimeError: If tokenizer type is unknown
        """
        if self._tokenizer_type == "tiktoken":
            content_tokens = len(self._tiktoken_encoding.encode(content))
            role_tokens = len(self._tiktoken_encoding.encode(role))
        elif self._tokenizer_type == "transformers":
            content_tokens = len(self._tokenizer.encode(content))
            role_tokens = len(self._tokenizer.encode(role))
        else:
            raise RuntimeError(f"Unknown tokenizer type: {self._tokenizer_type}")
        
        # Add message overhead (includes formatting tokens)
        return content_tokens + role_tokens + MESSAGE_OVERHEAD_TOKENS
    

# =============================================================================
# SIMPLE MEMORY MANAGER CLASS
# =============================================================================

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
    
    def __init__(self, max_token_limit: int = DEFAULT_MAX_TOKEN_LIMIT, llm=None):
        """
        Initialize simple memory manager with ConversationSummaryBufferMemory.
        
        Args:
            max_token_limit: Maximum number of tokens to keep in buffer before summarizing
            llm: Language model for summarization (default: DeepSeek with tiktoken)
            
        Raises:
            Exception: If LLM creation or memory initialization fails
        """
        self.max_token_limit = max_token_limit
        self.llm = llm or self._create_default_llm()
        self.memory = self._create_memory()
        
        logger.info(f"Memory manager initialized with {max_token_limit} token limit")
    
    def _create_default_llm(self) -> TokenCountingChatOpenAI:
        """Create default LLM instance for memory management.
        
        Returns:
            Configured TokenCountingChatOpenAI instance
            
        Raises:
            Exception: If configuration loading or LLM creation fails
        """
        from ..utils.common import get_config_value
        
        api_key = get_config_value('llm', 'api_key')
        base_url = get_config_value('llm', 'base_url')
        model = get_config_value('llm', 'model', default=DEFAULT_MODEL)
        
        return TokenCountingChatOpenAI(
            model=model,
            openai_api_key=api_key,
            base_url=base_url,
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS
        )
    
    def _create_memory(self) -> ConversationSummaryBufferMemory:
        """Create and configure conversation memory.
        
        Returns:
            Configured ConversationSummaryBufferMemory instance
        """
        return ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=self.max_token_limit,
            return_messages=True,
            memory_key="chat_history"
        )
    
    def add_message(self, human_input: str, ai_output: str) -> None:
        """
        Add a conversation turn to memory.
        
        Args:
            human_input: User's message
            ai_output: AI's response
            
        Raises:
            Exception: If memory save operation fails
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
    
    def clear_memory(self) -> None:
        """Clear all memory content.
        
        Raises:
            Exception: If memory clear operation fails
        """
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
            
        Raises:
            Exception: If memory variable loading fails
        """
        memory_vars = self.get_memory_variables()
        return memory_vars.get('chat_history', '')
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics and information.
        
        Returns:
            Dictionary with memory statistics including message count,
            token limit, and tokenizer type
        """
        history = self.get_chat_history()
        return {
            'message_count': len(history),
            'max_token_limit': self.max_token_limit,
            'tokenizer_type': getattr(self.llm, '_tokenizer_type', 'unknown'),
            'memory_key': self.memory.memory_key,
            'llm_model': getattr(self.llm, 'model_name', 'unknown')
        }


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_memory_manager(max_token_limit: int = DEFAULT_MAX_TOKEN_LIMIT, llm=None) -> SimpleMemoryManager:
    """
    Factory function to create a memory manager.
    
    Args:
        max_token_limit: Maximum number of tokens to keep in buffer
        llm: Language model for summarization (default: DeepSeek)
        
    Returns:
        Configured SimpleMemoryManager instance
    """
    return SimpleMemoryManager(max_token_limit=max_token_limit, llm=llm)


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def _test_memory() -> None:
    """Test the simple memory manager.
    
    Comprehensive test suite covering:
    - Memory manager creation
    - Conversation turn addition
    - History retrieval and statistics
    - Memory variables and string formatting
    - Error handling and logging
    """
    print("🧪 Testing Simple Memory Manager...")
    
    try:
        # Create memory manager with small limit for testing
        test_token_limit = 500
        print(f"📝 Creating memory manager with {test_token_limit} token limit...")
        memory_manager = create_memory_manager(max_token_limit=test_token_limit)
        
        # Test conversation turns
        test_conversations = [
            ("Hello, how are you?", "I'm doing well, thank you!"),
            ("What's the weather like?", "I don't have access to current weather data."),
            ("Tell me a joke", "Why don't scientists trust atoms? Because they make up everything!")
        ]
        
        print(f"💬 Adding {len(test_conversations)} conversation turns...")
        for i, (human_input, ai_output) in enumerate(test_conversations, 1):
            memory_manager.add_message(human_input, ai_output)
            print(f"   {i}. Added: '{human_input[:30]}...' -> '{ai_output[:30]}...'")
        
        # Verify functionality
        history = memory_manager.get_chat_history()
        print(f"📊 Chat history has {len(history)} messages")
        
        memory_vars = memory_manager.get_memory_variables()
        print(f"🔧 Memory variables: {list(memory_vars.keys())}")
        
        history_str = memory_manager.get_history_string()
        print(f"📜 History string length: {len(history_str)} characters")
        
        # Test new memory stats function
        stats = memory_manager.get_memory_stats()
        print(f"📈 Memory stats: {stats}")
        
        print("✅ Memory manager test completed successfully!")
        
    except Exception as e:
        logger.error(f"Memory manager test failed: {e}")
        print(f"❌ Memory manager test failed: {e}")
        raise


if __name__ == "__main__":
    _test_memory()