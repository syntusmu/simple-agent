"""
ReAct Agent implementation using LangChain create_react_agent.

This module provides a ReAct agent with RAG retrieval and Excel analysis capabilities.
"""

import logging
import re
import os
from pathlib import Path
from typing import Optional, List

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from ..service.llm.llm_interface import create_llm
from ..service.rag.retriever import RRFRetriever
from ..service.prompt.prompt import create_react_prompt
from ..tools.data_analyzer import analyze_excel_with_pandas
from ..tools.contextual_analyzer import analyze_document_contextually
from ..tools.retriever_tool import DocumentRetrieverWrapper
# from ..tools.postgresql_tool import query_postgresql_database
from .memory import SimpleMemoryManager, create_memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReActAgent:
    """
    ReAct Agent using LangChain create_react_agent.
    
    Features:
    - RAG document retrieval using RRFRetriever
    - Excel/CSV data analysis using pandas agent
    - Configurable LLM backend
    """
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __init__(
        self,
        llm=None,
        retriever: Optional[RRFRetriever] = None,
        memory_manager: Optional[SimpleMemoryManager] = None
    ):
        """
        Initialize ReAct Agent.
        
        Args:
            llm: Language model instance (default: DeepSeek via centralized LLM service)
            retriever: RRF retriever instance
            memory_manager: Memory manager for chat history
        """
        # Initialize LLM using centralized service
        self.llm = llm or create_llm()
        
        # Initialize services
        self.retriever = retriever or RRFRetriever()
        self.memory_manager = memory_manager or create_memory_manager()
        
        # Initialize file dictionary list for processing uploaded files
        self.file_dict_list = []
        
        # Setup tools and agent
        self._setup_tools()
        self._setup_agent()
        
        logger.info("ReAct Agent initialized successfully")
    
    # =========================================================================
    # SETUP METHODS
    # =========================================================================
    
    def _setup_tools(self) -> None:
        """Setup agent tools."""
        # Create retriever wrapper for LangChain compatibility
        retriever_wrapper = DocumentRetrieverWrapper(self.retriever)
        
        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever=retriever_wrapper,
            name="document_retriever",
            description=(
                "WHEN TO USE: Search for information across previously processed documents in the knowledge base.\n"
                "PURPOSE: Find relevant content from stored documents using semantic similarity search.\n"
                "INPUT: Natural language search query describing the information you need.\n"
                "OUTPUT: Returns relevant document chunks with SOURCE FILENAMES prominently displayed as '📄 **Source: filename**'.\n"
                "SCOPE: Searches across all documents that have been previously processed and stored in the vector database.\n"
                "CITATION REQUIREMENT: Always include the source filename in your final answer when using retrieved information.\n"
                "COMPLEMENTARY USE: Use alongside list_file and analyzer tools for comprehensive file analysis.\n"
                "FALLBACK: If no tools can answer the question, answer directly without using tools.\n"
                "EXAMPLES: 'financial performance metrics', 'project timeline details', 'risk assessment findings'"
            )
        )
        
        # Create data analyzer tool
        data_analyzer_tool = Tool(
            name="data_analyzer",
            func=self._excel_analyzer_wrapper,
            description=(
                "WHEN TO USE: Analyze Excel/CSV files with numerical data, tables, spreadsheets, or structured data.\n"
                "PURPOSE: Perform statistical analysis, data visualization, generate insights, answer quantitative questions.\n"
                "CRITICAL PREREQUISITE: When files are uploaded, ALWAYS call list_file tool FIRST to get file info.\n"
                "INPUT FORMAT: Use file_path from list_file output: 'data/filename.xlsx' or 'data/filename.xlsx|your_question'\n"
                "SUPPORTED FILES: .xlsx, .xls, .csv (Excel and CSV data files)\n"
                "MANDATORY WORKFLOW: list_file → data_analyzer with file_path\n"
                "FALLBACK STRATEGY: If contextual_analyzer cannot handle Excel data analysis, use this pandas-based tool instead.\n"
                "CAPABILITIES: Data aggregation, statistical calculations, trend analysis, filtering, sorting, pivot tables.\n"
                "GENERAL FALLBACK: If no tools can answer the question, answer directly without using tools.\n"
                "EXAMPLE: 'data/sales_data.csv|analyze monthly revenue trends and identify top 5 products'"
            )
        )
        
        # Create list_file tool
        list_file_tool = Tool(
            name="list_file",
            func=self._list_file_wrapper,
            description=(
                "WHEN TO USE: Check what files are available in the current session before any file analysis.\n"
                "PURPOSE: Discover uploaded files and get correct file paths for subsequent analysis tools.\n"
                "INPUT: 'list' to show all files, or partial filename to search specific files.\n"
                "OUTPUT: Returns file_path values (e.g. 'data/filename.xlsx') needed for analyzer tools.\n"
                "CRITICAL IMPORTANCE: MANDATORY FIRST STEP for any file-related question or analysis.\n"
                "WORKFLOW INTEGRATION: list_file → choose appropriate analyzer (data_analyzer for Excel/CSV, contextual_analyzer for documents).\n"
                "FALLBACK: If no tools can answer the question, answer directly without using tools.\n"
                "EXAMPLES: 'list' (show all), 'sales' (find files containing 'sales'), 'report.pdf' (find specific file)"
            )
        )
        
        # Create contextual analyzer tool
        contextual_tool = Tool(
            name="contextual_analyzer",
            func=self._contextual_analyzer_wrapper,
            description=(
                "WHEN TO USE: Analyze text-based documents (PDF, Word, TXT, Markdown) for content understanding.\n"
                "PURPOSE: Extract insights, summarize content, answer qualitative questions about documents.\n"
                "CRITICAL PREREQUISITE: When files are uploaded, ALWAYS call list_file tool FIRST to get file info.\n"
                "INPUT FORMAT: 'file_path' or 'file_path|query'\n"
                "SUPPORTED FILES: .pdf, .docx, .txt, .md, .markdown (text-based documents)\n"
                "NOT SUITABLE FOR: Excel/CSV data analysis - use data_analyzer tool for numerical/tabular data.\n"
                "MANDATORY WORKFLOW: list_file → contextual_analyzer (with file path from list_file)\n"
                "EXCEL LIMITATION: If you encounter Excel files that need content analysis (not data analysis), recommend using data_analyzer instead.\n"
                "CAPABILITIES: Text summarization, content extraction, qualitative analysis, document Q&A.\n"
                "FALLBACK: If no tools can answer the question, answer directly without using tools.\n"
                "EXAMPLE: '/path/to/document.pdf|summarize key findings and recommendations'"
            )
        )
        
        # Create PostgreSQL database tool
        # postgresql_tool = Tool(
        #     name="postgresql_database",
        #     func=self._postgresql_wrapper,
        #     description=(
        #         "WHEN TO USE: Query PostgreSQL database with natural language or analyze database structure.\n"
        #         "PURPOSE: Execute database queries, get schema information, analyze data in PostgreSQL.\n"
        #         "INPUT FORMAT: 'connection_string|query' where connection_string is PostgreSQL URI.\n"
        #         "CONNECTION: postgresql://username:password@host:port/database\n"
        #         "EXAMPLES:\n"
        #         "- 'postgresql://user:pass@localhost:5432/mydb|show me top 10 customers'\n"
        #         "- 'postgresql://user:pass@localhost:5432/mydb|get database schema'\n"
        #         "- 'postgresql://user:pass@localhost:5432/mydb|analyze sales trends by month'\n"
        #         "SECURITY: Use environment variables for credentials in production."
        #     )
        # )
        
        self.tools = [retriever_tool, data_analyzer_tool, list_file_tool, contextual_tool]  # postgresql_tool commented out
        logger.info(f"Setup {len(self.tools)} tools: {[tool.name for tool in self.tools]}")
    
    def _setup_agent(self) -> None:
        """Setup ReAct agent with tools and memory integration."""
        try:
            # Get custom ReAct prompt
            prompt = create_react_prompt()
            
            # Create ReAct agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )
            
            # Create agent executor with memory integration
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory_manager.memory,
                verbose=True,
                handle_parsing_errors="Check your output and make sure it conforms to the format instructions. Always include 'Action:' after 'Thought:' and use one of the available tool names.",
                max_iterations=3,
                early_stopping_method="force"
            )
            
            logger.info("ReAct agent setup completed with memory integration")
            
        except Exception as e:
            logger.error(f"Error setting up agent: {e}")
            raise
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def _resolve_file_path(self, file_path: str) -> str:
        """
        Resolve file path by checking multiple locations.
        
        Args:
            file_path: Input file path (could be filename or full path)
            
        Returns:
            Resolved full path or None if not found
        """
        
        # If it's already a full path and exists, return it
        if os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path
        
        # Check in uploaded files list first
        if self.file_dict_list:
            for file_dict in self.file_dict_list:
                file_name = file_dict.get('file_name', '')
                stored_path = file_dict.get('file_path', '')
                
                # Match by filename or exact path match
                if (file_name == file_path or file_name.lower() == file_path.lower() or 
                    stored_path == file_path):
                    # Check if stored_path exists as absolute path
                    if os.path.exists(stored_path):
                        return stored_path
                    # Check if stored_path exists as relative path from project root
                    project_root = Path(__file__).parent.parent.parent
                    relative_path = project_root / stored_path
                    if relative_path.exists():
                        return str(relative_path)
        
        # Check in data directory
        data_dir = Path(__file__).parent.parent.parent / "data"
        if data_dir.exists():
            # Remove "data/" prefix if present
            clean_file_path = file_path
            if file_path.startswith("data/"):
                clean_file_path = file_path[5:]
            
            # Try exact match
            candidate_path = data_dir / clean_file_path
            if candidate_path.exists():
                return str(candidate_path)
            
            # Try case-insensitive match and fuzzy matching for filenames with spaces
            for item in data_dir.iterdir():
                if item.is_file():
                    # Exact match (case insensitive)
                    if item.name.lower() == clean_file_path.lower():
                        return str(item)
                    
                    # Normalize spaces and compare (handle multiple spaces)
                    normalized_item = re.sub(r'\s+', ' ', item.name.lower().strip())
                    normalized_search = re.sub(r'\s+', ' ', clean_file_path.lower().strip())
                    
                    if normalized_item == normalized_search:
                        return str(item)
                    
                    # Partial match for filenames with spaces
                    if clean_file_path.lower() in item.name.lower() or item.name.lower() in clean_file_path.lower():
                        return str(item)
        
        return None
    
    def _parse_tool_input(self, input_str: str) -> tuple[str, Optional[str]]:
        """
        Parse tool input string into file_path and optional query.
        
        Args:
            input_str: Either file path or 'file_path|query'
            
        Returns:
            Tuple of (file_path, query) where query can be None
        """
        if '|' in input_str:
            file_path, query = input_str.split('|', 1)
            return file_path.strip(), query.strip()
        else:
            return input_str.strip(), None
    
    def _handle_tool_error(self, error: Exception, tool_name: str) -> str:
        """
        Handle tool execution errors consistently.
        
        Args:
            error: The exception that occurred
            tool_name: Name of the tool for logging
            
        Returns:
            Formatted error message
        """
        error_msg = f"Error in {tool_name}: {str(error)}"
        logger.error(error_msg)
        return error_msg
    
    # =========================================================================
    # TOOL WRAPPER METHODS
    # =========================================================================
    
    def _excel_analyzer_wrapper(self, input_str: str) -> str:
        """
        Wrapper for Excel analyzer tool.
        
        Args:
            input_str: Either file path or 'file_path|query'
            
        Returns:
            Analysis results or error message
        """
        try:
            # Parse input using shared method
            file_path, query = self._parse_tool_input(input_str)
            
            if not file_path:
                return "Error: Please provide a valid file path"
            
            # Resolve file path - check if it's just a filename and needs full path
            resolved_path = self._resolve_file_path(file_path)
            if not resolved_path:
                return f"Error: File not found: {file_path}"
            
            # Call Excel analyzer
            result = analyze_excel_with_pandas(resolved_path, query)
            return result
            
        except Exception as e:
            return self._handle_tool_error(e, "Excel analysis")
    
    def _list_file_wrapper(self, input_str: str) -> str:
        """
        Wrapper for list_file tool to process uploaded file dictionary list.
        
        Args:
            input_str: Query about files or 'list' to show all files
            
        Returns:
            Information about available files or specific file details
        """
        try:
            if not self.file_dict_list:
                return "No files have been uploaded in this session."
            
            # Clean input string - remove quotes and extra whitespace
            clean_input = input_str.strip().strip("'\"").lower()
            
            # If user wants to list all files
            if clean_input in ['list', 'show files', 'available files', '']:
                file_info = []
                for i, file_dict in enumerate(self.file_dict_list, 1):
                    file_name = file_dict.get('file_name', 'Unknown')
                    file_path = file_dict.get('file_path', 'Unknown')
                    file_info.append(f"{i}. {file_name} (Path: {file_path})")
                
                return f"Available files ({len(self.file_dict_list)}):\n" + "\n".join(file_info)
            
            # Search for specific file
            query = clean_input
            matching_files = []
            
            for file_dict in self.file_dict_list:
                file_name = file_dict.get('file_name', '').lower()
                file_path = file_dict.get('file_path', '').lower()
                
                if query in file_name or query in file_path:
                    matching_files.append({
                        'name': file_dict.get('file_name', 'Unknown'),
                        'path': file_dict.get('file_path', 'Unknown')
                    })
            
            if matching_files:
                result = f"Found {len(matching_files)} matching file(s):\n"
                for i, file_info in enumerate(matching_files, 1):
                    result += f"{i}. {file_info['name']} (Path: {file_info['path']})\n"
                result += "\nYou can use the data_analyzer tool with the file path to analyze these files."
                return result
            else:
                return f"No files found matching '{input_str}'. Available files: {len(self.file_dict_list)}"
                
        except Exception as e:
            return self._handle_tool_error(e, "file listing")
    
    def _contextual_analyzer_wrapper(self, input_str: str) -> str:
        """
        Wrapper for contextual analyzer tool.
        
        Args:
            input_str: Either file path or 'file_path|query'
            
        Returns:
            Analysis results or error message. For successful analysis, 
            returns result with DIRECT_OUTPUT marker for agent to output directly.
        """
        try:
            # Parse input using shared method
            file_path, query = self._parse_tool_input(input_str)
            
            if not file_path:
                return "Error: Please provide a valid file path"
            
            # Resolve file path - check if it's just a filename and needs full path
            resolved_path = self._resolve_file_path(file_path)
            if not resolved_path:
                return f"Error: File not found: {file_path}"
            
            # Call contextual analyzer with resolved file path
            result = analyze_document_contextually(resolved_path, query)
            
            # Check if analysis was successful (not an error)
            if result and not result.startswith("❌"):
                # Mark successful analysis for direct output
                return f"DIRECT_OUTPUT:{result}"
            else:
                # Return error as-is for normal processing
                return result
            
        except Exception as e:
            return self._handle_tool_error(e, "contextual analysis")
    
    def _postgresql_wrapper(self, input_str: str) -> str:
        """
        Wrapper for PostgreSQL database tool.
        
        Args:
            input_str: 'connection_string|query' format
            
        Returns:
            Database query results or error message
        """
        try:
            # Parse input - must have connection string and query
            if '|' not in input_str:
                return "Error: Please provide input in format 'connection_string|query'"
            
            connection_string, query = input_str.split('|', 1)
            connection_string = connection_string.strip()
            query = query.strip()
            
            if not connection_string:
                return "Error: Please provide a valid PostgreSQL connection string"
            
            if not query:
                return "Error: Please provide a valid query"
            
            # Special handling for schema requests
            # if query.lower() in ['schema', 'get schema', 'show schema', 'database schema', 'get database schema']:
            #     from ..tools.postgresql_tool import get_postgresql_schema
            #     result = get_postgresql_schema(connection_string)
            # else:
            #     # Execute natural language query
            #     result = query_postgresql_database(connection_string, query)
            result = "PostgreSQL tool is currently disabled due to Docker dependency requirements."
            
            return result
            
        except Exception as e:
            return self._handle_tool_error(e, "PostgreSQL query")
    
    # =========================================================================
    # MAIN EXECUTION METHODS
    # =========================================================================
    
    def run(self, query: str) -> str:
        """
        Run the agent with a query and maintain chat history.
        
        For contextual analysis queries, checks for direct output first.
        
        Args:
            query: User query or question
            
        Returns:
            Agent response
        """
        if not query or not query.strip():
            return "Please provide a valid query."
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Check for direct output from contextual analyzer
            is_direct, direct_content = self._check_for_direct_output(query)
            if is_direct and direct_content:
                logger.info("Using direct output from contextual analyzer")
                # Save to memory and return direct content
                self.memory_manager.add_message(query, direct_content)
                return direct_content
            
            # Run agent with memory integration (normal flow)
            result = self.agent_executor.invoke({"input": query})
            
            # Extract output
            output = result.get("output", "No response generated")
            
            # Explicitly save conversation turn to memory for session persistence
            self.memory_manager.add_message(query, output)
            
            logger.info("Query processed successfully and saved to memory")
            return output
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def run_stream(self, query: str):
        """
        Run the agent with streaming output using LLM interface streaming capabilities.
        
        For contextual analysis queries, checks for direct output first and streams it.
        
        Args:
            query: User query or question
            
        Yields:
            Streaming chunks of the agent response
        """
        if not query or not query.strip():
            yield "Please provide a valid query."
            return
        
        try:
            logger.info(f"Processing streaming query: {query[:100]}...")
            
            # Check for direct output from contextual analyzer
            is_direct, direct_content = self._check_for_direct_output(query)
            if is_direct and direct_content:
                logger.info("Using direct output from contextual analyzer (streaming)")
                # Stream the direct content
                from ..service.llm.llm_interface import create_fallback_stream
                for chunk in create_fallback_stream(direct_content):
                    yield chunk
                # Save to memory
                self.memory_manager.add_message(query, direct_content)
                logger.info("Direct streaming query processed successfully and saved to memory")
                return
            
            # Import streaming components from LLM interface
            from ..service.llm.llm_interface import (
                EnhancedStreamingCallbackHandler,
                create_fallback_stream
            )
            import threading
            
            # Create streaming callback handler
            streaming_handler = EnhancedStreamingCallbackHandler()
            
            # Run agent with streaming callback
            tokens_yielded = 0
            agent_result = None
            
            def run_agent():
                """Execute agent in separate thread."""
                nonlocal agent_result
                try:
                    agent_result = self.agent_executor.invoke(
                        {"input": query},
                        config={"callbacks": [streaming_handler]}
                    )
                    # Ensure completion signal
                    if not streaming_handler.finished:
                        streaming_handler.agent_finished = True
                        streaming_handler.finished = True
                        streaming_handler.token_queue.put(None)
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    streaming_handler.token_queue.put(error_msg)
                    streaming_handler.finished = True
                    streaming_handler.token_queue.put(None)
            
            # Start agent execution
            agent_thread = threading.Thread(target=run_agent)
            agent_thread.start()
            
            # Stream tokens using the enhanced callback handler
            try:
                for token in streaming_handler.get_tokens():
                    yield token
                    tokens_yielded += 1
            except Exception as e:
                logger.error(f"Streaming error: {e}")
            
            # Wait for agent completion
            agent_thread.join(timeout=10.0)
            
            logger.info(f"Streaming completed: yielded {tokens_yielded} tokens, agent_finished: {streaming_handler.agent_finished}")
            
            # Handle fallback scenarios
            if self._should_use_fallback(tokens_yielded, streaming_handler):
                yield from self._handle_fallback_streaming(query, streaming_handler, agent_result)
            else:
                # Save successful streaming response to memory
                self.memory_manager.add_message(query, streaming_handler.current_response)
            
            logger.info("Streaming query processed successfully and saved to memory")
            
        except Exception as e:
            error_msg = f"Error processing streaming query: {str(e)}"
            logger.error(error_msg)
            yield error_msg
    
    def _should_use_fallback(self, tokens_yielded: int, streaming_handler) -> bool:
        """
        Determine if fallback streaming should be used.
        
        Args:
            tokens_yielded: Number of tokens successfully yielded
            streaming_handler: The streaming callback handler
            
        Returns:
            True if fallback should be used
        """
        return (
            tokens_yielded == 0 or 
            not streaming_handler.current_response or 
            not streaming_handler.agent_finished
        )
    
    def _handle_fallback_streaming(self, query: str, streaming_handler, agent_result):
        """
        Handle fallback streaming when primary streaming fails.
        
        Args:
            query: Original user query
            streaming_handler: The streaming callback handler
            agent_result: Result from agent execution (may be None)
            
        Yields:
            Fallback streaming chunks
        """
        from ..service.llm.llm_interface import create_fallback_stream
        
        logger.warning(
            f"Streaming incomplete, falling back to static execution. "
            f"Response: {bool(streaming_handler.current_response)}, "
            f"Finished: {streaming_handler.agent_finished}"
        )
        
        # Get final output
        if agent_result:
            output = agent_result.get("output", "No response generated")
        else:
            # Re-run agent if needed
            result = self.agent_executor.invoke({"input": query})
            output = result.get("output", "No response generated")
        
        # Determine what content to stream
        content_to_stream = self._get_remaining_content(streaming_handler.current_response, output)
        
        # Stream the content using fallback streaming
        fallback_chunks = 0
        for chunk in create_fallback_stream(content_to_stream):
            yield chunk
            fallback_chunks += 1
        
        logger.info(f"Fallback streaming: yielded {fallback_chunks} chunks")
        
        # Save to memory
        self.memory_manager.add_message(query, output)
    
    def _get_remaining_content(self, streamed_content: str, full_output: str) -> str:
        """
        Get the content that still needs to be streamed.
        
        Args:
            streamed_content: Content already streamed
            full_output: Complete output from agent
            
        Returns:
            Content that still needs to be streamed
        """
        if not streamed_content or not full_output:
            return full_output or ""
        
        streamed_content = streamed_content.strip()
        
        # Check if streamed content is part of full output
        if streamed_content not in full_output:
            return full_output
        
        # Find remaining content
        if full_output.startswith(streamed_content):
            remaining = full_output[len(streamed_content):].strip()
            return remaining if remaining else full_output
        
        return full_output
    
    def _check_for_direct_output(self, query: str) -> tuple[bool, str]:
        """
        Check if query should trigger direct output from contextual analyzer.
        
        This method simulates tool execution to check for DIRECT_OUTPUT marker
        without going through the full ReAct loop.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (is_direct_output, content) where is_direct_output indicates
            if direct output should be used, and content is the direct output content
        """
        try:
            # Check if this looks like a contextual analysis request
            query_lower = query.lower()
            contextual_keywords = [
                'analyze', 'analysis', 'summarize', 'summary', 'explain', 'describe',
                '分析', '总结', '解释', '说明', '描述'
            ]
            
            # Only check for direct output if query contains contextual analysis keywords
            if not any(keyword in query_lower for keyword in contextual_keywords):
                return False, ""
            
            # If we have files uploaded, try to detect if contextual analyzer would be used
            if self.file_dict_list:
                # Look for document files (not Excel/CSV which would use data_analyzer)
                doc_files = []
                for file_dict in self.file_dict_list:
                    file_name = file_dict.get('file_name', '').lower()
                    if any(ext in file_name for ext in ['.pdf', '.docx', '.txt', '.md', '.markdown']):
                        doc_files.append(file_dict)
                
                # If we have document files and this is an analysis query,
                # try contextual analyzer directly
                if doc_files:
                    file_path = doc_files[0].get('file_path', '')
                    if file_path:
                        # Try contextual analyzer wrapper
                        result = self._contextual_analyzer_wrapper(file_path + "|" + query)
                        
                        # Check if it returned direct output
                        if result.startswith("DIRECT_OUTPUT:"):
                            direct_content = result[14:]  # Remove "DIRECT_OUTPUT:" prefix
                            return True, direct_content
            
            return False, ""
            
        except Exception as e:
            logger.debug(f"Error checking for direct output: {e}")
            return False, ""


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_react_agent_instance(
    llm=None,
    retriever: Optional[RRFRetriever] = None,
    memory_manager: Optional[SimpleMemoryManager] = None
) -> ReActAgent:
    """
    Factory function to create a ReAct agent instance.
    
    Args:
        llm: Language model instance
        retriever: RRF retriever instance
        memory_manager: Memory manager for chat history
        
    Returns:
        Configured ReAct agent
    """
    return ReActAgent(llm=llm, retriever=retriever, memory_manager=memory_manager)


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def _test_agent() -> None:
    """Test function for the ReAct agent."""
    print("Testing ReAct Agent...")
    
    try:
        # Create agent
        agent = create_react_agent_instance()
        
        # Test queries
        test_queries = [
            "What tools do you have available?",
            "Can you help me analyze data?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = agent.run(query)
            print(f"Response: {response}")
        
        # Test direct output functionality
        print("\n" + "="*50)
        print("Testing Direct Output Functionality")
        print("="*50)
        
        # Simulate having a document file uploaded
        agent.file_dict_list = [
            {
                'file_name': 'test_document.pdf',
                'file_path': '/path/to/test_document.pdf'
            }
        ]
        
        # Test direct output detection
        test_analysis_queries = [
            "Analyze this document",
            "Please summarize the content",
            "分析这个文档"
        ]
        
        for query in test_analysis_queries:
            print(f"\nTesting direct output for: {query}")
            is_direct, content = agent._check_for_direct_output(query)
            print(f"  Direct output detected: {is_direct}")
            if is_direct:
                print(f"  Content preview: {content[:100]}...")
        
        # Show basic agent info
        print("\nAgent Information:")
        print(f"  Tools: {[tool.name for tool in agent.tools]}")
        print(f"  LLM Model: {getattr(agent.llm, 'model_name', 'Unknown')}")
        print(f"  Direct output functionality: ✅ Implemented")
        
    except Exception as e:
        logger.error(f"Error testing agent: {e}")


if __name__ == "__main__":
    _test_agent()