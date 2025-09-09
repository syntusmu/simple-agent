"""
ReAct Agent implementation using LangChain create_react_agent.

This module provides a ReAct agent with RAG retrieval and Excel analysis capabilities.
"""

import logging
from typing import Optional, List

from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.tools.retriever import create_retriever_tool
from langchain.schema import BaseRetriever, Document
from ..service.llm.llm_interface import create_llm
from ..service.rag.retriever import RRFRetriever
from ..service.prompt.prompt import create_react_prompt
from ..tools.data_analyzer import analyze_excel_with_pandas
from ..tools.contextual_analyzer import analyze_document_contextually
from .memory import SimpleMemoryManager, create_memory_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentRetrieverWrapper(BaseRetriever):
    """Wrapper to make RRFRetriever compatible with LangChain BaseRetriever."""
    
    rrfretriever: RRFRetriever
    
    def __init__(self, rrfretriever: RRFRetriever):
        """Initialize wrapper with RRFRetriever instance."""
        super().__init__(rrfretriever=rrfretriever)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using RRFRetriever."""
        try:
            return self.rrfretriever.retrieve(query)
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            return []


class ReActAgent:
    """
    ReAct Agent using LangChain create_react_agent.
    
    Features:
    - RAG document retrieval using RRFRetriever
    - Excel/CSV data analysis using pandas agent
    - Configurable LLM backend
    """
    
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
    
    def _setup_tools(self) -> None:
        """Setup agent tools."""
        # Create retriever wrapper for LangChain compatibility
        retriever_wrapper = DocumentRetrieverWrapper(self.retriever)
        
        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever=retriever_wrapper,
            name="document_retriever",
            description=(
                "WHEN TO USE: Search for information in previously uploaded documents or knowledge base.\n"
                "PURPOSE: Find relevant content from stored documents using semantic search.\n"
                "INPUT: A search query describing what information you need.\n"
                "EXAMPLE: 'financial performance metrics' or 'project timeline details'"
            )
        )
        
        # Create data analyzer tool
        data_analyzer_tool = Tool(
            name="data_analyzer",
            func=self._excel_analyzer_wrapper,
            description=(
                "WHEN TO USE: Analyze Excel/CSV files with numerical data, tables, or spreadsheets.\n"
                "PURPOSE: Perform statistical analysis, generate insights, answer data questions.\n"
                "PREREQUISITE: MUST use list_file tool first to get the exact file path.\n"
                "INPUT FORMAT: Use file_path from list_file output, e.g. 'data/filename.xlsx' or 'data/filename.xlsx|your_question'\n"
                "SUPPORTED: .xlsx, .xls, .csv files\n"
                "WORKFLOW: list_file → data_analyzer with file_path\n"
                "EXAMPLE: 'data/abc.csv|what are the top performing products?'"
            )
        )
        
        # Create list_file tool
        list_file_tool = Tool(
            name="list_file",
            func=self._list_file_wrapper,
            description=(
                "WHEN TO USE: Check what files are available in the current session.\n"
                "PURPOSE: See uploaded files before choosing analysis tools.\n"
                "INPUT: 'list' to show all files, or filename to search specific files.\n"
                "OUTPUT: Returns file_path values (e.g. 'data/abc.csv') for use in analyzer tools.\n"
                "CRITICAL: ALWAYS use this FIRST for any file-related question.\n"
                "EXAMPLE: 'list' or 'report.pdf'"
            )
        )
        
        # Create contextual analyzer tool
        contextual_tool = Tool(
            name="contextual_analyzer",
            func=self._contextual_analyzer_wrapper,
            description=(
                "WHEN TO USE: Analyze document content (PDF, Word, TXT, Markdown) with AI understanding.\n"
                "PURPOSE: Extract insights, summarize, answer questions about document content.\n"
                "PREREQUISITE: MUST use list_file tool first to identify available files.\n"
                "INPUT: Takes file_path from list_file (e.g. 'data/document.pdf') with query.\n"
                "INPUT FORMAT: 'your_question' or 'your_question|filename_filter'\n"
                "SUPPORTED: .pdf, .docx, .txt, .md, .markdown (NOT for data analysis)\n"
                "WORKFLOW: list_file → contextual_analyzer with file_path\n"
                "EXAMPLE: 'summarize key findings|data/research_report.pdf'"
            )
        )
        
        self.tools = [retriever_tool, data_analyzer_tool, list_file_tool, contextual_tool]
        logger.info(f"Setup {len(self.tools)} tools: {[tool.name for tool in self.tools]}")
    
    def _excel_analyzer_wrapper(self, input_str: str) -> str:
        """
        Wrapper for Excel analyzer tool.
        
        Args:
            input_str: Either file path or 'file_path|query'
            
        Returns:
            Analysis results or error message
        """
        try:
            # Parse input
            if '|' in input_str:
                file_path, query = input_str.split('|', 1)
                file_path = file_path.strip()
                query = query.strip()
            else:
                file_path = input_str.strip()
                query = None
            
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
            error_msg = f"Error in Excel analysis: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _resolve_file_path(self, file_path: str) -> str:
        """
        Resolve file path by checking multiple locations.
        
        Args:
            file_path: Input file path (could be filename or full path)
            
        Returns:
            Resolved full path or None if not found
        """
        from pathlib import Path
        import os
        
        # If it's already a full path and exists, return it
        if os.path.isabs(file_path) and os.path.exists(file_path):
            return file_path
        
        # Check in uploaded files list first
        if self.file_dict_list:
            for file_dict in self.file_dict_list:
                file_name = file_dict.get('file_name', '')
                stored_path = file_dict.get('file_path', '')
                
                # Match by filename
                if file_name == file_path or file_name.lower() == file_path.lower():
                    if os.path.exists(stored_path):
                        return stored_path
        
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
                    import re
                    normalized_item = re.sub(r'\s+', ' ', item.name.lower().strip())
                    normalized_search = re.sub(r'\s+', ' ', clean_file_path.lower().strip())
                    
                    if normalized_item == normalized_search:
                        return str(item)
                    
                    # Partial match for filenames with spaces
                    if clean_file_path.lower() in item.name.lower() or item.name.lower() in clean_file_path.lower():
                        return str(item)
        
        return None
    
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
                result += "\nYou can use the excel_analyzer tool with the file path to analyze these files."
                return result
            else:
                return f"No files found matching '{input_str}'. Available files: {len(self.file_dict_list)}"
                
        except Exception as e:
            error_msg = f"Error listing files: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _contextual_analyzer_wrapper(self, input_str: str) -> str:
        """
        Wrapper for contextual analyzer tool.
        
        Args:
            input_str: Either query or 'query|filename_filter'
            
        Returns:
            Analysis results or error message
        """
        try:
            # Parse input
            if '|' in input_str:
                query, filename_filter = input_str.split('|', 1)
                query = query.strip()
                filename_filter = filename_filter.strip()
            else:
                query = input_str.strip()
                filename_filter = None
            
            if not query:
                return "Error: Please provide a valid analysis query"
            
            # Call contextual analyzer with file dict list
            result = analyze_document_contextually(
                file_dict_list=self.file_dict_list,
                query=query,
                filename_filter=filename_filter
            )
            return result
            
        except Exception as e:
            error_msg = f"Error in contextual analysis: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
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
                handle_parsing_errors=True,
                max_iterations=5,
                early_stopping_method="force"
            )
            
            logger.info("ReAct agent setup completed with memory integration")
            
        except Exception as e:
            logger.error(f"Error setting up agent: {e}")
            raise
    
    def run(self, query: str) -> str:
        """
        Run the agent with a query and maintain chat history.
        
        Args:
            query: User query or question
            
        Returns:
            Agent response
        """
        if not query or not query.strip():
            return "Please provide a valid query."
        
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Run agent with memory integration
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
        
        # Show basic agent info
        print("\nAgent Information:")
        print(f"  Tools: {[tool.name for tool in agent.tools]}")
        print(f"  LLM Model: {getattr(agent.llm, 'model_name', 'Unknown')}")
    except Exception as e:
        logger.error(f"Error testing agent: {e}")


if __name__ == "__main__":
    _test_agent()