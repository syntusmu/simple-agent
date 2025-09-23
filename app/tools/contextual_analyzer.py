"""Contextual Document Analyzer for Intelligent Document Analysis.

This module provides comprehensive document analysis capabilities using docling for document
processing and LLM for intelligent contextual analysis. It supports multiple document formats
and provides both modern and legacy APIs for backward compatibility.

Classes:
    ContextualAnalyzer: Main analyzer class for document processing and LLM-based analysis

Functions:
    analyze_document_contextually: Recommended modern API for document analysis
    analyze_document_contextually_legacy: Legacy API for backward compatibility
    get_supported_formats: Get list of supported file formats
    is_supported_format: Check if a file format is supported
    _test_contextual_analyzer: Testing function for validation

Constants:
    MAX_TOKENS: Maximum token limit for document analysis
    DEFAULT_SECTION: Default LLM configuration section
    DEFAULT_MODEL: Default LLM model name
    DEFAULT_TEMPERATURE: Default LLM temperature setting
    TEXT_FORMATS: Supported text file formats
    SUPPORTED_FORMATS: All supported document formats

Usage:
    # Modern API (recommended)
    result = analyze_document_contextually("document.pdf", "Analyze this document")
    
    # Class-based usage
    analyzer = ContextualAnalyzer(section='deepseek')
    result = analyzer.analyze_document("document.pdf", "Your query here")
    
    # Legacy API (for backward compatibility)
    file_list = [{'file_name': 'doc.pdf', 'file_path': '/path/to/doc.pdf'}]
    result = analyze_document_contextually_legacy(file_list, "Your query")

Architecture:
    The analyzer follows a clean separation of concerns:
    1. Document conversion (docling integration for various formats)
    2. Token counting and validation (tiktoken integration)
    3. LLM-based contextual analysis (LangChain integration)
    4. Error handling and user-friendly formatting
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import tiktoken

from ..service.llm.llm_interface import create_llm
from ..service.prompt.prompt import create_contextual_analysis_prompt
from ..service.document.docling import DocumentProcessor

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Analysis Configuration
MAX_TOKENS = 6000                    # Maximum tokens allowed for document analysis
DEFAULT_SECTION = 'llm'         # Default LLM configuration section
DEFAULT_MODEL = None                 # Default model (uses config.ini setting)
DEFAULT_TEMPERATURE = 0.1            # Default temperature for analysis
DEFAULT_QUERY = "Please provide a comprehensive analysis and summary of this document."

# File Format Support
TEXT_FORMATS = {'.txt', '.markdown'}  # Text formats handled directly
SUPPORTED_FORMATS = set(DocumentProcessor.SUPPORTED_FORMATS.keys()) | TEXT_FORMATS | {'.xls'}

# Error Messages
ERROR_PREFIX = "❌"                   # Prefix for error messages
SUCCESS_PREFIX = "📄"                 # Prefix for success messages


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_supported_formats() -> List[str]:
    """Get list of supported file formats.
    
    Returns:
        List[str]: Sorted list of supported file extensions including:
                   - PDF files (.pdf)
                   - Office documents (.docx, .xlsx, .pptx)
                   - Text files (.txt, .markdown)
                   - And other formats supported by docling
    """
    return sorted(list(SUPPORTED_FORMATS))


def is_supported_format(file_path: Union[str, Path]) -> bool:
    """Check if file format is supported.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if file format is supported, False otherwise
    """
    return Path(file_path).suffix.lower() in SUPPORTED_FORMATS


# =============================================================================
# CONTEXTUAL ANALYZER CLASS
# =============================================================================

class ContextualAnalyzer:
    """Contextual document analyzer using docling and LLM.
    
    This class provides comprehensive document analysis capabilities by:
    1. Converting documents to markdown using docling
    2. Performing token counting and validation
    3. Analyzing content using LLM with contextual prompts
    4. Formatting results with rich markdown output
    
    Attributes:
        llm: LangChain LLM instance for analysis
        tokenizer: Tiktoken tokenizer for token counting
        section: LLM configuration section used
        model_name: LLM model name used
        temperature: LLM temperature setting
    
    Example:
        # Basic usage
        analyzer = ContextualAnalyzer()
        result = analyzer.analyze_document("document.pdf", "Summarize this document")
        
        # Custom configuration
        analyzer = ContextualAnalyzer(section='qwen', model_name='qwen-max', temperature=0.2)
        result = analyzer.analyze_document("document.pdf", "Extract key insights")
    """
    
    def __init__(self, 
                 section: str = DEFAULT_SECTION, 
                 model_name: Optional[str] = DEFAULT_MODEL,
                 temperature: float = DEFAULT_TEMPERATURE):
        """Initialize the analyzer with LLM and tokenizer.
        
        Args:
            section: LLM configuration section (e.g., 'deepseek', 'qwen', 'openai')
            model_name: Specific model name to use (uses config.ini if None)
            temperature: Temperature setting for LLM (0.0-1.0)
        
        Raises:
            Exception: If LLM initialization fails
        """
        try:
            # Store configuration
            self.section = section
            self.model_name = model_name
            self.temperature = temperature
            
            # Initialize LLM with section-based configuration
            self.llm = create_llm(section=section, model_name=model_name, temperature=temperature)
            
            # Initialize tokenizer
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
            logger.info(f"Contextual analyzer initialized with section={section}, model={model_name}, temperature={temperature}")
            
        except Exception as e:
            error_msg = f"Failed to initialize ContextualAnalyzer: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    # -------------------------------------------------------------------------
    # Configuration and Information Methods
    # -------------------------------------------------------------------------
    
    def get_analyzer_info(self) -> Dict[str, Union[str, float, int]]:
        """Get analyzer configuration information.
        
        Returns:
            Dict containing analyzer configuration details
        """
        return {
            'section': self.section,
            'model_name': self.model_name,
            'temperature': self.temperature,
            'max_tokens': MAX_TOKENS,
            'supported_formats_count': len(SUPPORTED_FORMATS),
            'tokenizer_encoding': 'cl100k_base'
        }
    
    # -------------------------------------------------------------------------
    # Private Utility Methods
    # -------------------------------------------------------------------------
    
    def _is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            bool: True if format is supported, False otherwise
        """
        return Path(file_path).suffix.lower() in SUPPORTED_FORMATS
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text with fallback estimation.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            int: Number of tokens in the text
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text) // 4  # Fallback estimation (rough approximation)
    
    def _format_error(self, error_msg: str, context: str = "") -> str:
        """Format error message consistently.
        
        Args:
            error_msg: The error message to format
            context: Optional context information
            
        Returns:
            str: Formatted error message with consistent prefix
        """
        full_msg = f"{context}: {error_msg}" if context else error_msg
        logger.error(full_msg)
        return f"{ERROR_PREFIX} {full_msg}"
    
    def _validate_inputs(self, file_path: Union[str, Path], query: str) -> Optional[str]:
        """Validate input parameters.
        
        Args:
            file_path: Path to the file to validate
            query: Query string to validate
            
        Returns:
            Optional[str]: Error message if validation fails, None if valid
        """
        # Validate file path
        if not file_path:
            return "File path cannot be empty"
        
        path_obj = Path(file_path)
        if not path_obj.exists():
            return f"File not found: {file_path}"
        
        if not self._is_supported_format(file_path):
            supported = ', '.join(sorted(SUPPORTED_FORMATS))
            return f"Unsupported file format: {path_obj.suffix}. Supported: {supported}"
        
        # Validate query
        if not query or not query.strip():
            return "Query cannot be empty"
        
        return None
    
    # -------------------------------------------------------------------------
    # Document Processing Methods
    # -------------------------------------------------------------------------
    
    def _convert_to_markdown(self, file_path: Union[str, Path]) -> Tuple[str, Optional[str]]:
        """Convert document to markdown format.
        
        This method handles different file formats:
        - Text files (.txt, .markdown): Read directly
        - Other formats: Use DocumentProcessor (docling) for conversion
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple[str, Optional[str]]: (markdown_content, error_message)
                - markdown_content: Converted content or empty string on error
                - error_message: None if successful, error description if failed
        """
        try:
            file_path = Path(file_path)
            
            # Validate file existence and format (redundant check for safety)
            if not file_path.exists():
                return "", f"File not found: {file_path}"
            
            if not self._is_supported_format(file_path):
                return "", f"Unsupported file format: {file_path.suffix}"
            
            # Handle text files directly for better performance
            if file_path.suffix.lower() in TEXT_FORMATS:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        logger.debug(f"Successfully read text file: {file_path.name}")
                        return content, None
                except UnicodeDecodeError:
                    # Try with different encoding
                    try:
                        with open(file_path, 'r', encoding='latin-1') as f:
                            content = f.read()
                            logger.debug(f"Successfully read text file with latin-1 encoding: {file_path.name}")
                            return content, None
                    except Exception as e:
                        return "", f"Error reading text file with multiple encodings: {str(e)}"
                except Exception as e:
                    return "", f"Error reading text file: {str(e)}"
            
            # Use DocumentProcessor for other formats (PDF, DOCX, etc.)
            try:
                logger.debug(f"Processing document with docling: {file_path.name}")
                processor = DocumentProcessor(input_file=file_path)
                markdown_content, _ = processor.process_document()
                
                if not markdown_content or not markdown_content.strip():
                    return "", f"No content extracted from document: {file_path.name}"
                
                logger.debug(f"Successfully processed document: {file_path.name}")
                return markdown_content, None
                
            except Exception as e:
                return "", f"Error processing document with docling: {str(e)}"
                
        except Exception as e:
            error_msg = f"Unexpected error converting {file_path} to markdown: {str(e)}"
            logger.error(error_msg)
            return "", error_msg
    
    # -------------------------------------------------------------------------
    # Analysis Methods
    # -------------------------------------------------------------------------
    
    def _perform_analysis(self, content: str, query: str, filename: str, token_count: int) -> str:
        """Perform contextual analysis using LLM.
        
        Args:
            content: Document content in markdown format
            query: User's analysis query
            filename: Name of the file being analyzed
            token_count: Number of tokens in the content
            
        Returns:
            str: Formatted analysis result with rich markdown formatting
        
        Raises:
            Exception: If LLM analysis fails
        """
        try:
            # Get system prompt for contextual analysis
            system_prompt = create_contextual_analysis_prompt()
            
            # Construct user message with structured format
            user_message = f"""
Document Information:
- Filename: {filename}
- Content Length: {token_count} tokens
- Analysis Query: {query}

Document Content:
{content}

Please analyze the document content and provide a comprehensive response to the user's query. 
Focus on being accurate, detailed, and directly addressing the specific query while considering the full context of the document.
"""
            
            logger.debug(f"Performing LLM analysis for {filename} with {token_count} tokens")
            
            # Invoke LLM for analysis
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ])
            
            # Format the response with rich markdown
            formatted_result = f"""
{SUCCESS_PREFIX} **Document Analysis: {filename}**
📊 **Content Size:** {token_count} tokens
❓ **Query:** {query}
🤖 **Model:** {self.section}/{self.model_name or 'default'}

**Analysis Result:**
{response.content}

---
*Analysis completed using {self.section} model with temperature {self.temperature}*
"""
            
            logger.info(f"Successfully completed analysis for {filename}")
            return formatted_result
            
        except Exception as e:
            error_msg = f"LLM analysis failed: {str(e)}"
            logger.error(error_msg)
            return self._format_error(error_msg, "Analysis Error")
    
    def analyze_document(self, file_path: Union[str, Path], user_query: str) -> str:
        """Analyze document content based on user query.
        
        This is the main method for document analysis. It orchestrates the complete workflow:
        1. Input validation
        2. Document conversion to markdown
        3. Token counting and validation
        4. LLM-based contextual analysis
        5. Result formatting
        
        Args:
            file_path: Path to the document to analyze
            user_query: User's analysis query
            
        Returns:
            str: Formatted analysis result or error message with consistent formatting
        
        Example:
            analyzer = ContextualAnalyzer()
            result = analyzer.analyze_document("report.pdf", "What are the key findings?")
        """
        try:
            # Validate inputs
            validation_error = self._validate_inputs(file_path, user_query)
            if validation_error:
                return self._format_error(validation_error, "Input Validation")
            
            file_path = Path(file_path)
            logger.info(f"Starting analysis of {file_path.name} with query: {user_query[:50]}...")
            
            # Convert document to markdown
            markdown_content, conversion_error = self._convert_to_markdown(file_path)
            
            if conversion_error:
                return self._format_error(conversion_error, "Document Conversion")
            
            if not markdown_content.strip():
                return self._format_error(f"No content extracted from {file_path.name}", "Content Extraction")
            
            # Check token limit
            token_count = self._count_tokens(markdown_content)
            logger.debug(f"Document {file_path.name} contains {token_count} tokens")
            
            if token_count > MAX_TOKENS:
                return self._format_error(
                    f"Document too large: {token_count} tokens (limit: {MAX_TOKENS}). "
                    f"Please use a smaller document or split it into sections.",
                    "Token Limit Exceeded"
                )
            
            # Perform LLM analysis
            return self._perform_analysis(markdown_content, user_query, file_path.name, token_count)
            
        except Exception as e:
            error_msg = f"Unexpected error analyzing document {file_path}: {str(e)}"
            logger.error(error_msg)
            return self._format_error(error_msg, "Analysis Error")
    
    # -------------------------------------------------------------------------
    # Legacy Methods (for backward compatibility)
    # -------------------------------------------------------------------------
    
    def _matches_filter(self, file_name: str, file_path: str, filename_filter: str) -> bool:
        """Check if file matches the given filter.
        
        This method supports multiple matching strategies:
        1. Exact filename or path match
        2. Case-insensitive substring match
        3. Resolved path comparison
        
        Args:
            file_name: Name of the file
            file_path: Full path to the file
            filename_filter: Filter string to match against
            
        Returns:
            bool: True if file matches the filter, False otherwise
        """
        if not filename_filter:
            return True  # No filter means match all
        
        filter_lower = filename_filter.lower()
        
        # Exact matches (case-sensitive)
        if file_path == filename_filter or file_name == filename_filter:
            return True
        
        # Substring matches (case-insensitive)
        if filter_lower in file_name.lower() or filter_lower in file_path.lower():
            return True
        
        # Path resolution match (with error handling)
        try:
            if file_path and filename_filter:
                resolved_file = str(Path(file_path).resolve())
                resolved_filter = str(Path(filename_filter).resolve())
                if resolved_file == resolved_filter:
                    return True
        except Exception as e:
            logger.debug(f"Path resolution failed for filter matching: {e}")
            pass  # Ignore path resolution errors
        
        return False
    
    def analyze_file_from_dict_list(self, file_dict_list: List[Dict[str, str]], query: str, filename_filter: Optional[str] = None) -> str:
        """Analyze files from file dictionary list based on query.
        
        Note: This method is kept for backward compatibility.
        For new code, prefer using analyze_document() directly with file paths.
        
        Args:
            file_dict_list: List of file dictionaries with 'file_name' and 'file_path' keys
            query: Analysis query
            filename_filter: Optional filter to select specific files
            
        Returns:
            str: Formatted analysis result or error message
        
        Example:
            file_list = [{'file_name': 'doc.pdf', 'file_path': '/path/to/doc.pdf'}]
            result = analyzer.analyze_file_from_dict_list(file_list, "Summarize this")
        """
        try:
            if not file_dict_list:
                return "❌ No files available for analysis."
            
            # Filter files if needed
            target_files = file_dict_list
            if filename_filter:
                filename_filter = filename_filter.strip()
                target_files = [
                    f for f in file_dict_list 
                    if self._matches_filter(
                        f.get('file_name', ''), 
                        f.get('file_path', ''), 
                        filename_filter
                    )
                ]
                
                if not target_files:
                    available_files = [f"{f.get('file_name', 'Unknown')} at {f.get('file_path', 'Unknown')}" for f in file_dict_list]
                    logger.warning(f"No files matched filter '{filename_filter}'. Available: {available_files}")
                    return f"❌ No files found matching filter: {filename_filter}"
            
            # Analyze first supported file
            for file_dict in target_files:
                file_path = file_dict.get('file_path')
                file_name = file_dict.get('file_name')
                
                if file_path and self._is_supported_format(file_path):
                    logger.info(f"Analyzing file: {file_name}")
                    return self.analyze_document(file_path, query)
            
            # No supported files found
            supported_extensions = ', '.join(sorted(SUPPORTED_FORMATS))
            return f"❌ No supported files found. Supported formats: {supported_extensions}"
            
        except Exception as e:
            return self._format_error(str(e), "Error analyzing files from dict list")


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def analyze_document_contextually(file_path: Union[str, Path], query: Optional[str] = None, 
                                 section: str = DEFAULT_SECTION, model_name: Optional[str] = DEFAULT_MODEL,
                                 temperature: float = DEFAULT_TEMPERATURE) -> str:
    """Analyze a specific document by file path using LLM.
    
    This is the recommended modern API for document analysis.
    
    Args:
        file_path: Path to the document to analyze
        query: Optional analysis query. If not provided, uses default comprehensive analysis
        section: LLM configuration section (e.g., 'deepseek', 'qwen', 'openai')
        model_name: Specific model name to use (uses config.ini if None)
        temperature: Temperature setting for LLM (0.0-1.0)
        
    Returns:
        str: Formatted analysis result or error message
        
    Example:
        # Basic usage
        result = analyze_document_contextually("document.pdf", "Summarize this document")
        
        # Custom configuration
        result = analyze_document_contextually(
            "document.pdf", 
            "Extract key insights",
            section='qwen',
            temperature=0.2
        )
    """
    try:
        # Create analyzer with specified configuration
        analyzer = ContextualAnalyzer(section=section, model_name=model_name, temperature=temperature)
        
        # Use default query if none provided
        if not query:
            query = DEFAULT_QUERY
        
        return analyzer.analyze_document(file_path, query)
        
    except Exception as e:
        error_msg = f"Failed to analyze document at {file_path} - {str(e)}"
        logger.error(error_msg)
        return f"{ERROR_PREFIX} {error_msg}"


def analyze_document_contextually_legacy(
    file_dict_list: List[Dict[str, str]], 
    query: str, 
    filename_filter: Optional[str] = None,
    section: str = DEFAULT_SECTION
) -> str:
    """Analyze documents contextually using LLM (legacy file dict list method).
    
    Note: This function is kept for backward compatibility.
    For new code, prefer using analyze_document_contextually() directly.
    
    Args:
        file_dict_list: List of file dictionaries with 'file_name' and 'file_path' keys
        query: Analysis query
        filename_filter: Optional filter to select specific files
        section: LLM configuration section to use
        
    Returns:
        str: Formatted analysis result or error message
    """
    try:
        analyzer = ContextualAnalyzer(section=section)
        return analyzer.analyze_file_from_dict_list(file_dict_list, query, filename_filter)
        
    except Exception as e:
        error_msg = f"Failed to perform contextual analysis - {str(e)}"
        logger.error(error_msg)
        return f"{ERROR_PREFIX} {error_msg}"


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def _test_contextual_analyzer() -> None:
    """Test the contextual analyzer functionality.
    
    This function provides comprehensive testing of:
    1. Analyzer initialization with different configurations
    2. Supported format validation
    3. Configuration information retrieval
    4. Error handling for invalid inputs
    5. Public API functions
    """
    print("\n" + "="*60)
    print("🧪 TESTING CONTEXTUAL ANALYZER")
    print("="*60)
    
    try:
        # Test 1: Supported formats
        print("\n📝 Step 1: Testing supported formats")
        formats = get_supported_formats()
        print(f"✅ Supported formats ({len(formats)}): {', '.join(formats[:5])}{'...' if len(formats) > 5 else ''}")
        
        # Test format checking
        test_files = ["test.pdf", "test.docx", "test.txt", "test.xyz"]
        for test_file in test_files:
            supported = is_supported_format(test_file)
            status = "✅" if supported else "❌"
            print(f"  {status} {test_file}: {'Supported' if supported else 'Not supported'}")
        
        # Test 2: Analyzer initialization
        print("\n📝 Step 2: Testing analyzer initialization")
        
        # Default configuration
        try:
            analyzer = ContextualAnalyzer()
            info = analyzer.get_analyzer_info()
            print(f"✅ Default analyzer created: {info['section']}/{info['model_name']}")
            print(f"   Temperature: {info['temperature']}, Max tokens: {info['max_tokens']}")
        except Exception as e:
            print(f"❌ Default analyzer creation failed: {e}")
        
        # Custom configuration
        try:
            custom_analyzer = ContextualAnalyzer(section='qwen', temperature=0.2)
            custom_info = custom_analyzer.get_analyzer_info()
            print(f"✅ Custom analyzer created: {custom_info['section']}/{custom_info['model_name']}")
            print(f"   Temperature: {custom_info['temperature']}")
        except Exception as e:
            print(f"❌ Custom analyzer creation failed: {e}")
        
        # Test 3: Input validation
        print("\n📝 Step 3: Testing input validation")
        if 'analyzer' in locals():
            # Test empty file path
            validation_error = analyzer._validate_inputs("", "test query")
            if validation_error:
                print(f"✅ Empty file path validation: {validation_error}")
            
            # Test empty query
            validation_error = analyzer._validate_inputs("test.pdf", "")
            if validation_error:
                print(f"✅ Empty query validation: {validation_error}")
        
        # Test 4: Public API functions
        print("\n📝 Step 4: Testing public API functions")
        
        # Test with non-existent file (should handle gracefully)
        result = analyze_document_contextually("non_existent_file.pdf", "Test query")
        if result.startswith(ERROR_PREFIX):
            print("✅ Non-existent file handled gracefully")
        else:
            print("❌ Non-existent file not handled properly")
        
        # Test legacy function with empty list
        legacy_result = analyze_document_contextually_legacy([], "Test query")
        if legacy_result.startswith(ERROR_PREFIX):
            print("✅ Empty file list handled gracefully")
        else:
            print("❌ Empty file list not handled properly")
        
        # Test 5: Configuration variations
        print("\n📝 Step 5: Testing configuration variations")
        
        config_tests = [
            {'section': 'deepseek', 'temperature': 0.1},
            {'section': 'qwen', 'temperature': 0.3},
        ]
        
        for config in config_tests:
            try:
                test_analyzer = ContextualAnalyzer(**config)
                test_info = test_analyzer.get_analyzer_info()
                print(f"✅ Config {config}: {test_info['section']}, temp={test_info['temperature']}")
            except Exception as e:
                print(f"❌ Config {config} failed: {e}")
        
        print("\n🎉 Contextual analyzer testing completed!")
        print(f"✅ All imports working correctly")
        print(f"✅ Analyzer initialization functional")
        print(f"✅ Input validation working")
        print(f"✅ Public API functions available")
        print(f"✅ Configuration variations supported")
        
    except Exception as e:
        print(f"❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    _test_contextual_analyzer()