"""Contextual document analyzer using docling and LLM."""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import tiktoken

from ..service.llm.llm_interface import create_llm
from ..service.prompt.prompt import create_contextual_analysis_prompt
from ..service.document.docling import DocumentProcessor

# Constants
MAX_TOKENS = 6000
SUPPORTED_FORMATS = set(DocumentProcessor.SUPPORTED_FORMATS.keys())
SUPPORTED_FORMATS.update({'.txt', '.markdown', '.xls'})

logger = logging.getLogger(__name__)


class ContextualAnalyzer:
    """Contextual document analyzer using docling and LLM."""
    
    def __init__(self, llm_provider: str = 'deepseek', model_name: str = None):
        """Initialize the analyzer with LLM and tokenizer."""
        # Use model from config.ini if not specified
        self.llm = create_llm(provider=llm_provider, model_name=model_name, temperature=0.1)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info(f"Contextual analyzer initialized with {llm_provider} {model_name}")
    
    def _is_supported_format(self, file_path: Union[str, Path]) -> bool:
        """Check if file format is supported."""
        return Path(file_path).suffix.lower() in SUPPORTED_FORMATS
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text with fallback estimation."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            return len(text) // 4  # Fallback estimation
    
    def _convert_to_markdown(self, file_path: Union[str, Path]) -> tuple[str, Optional[str]]:
        """Convert document to markdown."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return "", f"File not found: {file_path}"
            
            if not self._is_supported_format(file_path):
                return "", f"Unsupported file format: {file_path.suffix}"
            
            # Handle text files directly
            if file_path.suffix.lower() in ['.txt', '.markdown']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return f.read(), None
                except Exception as e:
                    return "", f"Error reading text file: {str(e)}"
            
            # Use DocumentProcessor for other formats
            processor = DocumentProcessor(input_file=file_path)
            markdown_content, _ = processor.process_document()
            return markdown_content, None
                
        except Exception as e:
            error_msg = f"Error converting {file_path} to markdown: {str(e)}"
            logger.error(error_msg)
            return "", error_msg
    
    def analyze_document(self, file_path: Union[str, Path], user_query: str) -> str:
        """Analyze document content based on user query."""
        try:
            # Convert document to markdown
            markdown_content, error = self._convert_to_markdown(file_path)
            
            if error:
                return f"❌ Conversion Error: {error}"
            
            if not markdown_content.strip():
                return f"❌ Error: No content extracted from {Path(file_path).name}"
            
            # Check token limit
            token_count = self._count_tokens(markdown_content)
            if token_count > MAX_TOKENS:
                return (f"❌ Document too large: {token_count} tokens "
                       f"(limit: {MAX_TOKENS}). Please use a smaller document.")
            
            # Perform analysis
            return self._perform_analysis(markdown_content, user_query, Path(file_path).name, token_count)
            
        except Exception as e:
            error_msg = f"Error analyzing document {file_path}: {str(e)}"
            logger.error(error_msg)
            return f"❌ Analysis Error: {error_msg}"
    
    def _perform_analysis(self, content: str, query: str, filename: str, token_count: int) -> str:
        """Perform contextual analysis using LLM."""
        try:
            system_prompt = create_contextual_analysis_prompt()
            user_message = f"""
Document: {filename}
Content Length: {token_count} tokens
User Query: {query}

Document Content:
{content}

Please analyze the document content and provide a comprehensive response to the user's query.
"""
            
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ])
            
            return f"""
📄 **Document Analysis: {filename}**
📊 **Content Size:** {token_count} tokens
❓ **Query:** {query}

**Analysis Result:**
{response.content}
"""
            
        except Exception as e:
            error_msg = f"Error in contextual analysis: {str(e)}"
            logger.error(error_msg)
            return f"❌ Analysis Error: {error_msg}"
    
    def analyze_file_from_dict_list(self, file_dict_list: List[Dict[str, str]], query: str, filename_filter: Optional[str] = None) -> str:
        """Analyze files from file dictionary list based on query."""
        try:
            if not file_dict_list:
                return "❌ No files available for analysis."
            
            # Filter files if needed
            target_files = file_dict_list
            if filename_filter:
                filter_lower = filename_filter.lower()
                target_files = [
                    f for f in file_dict_list 
                    if filter_lower in f.get('file_name', '').lower()
                ]
            
            if not target_files:
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
            error_msg = f"Error analyzing files from dict list: {str(e)}"
            logger.error(error_msg)
            return f"❌ Error: {error_msg}"


def analyze_document_contextually(
    file_dict_list: List[Dict[str, str]], 
    query: str, 
    filename_filter: Optional[str] = None
) -> str:
    """Analyze documents contextually using LLM."""
    try:
        analyzer = ContextualAnalyzer()
        return analyzer.analyze_file_from_dict_list(file_dict_list, query, filename_filter)
        
    except Exception as e:
        logger.error(f"analyze_document_contextually failed: {str(e)}")
        return f"❌ Error: Failed to perform contextual analysis - {str(e)}"


def get_supported_formats() -> List[str]:
    """Get list of supported file formats."""
    return sorted(list(SUPPORTED_FORMATS))