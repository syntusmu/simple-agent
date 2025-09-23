"""
Document Retriever Tool for RAG functionality.

This module provides a wrapper to make RRFRetriever compatible with LangChain BaseRetriever
and enhances documents with file name information for better context.
"""

import logging
from typing import List

from langchain.schema import BaseRetriever, Document
from ..service.rag.retriever import RRFRetriever

# Configure logging
logger = logging.getLogger(__name__)


class DocumentRetrieverWrapper(BaseRetriever):
    """Wrapper to make RRFRetriever compatible with LangChain BaseRetriever."""
    
    rrfretriever: RRFRetriever
    
    def __init__(self, rrfretriever: RRFRetriever):
        """Initialize wrapper with RRFRetriever instance."""
        super().__init__(rrfretriever=rrfretriever)
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents using RRFRetriever with file names included."""
        try:
            documents = self.rrfretriever.retrieve(query)
            
            # If no documents found, return a document indicating this
            if not documents:
                no_results_doc = Document(
                    page_content="NO_DOCUMENTS_FOUND: The knowledge base search returned no relevant documents for this query. Consider using your built-in knowledge to answer the question.",
                    metadata={"source": "system", "query": query}
                )
                return [no_results_doc]
            
            # Enhance documents with file name information
            enhanced_documents = []
            for doc in documents:
                # Extract file name from metadata
                file_name = self._extract_file_name(doc.metadata)
                
                # Format content with file name header
                if file_name:
                    enhanced_content = f"📄 **Source: {file_name}**\n\n{doc.page_content}"
                else:
                    enhanced_content = doc.page_content
                
                # Create enhanced document
                enhanced_doc = Document(
                    page_content=enhanced_content,
                    metadata=doc.metadata
                )
                enhanced_documents.append(enhanced_doc)
            
            return enhanced_documents
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            # Return error document instead of empty list
            error_doc = Document(
                page_content=f"RETRIEVAL_ERROR: An error occurred during document retrieval: {str(e)}. Consider using your built-in knowledge to answer the question.",
                metadata={"source": "system", "error": str(e)}
            )
            return [error_doc]
    
    def _extract_file_name(self, metadata: dict) -> str:
        """Extract file name from document metadata."""
        # Try different possible metadata keys for file name
        possible_keys = ['source', 'file_name', 'filename', 'file', 'document_name', 'name']
        
        for key in possible_keys:
            if key in metadata and metadata[key]:
                file_path = str(metadata[key])
                # Extract just the filename from full path
                if '/' in file_path:
                    return file_path.split('/')[-1]
                elif '\\' in file_path:
                    return file_path.split('\\')[-1]
                else:
                    return file_path
        
        # If no file name found, return empty string
        return ""


def create_document_retriever_wrapper(rrfretriever: RRFRetriever) -> DocumentRetrieverWrapper:
    """
    Factory function to create a DocumentRetrieverWrapper instance.
    
    Args:
        rrfretriever: RRFRetriever instance
        
    Returns:
        Configured DocumentRetrieverWrapper
    """
    return DocumentRetrieverWrapper(rrfretriever)
