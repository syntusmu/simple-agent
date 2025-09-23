import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Union, Any

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    BSHTMLLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader,
)

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from .docling import DocumentProcessor
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

logger = logging.getLogger(__name__)

if not DOCLING_AVAILABLE:
    logger.warning("Docling service not available. PDF and Word files will use standard loaders.")

class MultiFileLoader:
    """Document loader supporting multiple file formats using LangChain loaders."""

    DOCLING_FORMATS = {".pdf", ".docx", ".doc"}
    
    FILE_LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".xlsx": "pandas_excel",  # Use pandas for better Excel content extraction
        ".xls": "pandas_excel",   # Use pandas for better Excel content extraction
        ".csv": CSVLoader,
        ".html": BSHTMLLoader,
        ".htm": BSHTMLLoader,
        ".txt": TextLoader,
        ".md": TextLoader,
    }

    def __init__(
        self, 
        path: Union[str, Path], 
        recurse: bool = False, 
        show_progress: bool = False,
        use_docling: bool = True,
        temp_dir: Union[str, Path, None] = None
    ):
        """Initialize MultiFileLoader."""
        self.path = Path(path)
        self.recurse = recurse
        self.show_progress = show_progress
        self.use_docling = use_docling and DOCLING_AVAILABLE
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "docling_temp"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._validate_path()
        
        if self.use_docling:
            logger.info("Docling preprocessing enabled for PDF and Word files")
        else:
            logger.info("Using standard loaders for all file types")

    def _validate_path(self) -> None:
        """Validate that the specified path exists."""
        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")
        
        if not (self.path.is_file() or self.path.is_dir()):
            raise ValueError(f"Path must be a file or directory: {self.path}")
    
    def _preprocess_with_docling(self, file_path: Path) -> Path:
        """Preprocess PDF/Word files with docling to markdown."""
        try:
            if self.show_progress:
                logger.info(f"Preprocessing {file_path.name} with docling...")
            
            processor = DocumentProcessor(
                input_file=str(file_path),
                output_dir=self.temp_dir
            )
            
            markdown_path = processor.process()
            
            if self.show_progress:
                logger.info(f"Docling preprocessing completed: {markdown_path.name}")
            
            return markdown_path
            
        except Exception as e:
            logger.error(f"Docling preprocessing failed for {file_path.name}: {e}")
            raise
    
    def _choose_loader(self, file_path: Path) -> tuple[Any, Path]:
        """Choose appropriate loader for the given file."""
        ext = file_path.suffix.lower()
        actual_path = file_path
        
        # Check if we should preprocess with docling
        if self.use_docling and ext in self.DOCLING_FORMATS:
            try:
                actual_path = self._preprocess_with_docling(file_path)
                loader_cls = TextLoader
                if self.show_progress:
                    logger.info(f"Using TextLoader for docling-processed {file_path.name}")
            except Exception as e:
                logger.warning(f"Docling failed for {file_path.name}, using standard loader: {e}")
                loader_cls = self.FILE_LOADER_MAP.get(ext)
        else:
            loader_cls = self.FILE_LOADER_MAP.get(ext)
        
        # Handle pandas Excel loader specially
        if loader_cls == "pandas_excel":
            if PANDAS_AVAILABLE:
                if self.show_progress:
                    logger.info(f"Using pandas Excel loader for {file_path.name}")
                return "pandas_excel", actual_path
            else:
                logger.warning(f"Pandas not available, falling back to UnstructuredExcelLoader for {file_path.name}")
                loader_cls = UnstructuredExcelLoader
        
        if loader_cls:
            if self.show_progress and actual_path == file_path:
                logger.info(f"Using {loader_cls.__name__} for {file_path.name}")
            return loader_cls(str(actual_path)), actual_path
        else:
            if self.show_progress:
                logger.info(f"Using UnstructuredFileLoader fallback for {file_path.name}")
            return UnstructuredFileLoader(str(actual_path)), actual_path

    def load_all(self) -> List[Document]:
        """Load all documents from the specified path."""
        docs = []
        files_processed = 0
        files_failed = 0
        
        if self.path.is_file():
            docs.extend(self._load_single_file(self.path))
            files_processed = 1
        elif self.path.is_dir():
            files_processed, files_failed = self._load_directory(docs)
        
        logger.info(f"Loaded {len(docs)} documents from {files_processed} files")
        if files_failed > 0:
            logger.warning(f"Failed to load {files_failed} files")
        
        return docs
    
    def _load_single_file(self, file_path: Path) -> List[Document]:
        """Load a single file."""
        loader, actual_path = self._choose_loader(file_path)
        try:
            if self.show_progress:
                logger.info(f"Loading: {file_path.name}")
            
            # Handle pandas Excel loader specially
            if loader == "pandas_excel":
                return self._load_excel_with_pandas(file_path)
            
            docs = loader.load()
            
            # Update metadata if we used docling
            if actual_path != file_path:
                for doc in docs:
                    doc.metadata['original_source'] = str(file_path)
                    doc.metadata['processed_with'] = 'docling'
                    doc.metadata['markdown_source'] = str(actual_path)
            
            return docs
            
        except Exception as e:
            logger.warning(f"Could not load {file_path.name}: {e}")
            
            # Try pandas fallback for Excel files (only if not already using pandas)
            ext = file_path.suffix.lower()
            if ext in ['.xlsx', '.xls'] and PANDAS_AVAILABLE and loader != "pandas_excel":
                logger.info(f"Trying pandas fallback for {file_path.name}")
                return self._load_excel_with_pandas(file_path)
            
            return []
    
    def _load_excel_with_pandas(self, file_path: Path) -> List[Document]:
        """Load Excel file using pandas as fallback."""
        try:
            # Read Excel file with pandas
            excel_data = pd.read_excel(str(file_path), sheet_name=None)
            
            docs = []
            if isinstance(excel_data, dict):
                # Multiple sheets
                for sheet_name, df in excel_data.items():
                    if not df.empty:
                        content = df.to_string(index=False)
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': str(file_path),
                                'sheet_name': sheet_name,
                                'loader': 'pandas_excel',
                                'rows': len(df),
                                'columns': len(df.columns)
                            }
                        )
                        docs.append(doc)
            else:
                # Single sheet
                if not excel_data.empty:
                    content = excel_data.to_string(index=False)
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': str(file_path),
                            'loader': 'pandas_excel',
                            'rows': len(excel_data),
                            'columns': len(excel_data.columns)
                        }
                    )
                    docs.append(doc)
            
            logger.info(f"Pandas fallback loaded {len(docs)} sheets from {file_path.name}")
            return docs
            
        except Exception as e:
            logger.error(f"Pandas fallback failed for {file_path.name}: {e}")
            return []
    
    def _load_directory(self, docs: List[Document]) -> tuple[int, int]:
        """Load all files from a directory."""
        pattern = "**/*" if self.recurse else "*"
        files_processed = 0
        files_failed = 0
        
        for path in self.path.glob(pattern):
            if not path.is_file():
                continue
                
            file_docs = self._load_single_file(path)
            if file_docs:
                docs.extend(file_docs)
                files_processed += 1
            else:
                files_failed += 1
        
        return files_processed, files_failed
    

def main() -> None:
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <path> [--recurse] [--progress] [--no-docling]")
        sys.exit(1)
    
    docs_path = sys.argv[1]
    recurse = "--recurse" in sys.argv
    show_progress = "--progress" in sys.argv
    use_docling = "--no-docling" not in sys.argv
    
    try:
        loader = MultiFileLoader(
            path=docs_path, 
            recurse=recurse, 
            show_progress=show_progress,
            use_docling=use_docling
        )
        
        docs = loader.load_all()
        
        if docs:
            logger.info(f"Successfully loaded {len(docs)} documents")
            
            # Show summary of first 3 documents
            for i, doc in enumerate(docs[:3]):
                source = doc.metadata.get('source', 'Unknown Source')
                content_length = len(doc.page_content)
                logger.info(f"Document {i+1}: {source} ({content_length} characters)")
            
            if len(docs) > 3:
                logger.info(f"... and {len(docs) - 3} more documents")
            
            # Show docling processing statistics
            docling_processed = sum(1 for doc in docs if doc.metadata.get('processed_with') == 'docling')
            if docling_processed > 0:
                logger.info(f"Documents processed with docling: {docling_processed}/{len(docs)}")
        else:
            logger.warning("No documents were loaded")
            
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        sys.exit(1)
    finally:
        # Clean up temporary files
        if 'loader' in locals() and hasattr(loader, 'temp_dir'):
            try:
                import shutil
                if loader.temp_dir.exists():
                    shutil.rmtree(loader.temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {loader.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory: {e}")

if __name__ == "__main__":
    main()
