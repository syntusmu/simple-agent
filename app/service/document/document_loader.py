import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Union, Any, Optional

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

# Optional imports with availability flags
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

try:
    from .excel_loader import ExcelLoader
    EXCEL_LOADER_AVAILABLE = True
except ImportError:
    EXCEL_LOADER_AVAILABLE = False

# Configure logger
logger = logging.getLogger(__name__)

if not DOCLING_AVAILABLE:
    logger.warning("Docling service not available. PDF and Word files will use standard loaders.")


class MultiFileLoader:
    """Document loader supporting multiple file formats using LangChain loaders."""

    # Class constants
    DOCLING_FORMATS = {".pdf", ".docx", ".doc"}
    
    FILE_LOADER_MAP = {
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
        ".xlsx": "excel_consolidated",
        ".xls": "excel_consolidated",
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
        temp_dir: Union[str, Path, None] = None,
        excel_header_row: Optional[int] = None
    ):
        """Initialize MultiFileLoader with configuration options."""
        self.path = Path(path)
        self.recurse = recurse
        self.show_progress = show_progress
        self.use_docling = use_docling and DOCLING_AVAILABLE
        self.excel_header_row = excel_header_row
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "docling_temp"
        
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._validate_path()
        
        if self.use_docling:
            logger.info("Docling preprocessing enabled for PDF and Word files")
        else:
            logger.info("Using standard loaders for all file types")

    # Validation methods
    def _validate_path(self) -> None:
        """Validate that the specified path exists and is accessible."""
        if not self.path.exists():
            raise FileNotFoundError(f"Path not found: {self.path}")
        
        if not (self.path.is_file() or self.path.is_dir()):
            raise ValueError(f"Path must be a file or directory: {self.path}")

    # Preprocessing methods
    def _preprocess_with_docling(self, file_path: Path) -> Path:
        """Preprocess PDF/Word files with docling to markdown format."""
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

    # Loader selection and configuration methods
    def _choose_loader(self, file_path: Path) -> tuple[Any, Path]:
        """Choose appropriate loader for the given file type."""
        ext = file_path.suffix.lower()
        actual_path = file_path
        
        # Handle docling preprocessing for supported formats
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
        
        # Handle special Excel loader cases
        if loader_cls == "excel_consolidated":
            return self._get_excel_loader(file_path, actual_path, "consolidated")
        
        # Handle standard loaders
        if loader_cls:
            if self.show_progress and actual_path == file_path:
                logger.info(f"Using {loader_cls.__name__} for {file_path.name}")
            return loader_cls(str(actual_path)), actual_path
        else:
            if self.show_progress:
                logger.info(f"Using UnstructuredFileLoader fallback for {file_path.name}")
            return UnstructuredFileLoader(str(actual_path)), actual_path

    def _get_excel_loader(self, file_path: Path, actual_path: Path, preferred_type: str) -> tuple[Any, Path]:
        """Get the best available Excel loader based on preferences and availability."""
        if preferred_type == "consolidated" and EXCEL_LOADER_AVAILABLE:
            if self.show_progress:
                logger.info(f"Using consolidated Excel loader for {file_path.name}")
            return "excel_consolidated", actual_path
        elif PANDAS_AVAILABLE:
            logger.warning(f"Consolidated Excel loader not available, falling back to pandas for {file_path.name}")
            if self.show_progress:
                logger.info(f"Using pandas Excel loader for {file_path.name}")
            return "pandas_excel", actual_path
        else:
            logger.warning(f"Neither consolidated nor pandas Excel loaders available, using UnstructuredExcelLoader for {file_path.name}")
            return UnstructuredExcelLoader(str(actual_path)), actual_path

    # Document loading methods
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
        """Load documents from a single file."""
        loader, actual_path = self._choose_loader(file_path)
        
        try:
            if self.show_progress:
                logger.info(f"Loading: {file_path.name}")
            
            # Handle special Excel loaders
            if loader == "excel_consolidated":
                return self._load_excel_with_consolidated_loader(file_path)
            elif loader == "pandas_excel":
                return self._load_excel_with_pandas(file_path)
            
            # Handle standard loaders
            docs = loader.load()
            
            # Update metadata for docling-processed files
            if actual_path != file_path:
                for doc in docs:
                    doc.metadata.update({
                        'original_source': str(file_path),
                        'processed_with': 'docling',
                        'markdown_source': str(actual_path)
                    })
            
            return docs
            
        except Exception as e:
            logger.warning(f"Could not load {file_path.name}: {e}")
            
            # Try pandas fallback for Excel files
            ext = file_path.suffix.lower()
            if ext in ['.xlsx', '.xls'] and PANDAS_AVAILABLE and loader == "excel_consolidated":
                logger.info(f"Trying pandas fallback for {file_path.name}")
                return self._load_excel_with_pandas(file_path)
            
            return []

    def _load_directory(self, docs: List[Document]) -> tuple[int, int]:
        """Load all supported files from a directory."""
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

    # Specialized Excel loading methods
    def _load_excel_with_consolidated_loader(self, file_path: Path) -> List[Document]:
        """Load Excel file using the consolidated Excel loader."""
        try:
            from .excel_loader import ExcelLoader
            
            loader = ExcelLoader(
                file_path=str(file_path),
                sheet_name=None,  # Load all sheets
                header_rows=[1, 2, 3, 4] if self.excel_header_row is None else [self.excel_header_row],
                preserve_merged_cells=True,
                backend='auto'
            )
            
            documents = loader.load()
            
            # Update metadata to match expected format
            for doc in documents:
                doc.metadata.update({
                    "loader": "consolidated_excel"
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading Excel file with consolidated loader: {e}")
            raise

    def _load_excel_with_pandas(self, file_path: Path) -> List[Document]:
        """Load Excel file using pandas as fallback."""
        try:
            excel_data = pd.read_excel(str(file_path), sheet_name=None)
            docs = []
            
            if isinstance(excel_data, dict):
                # Multiple sheets
                for sheet_name, df in excel_data.items():
                    if not df.empty:
                        df_cleaned = df.fillna('')
                        content = df_cleaned.to_string(index=False)
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
                    excel_data_cleaned = excel_data.fillna('')
                    content = excel_data_cleaned.to_string(index=False)
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

    # Cleanup methods
    def cleanup_temp_files(self) -> None:
        """Clean up temporary files created during processing."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")


def main() -> None:
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <path> [--recurse] [--progress] [--no-docling]")
        sys.exit(1)
    
    docs_path = sys.argv[1]
    recurse = "--recurse" in sys.argv
    show_progress = "--progress" in sys.argv
    use_docling = "--no-docling" not in sys.argv
    
    loader = None
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
        if loader:
            loader.cleanup_temp_files()


if __name__ == "__main__":
    main()
