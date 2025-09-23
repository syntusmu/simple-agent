import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

# Import common configuration for offline mode
from ...utils.common import initialize_offline_document_processing

from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.base import ImageRefMode
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    CsvFormatOption,
)
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    EasyOcrOptions,
    TableStructureOptions,
)
from docling.datamodel.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Offline mode and cache configuration is handled by common.py import above

class DocumentProcessor:
    """Document processor using Docling library for converting various formats to Markdown."""
    
    SUPPORTED_FORMATS = {
        '.pdf': InputFormat.PDF,
        '.docx': InputFormat.DOCX,
        '.html': InputFormat.HTML,
        '.pptx': InputFormat.PPTX,
        '.csv': InputFormat.CSV,
        '.xlsx': InputFormat.XLSX,
        '.md': InputFormat.MD,
        '.jpg': InputFormat.IMAGE,
        '.jpeg': InputFormat.IMAGE,
        '.png': InputFormat.IMAGE,
        '.bmp': InputFormat.IMAGE,
        '.tiff': InputFormat.IMAGE,
    }
    
    def __init__(self, input_file: Union[str, Path], output_dir: Optional[Union[str, Path]] = None):
        """Initialize DocumentProcessor.
        
        Args:
            input_file: Path to the input document
            output_dir: Output directory for processed files (defaults to 'data')
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file format is not supported
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir) if output_dir else Path("data")
        
        self._validate_input_file()
        self.doc_converter = self._initialize_document_converter()
        self._configure_settings()

    def _validate_input_file(self) -> None:
        """Validate that the input file exists and is supported.
        
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If file format is not supported
        """
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        if not self.input_file.is_file():
            raise ValueError(f"Input path is not a file: {self.input_file}")
        
        file_extension = self.input_file.suffix.lower()
        if file_extension not in self.SUPPORTED_FORMATS:
            supported_exts = ', '.join(self.SUPPORTED_FORMATS.keys())
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {supported_exts}"
            )

    def _initialize_document_converter(self) -> DocumentConverter:
        """Initialize the document converter with offline mode configuration using local models.
        
        Returns:
            Configured DocumentConverter instance
        """
        logger.info("Docling configured for offline mode using local model artifacts")
        
        pdf_pipeline = None
        if self.input_file.suffix.lower() == '.pdf':
            # Configure OCR options
            ocr_options = EasyOcrOptions(
                force_full_page_ocr=True, 
                lang=["en"]
            )
            
            # Configure local model paths for offline mode
            model_artifacts_path = Path("model_artifacts/docling").resolve()
            
            # Use TableFormer ACCURATE mode with local models
            pdf_pipeline = PdfPipelineOptions(
                artifacts_path=str(model_artifacts_path),
                do_table_structure=True,
                do_ocr=False,  # Disable OCR for offline mode
                ocr_options=ocr_options,
                table_structure_options=TableStructureOptions(
                    do_cell_matching=True,
                    mode=TableFormerMode.ACCURATE,  # Uses local accurate model
                    model_path=str(model_artifacts_path / "accurate" / "tableformer_accurate.safetensors")
                ),
                generate_page_images=False,
                generate_picture_images=False,
                images_scale=2.0,
                # Specify layout detection model path
                layout_model_path=str(model_artifacts_path / "model.safetensors")
            )

        return DocumentConverter(
            allowed_formats=list(self.SUPPORTED_FORMATS.values()),
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline),
                InputFormat.CSV: CsvFormatOption(pipeline_cls=SimplePipeline),
            },
        )


    def _configure_settings(self) -> None:
        """Configure Docling settings."""
        settings.debug.profile_pipeline_timings = True

    def process_document(self) -> Tuple[str, str]:
        """Process the document and convert to markdown.
        
        Returns:
            Tuple of (markdown_content, document_filename)
            
        Raises:
            Exception: If document conversion fails
        """
        try:
            logger.info(f"Converting document: {self.input_file}")
            result = self.doc_converter.convert(str(self.input_file))
            logger.debug(f"Conversion completed, result type: {type(result)}")
            return self._get_markdown(result)
        except Exception as e:
            logger.error(f"Failed to convert document {self.input_file}: {e}")
            raise

    def _get_markdown(self, result) -> Tuple[str, str]:
        """Extract markdown content from conversion result.
        
        Args:
            result: Document conversion result
            
        Returns:
            Tuple of (markdown_content, document_filename)
        """
        doc_filename = self.input_file.stem
        # md_content = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
        md_content = result.document.export_to_markdown(image_mode=ImageRefMode.PLACEHOLDER)
        logger.debug(f"Markdown content generated for {doc_filename}")
        return md_content, doc_filename

    def process(self) -> Path:
        """Complete processing: convert, get markdown, and save to output folder.
        
        Returns:
            Path to the saved markdown file
            
        Raises:
            OSError: If unable to create output directory or write file
        """
        try:
            markdown_content, doc_filename = self.process_document()
            
            # Create output directory
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save markdown file
            output_path = self.output_dir / f"{doc_filename}.md"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            logger.info(f"Markdown saved to {output_path.resolve()}")
            return output_path
            
        except OSError as e:
            logger.error(f"Failed to save markdown file: {e}")
            raise
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise

def main() -> None:
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python docling_service.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        processor = DocumentProcessor(input_file=input_file)
        output_path = processor.process()
        print(f"Successfully processed document. Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to process document: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()