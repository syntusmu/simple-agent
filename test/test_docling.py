#!/usr/bin/env python3
"""
Simple test script for docling_service.py
"""
import sys
import traceback
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.service.document.docling_service import DocumentProcessor

def test_docling_service():
    """Test the DocumentProcessor with the PDF file."""
    pdf_path = "data/2411.04602v1.pdf"
    
    print(f"Testing DocumentProcessor with: {pdf_path}")
    print(f"File exists: {Path(pdf_path).exists()}")
    
    try:
        # Initialize processor
        print("Initializing DocumentProcessor...")
        processor = DocumentProcessor(input_file=pdf_path)
        print("✓ DocumentProcessor initialized successfully")
        
        # Process the document
        print("Processing document...")
        output_path = processor.process()
        print(f"✓ Document processed successfully!")
        print(f"✓ Output saved to: {output_path}")
        
        # Check if output file exists and get its size
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"✓ Output file size: {size} bytes")
        else:
            print("✗ Output file was not created")
            
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_docling_service()
