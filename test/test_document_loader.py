#!/usr/bin/env python3
"""
Test script for document_loader.py functionality
"""
import sys
import traceback
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

def test_document_loader():
    """Test the MultiFileLoader functionality."""
    try:
        from app.service.document.document_loader import MultiFileLoader
        
        pdf_path = "data/2411.04602v1.pdf"
        
        print(f"Testing MultiFileLoader with: {pdf_path}")
        print(f"File exists: {Path(pdf_path).exists()}")
        
        # Test 1: Initialize loader with docling enabled
        print("\n=== Test 1: Initialize with docling ===")
        loader = MultiFileLoader(
            path=pdf_path,
            show_progress=True,
            use_docling=True
        )
        print("✓ MultiFileLoader initialized successfully with docling")
        
        # Test 2: Initialize loader without docling
        print("\n=== Test 2: Initialize without docling ===")
        loader_no_docling = MultiFileLoader(
            path=pdf_path,
            show_progress=True,
            use_docling=False
        )
        print("✓ MultiFileLoader initialized successfully without docling")
        
        # Test 3: Load documents (this might take time due to docling processing)
        print("\n=== Test 3: Load documents with docling ===")
        print("Loading documents... (this may take a while for docling processing)")
        docs = loader.load_all()
        
        if docs:
            print(f"✓ Successfully loaded {len(docs)} documents")
            
            # Check first document
            first_doc = docs[0]
            print(f"✓ First document has {len(first_doc.page_content)} characters")
            print(f"✓ Metadata: {first_doc.metadata}")
            
            # Check if docling was used
            if first_doc.metadata.get('processed_with') == 'docling':
                print("✓ Document was processed with docling")
            else:
                print("! Document was not processed with docling")
                
        else:
            print("✗ No documents were loaded")
            
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure the document_loader module is accessible")
        return False
        
    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality without docling."""
    try:
        from app.service.document.document_loader import MultiFileLoader
        
        # Test with a simple text file if PDF processing is slow
        test_content = "This is a test document for testing the loader."
        test_file = Path("test_doc.txt")
        
        # Create a test file
        with open(test_file, "w") as f:
            f.write(test_content)
        
        print(f"\n=== Basic Functionality Test ===")
        print(f"Created test file: {test_file}")
        
        # Load the test file
        loader = MultiFileLoader(
            path=str(test_file),
            show_progress=True,
            use_docling=False  # Disable docling for quick test
        )
        
        docs = loader.load_all()
        
        if docs:
            print(f"✓ Successfully loaded {len(docs)} documents")
            print(f"✓ Content matches: {docs[0].page_content.strip() == test_content}")
        else:
            print("✗ Failed to load test document")
        
        # Clean up
        test_file.unlink()
        print("✓ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting document_loader.py tests...\n")
    
    # Run basic functionality test first
    basic_success = test_basic_functionality()
    
    if basic_success:
        print("\n" + "="*50)
        # Run full PDF test
        pdf_success = test_document_loader()
        
        if pdf_success:
            print("\n🎉 All tests passed!")
        else:
            print("\n❌ PDF test failed")
    else:
        print("\n❌ Basic functionality test failed")
