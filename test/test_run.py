#!/usr/bin/env python3
"""Test script to verify Python execution"""

import sys
import os
from pathlib import Path

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Script location:", __file__)

# Test import
try:
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    from app.utils.store_utils import store_document
    print("✅ Successfully imported store_document")
    
    # Test file existence
    input_path = "data/HR 产假智能助手.pdf"
    full_path = project_root / input_path
    print(f"Target file path: {full_path}")
    print(f"File exists: {full_path.exists()}")
    
    if full_path.exists():
        print("🚀 Ready to store document!")
        result = store_document(str(full_path))
        print("Storage result:", result)
    else:
        print("❌ File not found")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
