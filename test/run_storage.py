#!/usr/bin/env python3
"""Direct execution of storage function"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app.utils.store_utils import store_document
    
    # Target file
    input_path = "data/各地产假规定一览表  to GE.xlsx"
    full_path = project_root / input_path
    
    print("🚀 Starting document storage process...")
    print(f"📄 Target document: {input_path}")
    print(f"📂 Full path: {full_path}")
    print(f"📁 File exists: {full_path.exists()}")
    
    if full_path.exists():
        print("📊 Storing document in ChromaDB...")
        result = store_document(str(full_path))
        
        print("\n" + "="*50)
        if result["success"]:
            print("🎉 Document storage completed successfully!")
            chunks_count = result.get('chunks_stored', 'unknown')
            print(f"📄 Chunks stored: {chunks_count}")
        else:
            print("💥 Document storage failed!")
            print(f"🚫 Error: {result.get('error', 'Unknown error')}")
        print("="*50)
    else:
        print("❌ File not found!")
        print("Available files in data directory:")
        data_dir = project_root / "data"
        if data_dir.exists():
            for file in data_dir.iterdir():
                print(f"  - {file.name}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
