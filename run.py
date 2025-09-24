#!/usr/bin/env python3
"""
Entry point script for the Maternity Agent Dialog application.

This script serves as the main entry point to run the Flask web server.
It imports and runs the Flask app from the app.web.main module.
"""

import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the Flask app
from app.web.main import app, logger

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5051))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Maternity Agent Dialog Server on port {port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host='0.0.0.0', port=port, debug=True)
