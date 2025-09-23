"""
Flask Web Server for Maternity Agent Dialog

This module creates a Flask web server with simplified API routes:
1. Stream chat with file upload for QA
2. Static chat with file upload for QA  
3. Upload documents to vector database
4. Delete documents from vector database
5. List documents in vector database
6. Static chat UI

Author: Maternity Agent Dialog Team
Version: 2.0 (Simplified API)
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
import sys
import tempfile
import logging
import secrets
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

# Third-party imports
from flask import Flask, request, jsonify, send_from_directory, Response, render_template, session
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage


# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

# Local imports
from app.utils.qa_utils import chat_with_agent, chat_with_agent_stream
from app.utils.store_utils import store_document



# =============================================================================
# CONSTANTS
# =============================================================================

# Allowed file extensions for security
ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.csv', '.json'}

# File upload limits
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size
MAX_QUERY_LENGTH = 10000  # 10KB limit for query

# Swagger UI configuration
SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.yaml'  # Our API url (can of course be a local resource)

# Cache file path



# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_allowed_file(filename: str) -> bool:
    """Check if the file extension is allowed.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file extension is allowed, False otherwise
    """
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_input_params(user_id: Optional[str], user_session: Optional[str], user_query: Optional[str]) -> Optional[Dict[str, Any]]:
    """Validate required input parameters.
    
    Args:
        user_id: User identifier
        user_session: User session identifier
        user_query: User's query text
        
    Returns:
        Error response dict if validation fails, None if valid
    """
    if not user_id or not user_id.strip():
        return {
            "success": False,
            "error": "user_id is required and cannot be empty"
        }
        
    if not user_session or not user_session.strip():
        return {
            "success": False,
            "error": "user_session is required and cannot be empty"
        }
        
    if not user_query or not user_query.strip():
        return {
            "success": False,
            "error": "user_query is required and cannot be empty"
        }
        
    # Basic length validation
    if len(user_query) > MAX_QUERY_LENGTH:
        return {
            "success": False,
            "error": f"user_query is too long (max {MAX_QUERY_LENGTH:,} characters)"
        }
        
    return None


def process_file_uploads(files: Dict[str, FileStorage]) -> Optional[List[Dict[str, str]]]:
    """Process uploaded files and return file information list using temporary files.
    
    Args:
        files: Dictionary of uploaded files from Flask request
        
    Returns:
        List of file information dictionaries or None if no valid files
    """
    if not files:
        return None
        
    file_upload_list = []
    
    for file_key in files:
        file = files[file_key]
        logger.debug(f"Processing uploaded file: {file.filename}")
        
        if file and file.filename:
            try:
                # Check file extension
                if not is_allowed_file(file.filename):
                    logger.warning(f"Skipping file with disallowed extension: {file.filename}")
                    continue
                
                # Secure the filename
                fn = secure_filename(file.filename)
                if not fn:  # Skip if filename becomes empty after securing
                    logger.warning(f"Skipping file with invalid filename: {file.filename}")
                    continue
                
                # Get file extension for temporary file
                _, ext = os.path.splitext(file.filename)
                
                # Create temporary file with proper extension and simple prefix
                temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="upload_")
                
                try:
                    # Save the file to temporary location
                    with os.fdopen(temp_fd, 'wb') as temp_file:
                        file.seek(0)  # Reset file pointer
                        temp_file.write(file.read())
                    
                    file_upload_list.append({
                        "file_name": file.filename,
                        "file_path": temp_path
                    })
                    
                    logger.info(f"Successfully saved file to temporary location: {temp_path}")
                    
                except Exception as e:
                    # Clean up the temporary file if something goes wrong
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    raise e
                
            except Exception as e:
                logger.error(f"Failed to save file {file.filename}: {str(e)}")
                continue
    
    return file_upload_list if file_upload_list else None



def create_error_response(error_message: str, status_code: int = 500) -> tuple:
    """Create a standardized error response.
    
    Args:
        error_message: The error message to include
        status_code: HTTP status code (default: 500)
        
    Returns:
        Tuple of (JSON response, status code)
    """
    return jsonify({
        "success": False,
        "error": error_message
    }), status_code



# =============================================================================
# FLASK APP CONFIGURATION
# =============================================================================

# Initialize logger
logger = logging.getLogger(__name__)



# =============================================================================
# FLASK APP CONFIGURATION
# =============================================================================

# Initialize logger
logger = logging.getLogger(__name__)

# Create Flask app with template folder configuration
app = Flask(__name__, template_folder='templates')
CORS(app, supports_credentials=True)  # Enable CORS for all routes with credentials support

# Configure Flask app
app.config['JSON_AS_ASCII'] = False  # Support for non-ASCII characters
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE  # 16MB max file size
app.config['SECRET_KEY'] = secrets.token_hex(16)  # Generate a random secret key for sessions

# Swagger UI blueprint configuration
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,  # Swagger UI static files will be mapped to '{SWAGGER_URL}/dist/'
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "Maternity Agent Dialog API v2.0",
        'supportedSubmitMethods': ['get', 'post'],
        'tryItOutEnabled': True,
        'docExpansion': 'list',
        'defaultModelsExpandDepth': 3,
        'defaultModelExpandDepth': 3,
    }
)

# Register Swagger UI blueprint
app.register_blueprint(swaggerui_blueprint)


# =============================================================================
# STATIC AND UTILITY ROUTES
# =============================================================================

@app.route('/static/swagger.yaml')
def swagger_spec():
    """Serve the Swagger specification file."""
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), 'swagger.yaml')


@app.route('/', methods=['GET'])
def home():
    """Home endpoint to check if the server is running and show available API endpoints."""
    return jsonify({
        "message": "Maternity Agent Dialog Server v2.0 is running!",
        "version": "2.0",
        "description": "Simplified API for maternity agent dialog with streaming chat, document management, and vector database operations",
        "endpoints": {
            "chat_stream": "/api/chat/stream (streaming chat with optional file upload for QA)",
            "chat_static": "/api/chat/static (non-streaming chat with optional file upload for QA)",
            "upload_vector": "/api/upload (upload documents to vector database)",
            "delete_vector": "/api/vector/delete (delete documents from vector database)",
            "list_vector": "/api/vector/list (list documents in vector database)",
            "chat_ui": "/chat (web interface)",
            "api_docs": "/api/docs (Swagger documentation)"
        },
        "supported_file_types": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024)
    })


@app.route('/chat', methods=['GET'])
def chat_ui():
    """Serve the chat interface HTML page."""
    return render_template('chat.html')


# =============================================================================
# CHAT API ROUTES
# =============================================================================


@app.route('/api/chat/stream', methods=['POST'])
def chat_stream_endpoint():
    """
    Streaming chat endpoint that returns Server-Sent Events (SSE).
    
    Supports both JSON and multipart/form-data:
    - JSON: {"user_id": "string", "user_session": "string", "user_query": "string"}
    - Form data: user_id, user_session, user_query as form fields + files as file uploads
    
    File uploads are stored in session and persist until webpage is closed/refreshed.
    """
    try:
        file_upload_list = None
        
        # Check if request has files (multipart/form-data)
        if request.files:
            # Handle multipart/form-data with file uploads
            user_id = request.form.get('user_id')
            user_session = request.form.get('user_session')
            user_query = request.form.get('user_query')
            logger.info(f"Processing streaming file uploads for user: {user_id}")
            # Process uploaded files
            file_upload_list = process_file_uploads(request.files)
                
        else:
            # Handle JSON payload or form data without files
            data = request.get_json(force=True, silent=True)
            
            if not data:
                # Try form data as fallback
                user_id = request.form.get('user_id')
                user_session = request.form.get('user_session')
                user_query = request.form.get('user_query')
                
                if not all([user_id, user_session, user_query]):
                    return jsonify({
                        "success": False,
                        "error": "No JSON data or form data provided"
                    }), 400
            else:
                # Extract required parameters from JSON
                user_id = data.get('user_id')
                user_session = data.get('user_session')
                user_query = data.get('user_query')
        
        # Validate required parameters
        validation_error = validate_input_params(user_id, user_session, user_query)
        if validation_error:
            return jsonify(validation_error), 400
        
        # Session key for storing files per user session
        session_key = f"files_{user_session}"
        
        # Handle file session management
        if file_upload_list:
            # Store new uploaded files in session
            session[session_key] = file_upload_list
            logger.info(f"Stored {len(file_upload_list)} files in session for user_session: {user_session}")
        else:
            # Retrieve existing files from session for this user_session
            file_upload_list = session.get(session_key, None)
            if file_upload_list:
                logger.info(f"Retrieved {len(file_upload_list)} files from session for user_session: {user_session}")
        
        def generate_stream():
            """Generator function for streaming response with improved buffering."""
            try:
                # Send initial metadata
                import json
                
                start_data = json.dumps({
                    'type': 'start', 
                    'user_id': user_id, 
                    'user_session': user_session, 
                    'files_processed': len(file_upload_list) if file_upload_list else 0
                })
                yield f"data: {start_data}\n\n"
                
                # Stream the agent response
                logger.info(f"Processing streaming chat request for user: {user_id}, session: {user_session}")
                if file_upload_list:
                    logger.info(f"Processing streaming request with {len(file_upload_list)} uploaded files: {[f['file_name'] for f in file_upload_list]}")
                
                chunk_count = 0
                try:
                    for chunk in chat_with_agent_stream(
                        user_id=str(user_id),
                        user_session=str(user_session),
                        user_query=str(user_query),
                        file_upload_list=file_upload_list
                    ):
                        if chunk and chunk.strip():  # Only send non-empty, non-whitespace chunks
                            chunk_count += 1
                            chunk_data = json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)
                            yield f"data: {chunk_data}\n\n"
                            
                            # Add small delay to prevent overwhelming the client
                            import time
                            time.sleep(0.01)
                except Exception as stream_error:
                    logger.error(f"Error during streaming: {str(stream_error)}")
                    error_data = json.dumps({'type': 'error', 'error': f"Streaming error: {str(stream_error)}"})
                    yield f"data: {error_data}\n\n"
                    return
                
                # Send completion signal with chunk count for debugging
                end_data = json.dumps({'type': 'end', 'chunks_sent': chunk_count})
                yield f"data: {end_data}\n\n"
                
                logger.info(f"Streaming completed for user {user_id}, sent {chunk_count} chunks")
                
            except Exception as e:
                import json
                error_msg = f"Streaming error: {str(e)}"
                logger.error(error_msg)
                error_data = json.dumps({'type': 'error', 'error': error_msg})
                yield f"data: {error_data}\n\n"
        
        # Return Server-Sent Events response with proper headers
        response = Response(
            generate_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type, Cache-Control',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'X-Accel-Buffering': 'no'  # Disable nginx buffering
            }
        )
        # Don't set Transfer-Encoding manually - Flask handles this
        response.charset = 'utf-8'
        return response
        
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500


@app.route('/api/chat/static', methods=['POST'])
def chat_static_endpoint():
    """
    Non-streaming chat endpoint that returns a complete response.
    
    Supports both JSON and multipart/form-data:
    - JSON: {"user_id": "string", "user_session": "string", "user_query": "string"}
    - Form data: user_id, user_session, user_query as form fields + files as file uploads
    
    File uploads are stored in session and persist until webpage is closed/refreshed.
    """
    try:
        file_upload_list = None
        
        # Check if request has files (multipart/form-data)
        if request.files:
            # Handle multipart/form-data with file uploads
            user_id = request.form.get('user_id')
            user_session = request.form.get('user_session')
            user_query = request.form.get('user_query')
            logger.info(f"Processing streaming file uploads for user: {user_id}")
            # Process uploaded files
            file_upload_list = process_file_uploads(request.files)
                
        else:
            # Handle JSON payload or form data without files
            data = request.get_json(force=True, silent=True)
            
            if not data:
                # Try form data as fallback
                user_id = request.form.get('user_id')
                user_session = request.form.get('user_session')
                user_query = request.form.get('user_query')
                
                if not all([user_id, user_session, user_query]):
                    return jsonify({
                        "success": False,
                        "error": "No JSON data or form data provided"
                    }), 400
            else:
                # Extract required parameters from JSON
                user_id = data.get('user_id')
                user_session = data.get('user_session')
                user_query = data.get('user_query')
        
        # Validate required parameters
        validation_error = validate_input_params(user_id, user_session, user_query)
        if validation_error:
            return jsonify(validation_error), 400
        
        # Session key for storing files per user session
        session_key = f"files_{user_session}"
        
        # Handle file session management
        if file_upload_list:
            # Store new uploaded files in session
            session[session_key] = file_upload_list
            logger.info(f"Stored {len(file_upload_list)} files in session for user_session: {user_session}")
        else:
            # Retrieve existing files from session for this user_session
            file_upload_list = session.get(session_key, None)
            if file_upload_list:
                logger.info(f"Retrieved {len(file_upload_list)} files from session for user_session: {user_session}")
        
        def generate_stream():
            """Generator function for streaming response with improved buffering."""
            try:
                # Send initial metadata
                import json
                
                start_data = json.dumps({
                    'type': 'start', 
                    'user_id': user_id, 
                    'user_session': user_session, 
                    'files_processed': len(file_upload_list) if file_upload_list else 0
                })
                yield f"data: {start_data}\n\n"
                
                # Stream the agent response
                logger.info(f"Processing streaming chat request for user: {user_id}, session: {user_session}")
                if file_upload_list:
                    logger.info(f"Processing streaming request with {len(file_upload_list)} uploaded files: {[f['file_name'] for f in file_upload_list]}")
                
                chunk_count = 0
                try:
                    for chunk in chat_with_agent_stream(
                        user_id=str(user_id),
                        user_session=str(user_session),
                        user_query=str(user_query),
                        file_upload_list=file_upload_list
                    ):
                        if chunk and chunk.strip():  # Only send non-empty, non-whitespace chunks
                            chunk_count += 1
                            chunk_data = json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)
                            yield f"data: {chunk_data}\n\n"
                            
                            # Add small delay to prevent overwhelming the client
                            import time
                            time.sleep(0.01)
                except Exception as stream_error:
                    logger.error(f"Error during streaming: {str(stream_error)}")
                    error_data = json.dumps({'type': 'error', 'error': f"Streaming error: {str(stream_error)}"})
                    yield f"data: {error_data}\n\n"
                    return
                
                # Send completion signal with chunk count for debugging
                end_data = json.dumps({'type': 'end', 'chunks_sent': chunk_count})
                yield f"data: {end_data}\n\n"
                
                logger.info(f"Streaming completed for user {user_id}, sent {chunk_count} chunks")
                
            except Exception as e:
                import json
                error_msg = f"Streaming error: {str(e)}"
                logger.error(error_msg)
                error_data = json.dumps({'type': 'error', 'error': error_msg})
                yield f"data: {error_data}\n\n"
        
        # Return Server-Sent Events response with proper headers
        response = Response(
            generate_stream(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type, Cache-Control',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'X-Accel-Buffering': 'no'  # Disable nginx buffering
            }
        )
        # Don't set Transfer-Encoding manually - Flask handles this
        response.charset = 'utf-8'
        return response
        
    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500


# =============================================================================
# VECTOR DATABASE API ROUTES
# =============================================================================

@app.route('/api/upload', methods=['POST'])
def upload_vector_endpoint():
    """
    Upload documents to ChromaDB vector database for RAG (Retrieval-Augmented Generation).
    
    Accepts multipart/form-data with file uploads and optional form fields:
    - files: Document files to upload
    - user_id (optional): User ID for document tracking
    - session_id (optional): Session ID for document tracking
    
    Files are processed, chunked, embedded, and stored permanently in the vector database.
    Supports filename-based upsert (latest upload replaces previous versions).
    """
    try:
        # Check if request has files
        if not request.files:
            return jsonify({
                "success": False,
                "error": "No files provided"
            }), 400
        
        results = []
        
        # Process each uploaded file
        for file_key in request.files:
            file = request.files[file_key]
            logger.info(f"Processing uploaded file for vector storage: {file.filename}")
            
            if file and file.filename:
                try:
                    # Check file extension
                    if not is_allowed_file(file.filename):
                        results.append({
                            "file_name": file.filename,
                            "success": False,
                            "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                        })
                        continue
                    
                    # Secure the filename
                    fn = secure_filename(file.filename)
                    if not fn:  # Skip if filename becomes empty after securing
                        results.append({
                            "file_name": file.filename,
                            "success": False,
                            "error": "Invalid filename"
                        })
                        continue
                    
                    # Get file extension for temporary file
                    _, ext = os.path.splitext(file.filename)
                    
                    # Create temporary file with proper extension and simple prefix
                    temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="upload_")
                    
                    # Save the file to temporary location
                    with os.fdopen(temp_fd, 'wb') as temp_file:
                        file.seek(0)  # Reset file pointer
                        temp_file.write(file.read())
                    
                    absolute_path = temp_path
                    temp_filename = os.path.basename(temp_path)  # Temporary filename for internal use
                    
                    logger.info(f"Successfully saved file '{file.filename}' to temporary location: {temp_path}")
                    
                    # Store in vector database using store_utils (with user tracking)
                    # Extract optional user_id and session_id from form data
                    user_id = request.form.get('user_id')
                    session_id = request.form.get('session_id')
                    
                    store_result = store_document(
                        file_path=absolute_path,
                        user_id=user_id,
                        session_id=session_id,
                        upsert=True,  # Enable filename-based replacement
                        original_filename=file.filename  # Pass original filename
                    )
                    
                    if store_result["success"]:
                        results.append({
                            "file_name": file.filename,
                            "saved_as": temp_filename,  # Show temp filename for debugging if needed
                            "file_path": absolute_path,
                            "success": True,
                            "chunks_stored": store_result.get("chunks_stored", 0),
                            "user_id": user_id,
                            "session_id": session_id,
                            "storage_type": "vector_database",
                            "message": f"Successfully stored {store_result.get('chunks_stored', 0)} chunks in vector database"
                        })
                        logger.info(f"Successfully stored file '{file.filename}' in vector database with {store_result.get('chunks_stored', 0)} chunks")
                        
                        # Clean up temporary file after successful processing
                        try:
                            os.unlink(temp_path)
                            logger.debug(f"Cleaned up temporary file: {temp_path}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temporary file {temp_path}: {cleanup_error}")
                    else:
                        results.append({
                            "file_name": file.filename,
                            "saved_as": temp_filename,
                            "file_path": absolute_path,
                            "success": False,
                            "error": f"Failed to store in vector database: {store_result.get('error', 'Unknown error')}"
                        })
                        logger.error(f"Failed to store file '{file.filename}' in vector database: {store_result.get('error')}")
                        
                        # Clean up temporary file after failed processing
                        try:
                            os.unlink(temp_path)
                            logger.debug(f"Cleaned up temporary file after failure: {temp_path}")
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temporary file {temp_path}: {cleanup_error}")
                    
                except Exception as e:
                    logger.error(f"Failed to process file {file.filename}: {str(e)}")
                    results.append({
                        "file_name": file.filename,
                        "success": False,
                        "error": f"Processing failed: {str(e)}"
                    })
        
        # Calculate summary statistics
        successful_uploads = [r for r in results if r["success"]]
        failed_uploads = [r for r in results if not r["success"]]
        total_chunks = sum(r.get("chunks_stored", 0) for r in successful_uploads)
        
        
            
        
        return jsonify({
            "success": len(successful_uploads) > 0,
            "total_files": len(results),
            "successful_uploads": len(successful_uploads),
            "failed_uploads": len(failed_uploads),
            "total_chunks_stored": total_chunks,
            "storage_type": "vector_database",
            "message": "Files uploaded and stored in vector database",
            "results": results
        })
        
    except Exception as e:
        error_msg = f"Vector upload server error: {str(e)}"
        logger.error(error_msg)
        return create_error_response(error_msg, 500)


@app.route('/api/vector/delete', methods=['POST'])
def delete_vector_endpoint():
    """
    Delete documents from ChromaDB vector database by filename.
    
    Accepts JSON payload with file_name parameter:
    {"file_name": "filename.ext"}
    
    The file_name should match exactly with documents stored in the vector database.
    All document chunks with the matching filename will be deleted.
    """
    try:
        # Enhanced logging for debugging
        logger.info(f"Delete request received - Content-Type: {request.content_type}")
        logger.info(f"Delete request raw data: {request.get_data()}")
        
        data = request.get_json(force=True, silent=True)
        
        if not data:
            logger.warning("Delete request failed: No JSON data provided")
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        logger.info(f"Delete request JSON data: {data}")
        
        file_name = data.get('file_name')
        if not file_name:
            logger.warning(f"Delete request failed: file_name parameter missing from data: {data}")
            return jsonify({
                "success": False,
                "error": "file_name parameter is required"
            }), 400
        
        logger.info(f"Delete request for file_name: '{file_name}' (type: {type(file_name)}, length: {len(str(file_name)) if file_name else 0})")
        
        # Use store_utils.delete_document for consistent error handling and logging
        from app.utils.store_utils import delete_document
        
        logger.info(f"API request to delete document: {file_name}")
        
        # Delete documents by filename using store_utils
        delete_result = delete_document(file_name)
        
        if delete_result["success"]:
            logger.info(f"Successfully deleted {delete_result.get('documents_deleted', 0)} chunks for file: {file_name}")
            
            return jsonify({
                "success": True,
                "file_name": file_name,
                "deleted_count": delete_result.get("documents_deleted", 0),
                "message": f"Successfully deleted {delete_result.get('documents_deleted', 0)} chunks for file {file_name}",
                "operation": "delete",
                "timestamp": delete_result.get("timestamp")
            })
        else:
            # Handle different error cases with appropriate HTTP status codes
            error_msg = delete_result.get("error", "Unknown error")
            
            if "No documents found" in error_msg:
                return jsonify({
                    "success": False,
                    "file_name": file_name,
                    "error": error_msg,
                    "operation": "delete"
                }), 404
            else:
                return jsonify({
                    "success": False,
                    "file_name": file_name,
                    "error": error_msg,
                    "operation": "delete"
                }), 500
        
    except ValueError as ve:
        # Handle validation errors (e.g., empty filename)
        return jsonify({
            "success": False,
            "error": f"Validation error: {str(ve)}"
        }), 400
        
    except Exception as e:
        error_msg = f"Vector delete server error: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500


@app.route('/api/vector/list', methods=['GET'])
def list_vector_documents():
    """List all documents in the vector database directly from vector store."""
    try:
        logger.info("Loading data directly from vector store...")
        from app.service.vector.vector_store import create_vector_store
        
        vector_store = create_vector_store()
        docs_info = vector_store.get_all_documents_info()
        
        return jsonify(docs_info)
        
    except Exception as e:
        error_msg = f"Failed to list vector documents: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return jsonify({
            "success": False,
            "error": error_msg,
            "total_documents": 0,
            "total_chunks": 0,
            "unique_files": 0,
            "files_summary": [],
            "all_chunks": []
        }), 500


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
