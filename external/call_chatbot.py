
import sys
import os
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.chat.chatbot import simple_chat, chat_with_agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Test function for the chatbot."""
print("Testing Simple Chatbot...")

try:
    # # Test simple chat
    # response1 = simple_chat("My name is Sunny")
    # print(f"Response 1: {response1}")
    
    # response2 = simple_chat("what is my name")
    # print(f"Response 2: {response2}")

    # Test with file upload
    file_list3 = [
        # {
        #     "file_name": "各地产假规定一览表  to GE.xlsx",
        #     "file_path": "data/各地产假规定一览表  to GE.xlsx"
        # }
    ]
    
    response3 = chat_with_agent(
        user_id="test_user",
        user_session="test_session",
        user_query="上海和北京的产假政策的区别",
        file_upload_list=file_list3
    )
    print(f"Response 3: {response3}")

    # file_list4 = [
    #     {
    #         "file_name": "atm_test_data.xlsx",
    #         "file_path": "data/atm_test_data.xlsx"
    #     }
    # ]
    # response4 = chat_with_agent(
    #     user_id="test_user",
    #     user_session="test_session",
    #     user_query="analyze the excel uploaded, Show summary statistics",
    #     file_upload_list=file_list4
    # )
    # print(f"Response 4: {response4}")
    
except Exception as e:
    logger.error(f"Error testing chatbot: {e}")