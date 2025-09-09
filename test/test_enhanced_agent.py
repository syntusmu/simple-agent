#!/usr/bin/env python3
"""
Test script for enhanced agent file workflow functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.agent.agent import create_react_agent_instance

def test_enhanced_agent():
    """Test the enhanced agent with file-related queries."""
    print("=== Testing Enhanced Agent File Workflow ===")
    
    try:
        # Create agent instance
        agent = create_react_agent_instance()
        
        # Add some mock files to test the workflow
        agent.file_dict_list = [
            {
                'file_name': 'sales_data.xlsx',
                'file_path': '/Users/suyu/Library/CloudStorage/OneDrive-SingaporeManagementUniversity/Code/workspace/simple-agent/data/atm_test_data.xlsx'
            },
            {
                'file_name': 'research_paper.pdf', 
                'file_path': '/Users/suyu/Library/CloudStorage/OneDrive-SingaporeManagementUniversity/Code/workspace/simple-agent/data/2411.04602v1.pdf'
            },
            {
                'file_name': 'policy_document.md',
                'file_path': '/Users/suyu/Library/CloudStorage/OneDrive-SingaporeManagementUniversity/Code/workspace/simple-agent/data/2411.04602v1.md'
            }
        ]
        
        # Test 1: General file listing query
        print("\n--- Test 1: What files are available? ---")
        response1 = agent.run("What files do I have available for analysis?")
        print("Response:", response1[:500] + "..." if len(response1) > 500 else response1)
        
        # Test 2: Specific file search
        print("\n--- Test 2: Looking for Excel files ---")
        response2 = agent.run("Do I have any Excel files I can analyze?")
        print("Response:", response2[:500] + "..." if len(response2) > 500 else response2)
        
        # Test 3: Document analysis query
        print("\n--- Test 3: Document analysis request ---")
        response3 = agent.run("Can you analyze the research paper and tell me what it's about?")
        print("Response:", response3[:500] + "..." if len(response3) > 500 else response3)
        
        print("\n=== Test completed successfully ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_agent()
