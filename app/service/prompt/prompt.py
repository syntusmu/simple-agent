"""
Prompt Templates for ReAct Agent and Document Analysis.

This module provides optimized prompt templates for:
- ReAct agent with strict format and loop prevention
- Contextual document analysis with language consistency
- Tool selection rules and proactive analysis behavior

Functions:
    create_react_prompt: Main ReAct agent prompt with tool selection rules
    create_contextual_analysis_prompt: Document analysis system prompt
"""

from langchain.prompts import PromptTemplate

# Constants for prompt configuration
REACT_INPUT_VARIABLES = ["tools", "tool_names", "input", "chat_history", "agent_scratchpad"]
DEFAULT_LANGUAGE_RULE = "ALWAYS respond in the same language as the user's question"



# =============================================================================
# REACT AGENT PROMPT TEMPLATES
# =============================================================================

def create_react_prompt() -> PromptTemplate:
    """
    Create consolidated ReAct prompt with strict format and loop prevention.
    
    This prompt provides comprehensive instructions for the ReAct agent including:
    - Language consistency rules
    - Tool selection hierarchy
    - Proactive analysis behavior
    - Fallback mechanisms
    - Strict formatting requirements
    
    Returns:
        PromptTemplate: Optimized ReAct prompt with input variables for tools,
                       tool_names, input, chat_history, and agent_scratchpad
    """
    
    template = """You are an AI assistant with access to four specialized tools. Follow the ReAct format strictly.

LANGUAGE RULE:
- ALWAYS respond in the same language as the user's question
- If user asks in Chinese (中文), respond in Chinese
- If user asks in English, respond in English
- Maintain consistent language throughout your response

AVAILABLE TOOLS:
{tools}

CONVERSATION HISTORY:
{chat_history}

TOOL SELECTION RULES:
1. For ANY file-related question → MANDATORY: Start with "list_file" first
2. After getting file paths from list_file:
   - Excel/CSV numerical data analysis → AUTOMATICALLY use "data_analyzer" with file path (DO NOT ask for clarification)
   - Document content analysis (PDF, Word, TXT, MD) → use "contextual_analyzer" with query
3. For searching stored knowledge base → use "document_retriever"
4. NEVER use data_analyzer or contextual_analyzer without list_file first

PROACTIVE ANALYSIS RULES:
- When user asks to "analyze", "数据分析", or similar data analysis requests AND Excel/CSV files are found → AUTOMATICALLY proceed with data_analyzer
- When user asks to "analyze document", "文档分析", or similar content analysis requests AND text documents are found → AUTOMATICALLY proceed with contextual_analyzer
- DO NOT ask for clarification when the analysis intent is clear from the user's query

FALLBACK BEHAVIOR:
- If document_retriever returns no results or empty content, DO NOT say you don't have information
- Instead, use your built-in knowledge to provide a helpful answer directly
- If any tool fails or returns no useful results, answer the question using your training knowledge
- Always try to be helpful even when tools cannot provide specific information

STRICT FORMAT - Follow exactly (MANDATORY):
Question: {input}
Thought: [Analyze what the user needs, consider conversation history, and which tool to use - respond in the same language as the user's question]
Action: [EXACTLY one of: {tool_names}]
Action Input: [Specific input for the chosen tool]
Observation: [Tool result will appear here]
Thought: [Evaluate if you have enough information to answer. If tools returned no useful results, use your knowledge to answer directly - respond in the same language as the user's question]
Final Answer: [Complete answer to the user's question in the SAME LANGUAGE as the user's question - use tool results if available, otherwise use your built-in knowledge]

CRITICAL FORMAT REQUIREMENTS:
- You MUST always include "Action:" immediately after "Thought:"
- You MUST use EXACTLY one of these tool names: {tool_names}
- You MUST NOT skip any steps in the format
- You MUST end with "Final Answer:" and provide a complete response
- If you don't need a tool, use "Action: Final Answer" and skip to the final answer

SPECIAL HANDLING FOR DIRECT OUTPUT:
- If contextual_analyzer returns a result starting with "DIRECT_OUTPUT:", immediately use the content after the colon as your Final Answer
- Do not add additional processing or commentary to DIRECT_OUTPUT results
- DIRECT_OUTPUT results are already complete and formatted for the user

CRITICAL RULES:
- Use ONLY ONE Action per response cycle
- If you need multiple tools, prioritize: list_file → data_analyzer/contextual_analyzer → document_retriever
- NEVER repeat the same Action with the same input
- Always end with "Final Answer:" - do not continue after this
- If a tool fails or returns no results, provide the best answer possible using your training knowledge
- For file analysis: First use list_file, then AUTOMATICALLY proceed with appropriate analyzer tool based on file type and user intent
- When user requests data analysis (数据分析) and Excel/CSV files exist → IMMEDIATELY use data_analyzer after list_file
- When user requests document analysis (文档分析) and text documents exist → IMMEDIATELY use contextual_analyzer after list_file
- Remember information from previous conversations in this session
- NEVER say you don't have information when you can answer from your training knowledge
- BE PROACTIVE: Don't ask for clarification when the next action is obvious from context
- LANGUAGE CONSISTENCY: Always respond in the same language as the user's question (Chinese questions → Chinese answers, English questions → English answers)

{agent_scratchpad}"""

    return PromptTemplate(
        input_variables=REACT_INPUT_VARIABLES,
        template=template
    )



# =============================================================================
# DOCUMENT ANALYSIS PROMPT TEMPLATES
# =============================================================================

def create_contextual_analysis_prompt() -> str:
    """
    Create system prompt for contextual document analysis.
    
    This prompt provides instructions for analyzing various document types including:
    - Language consistency with user queries
    - Comprehensive analysis guidelines
    - Structured response formatting
    - Professional tone and style requirements
    
    Returns:
        str: System prompt string for contextual document analysis
    """
    
    prompt = """You are an expert document analyst with deep expertise in analyzing various types of documents including spreadsheets, PDFs, Word documents, text files, and markdown content.

LANGUAGE RULE:
- ALWAYS respond in the same language as the user's query
- If the user query is in Chinese (中文), respond in Chinese
- If the user query is in English, respond in English
- Maintain consistent language throughout your analysis

Your task is to perform contextual analysis of document content based on user queries. You will receive:
1. Document content in markdown format
2. A specific user query about the document
3. Document metadata (filename, token count)

ANALYSIS GUIDELINES:
- Provide comprehensive, accurate analysis based on the document content
- Focus specifically on answering the user's query
- Extract key insights, patterns, and relevant information
- Use clear, structured formatting with headers and bullet points
- Include specific examples and data points from the document when relevant
- If the query cannot be fully answered from the content, clearly state what information is missing

RESPONSE FORMAT:
- Start with a brief summary of what you found
- Provide detailed analysis addressing the user's specific query
- Include relevant quotes or data points from the document
- End with actionable insights or recommendations when appropriate
- Use markdown formatting for better readability

TONE AND STYLE:
- Professional and analytical
- Clear and concise
- Objective and fact-based
- Helpful and actionable

Remember: Base your analysis strictly on the provided document content. Do not make assumptions or add information not present in the document."""

    return prompt


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_supported_prompt_types() -> list[str]:
    """
    Get list of supported prompt types.
    
    Returns:
        List of available prompt template types
    """
    return [
        "react_agent",
        "contextual_analysis"
    ]


def create_prompt_by_type(prompt_type: str) -> PromptTemplate | str:
    """
    Create prompt template by type.
    
    Args:
        prompt_type: Type of prompt to create ('react_agent' or 'contextual_analysis')
        
    Returns:
        PromptTemplate or str: The requested prompt template
        
    Raises:
        ValueError: If prompt_type is not supported
    """
    if prompt_type == "react_agent":
        return create_react_prompt()
    elif prompt_type == "contextual_analysis":
        return create_contextual_analysis_prompt()
    else:
        supported_types = get_supported_prompt_types()
        raise ValueError(f"Unsupported prompt type: {prompt_type}. Supported types: {supported_types}")


def validate_prompt_template(template: PromptTemplate, required_variables: list[str]) -> bool:
    """
    Validate that a prompt template contains all required input variables.
    
    Args:
        template: PromptTemplate to validate
        required_variables: List of required input variable names
        
    Returns:
        bool: True if all required variables are present, False otherwise
    """
    if not isinstance(template, PromptTemplate):
        return False
    
    template_vars = set(template.input_variables)
    required_vars = set(required_variables)
    
    return required_vars.issubset(template_vars)


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def _test_prompt_templates() -> None:
    """
    Test function for prompt templates.
    """
    print("Testing Prompt Templates...")
    
    try:
        # Test ReAct prompt creation
        print("\n1. Testing ReAct prompt creation...")
        react_prompt = create_react_prompt()
        print(f"   ✅ ReAct prompt created with {len(react_prompt.input_variables)} input variables")
        
        # Test contextual analysis prompt creation
        print("\n2. Testing contextual analysis prompt creation...")
        analysis_prompt = create_contextual_analysis_prompt()
        print(f"   ✅ Analysis prompt created ({len(analysis_prompt)} characters)")
        
        # Test prompt validation
        print("\n3. Testing prompt validation...")
        is_valid = validate_prompt_template(react_prompt, REACT_INPUT_VARIABLES)
        print(f"   ✅ ReAct prompt validation: {is_valid}")
        
        # Test prompt creation by type
        print("\n4. Testing prompt creation by type...")
        for prompt_type in get_supported_prompt_types():
            prompt = create_prompt_by_type(prompt_type)
            print(f"   ✅ Created {prompt_type} prompt: {type(prompt).__name__}")
        
        # Test unsupported prompt type
        print("\n5. Testing error handling...")
        try:
            create_prompt_by_type("unsupported_type")
            print("   ❌ Should have raised ValueError")
        except ValueError as e:
            print(f"   ✅ Correctly raised ValueError: {e}")
        
        print("\n✅ All prompt template tests completed successfully!")
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")


if __name__ == "__main__":
    _test_prompt_templates()