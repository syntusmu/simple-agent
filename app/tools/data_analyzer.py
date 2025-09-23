"""
Data Analyzer Tool for Excel and CSV Analysis.

This module provides comprehensive data analysis capabilities for Excel and CSV files
using LangChain's pandas dataframe agent with natural language queries.

Classes:
    ExcelPandasAnalyzer: Main analyzer class for Excel/CSV data analysis

Functions:
    analyze_excel_with_pandas(): Convenience function for file analysis
    _test_data_analyzer(): Testing function for analyzer functionality

Constants:
    MAX_ROWS: Maximum rows to load from files
    MAX_COLUMNS: Maximum columns to process
    SUPPORTED_FORMATS: List of supported file formats
    DEFAULT_PROVIDER: Default LLM provider for analysis
    DEFAULT_TEMPERATURE: Default temperature for LLM
    DEFAULT_MAX_ITERATIONS: Default max iterations for agent
"""

from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
import logging

# Import common SSL configuration
from ..utils.common import initialize_document_processing

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from ..service.llm.llm_interface import create_llm

# =============================================================================
# CONSTANTS
# =============================================================================

MAX_ROWS = 10000
MAX_COLUMNS = 100
SUPPORTED_FORMATS = ['.xlsx', '.xls', '.csv']
DEFAULT_PROVIDER = 'deepseek'
DEFAULT_TEMPERATURE = 0
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_QUERY = "Provide a comprehensive analysis of this dataset including summary statistics, data types, missing values, and key insights."

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN DATA ANALYZER CLASS
# =============================================================================

class ExcelPandasAnalyzer:
    """
    Excel and CSV data analysis using LangChain's pandas dataframe agent.
    
    This class provides comprehensive data analysis capabilities for Excel and CSV files
    using natural language queries powered by LLM agents.
    
    Attributes:
        llm: Language model for analysis
        current_df: Currently loaded pandas DataFrame
        current_agent: Current pandas dataframe agent
        provider: LLM provider used for analysis
        temperature: Temperature setting for LLM
        max_iterations: Maximum iterations for agent analysis
    
    Example:
        analyzer = ExcelPandasAnalyzer()
        result = analyzer.analyze_file("data.xlsx", "What are the top 5 values?")
    """
    
    def __init__(self, provider: str = DEFAULT_PROVIDER, temperature: float = DEFAULT_TEMPERATURE, 
                 max_iterations: int = DEFAULT_MAX_ITERATIONS):
        """
        Initialize the analyzer with specified LLM configuration.
        
        Args:
            provider (str): LLM provider to use (default: 'deepseek')
            temperature (float): Temperature for LLM responses (default: 0)
            max_iterations (int): Maximum iterations for agent (default: 5)
        """
        self.provider = provider
        self.temperature = temperature
        self.max_iterations = max_iterations
        
        try:
            # Initialize LLM with specified configuration
            self.llm = create_llm(section=provider, temperature=temperature)
            logger.info(f"Initialized data analyzer with {provider} LLM")
        except Exception as e:
            logger.error(f"Failed to initialize LLM with provider {provider}: {e}")
            self.llm = None
        
        self.current_df = None
        self.current_agent = None
    
    def _validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate file existence and format.
        
        Args:
            file_path (Union[str, Path]): Path to the file to validate
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported file format: {file_path.suffix}. Supported: {SUPPORTED_FORMATS}")
            return False
            
        return True
    
    def _load_dataframe(self, file_path: Path, sheet_name: Optional[Union[str, int]] = None) -> Optional[pd.DataFrame]:
        """
        Load data from file into pandas DataFrame.
        
        Args:
            file_path (Path): Path to the data file
            sheet_name (Optional[Union[str, int]]): Sheet name or index for Excel files
            
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if failed
        """
        try:
            # Load data based on file type
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=MAX_ROWS)
            else:
                # For Excel files, ensure we get a DataFrame, not a dict
                if sheet_name is None:
                    sheet_name = 0  # Default to first sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=MAX_ROWS)
                
                # If pd.read_excel returns a dict (multiple sheets), get the first one
                if isinstance(df, dict):
                    df = list(df.values())[0]
            
            # Limit columns if necessary
            if len(df.columns) > MAX_COLUMNS:
                df = df.iloc[:, :MAX_COLUMNS]
                logger.warning(f"Limited to first {MAX_COLUMNS} columns")
            
            logger.info(f"Loaded DataFrame with shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from {file_path}: {str(e)}")
            return None
    
    def _create_agent(self, df: pd.DataFrame) -> bool:
        """
        Create pandas dataframe agent for the loaded DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame to create agent for
            
        Returns:
            bool: True if agent created successfully, False otherwise
        """
        try:
            if self.llm is None:
                logger.error("Cannot create agent: LLM not available")
                return False
                
            self.current_agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=self.max_iterations,
                allow_dangerous_code=True
            )
            
            logger.info("Pandas dataframe agent created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating pandas agent: {str(e)}")
            return False
    
    def _load_data(self, file_path: Union[str, Path], sheet_name: Optional[Union[str, int]] = None) -> bool:
        """
        Load Excel/CSV data into pandas DataFrame and create agent.
        
        Args:
            file_path (Union[str, Path]): Path to the data file
            sheet_name (Optional[Union[str, int]]): Sheet name or index for Excel files
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        file_path = Path(file_path)
        
        # Validate file
        if not self._validate_file(file_path):
            return False
        
        # Load DataFrame
        df = self._load_dataframe(file_path, sheet_name)
        if df is None:
            return False
        
        self.current_df = df
        
        # Create agent
        return self._create_agent(df)
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded DataFrame.
        
        Returns:
            Dict[str, Any]: Information about the DataFrame including shape, columns, dtypes
        """
        if self.current_df is None:
            return {"error": "No data loaded"}
        
        return {
            "shape": self.current_df.shape,
            "columns": list(self.current_df.columns),
            "dtypes": self.current_df.dtypes.to_dict(),
            "memory_usage": self.current_df.memory_usage(deep=True).sum(),
            "null_counts": self.current_df.isnull().sum().to_dict()
        }
    
    def analyze_file(self, file_path: Union[str, Path], user_query: Optional[str] = None, 
                    sheet_name: Optional[Union[str, int]] = None) -> str:
        """
        Analyze Excel/CSV file with natural language query.
        
        Args:
            file_path (Union[str, Path]): Path to the Excel/CSV file
            user_query (Optional[str]): Natural language query for analysis
            sheet_name (Optional[Union[str, int]]): Sheet name or index for Excel files
            
        Returns:
            str: Analysis result or error message
            
        Example:
            result = analyzer.analyze_file("data.xlsx", "What are the top 5 sales by region?")
        """
        if self.llm is None:
            return f"Error: LLM not available. Please check your {self.provider} API configuration."
        
        # Load data
        if not self._load_data(file_path, sheet_name):
            return f"Error: Failed to load data from {file_path}"
        
        # Use default query if none provided
        if not user_query:
            user_query = DEFAULT_QUERY
        
        # Run analysis
        try:
            if self.current_agent is None:
                return "Error: Agent not available."
            
            logger.info(f"Running analysis with query: {user_query}")
            response = self.current_agent.run(user_query)
            return str(response)
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            logger.error(error_msg)
            return error_msg


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_excel_with_pandas(file_path: str, user_query: Optional[str] = None, **kwargs) -> str:
    """
    Analyze Excel/CSV file using pandas dataframe agent.
    
    This is a convenience function that creates an analyzer instance and performs
    the analysis with formatted output.
    
    Args:
        file_path (str): Path to the Excel/CSV file
        user_query (Optional[str]): Natural language query for analysis
        **kwargs: Additional arguments including:
            - sheet_name: Sheet name or index for Excel files
            - provider: LLM provider to use (default: 'deepseek')
            - temperature: Temperature for LLM (default: 0)
            - max_iterations: Max iterations for agent (default: 5)
            
    Returns:
        str: Formatted analysis result with emojis and structure
        
    Example:
        result = analyze_excel_with_pandas("sales.xlsx", "Show top 10 products by revenue")
    """
    try:
        # Extract configuration parameters
        provider = kwargs.get('provider', DEFAULT_PROVIDER)
        temperature = kwargs.get('temperature', DEFAULT_TEMPERATURE)
        max_iterations = kwargs.get('max_iterations', DEFAULT_MAX_ITERATIONS)
        sheet_name = kwargs.get('sheet_name')
        
        # Create analyzer with specified configuration
        analyzer = ExcelPandasAnalyzer(
            provider=provider,
            temperature=temperature,
            max_iterations=max_iterations
        )
        
        # Perform analysis
        analysis = analyzer.analyze_file(file_path, user_query, sheet_name)
        
        # Handle errors
        if analysis.startswith("Error:"):
            return f"❌ {analysis}"
        
        # Format output with enhanced information
        file_name = Path(file_path).name
        data_info = analyzer.get_data_info()
        
        formatted_result = f"""
📊 Excel/CSV Analysis Results
📁 File: {file_name}
❓ Query: {user_query or 'Default comprehensive analysis'}
📈 Data Shape: {data_info.get('shape', 'Unknown')}
🔧 LLM Provider: {provider}

🔍 Analysis:
{analysis}
"""
        
        return formatted_result
        
    except Exception as e:
        logger.error(f"analyze_excel_with_pandas failed: {str(e)}")
        return f"❌ Error: Failed to analyze Excel file - {str(e)}"


def get_supported_formats() -> list:
    """
    Get list of supported file formats.
    
    Returns:
        list: List of supported file extensions
    """
    return SUPPORTED_FORMATS.copy()


def is_supported_format(file_path: Union[str, Path]) -> bool:
    """
    Check if file format is supported for analysis.
    
    Args:
        file_path (Union[str, Path]): Path to the file
        
    Returns:
        bool: True if format is supported, False otherwise
    """
    file_path = Path(file_path)
    return file_path.suffix.lower() in SUPPORTED_FORMATS


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def _test_data_analyzer() -> None:
    """
    Test the data analyzer functionality.
    
    This function tests the basic functionality of the data analyzer
    including initialization, format checking, and basic operations.
    """
    print("🧪 Testing Data Analyzer...")
    
    try:
        # Test 1: Analyzer creation
        print("\n📝 Test 1: Creating analyzer...")
        analyzer = ExcelPandasAnalyzer()
        print(f"✅ Analyzer created with provider: {analyzer.provider}")
        print(f"✅ LLM available: {analyzer.llm is not None}")
        
        # Test 2: Format support
        print("\n📝 Test 2: Testing format support...")
        supported = get_supported_formats()
        print(f"✅ Supported formats: {supported}")
        
        test_files = ['test.xlsx', 'test.csv', 'test.txt', 'test.xls']
        for test_file in test_files:
            is_supported = is_supported_format(test_file)
            status = "✅" if is_supported else "❌"
            print(f"{status} {test_file}: {'Supported' if is_supported else 'Not supported'}")
        
        # Test 3: Data info (without actual file)
        print("\n📝 Test 3: Testing data info (no data loaded)...")
        info = analyzer.get_data_info()
        print(f"✅ Data info (empty): {info}")
        
        # Test 4: Configuration info
        print("\n📝 Test 4: Testing analyzer configuration...")
        print(f"✅ Provider: {analyzer.provider}")
        print(f"✅ Temperature: {analyzer.temperature}")
        print(f"✅ Max iterations: {analyzer.max_iterations}")
        print(f"✅ Max rows limit: {MAX_ROWS}")
        print(f"✅ Max columns limit: {MAX_COLUMNS}")
        
        # Test 5: Enhanced analyzer with custom parameters
        print("\n📝 Test 5: Testing custom analyzer configuration...")
        custom_analyzer = ExcelPandasAnalyzer(
            provider='qwen',
            temperature=0.3,
            max_iterations=10
        )
        print(f"✅ Custom analyzer provider: {custom_analyzer.provider}")
        print(f"✅ Custom analyzer temperature: {custom_analyzer.temperature}")
        print(f"✅ Custom analyzer max_iterations: {custom_analyzer.max_iterations}")
        print(f"✅ Custom analyzer LLM available: {custom_analyzer.llm is not None}")
        
        print("\n🎉 Data analyzer test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise


if __name__ == "__main__":
    _test_data_analyzer()
