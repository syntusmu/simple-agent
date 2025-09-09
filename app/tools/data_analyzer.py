from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
import logging

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

from ..service.llm.llm_interface import create_llm

# Constants
MAX_ROWS = 10000
MAX_COLUMNS = 100
SUPPORTED_FORMATS = ['.xlsx', '.xls', '.csv']

logger = logging.getLogger(__name__)


class ExcelPandasAnalyzer:
    """Excel data analysis using LangChain's pandas dataframe agent."""
    
    def __init__(self):
        """Initialize the analyzer with DeepSeek LLM."""
        try:
            # Use model from config.ini instead of hardcoded value
            self.llm = create_llm(provider='deepseek', temperature=0)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
        
        self.current_df = None
        self.current_agent = None
    
    def _load_data(self, file_path: Union[str, Path], sheet_name: Optional[Union[str, int]] = None) -> bool:
        """Load Excel/CSV data into pandas DataFrame."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
            
            if file_path.suffix.lower() not in SUPPORTED_FORMATS:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return False
            
            # Load data
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
            
            self.current_df = df
            
            # Create pandas agent
            if self.llm is not None:
                self.current_agent = create_pandas_dataframe_agent(
                    llm=self.llm,
                    df=self.current_df,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    max_iterations=5,
                    allow_dangerous_code=True
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            return False
    
    def analyze_file(self, file_path: Union[str, Path], user_query: Optional[str] = None, 
                    sheet_name: Optional[Union[str, int]] = None) -> str:
        """Analyze Excel/CSV file with natural language query."""
        if self.llm is None:
            return "Error: LLM not available. Please check your DeepSeek API configuration."
        
        # Load data
        if not self._load_data(file_path, sheet_name):
            return f"Error: Failed to load data from {file_path}"
        
        # Default query if none provided
        if not user_query:
            user_query = "Provide a comprehensive analysis of this dataset including summary statistics, data types, missing values, and key insights."
        
        # Run analysis
        try:
            if self.current_agent is None:
                return "Error: Agent not available."
            
            response = self.current_agent.run(user_query)
            return str(response)
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            logger.error(error_msg)
            return error_msg


def analyze_excel_with_pandas(file_path: str, user_query: Optional[str] = None, **kwargs) -> str:
    """Analyze Excel/CSV file using pandas dataframe agent."""
    try:
        analyzer = ExcelPandasAnalyzer()
        analysis = analyzer.analyze_file(file_path, user_query, kwargs.get('sheet_name'))
        
        if analysis.startswith("Error:"):
            return f"❌ {analysis}"
        
        # Format output
        file_name = Path(file_path).name
        return f"""
📊 Excel Analysis Results
📁 File: {file_name}
❓ Query: {user_query or 'Default analysis'}

🔍 Analysis:
{analysis}
"""
        
    except Exception as e:
        logger.error(f"analyze_excel_with_pandas failed: {str(e)}")
        return f"❌ Error: Failed to analyze Excel file - {str(e)}"

