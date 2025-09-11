"""PostgreSQL agent tool for ReAct Agent using LangChain SQL toolkit."""

import logging
from urllib.parse import urlparse
from typing import Optional

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType

from ..service.llm.llm_interface import create_llm
from ..utils.common import load_section_config, get_config_value

# Constants
MAX_ROWS_LIMIT = 1000
DEFAULT_SAMPLE_ROWS = 3

logger = logging.getLogger(__name__)


class PostgreSQLAgent:
    """PostgreSQL database agent using LangChain SQL toolkit."""
    
    def __init__(self, llm_provider: str = 'deepseek', model_name: str = None, connection_string: Optional[str] = None):
        """Initialize the PostgreSQL agent with LLM."""
        try:
            # Use model from config.ini if not specified
            self.llm = create_llm(provider=llm_provider, model_name=model_name, temperature=0)
            self.db = None
            self.agent = None
            
            # Load PostgreSQL configuration
            self.config = self._load_postgresql_config()
            
            # Use provided connection string or build from config
            self.connection_string = connection_string or self._build_connection_string()
            
            logger.info(f"PostgreSQL agent initialized with {llm_provider} {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
            self.config = {}
            self.connection_string = None
    
    def connect_database(self, connection_string: Optional[str] = None) -> bool:
        """Connect to PostgreSQL database."""
        try:
            if self.llm is None:
                logger.error("LLM not available")
                return False
            
            # Use provided connection string or default
            conn_str = connection_string or self.connection_string
            if not conn_str:
                logger.error("No connection string available")
                return False
            
            # Validate connection string format
            if not self._validate_connection_string(conn_str):
                logger.error("Invalid PostgreSQL connection string format")
                return False
            
            # Create database connection
            self.db = SQLDatabase.from_uri(conn_str)
            
            # Create SQL agent with config settings
            max_iterations = self.config.get('max_iterations', 5)
            query_timeout = self.config.get('query_timeout', 30)
            
            self.agent = create_sql_agent(
                llm=self.llm,
                db=self.db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                max_iterations=max_iterations,
                max_execution_time=query_timeout,
                early_stopping_method="generate"
            )
            
            logger.info("Successfully connected to PostgreSQL database")
            return True
            
        except Exception as e:
            error_msg = f"Error connecting to database: {str(e)}"
            logger.error(error_msg)
            return False
    
    def _load_postgresql_config(self) -> dict:
        """Load PostgreSQL configuration from config.ini with environment fallbacks."""
        try:
            env_mapping = {
                'host': 'POSTGRES_HOST',
                'port': 'POSTGRES_PORT', 
                'database': 'POSTGRES_DB',
                'username': 'POSTGRES_USER',
                'password': 'POSTGRES_PASSWORD',
                'connection_string': 'DATABASE_URL'
            }
            
            config = load_section_config(
                'postgresql',
                env_mapping=env_mapping
            )
            
            # Set defaults for missing values
            config.setdefault('host', 'localhost')
            config.setdefault('port', 5432)
            config.setdefault('database', 'postgres')
            config.setdefault('username', 'postgres')
            config.setdefault('max_rows_limit', MAX_ROWS_LIMIT)
            config.setdefault('query_timeout', 30)
            config.setdefault('max_iterations', 5)
            
            return config
            
        except Exception as e:
            logger.warning(f"Failed to load PostgreSQL config: {e}")
            return {
                'host': 'localhost',
                'port': 5432,
                'database': 'postgres',
                'username': 'postgres',
                'max_rows_limit': MAX_ROWS_LIMIT,
                'query_timeout': 30,
                'max_iterations': 5
            }
    
    def _build_connection_string(self) -> Optional[str]:
        """Build PostgreSQL connection string from config."""
        try:
            # Check if connection_string is directly provided in config
            if 'connection_string' in self.config and self.config['connection_string']:
                conn_str = self.config['connection_string']
                # Skip dummy values
                if not conn_str.startswith('postgresql://postgres:dummy-password'):
                    return conn_str
            
            # Build from individual components
            host = self.config.get('host', 'localhost')
            port = self.config.get('port', 5432)
            database = self.config.get('database', 'postgres')
            username = self.config.get('username', 'postgres')
            password = self.config.get('password')
            
            if not password or password.startswith('dummy-password'):
                logger.warning("No valid PostgreSQL password found in config or environment")
                return None
            
            return f"postgresql://{username}:{password}@{host}:{port}/{database}"
            
        except Exception as e:
            logger.error(f"Error building connection string: {e}")
            return None
    
    def _validate_connection_string(self, connection_string: str) -> bool:
        """Validate PostgreSQL connection string format."""
        try:
            parsed = urlparse(connection_string)
            return (
                parsed.scheme in ['postgresql', 'postgres'] and
                parsed.hostname is not None and
                parsed.username is not None
            )
        except Exception:
            return False
    
    def get_database_info(self) -> str:
        """Get database schema information."""
        try:
            if self.db is None:
                return "❌ Error: Not connected to database"
            
            # Get table information
            table_info = self.db.get_table_info()
            
            # Get list of tables
            tables = self.db.get_usable_table_names()
            
            return f"""
📊 **Database Schema Information**

**Available Tables:** {len(tables)}
{', '.join(tables)}

**Schema Details:**
```sql
{table_info}
```
"""
        except Exception as e:
            error_msg = f"Error getting database info: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def execute_query(self, user_query: str) -> str:
        """Execute natural language query using SQL agent."""
        try:
            if self.agent is None:
                return "❌ Error: Database agent not initialized. Please connect to database first."
            
            if not user_query.strip():
                return "❌ Error: Empty query provided"
            
            # Add context to help the agent understand the task better
            max_rows = self.config.get('max_rows_limit', MAX_ROWS_LIMIT)
            enhanced_query = f"""
Please analyze the following request and provide a comprehensive response:

User Query: {user_query}

Instructions:
1. If this is a data retrieval request, write and execute appropriate SQL queries
2. Limit results to {max_rows} rows maximum for performance
3. Provide clear explanations of your findings
4. Format results in a readable way
5. If you encounter errors, explain what went wrong and suggest alternatives
"""
            
            # Execute query using the agent
            response = self.agent.run(enhanced_query)
            
            return f"""
🔍 **Query Analysis Results**
❓ **User Query:** {user_query}

**Response:**
{response}
"""
            
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def execute_raw_sql(self, sql_query: str) -> str:
        """Execute raw SQL query directly."""
        try:
            if self.db is None:
                return "❌ Error: Not connected to database"
            
            if not sql_query.strip():
                return "❌ Error: Empty SQL query provided"
            
            # Execute the query
            result = self.db.run(sql_query)
            
            return f"""
📊 **SQL Query Results**
**Query:** 
```sql
{sql_query}
```

**Results:**
```
{result}
```
"""
            
        except Exception as e:
            error_msg = f"Error executing SQL: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def get_table_sample(self, table_name: str, limit: int = DEFAULT_SAMPLE_ROWS) -> str:
        """Get sample data from a specific table."""
        try:
            if self.db is None:
                return "❌ Error: Not connected to database"
            
            # Validate table exists
            tables = self.db.get_usable_table_names()
            if table_name not in tables:
                return f"❌ Error: Table '{table_name}' not found. Available tables: {', '.join(tables)}"
            
            # Get sample data
            max_rows = self.config.get('max_rows_limit', MAX_ROWS_LIMIT)
            sql_query = f"SELECT * FROM {table_name} LIMIT {min(limit, max_rows)}"
            result = self.db.run(sql_query)
            
            return f"""
📋 **Sample Data from {table_name}**
**Query:** `SELECT * FROM {table_name} LIMIT {limit}`

**Results:**
```
{result}
```
"""
            
        except Exception as e:
            error_msg = f"Error getting table sample: {str(e)}"
            logger.error(error_msg)
            return f"❌ {error_msg}"
    
    def disconnect(self):
        """Clean up database connection."""
        self.db = None
        self.agent = None
        logger.info("Disconnected from database")


def create_postgresql_agent(connection_string: Optional[str] = None, llm_provider: str = 'deepseek', model_name: str = None) -> PostgreSQLAgent:
    """Create and connect PostgreSQL agent."""
    try:
        agent = PostgreSQLAgent(llm_provider=llm_provider, model_name=model_name, connection_string=connection_string)
        
        if agent.connect_database():
            return agent
        else:
            raise Exception("Failed to connect to database")
            
    except Exception as e:
        logger.error(f"Failed to create PostgreSQL agent: {str(e)}")
        raise


def query_postgresql_database(connection_string: Optional[str] = None, user_query: str = "", **kwargs) -> str:
    """Query PostgreSQL database with natural language."""
    try:
        agent = create_postgresql_agent(connection_string, **kwargs)
        result = agent.execute_query(user_query)
        agent.disconnect()
        return result
        
    except Exception as e:
        logger.error(f"query_postgresql_database failed: {str(e)}")
        return f"❌ Error: Failed to query PostgreSQL database - {str(e)}"


def execute_postgresql_sql(connection_string: Optional[str] = None, sql_query: str = "", **kwargs) -> str:
    """Execute raw SQL query on PostgreSQL database."""
    try:
        agent = create_postgresql_agent(connection_string, **kwargs)
        result = agent.execute_raw_sql(sql_query)
        agent.disconnect()
        return result
        
    except Exception as e:
        logger.error(f"execute_postgresql_sql failed: {str(e)}")
        return f"❌ Error: Failed to execute SQL query - {str(e)}"


def get_postgresql_schema(connection_string: Optional[str] = None, **kwargs) -> str:
    """Get PostgreSQL database schema information."""
    try:
        agent = create_postgresql_agent(connection_string, **kwargs)
        result = agent.get_database_info()
        agent.disconnect()
        return result
        
    except Exception as e:
        logger.error(f"get_postgresql_schema failed: {str(e)}")
        return f"❌ Error: Failed to get database schema - {str(e)}"
