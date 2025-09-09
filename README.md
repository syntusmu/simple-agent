# HR Maternity Leave Agent 🤖

**Your AI-powered document analysis and chat companion**

Transform how you interact with documents and data through natural language conversations. Simple Agent combines advanced AI capabilities with an intuitive chat interface to help you analyze documents, explore Excel data, and get instant insights.

## ✨ What Can Simple Agent Do?

### 📄 Smart Document Analysis
- **Chat with your documents**: Upload PDFs, Word docs, or text files and ask questions in plain English
- **Multi-format support**: Works with PDF, DOCX, Excel, CSV, HTML, TXT, and Markdown files
- **Intelligent search**: Find relevant information across all your uploaded documents
- **Batch processing**: Upload entire folders and analyze multiple documents at once

### 📊 Excel & Data Analysis
- **Natural language queries**: "What are the top 5 sales regions?" or "Show me trends in Q3 data"
- **Instant insights**: Get summaries, statistics, and analysis without writing code
- **Large dataset support**: Handle files up to 10,000 rows and 100 columns
- **Interactive exploration**: Follow up with additional questions about your data

### 🧠 AI-Powered Intelligence
- **Multiple AI providers**: Uses DeepSeek, OpenAI, and Qwen for optimal performance
- **Semantic search**: Understands context and meaning, not just keywords
- **Memory**: Remembers your conversation history within each session
- **Smart reasoning**: Uses ReAct (Reasoning + Acting) pattern for better responses

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd simple-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional required packages
pip install chromadb pypdf unstructured msoffcrypto-tool sentence-transformers
```

### 2. Configuration

Create or update `config.ini` with your API keys:

```ini
[embedding]
api_key = your_qwen_api_key_here
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
model = text-embedding-v4

[deepseek]
api_key = your_deepseek_api_key_here
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
model = deepseek-v3
```

### 3. Start Chatting!

```bash
# Start the interactive chat
python external/call_chatbot.py
```

That's it! You can now upload documents and start asking questions.

## 💡 Usage Examples

### Chat with Documents
```
> python external/call_chatbot.py

Simple Agent: Hello! I can help you analyze documents and data. 
What would you like to do?

You: I want to upload a research paper
Simple Agent: Great! Please provide the file path to your document.

You: data/research_paper.pdf
Simple Agent: ✅ Document uploaded successfully! 
What would you like to know about it?

You: What are the main findings?
Simple Agent: Based on the research paper, the main findings are...
```

### Analyze Excel Data
```
You: I have sales data in data/sales_report.xlsx
Simple Agent: ✅ Excel file loaded! I can see sales data with multiple columns.

You: What are the top 5 performing products?
Simple Agent: 📊 Here are the top 5 performing products by sales:
1. Product A: $125,000
2. Product B: $98,500
...
```

### Batch Document Processing
```python
# Upload multiple documents at once
from external.call_storage import call_storage

# Process single document
result = call_storage("data/report.pdf")
print(result)

# Process with custom collection name
result = call_storage("data/analysis.xlsx", collection_name="quarterly_reports")

# Process without chunking (for smaller files)
result = call_storage("data/summary.txt", chunk_documents=False)

# Use different embedding provider
result = call_storage("data/document.pdf", embedding_provider="qwen")
```

## 🛠️ Available Tools

Simple Agent comes with powerful built-in tools:

| Tool | What it does | Example Usage |
|------|-------------|---------------|
| **Document Retriever** | Searches through uploaded documents | "Find information about project timeline" |
| **Data Analyzer** | Analyzes Excel/CSV files with pandas | "What's the average sales by region?" |
| **Contextual Analyzer** | Understands document content and context | "Summarize the key points in this report" |
| **Storage Utils** | Batch document processing and storage | "Upload and process multiple documents" |
| **File Manager** | Lists and manages uploaded files | "Show me all uploaded files" |

### Supported File Formats

✅ **Documents**: PDF, DOCX, TXT, Markdown, HTML  
✅ **Data Files**: Excel (.xlsx, .xls), CSV  
✅ **Batch Processing**: Upload entire folders  
✅ **Large Files**: Up to 100MB per file

## ⚙️ Configuration

### API Keys Setup

You'll need API keys from these providers:

1. **Qwen (Alibaba Cloud)** - For embeddings (text understanding)
   - Get your key at: [DashScope Console](https://dashscope.console.aliyun.com/)
   - Used for: Document search and semantic understanding

2. **DeepSeek** - For AI reasoning and analysis  
   - Get your key at: [DeepSeek Platform](https://platform.deepseek.com/)
   - Used for: Chat responses and data analysis

### Configuration File

Update `config.ini` with your keys:

```ini
[embedding]
api_key = your_qwen_api_key
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings
model = text-embedding-v4

[deepseek] 
api_key = your_deepseek_api_key
base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
model = deepseek-v3
temperature = 0
max_tokens = 2048
```

### Alternative: Environment Variables

You can also set these as environment variables:
```bash
export QWEN_API_KEY="your_qwen_key"
export DEEPSEEK_API_KEY="your_deepseek_key"
```

## 🎯 Real-World Use Cases

### 📚 Research & Academia
- **Literature Review**: Upload research papers and ask "What are the common themes across these studies?"
- **Data Analysis**: Analyze survey results with "Show me correlations between age and satisfaction scores"
- **Citation Finding**: "Find all papers that mention machine learning applications"

### 💼 Business & Analytics  
- **Report Analysis**: Upload quarterly reports and ask "What were the key performance indicators?"
- **Sales Data**: "Which products had the highest growth rate this quarter?"
- **Market Research**: Analyze competitor reports and extract insights

### 📊 Data Science
- **Dataset Exploration**: "What's the distribution of values in this dataset?"
- **Trend Analysis**: "Show me patterns in the time series data"
- **Data Quality**: "Are there any missing values or outliers?"

### 📄 Document Management
- **Content Search**: Find specific information across hundreds of documents
- **Summarization**: Get quick summaries of long reports
- **Comparison**: "How do these two contracts differ?"

## 🔧 Troubleshooting

### Common Issues

**"API key not found" error**
- Check your `config.ini` file has the correct API keys
- Verify the API keys are valid and active (not dummy/test keys)
- Try setting environment variables as an alternative
- Ensure Qwen API key is from DashScope console, not a placeholder

**"File not found" error**
- Make sure file paths are correct (use forward slashes)
- Check if files are in the `data/` directory
- Try using absolute file paths

**"Model not responding" error**
- Check your internet connection
- Verify API keys have sufficient credits
- Try restarting the application

**Excel analysis not working**
- Ensure file is in .xlsx, .xls, or .csv format
- Check file size (max 10,000 rows, 100 columns)
- Verify the file isn't password protected
- Make sure pandas agent has allow_dangerous_code=True (handled automatically)
- Check file path resolution - use exact filenames including spaces

### Getting Help

- 📖 Check the technical documentation in `tech_documentation.md`
- 🐛 Report issues on the repository
- 💬 Review example usage in the code comments

## 🏗️ Architecture Overview

Simple Agent uses a modular architecture:

```
┌─────────────────┐
│   Chat Interface │ ← You interact here
├─────────────────┤
│   ReAct Agent   │ ← Decides what tools to use
├─────────────────┤
│     Tools       │ ← Document search, data analysis
├─────────────────┤
│   AI Services   │ ← LLM, embeddings, vector DB
└─────────────────┘
```

**Key Components:**
- **ReAct Agent**: Smart reasoning engine that selects appropriate tools
- **Document Processor**: Handles multiple file formats with intelligent chunking (storage_utils.py)
- **Vector Database**: ChromaDB with BM25 hybrid search for fast semantic search
- **Data Analyzer**: Pandas-powered Excel/CSV analysis with DeepSeek LLM integration
- **Memory System**: ConversationSummaryBufferMemory maintains conversation context
- **Centralized Config**: Common configuration system across all services

## 🚀 What's Next?

**Planned Features:**
- 🌐 Web-based interface for easier document upload
- 📊 Advanced data visualization and charts
- 🔄 Real-time collaborative analysis
- 🌍 Multi-language document support
- 🐳 Docker containerization for easy deployment
- 🔌 REST API for integration with other tools

---

**Ready to get started?** Run `python external/call_chatbot.py` and start chatting with your documents! 🎉