# Simple Agent 🤖

**Your AI-powered document analysis and chat companion**

Transform how you interact with documents and data through natural language conversations. Simple Agent combines advanced AI capabilities with an intuitive chat interface to help you analyze documents, explore Excel data, and get instant insights.

## ✨ Features

### 📄 Smart Document Analysis
- **Multi-format support**: PDF, DOCX, Excel, CSV, HTML, TXT, and Markdown files
- **Natural language queries**: Ask questions in plain English
- **Intelligent search**: Semantic understanding with ChromaDB and BM25 hybrid search
- **Batch processing**: Upload entire folders and analyze multiple documents

### 📊 Data Analysis
- **Excel/CSV analysis**: Handle files up to 10,000 rows and 100 columns
- **Natural language queries**: "What are the top 5 sales regions?" or "Show me trends in Q3 data"
- **Interactive exploration**: Follow up with additional questions about your data

### 🧠 AI-Powered Intelligence
- **Multiple AI providers**: DeepSeek, OpenAI, and Qwen for optimal performance
- **ReAct reasoning**: Smart tool selection and reasoning patterns
- **Conversation memory**: Maintains context within sessions

## 🚀 Quick Start

### Installation & Setup

```bash
# Clone and setup
git clone <repository-url>
cd simple-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Create `config.ini` with your API keys:

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

**Get API Keys:**
- **Qwen**: [DashScope Console](https://dashscope.console.aliyun.com/) (for embeddings)
- **DeepSeek**: [DeepSeek Platform](https://platform.deepseek.com/) (for AI reasoning)

### Run the Application

```bash
# Web interface
python run.py

# Interactive chat
python external/call_chatbot.py
```

## 💡 Usage Examples

### Document Analysis
```bash
> python external/call_chatbot.py

You: I want to upload a research paper
Agent: Please provide the file path to your document.

You: data/research_paper.pdf
Agent: ✅ Document uploaded! What would you like to know about it?

You: What are the main findings?
Agent: Based on the research paper, the main findings are...
```

### Data Analysis
```bash
You: I have sales data in data/sales_report.xlsx
Agent: ✅ Excel file loaded! I can see sales data with multiple columns.

You: What are the top 5 performing products?
Agent: 📊 Here are the top 5 performing products by sales:
1. Product A: $125,000
2. Product B: $98,500
...
```

### Batch Processing
```python
from external.call_storage import call_storage

# Process single document
result = call_storage("data/report.pdf")

# Process with custom collection
result = call_storage("data/analysis.xlsx", collection_name="quarterly_reports")

# Different embedding provider
result = call_storage("data/document.pdf", embedding_provider="qwen")
```

## 🛠️ Built-in Tools

| Tool | Purpose | Example |
|------|---------|---------|
| **Document Retriever** | Search uploaded documents | "Find project timeline information" |
| **Data Analyzer** | Excel/CSV analysis with pandas | "Average sales by region?" |
| **Contextual Analyzer** | Content understanding | "Summarize key points" |
| **Storage Utils** | Batch document processing | "Process multiple documents" |

**Supported Formats**: PDF, DOCX, TXT, Markdown, HTML, Excel (.xlsx, .xls), CSV  
**File Limits**: Up to 100MB per file, 10,000 rows × 100 columns for data files

## 🎯 Use Cases

- **Research**: Literature reviews, citation finding, data analysis
- **Business**: Report analysis, sales insights, market research
- **Data Science**: Dataset exploration, trend analysis, data quality checks
- **Document Management**: Content search, summarization, document comparison

## 🔧 Troubleshooting

**Common Issues:**

- **API key errors**: Verify keys in `config.ini` or set environment variables (`QWEN_API_KEY`, `DEEPSEEK_API_KEY`)
- **File not found**: Use correct paths, check `data/` directory
- **Excel issues**: Ensure .xlsx/.csv format, check file size limits, verify no password protection
- **Model not responding**: Check internet connection and API key credits

## 🏗️ Architecture

```
┌─────────────────┐
│   Chat Interface │ ← User interaction
├─────────────────┤
│   ReAct Agent   │ ← Tool selection & reasoning
├─────────────────┤
│     Tools       │ ← Document search, data analysis
├─────────────────┤
│   AI Services   │ ← LLM, embeddings, vector DB
└─────────────────┘
```

**Key Components:**
- **ReAct Agent**: Smart reasoning engine for tool selection
- **Document Processor**: Multi-format support with intelligent chunking
- **Vector Database**: ChromaDB with hybrid search
- **Data Analyzer**: Pandas-powered Excel/CSV analysis
- **Memory System**: Conversation context management

## 🚀 What's Next

- 🌐 Enhanced web interface
- 📊 Data visualization and charts
- 🔄 Real-time collaborative analysis
- 🌍 Multi-language support
- 🐳 Docker containerization
- 🔌 REST API integration

---

**Ready to start?** Run `python external/call_chatbot.py` and begin chatting with your documents! 🎉