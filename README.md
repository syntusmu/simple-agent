# Simple Agent 🤖

**Your AI-Powered Assistant for Document and Data Analysis**

Simple Agent is designed to help teams and professionals manage, analyze, and query policy documents, business data, and regulations across regions. With natural language conversations, you can easily process documents, analyze datasets, and compare rules or policies, making your workflow more efficient and accurate.

## ✨ What Can Simple Agent Do?

### 📄 Intelligent Document Analysis
- **Conversational Document Query**: Upload policy, business, or regulatory documents and ask questions in plain language.
- **Multi-format Support**: Works with PDF, Word, Excel, and more for both documents and data.
- **Smart Policy Search**: Quickly find specific rules or details for any region or scenario.
- **Batch Comparison**: Analyze and compare multiple documents or datasets at once.

### 📊 Data Analysis & Reporting
- **Data Query**: "Which projects are due next month?" or "What are the sales trends this quarter?"
- **Policy Comparison**: "How do the regulations differ between Region A and Region B?"
- **Automated Reports**: Generate summaries, statistics, and budget estimates from your data.
- **Compliance Checks**: Verify if records or applications meet local or organizational requirements.

### 🧠 Professional AI Services
- **Multiple AI Engines**: Integrates DeepSeek, Qwen, and other models, optimized for business scenarios.
- **Semantic Understanding**: Accurately interprets complex document clauses and user queries.
- **Conversation Memory**: Remembers context for coherent, multi-turn conversations.
- **Smart Reasoning**: Automatically calculates figures, standards, or recommendations based on document rules.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd simple-agent

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional required packages
pip install chromadb pypdf unstructured msoffcrypto-tool sentence-transformers
```

### 2. Configuration

Create or update a `config.ini` file with your API keys:

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
# Launch the interactive chat
python external/call_chatbot.py
```

You’re ready to upload documents and ask questions!

## 💡 Usage Examples

### Document Query
```
> python external/call_chatbot.py

Simple Agent: Hello! I am your professional document and data assistant.
I can help you analyze policies, business documents, and datasets.

You: I want to check the latest business policy for Region A.
Simple Agent: Please upload the relevant policy document, or tell me your specific question.

You: data/Regional_Policies.xlsx
Simple Agent: ✅ Document loaded!
I can now access the policy data for all regions. What would you like to know?

You: What are the main requirements for Region A? Any special notes?
Simple Agent: 📋 According to the document, Region A’s policy includes:
- Standard requirement: ...
- Additional notes: ...
```

### Data Analysis
```
You: I have project data in data/project_overview.xlsx
Simple Agent: ✅ Data loaded! I can see the project records.

You: How many projects are scheduled for Q4?
Simple Agent: 📊 Q4 Project Summary:
- Total projects: 12
- Completed: 8
- In progress: 4
- Average duration: 3 months
- Estimated total cost: $120,000
```

### Batch Document Processing
```python
# Upload multiple documents at once
from external.call_storage import call_storage

# Process a single document
result = call_storage("data/report.pdf")
print(result)

# Use a custom collection name
result = call_storage("data/analysis.xlsx", collection_name="quarterly_reports")

# Process without chunking (for small files)
result = call_storage("data/summary.txt", chunk_documents=False)

# Use a different embedding provider
result = call_storage("data/document.pdf", embedding_provider="qwen")
```

## 🛠️ Built-in Tools

Simple Agent comes with powerful built-in tools:

| Tool                | Function                        | Example Use Case                      |
|---------------------|---------------------------------|---------------------------------------|
| **Document Retriever** | Search policy/business docs      | "Find the section on export rules"    |
| **Data Analyzer**      | Analyze business/project data    | "What’s the average project duration?"|
| **Policy Interpreter** | Understand document clauses      | "Explain this policy requirement"     |
| **Batch Processor**    | Handle multiple documents        | "Compare policies across all regions" |
| **File Manager**       | Manage uploaded files            | "Show all uploaded documents"         |

### Supported File Formats

✅ **Documents**: PDF, DOCX, TXT, Markdown, HTML  
✅ **Data Files**: Excel (.xlsx, .xls), CSV  
✅ **Batch Processing**: Upload entire folders  
✅ **Large Files**: Up to 100MB per file

## ⚙️ Configuration

### API Key Setup

You’ll need API keys from these providers:

1. **Qwen (Aliyun DashScope)** – for embeddings (text understanding)
   - Get your key here: [DashScope Console](https://dashscope.console.aliyun.com/)
   - Used for document search and semantic understanding

2. **DeepSeek** – for AI reasoning and analysis  
   - Get your key here: [DeepSeek Platform](https://platform.deepseek.com/)
   - Used for chat replies and data analysis

### Config File

Update your `config.ini` with your keys:

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

## 🎯 Real-World Application Scenarios

### 📄 Policy & Document Query
- **Policy Lookup**: "What are the main requirements for Region B?"
- **Clause Explanation**: "What does this section mean?"
- **Regional Comparison**: "Compare the rules for Region A and Region C."
- **Special Cases**: "How are exceptions handled in this policy?"

### 📋 Project & Data Management
- **Application Review**: "Does this project meet all requirements?"
- **Duration Calculation**: "How long should this process take?"
- **Budget Estimation**: "What’s the estimated cost for Q2?"
- **Compliance Check**: "Are all records compliant with regulations?"

### 📊 Data Statistics & Analysis
- **Trend Analysis**: "What are the sales trends over the past three years?"
- **Department Stats**: "Which department completed the most projects?"
- **Cost Analysis**: "What percentage of costs are due to logistics?"
- **Forecasting**: "What’s the projected budget for next year?"

### 📄 Document Management
- **Policy Updates**: Quickly identify changes in regulations.
- **Document Summarization**: Extract key points from multiple documents.
- **Compliance Review**: "Does our current policy align with the latest standards?"

## 🔧 Troubleshooting

### Common Issues

**"API key not found" error**
- Check your `config.ini` for valid API keys.
- Make sure your keys are active and not placeholders.
- Try using environment variables as an alternative.
- Ensure your Qwen API key is from the DashScope console.

**"File not found" error**
- Double-check your file paths (use forward slashes).
- Make sure files are in the `data/` directory.
- Try using absolute file paths.

**"Model not responding" error**
- Check your network connection.
- Verify your API key quota.
- Try restarting the application.

**Excel analysis not working**
- Ensure the file is .xlsx, .xls, or .csv.
- Check file size (max 10,000 rows, 100 columns).
- Make sure the file is not password-protected.
- Confirm pandas is set up correctly.
- Use the exact filename if it contains spaces.

### Need Help?

- 📖 See `tech_documentation.md` for technical docs
- 🐛 Report issues in the repository
- 💬 Check code comments for usage examples

## 🏗️ Technical Architecture Overview

Simple Agent uses a ReAct (Reasoning and Acting) agent architecture to automate complex tasks through a reasoning-action loop:

### 🧠 Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Chat Interface Layer                         │
│                   (chat_interface.py, chatbot.py)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                        ReAct Agent Orchestration Layer                 │
│                          (agent.py)                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │   Reasoning     │  │   Acting        │  │   Memory Mgmt   │        │
│  │  (Reasoning)    │  │   (Acting)      │  │  (memory.py)    │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
├─────────────────────────────────────────────────────────────────────────┤
│                            Tool Execution Layer                        │
│  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐ │
│  │   RAG Retriever │  │   Pandas Analyzer   │  │     PostgreSQL      │ │
│  │ (Semantic Search)│  │(excel analysis agent)│  │ (postgresql agent)  │ │
│  └─────────────────┘  └─────────────────────┘  └─────────────────────┘ │
│  ┌─────────────────────┐  ┌─────────────────┐                         │
│  │  Multimodal Analyzer│  │  Contextualizer │                         │
│  │(contextual analyzer)│  │ (semantic boost)│                         │
│  └─────────────────────┘  └─────────────────┘                         │
├─────────────────────────────────────────────────────────────────────────┤
│                          AI Service Layer                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│  │   LLM Service   │  │   Embedding     │  │  Vector DB      │        │
│  │ (DeepSeek/Qwen) │  │ (text-embedding)│  │  (ChromaDB)     │        │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 🎯 ReAct Agent Orchestration

**ReAct Agent** is the core orchestrator, enabling intelligent task decomposition and tool selection:

- **Reasoning**: Analyzes user intent and task complexity
- **Acting**: Selects and executes the best tool combination
- **Observation**: Evaluates results and determines next steps
- **Iteration**: Continuously improves strategies based on feedback

### 🛠️ Professional Tool Integration

#### 1. **RAG Retrieval System**
- **Hybrid Search**: ChromaDB vector + BM25 keyword search
- **Smart Chunking**: Adaptive document splitting for semantic integrity
- **Multi-format Support**: PDF, Word, Excel, Markdown, etc.
- **Semantic Understanding**: Deep matching with text-embedding-v4

#### 2. **Pandas Data Analysis Agent**
- **Natural Language Query**: Converts questions into Pandas operations
- **Safe Execution**: Runs analysis in a sandboxed environment
- **Smart Reasoning**: Auto-generates analysis strategies
- **Visualization**: Auto-generates charts and reports

#### 3. **PostgreSQL Database Agent**
- **Auto SQL Generation**: Natural language to SQL
- **Schema Exploration**: Analyzes database structure
- **Secure Connection**: Connection pooling and SQL injection protection
- **Performance Optimization**: Query tuning and result caching

#### 4. **Multimodal Context Analyzer**
- **Deep Semantic Analysis**: Understands document structure and context
- **Cross-modal Understanding**: Integrates text, tables, images
- **Context Boost**: Enhances local understanding with global info
- **Smart Summarization**: Extracts key points and summaries

### 🔄 System Workflow

1. **User Input** → Chat interface receives natural language requests
2. **Intent Recognition** → ReAct Agent analyzes task type and complexity
3. **Tool Selection** → Chooses the best tool(s) for the job
4. **Parallel Execution** → Tools collaborate to handle complex tasks
5. **Result Integration** → Aggregates outputs for unified answers
6. **Memory Update** → Saves context for continuous interaction

### 💡 Technical Highlights

- **Smart Orchestration**: ReAct pattern for complex task automation
- **Tool Collaboration**: Seamless integration of multiple tools
- **Context Awareness**: Global memory for coherent conversations
- **Secure Execution**: Sandbox environment for safe code
- **High Performance**: Vector search + caching for fast responses
- **Extensible**: Modular design for easy tool/function expansion

## 🚀 Roadmap

**Planned Features:**
- 🌐 Web-based interface for easy uploads
- 📊 Advanced data visualization and charts
- 🔄 Real-time collaborative analysis
- 🌍 Multi-language document support
- 🐳 Docker containerization for deployment
- 🔌 REST API for integration

---

**Ready to get started?** Run `python external/call_chatbot.py` and start chatting with your documents! 🎉