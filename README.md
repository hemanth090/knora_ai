# 🔍 KnoRa AI Knowledge Assistant

[![GitHub](https://img.shields.io/github/license/hemanth090/knora_ai)](https://github.com/hemanth090/knora_ai)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/hemanth090/knora_ai)
[![GitHub stars](https://img.shields.io/github/stars/hemanth090/knora_ai)](https://github.com/hemanth090/knora_ai)

A modular, production-ready Retrieval-Augmented Generation (RAG) system for intelligent document analysis and question answering using state-of-the-art AI models.

## 📌 Project Overview

KnoRa AI Knowledge Assistant is a sophisticated document processing and question-answering system that leverages Retrieval-Augmented Generation (RAG) technology. It enables users to upload various document formats, process them into searchable knowledge bases, and ask natural language questions to extract insights from their documents.

### Key Features

- **Multi-format Document Support**: Process PDF, TXT, DOCX, CSV, XLSX, MD, and PPTX files
- **Intelligent Text Chunking**: Automatically segments documents with overlap for better context retention
- **Semantic Search**: Uses FAISS vector database with sentence transformers for similarity search
- **AI-Powered Q&A**: Integrates with Groq's LLMs to provide contextual answers with source citations
- **Interactive Web Interface**: Streamlit-based UI with document ingestion, querying, and analytics tabs
- **Persistent Storage**: Saves processed documents and embeddings for future sessions
- **Advanced Configuration**: Adjustable parameters for chunk size, similarity thresholds, and LLM settings

## 🛠️ Technologies & Dependencies

### Core Technologies

- **Python 3.8+**: Primary programming language
- **Streamlit**: Web framework for the user interface
- **FAISS**: Facebook AI Similarity Search for vector storage and retrieval
- **Sentence Transformers**: For generating document embeddings
- **Groq API**: Integration with LLMs (LLaMA, Mixtral, Gemma models)
- **Docker**: Containerization support

### Python Libraries

| Category | Libraries |
|----------|-----------|
| Core Framework | `streamlit>=1.36.0` |
| Numerical Computing | `numpy>=1.24.0`, `faiss-cpu>=1.7.4` |
| NLP & Embeddings | `sentence-transformers>=2.6.1` |
| LLM Integration | `groq>=0.5.0` |
| Document Processing | `PyPDF2>=3.0.1`, `pandas>=2.2.0`, `python-docx>=1.1.0`, `python-pptx>=0.6.23`, `markdown>=3.5.2` |
| Configuration | `python-dotenv>=1.0.1` |

## 🚀 Installation Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hemanth090/knora_ai.git
   cd knora_ai
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Edit the `.env` file and add your Groq API key:
   ```bash
   GROQ_API_KEY=your_actual_api_key_here
   ```
   Get your free API key at [console.groq.com](https://console.groq.com)

5. **Run the application**:
   ```bash
   streamlit run knora_ai/app.py
   ```

### Docker Deployment

1. **Configure your API key**:
   Edit the `.env` file and add your Groq API key

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up
   ```

3. **Access the application** at [http://localhost:8501](http://localhost:8501)

## 🎯 Usage Instructions

### 1. Document Ingestion

- Navigate to the "📁 Document Ingestion" tab
- Upload supported document files (PDF, TXT, DOCX, CSV, XLSX, MD, PPTX)
- Click "🚀 Process Files" to analyze and index documents
- View processing results and chunk statistics

### 2. Intelligent Query

- Switch to the "🔍 Intelligent Query" tab
- Enter natural language questions about your documents
- Adjust advanced settings:
  - Number of sources (1-10)
  - Similarity threshold (0.0-1.0)
  - Response creativity (temperature)
  - Maximum response length
- Click "🔍 Search & Analyze" to get AI-powered answers
- View answers with source citations and similarity scores

### 3. Knowledge Analytics

- Use the "📊 Knowledge Analytics" tab to:
  - View document collection statistics
  - Monitor vector embeddings and storage usage
  - See system configuration details
  - Check average chunks per document

### Example Workflow

1. Upload a research paper in PDF format
2. Process the document
3. Ask questions like:
   - "What are the key findings of this research?"
   - "What methodology was used in the study?"
   - "What are the limitations mentioned by the authors?"

## 📁 Project Structure

```
knora_ai/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── .env                  # Environment variables
├── .gitignore            # Git ignore patterns
├── config/               # Configuration settings
│   ├── __init__.py
│   └── settings.py       # Application configuration
├── models/               # Data models
│   ├── __init__.py
│   └── document.py       # Dataclasses for documents and responses
└── services/             # Business logic services
    ├── __init__.py
    ├── document_processor.py  # Document parsing and chunking
    ├── vector_store.py        # FAISS vector database
    └── llm_handler.py         # Groq LLM integration
```

### Module Details

- **config/settings.py**: Centralized configuration management
- **models/document.py**: Dataclasses for type-safe data handling
- **services/document_processor.py**: Handles extraction from 7+ file formats
- **services/vector_store.py**: FAISS-based semantic search implementation
- **services/llm_handler.py**: Groq API integration with multiple model support

## ⚙️ Configuration

### Environment Variables

The application uses the following environment variables, configured in the `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `GROQ_API_KEY` | Required API key for Groq LLM access | None |
| `VECTOR_STORE_PATH` | Directory for storing processed documents | `/app/data/vector_store` |

To obtain a Groq API key:
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Create an API key
4. Add it to the `.env` file

### Application Settings

Configurable in `config/settings.py`:

- **Chunk Size**: 1000 characters (adjustable in UI)
- **Chunk Overlap**: 200 characters
- **Default Embedding Model**: `all-MiniLM-L6-v2`
- **Supported LLM Models**: 
  - `openai/gpt-oss-120b` (default)
  - `llama-3.1-70b-versatile`
  - `llama-3.1-8b-instant`
  - `mixtral-8x7b-32768`
  - `gemma2-9b-it`

## 🧪 Testing

To verify your setup:

```bash
python -c "from config.settings import GROQ_API_KEY; print('API Key configured' if GROQ_API_KEY else 'API Key missing')"
```

Note: The application will run without an API key for document processing, but LLM features will be disabled until you add a valid Groq API key to the `.env` file.

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository from [https://github.com/hemanth090/knora_ai](https://github.com/hemanth090/knora_ai)
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

Areas for contribution:
- Additional document format support
- Enhanced text chunking algorithms
- New embedding models
- UI/UX improvements
- Performance optimizations
- Additional LLM provider integrations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Additional Notes

### System Requirements

- **RAM**: Minimum 4GB recommended (8GB+ for large documents)
- **Storage**: Depends on document collection size
- **Internet**: Required for initial setup and LLM API access

### Limitations

- **API Dependency**: Requires active Groq API key for LLM functionality
- **Processing Time**: Large documents may take several minutes to process
- **Memory Usage**: Embedding generation can be memory-intensive
- **Session Isolation**: Each browser session creates isolated vector stores

### Security

- **API Key Protection**: Never commit your API key to version control
- **Environment Variables**: Always use `.env` file for sensitive credentials
- **Public Sharing**: The repository is safe to share as no keys are hardcoded

### Future Enhancements

- Multi-user support with shared knowledge bases
- Advanced document preprocessing (tables, images)
- Custom embedding model training
- Export/import functionality for knowledge bases
- Additional LLM providers (OpenAI, Anthropic)
- Mobile-responsive UI
- Batch document processing

### Troubleshooting

**Common Issues**:
1. **API Key Errors**: Verify `.env` file contains valid Groq API key
2. **Import Errors**: Ensure all dependencies installed via `requirements.txt`
3. **Memory Issues**: Process smaller batches of large documents
4. **Port Conflicts**: Change port in Docker configuration or Streamlit command
5. **LLM Features Not Working**: Ensure you have added a valid Groq API key to `.env`

**Support**:
- Report bugs on [GitHub Issues](https://github.com/hemanth090/knora_ai/issues)
- Get help on [Project Documentation](https://github.com/hemanth090/knora_ai)