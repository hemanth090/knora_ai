"""
Configuration settings for the KnoRa AI application.
"""
import os
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Application settings
APP_NAME = "KnoRa AI Knowledge Assistant"
APP_VERSION = "2.0.0"

# Vector store settings
DEFAULT_VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "/app/data/vector_store")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# LLM settings
DEFAULT_LLM_MODEL = "openai/gpt-oss-120b"
SUPPORTED_LLM_MODELS = {
    "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B",
    "llama-3.1-70b-versatile": "LLaMA 3.1 70B Versatile",
    "llama-3.1-8b-instant": "LLaMA 3.1 8B Instant",
    "mixtral-8x7b-32768": "Mixtral 8x7B",
    "gemma2-9b-it": "Gemma2 9B IT"
}

# Document processing settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
SUPPORTED_FILE_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx', '.csv', '.xlsx', '.xls', '.md', '.pptx'}

# API settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")