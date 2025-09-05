"""
LLM handler service for generating responses using various AI models.
"""
import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Third-party imports
try:
    from groq import Groq
except ImportError as e:
    raise ImportError(f"Missing required package: {e}")


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate_response(self, query: str, context: str, **kwargs) -> str:
        """Generate a response to a query using the provided context."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the LLM model."""
        pass


class GroqLLM(BaseLLM):
    """Professional Groq API implementation with streaming support."""
    
    SUPPORTED_MODELS = {
        "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B",
        "llama-3.1-70b-versatile": "LLaMA 3.1 70B Versatile",
        "llama-3.1-8b-instant": "LLaMA 3.1 8B Instant",
        "mixtral-8x7b-32768": "Mixtral 8x7B",
        "gemma2-9b-it": "Gemma2 9B IT"
    }

    def __init__(self, api_key: Optional[str] = None, model: str = "openai/gpt-oss-120b"):
        """
        Initialize the Groq LLM.
        
        Args:
            api_key: Groq API key
            model: Model to use for generation
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY environment variable.")

        if model not in self.SUPPORTED_MODELS:
            self.model = "openai/gpt-oss-120b"

        self.client = Groq(api_key=self.api_key)
    
    def generate_response(self, query: str, context: str, 
                         max_tokens: int = 8192,
                         temperature: float = 1.0, 
                         top_p: float = 1.0,
                         reasoning_effort: str = "medium", 
                         enable_streaming: bool = True) -> str:
        """
        Generate response using Groq API with advanced parameters.
        
        Args:
            query: User's question
            context: Context information for the query
            max_tokens: Maximum number of tokens in the response
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            reasoning_effort: Reasoning effort level
            enable_streaming: Whether to enable streaming responses
            
        Returns:
            Generated response text
        """
        system_prompt = """You are an expert AI assistant specializing in document analysis and knowledge extraction. 

Your responsibilities:
- Provide accurate, well-structured answers based solely on the provided context
- Cite specific information from the context when possible
- Clearly state when information is insufficient to answer the question
- Maintain professional, concise communication
- Focus on factual accuracy over speculation"""

        user_prompt = f"""Context Information:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information, clearly state this limitation."""

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=top_p,
                reasoning_effort=reasoning_effort,
                stream=enable_streaming,
                stop=None
            )

            if enable_streaming:
                response_text = ""
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        response_text += chunk.choices[0].delta.content
                return response_text.strip()
            else:
                return completion.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get Groq model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'provider': 'groq',
            'model': self.model,
            'model_name': self.SUPPORTED_MODELS.get(self.model, self.model),
            'supports_streaming': True,
            'max_tokens': 8192
        }


class LLMHandler:
    """Professional LLM handler with intelligent provider management."""

    def __init__(self, llm_type: str = "groq", **kwargs):
        """
        Initialize the LLM handler.
        
        Args:
            llm_type: Type of LLM to use
            **kwargs: Additional arguments for the LLM
        """
        self.llm_type = llm_type

        if llm_type == "groq":
            self.llm = GroqLLM(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def generate_answer(self,
                       query: str,
                       retrieved_chunks: List[Dict[str, Any]],
                       max_tokens: int = 8192,
                       **kwargs) -> Dict[str, Any]:
        """
        Generate an answer to a query using retrieved chunks.
        
        Args:
            query: User's question
            retrieved_chunks: List of retrieved document chunks
            max_tokens: Maximum number of tokens in the response
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary with the answer and metadata
        """
        if not retrieved_chunks:
            return {
                'answer': "I couldn't find any relevant information in the knowledge base to answer your question.",
                'sources': [],
                'context_used': "",
                'num_sources': 0,
                'llm_type': self.llm_type,
                'model_used': getattr(self.llm, 'model', 'unknown')
            }
        
        # Prepare context from top chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks[:5]):
            context_parts.append(f"[Source {i+1}] {chunk['text']}")
            sources.append({
                'file_name': chunk['file_name'],
                'file_path': chunk['file_path'],
                'similarity_score': chunk.get('similarity_score', 0.0),
                'chunk_id': chunk.get('chunk_id', 0)
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        answer = self.llm.generate_response(
            query, context, max_tokens=max_tokens, **kwargs
        )
        
        return {
            'answer': answer,
            'sources': sources,
            'context_used': context,
            'num_sources': len(sources),
            'llm_type': self.llm_type,
            'model_used': getattr(self.llm, 'model', 'unknown')
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model information
        """
        return self.llm.get_model_info()


def create_llm_handler(llm_type: str = "auto", **kwargs) -> LLMHandler:
    """
    Create LLM handler with intelligent provider selection.
    
    Args:
        llm_type: Type of LLM to use ("auto" to automatically select)
        **kwargs: Additional arguments for the LLM
        
    Returns:
        LLMHandler instance
    """
    if llm_type == "auto":
        # Try Groq first
        if os.getenv("GROQ_API_KEY"):
            try:
                return LLMHandler("groq", **kwargs)
            except Exception:
                pass

    return LLMHandler("groq", **kwargs)