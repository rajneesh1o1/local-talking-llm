"""LLM chat interface supporting multiple providers (Gemini, OpenAI, Ollama)."""

import json
import logging
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - Edit these settings to switch LLM providers
# ============================================================================

# Provider options: "gemini", "openai", "ollama"
LLM_PROVIDER = "ollama"

# Gemini Configuration
GEMINI_API_KEY = "AIzaSyDOKZB0LcVgassqpZAPQpXRJnNcssMsv04"
GEMINI_MODEL = "gemini-3-flash"

# OpenAI Configuration
OPENAI_API_KEY = ""  # Set your OpenAI API key here
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_BASE_URL = None  # Use None for default, or set custom endpoint

# Ollama Configuration (for local models)
# OLLAMA_BASE_URL = "http://192.168.1.146:11434"  # Default Ollama URL
OLLAMA_BASE_URL = "localhost:11434"
OLLAMA_MODEL = "llama3.2:latest"  # Change to your preferred local model

# System Prompt (used by all providers)
SYSTEM_PROMPT = """You are a helpful assistant. Format your responses with these rules:
1. Break your response into small chunks of 13-15 words maximum
2. Add "#" after each meaningful chunk
3. Use simple, everyday words
4. Make each chunk emotionally complete and satisfying on its own
5. Keep chunks natural and conversational
6. Keep overall response short and funny.
Example format:
Hello there friend# How are you doing today? # I hope you are feeling well # What can I help you with? # 

CRITICAL: You MUST respond with valid JSON only, no markdown, no explanations. Format:
{
  "response": "your actual response text here with # chunk markers",
  "user_message_metadata": {
    "type": "random_talk | informative | about_user",
    "priority": 0.0-1.0
  },
  "previous_llm_message_metadata": {
    "type": "random_talk | informative | about_user",
    "priority": 0.0-1.0
  }
}

The "response" field contains your actual reply with # chunk markers (e.g., "Hello there friend# How are you doing today? #"). 
The metadata fields help categorize messages for future reference.

CRITICAL METADATA CLASSIFICATION RULES:
- type: Choose EXACTLY ONE value from this list (do NOT return multiple values separated by |):
  * "about_user" - Personal information about the user (name, preferences, skills, hobbies, location, job, interests, what they like/dislike, personal facts). 
    Examples: "I like C++", "I'm a developer", "I live in NYC", "I enjoy coding", "My name is John", "I work at Google", "im rajneesh"
  * "informative" - Factual information, explanations, or educational content that could be useful later
  * "random_talk" - Casual conversation, jokes, greetings, small talk with no lasting value
  
- priority: 0.0-1.0, higher = more useful for future reference
  * "about_user" messages should typically have priority >= 0.7 (high priority for personalization)
  * "informative" messages should have priority 0.5-0.8
  * "random_talk" should have priority <= 0.3

IMPORTANT RULES:
1. If the user mentions ANY personal information (what they do, like, prefer, their skills, location, name, etc.), 
   it MUST be classified as "about_user" with priority >= 0.7, NOT "random_talk"!
2. Return ONLY ONE type value, NOT multiple values separated by "|". Examples:
   ✅ CORRECT: "type": "about_user"
   ❌ WRONG: "type": "random_talk | informative | about_user"
3. The type field must be a single string value, not a list or multiple values.

- Respond ONLY with the JSON object, nothing else."""

# ============================================================================


class BaseLLM(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    def send_message(self, message: str, retry_on_parse_failure: bool = True) -> Dict[str, Any]:
        """Send a message and get parsed response."""
        pass
    
    def parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON response."""
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        try:
            parsed = json.loads(response_text)
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:200]}")
            return None


class GeminiLLM(BaseLLM):
    """Google Gemini LLM provider."""
    
    def __init__(self, api_key: str, model_name: str, system_prompt: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt
        )
        self.chat = self.model.start_chat(history=[])
        self.model_name = model_name
    
    def send_message(self, message: str, retry_on_parse_failure: bool = True) -> Dict[str, Any]:
        parsed_response = None
        response_text = None
        full_prompt = message
        
        for attempt in range(2 if retry_on_parse_failure else 1):
            try:
                response = self.chat.send_message(full_prompt)
                response_text = response.text
                parsed_response = self.parse_response(response_text)
                
                if parsed_response:
                    break
                elif attempt == 0 and retry_on_parse_failure:
                    logger.warning("Failed to parse JSON response, retrying...")
                    full_prompt = f"{full_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON, no markdown, no explanations."
            except Exception as e:
                logger.error(f"Error getting LLM response (attempt {attempt + 1}): {e}")
                if attempt == 0 and retry_on_parse_failure:
                    continue
                else:
                    break
        
        if parsed_response:
            return {
                'response': parsed_response.get('response', response_text or ''),
                'user_message_metadata': parsed_response.get('user_message_metadata'),
                'previous_llm_message_metadata': parsed_response.get('previous_llm_message_metadata'),
                'raw_response': response_text
            }
        else:
            return {
                'response': response_text or "I apologize, but I'm having trouble formatting my response.",
                'user_message_metadata': None,
                'previous_llm_message_metadata': None,
                'raw_response': response_text
            }


class OpenAILLM(BaseLLM):
    """OpenAI LLM provider."""
    
    def __init__(self, api_key: str, model_name: str, system_prompt: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]
    
    def send_message(self, message: str, retry_on_parse_failure: bool = True) -> Dict[str, Any]:
        parsed_response = None
        response_text = None
        full_prompt = message
        
        for attempt in range(2 if retry_on_parse_failure else 1):
            try:
                # Add user message
                messages = self.messages + [{"role": "user", "content": full_prompt}]
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7
                )
                
                response_text = response.choices[0].message.content
                parsed_response = self.parse_response(response_text)
                
                if parsed_response:
                    # Update conversation history
                    self.messages.append({"role": "user", "content": full_prompt})
                    self.messages.append({"role": "assistant", "content": response_text})
                    break
                elif attempt == 0 and retry_on_parse_failure:
                    logger.warning("Failed to parse JSON response, retrying...")
                    full_prompt = f"{full_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON, no markdown, no explanations."
            except Exception as e:
                logger.error(f"Error getting LLM response (attempt {attempt + 1}): {e}")
                if attempt == 0 and retry_on_parse_failure:
                    continue
                else:
                    break
        
        if parsed_response:
            return {
                'response': parsed_response.get('response', response_text or ''),
                'user_message_metadata': parsed_response.get('user_message_metadata'),
                'previous_llm_message_metadata': parsed_response.get('previous_llm_message_metadata'),
                'raw_response': response_text
            }
        else:
            return {
                'response': response_text or "I apologize, but I'm having trouble formatting my response.",
                'user_message_metadata': None,
                'previous_llm_message_metadata': None,
                'raw_response': response_text
            }


class OllamaLLM(BaseLLM):
    """Ollama LLM provider (for local models)."""
    
    def __init__(self, base_url: str, model_name: str, system_prompt: str):
        try:
            import ollama
        except ImportError:
            raise ImportError("ollama package not installed. Install with: pip install ollama")
        
        self.client = ollama.Client(host=base_url)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.messages = []
    
    def send_message(self, message: str, retry_on_parse_failure: bool = True) -> Dict[str, Any]:
        parsed_response = None
        response_text = None
        full_prompt = message
        
        # Build messages with system prompt
        messages = [{"role": "system", "content": self.system_prompt}] + self.messages
        messages.append({"role": "user", "content": full_prompt})
        
        for attempt in range(2 if retry_on_parse_failure else 1):
            try:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages
                )
                
                response_text = response['message']['content']
                parsed_response = self.parse_response(response_text)
                
                if parsed_response:
                    # Update conversation history
                    self.messages.append({"role": "user", "content": full_prompt})
                    self.messages.append({"role": "assistant", "content": response_text})
                    break
                elif attempt == 0 and retry_on_parse_failure:
                    logger.warning("Failed to parse JSON response, retrying...")
                    full_prompt = f"{full_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON, no markdown, no explanations."
                    messages[-1] = {"role": "user", "content": full_prompt}
            except Exception as e:
                logger.error(f"Error getting LLM response (attempt {attempt + 1}): {e}")
                if attempt == 0 and retry_on_parse_failure:
                    continue
                else:
                    break
        
        if parsed_response:
            return {
                'response': parsed_response.get('response', response_text or ''),
                'user_message_metadata': parsed_response.get('user_message_metadata'),
                'previous_llm_message_metadata': parsed_response.get('previous_llm_message_metadata'),
                'raw_response': response_text
            }
        else:
            return {
                'response': response_text or "I apologize, but I'm having trouble formatting my response.",
                'user_message_metadata': None,
                'previous_llm_message_metadata': None,
                'raw_response': response_text
            }


def create_llm() -> BaseLLM:
    """
    Create LLM instance based on configuration.
    
    To switch providers, change LLM_PROVIDER at the top of this file.
    """
    provider = LLM_PROVIDER.lower()
    
    if provider == "gemini":
        return GeminiLLM(
            api_key=GEMINI_API_KEY,
            model_name=GEMINI_MODEL,
            system_prompt=SYSTEM_PROMPT
        )
    elif provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY must be set in llm/chat.py")
        return OpenAILLM(
            api_key=OPENAI_API_KEY,
            model_name=OPENAI_MODEL,
            system_prompt=SYSTEM_PROMPT,
            base_url=OPENAI_BASE_URL
        )
    elif provider == "ollama":
        return OllamaLLM(
            base_url=OLLAMA_BASE_URL,
            model_name=OLLAMA_MODEL,
            system_prompt=SYSTEM_PROMPT
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose from: gemini, openai, ollama")


# Convenience class for backward compatibility
class GeminiChat:
    """Backward compatibility wrapper."""
    
    def __init__(self, api_key: str = None, model_name: str = None):
        if api_key is None:
            api_key = GEMINI_API_KEY
        if model_name is None:
            model_name = GEMINI_MODEL
        
        self._llm = GeminiLLM(api_key=api_key, model_name=model_name, system_prompt=SYSTEM_PROMPT)
    
    def send_message(self, message: str, retry_on_parse_failure: bool = True) -> Dict[str, Any]:
        return self._llm.send_message(message, retry_on_parse_failure)


# Default exports
DEFAULT_API_KEY = GEMINI_API_KEY
DEFAULT_MODEL = GEMINI_MODEL
