import google.generativeai as genai
import json
import uuid
import logging
from typing import Optional, Dict, Any
from tts.xtts import tts_with_live_playback, speaker_wav
from memory import (
    init_db, close_pool, embed,
    search_similar_messages, format_memories_for_context,
    write_user_message, write_llm_message, update_metadata_from_response
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = "AIzaSyDQASCxtJ4TIbXbTgqGP0GOkrWK1p8kf-Y"
MODEL = "gemini-2.5-flash"

# Memory configuration
MEMORY_CONFIG = {
    'top_k': 5,
    'priority_threshold': 0.3,
    'exclude_type': 'random_talk',
    'memory_window': 10  # Last N messages for short-term context
}

SYSTEM_PROMPT_BASE = """You are a helpful assistant. Format your responses with these rules:
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
- type: "random_talk" for casual chat, "informative" for factual info, "about_user" for personal details
- priority: 0.0-1.0, higher = more useful for future reference
- Respond ONLY with the JSON object, nothing else."""

def parse_llm_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM JSON response, retry once if invalid."""
    # Try to extract JSON from response
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


def get_memory_context(conversation_id: uuid.UUID, query_text: str) -> str:
    """Retrieve relevant memories and format for context."""
    try:
        query_embedding = embed(query_text)
        memories = search_similar_messages(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=MEMORY_CONFIG['top_k'],
            priority_threshold=MEMORY_CONFIG['priority_threshold'],
            exclude_type=MEMORY_CONFIG['exclude_type']
        )
        return format_memories_for_context(memories)
    except Exception as e:
        logger.error(f"Failed to retrieve memory context: {e}")
        return ""


# Initialize memory system
print("Initializing memory system...")
try:
    if init_db():
        print("Memory system initialized successfully")
    else:
        print("Warning: Memory system initialization failed, continuing without memory")
except Exception as e:
    print(f"Warning: Memory system initialization error: {e}")
    print("Continuing without memory...")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Generate conversation ID for this session
conversation_id = uuid.uuid4()
message_index = 0

# Short-term memory window (in-memory only, for context)
short_term_memory = []

try:
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_PROMPT_BASE
    )
    chat = model.start_chat(history=[])
except Exception as e:
    print(f"Error initializing model {MODEL}: {e}")
    print("\nTrying alternative model: gemini-pro")
    MODEL = "gemini-pro"
    model = genai.GenerativeModel(
        model_name=MODEL,
        system_instruction=SYSTEM_PROMPT_BASE
    )
    chat = model.start_chat(history=[])

print("Chat with Gemini + TTS + Memory")
print("Type your message and press Enter. Type 'quit' or 'exit' to stop.\n")

while True:
    try:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Step 1: Generate embedding for user message
        user_embedding = None
        try:
            user_embedding = embed(user_input)
        except Exception as e:
            logger.error(f"Failed to generate user embedding: {e}")
        
        # Step 2: Insert user message into memory
        user_message_id = None
        try:
            user_message_id = write_user_message(
                conversation_id=conversation_id,
                message_index=message_index,
                text=user_input,
                embedding=user_embedding
            )
        except Exception as e:
            logger.error(f"Failed to write user message: {e}")
        
        # Step 3: Retrieve relevant memories
        memory_context = get_memory_context(conversation_id, user_input)
        
        # Step 4: Build prompt with memory context
        full_prompt = user_input
        if memory_context:
            full_prompt = f"{memory_context}\n\nUser: {user_input}"
        
        # Step 5: Get LLM response (with retry on JSON parse failure)
        print("Thinking...")
        parsed_response = None
        response_text = None
        
        for attempt in range(2):  # Try twice
            try:
                response = chat.send_message(full_prompt)
                response_text = response.text
                parsed_response = parse_llm_response(response_text)
                
                if parsed_response:
                    break  # Success, exit retry loop
                elif attempt == 0:
                    logger.warning("Failed to parse JSON response, retrying...")
                    # Add more explicit instruction on retry
                    full_prompt = f"{full_prompt}\n\nIMPORTANT: Respond ONLY with valid JSON, no markdown, no explanations."
            except Exception as e:
                logger.error(f"Error getting LLM response (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    continue  # Retry once
                else:
                    break
        
        # Step 6: Extract response and metadata
        if parsed_response:
            assistant_message = parsed_response.get('response', response_text or '')
            user_metadata = parsed_response.get('user_message_metadata')
            prev_llm_metadata = parsed_response.get('previous_llm_message_metadata')
        else:
            # Fallback: use raw response if JSON parsing fails after retry
            assistant_message = response_text or "I apologize, but I'm having trouble formatting my response."
            user_metadata = None
            prev_llm_metadata = None
            logger.warning("Failed to parse JSON response after retry, using raw text")
        
        print(f"\nAssistant: {assistant_message}\n")
        
        # Step 7: Generate embedding for assistant message
        assistant_embedding = None
        try:
            assistant_embedding = embed(assistant_message)
        except Exception as e:
            logger.error(f"Failed to generate assistant embedding: {e}")
        
        # Step 8: Insert assistant message into memory
        assistant_message_id = None
        try:
            assistant_message_id = write_llm_message(
                conversation_id=conversation_id,
                message_index=message_index + 1,
                text=assistant_message,
                embedding=assistant_embedding
            )
        except Exception as e:
            logger.error(f"Failed to write assistant message: {e}")
        
        # Step 9: Update metadata
        if user_metadata or prev_llm_metadata:
            try:
                update_metadata_from_response(
                    conversation_id=conversation_id,
                    user_message_metadata=user_metadata,
                    previous_llm_message_metadata=prev_llm_metadata
                )
            except Exception as e:
                logger.error(f"Failed to update metadata: {e}")
        
        # Step 10: Update short-term memory window (FIFO)
        short_term_memory.append({
            'role': 'user',
            'content': user_input
        })
        short_term_memory.append({
            'role': 'assistant',
            'content': assistant_message
        })
        
        # Trim to memory window size
        if len(short_term_memory) > MEMORY_CONFIG['memory_window'] * 2:
            short_term_memory = short_term_memory[-MEMORY_CONFIG['memory_window'] * 2:]
        
        # Step 11: Increment message index
        message_index += 2
        
        # Step 12: Play TTS
        if assistant_message.strip():
            tts_with_live_playback(
                text=assistant_message,
                speaker_wav=speaker_wav,
                language="en",
                speed=1
            )
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        logger.error(f"Error in chat loop: {e}")
        print(f"Error: {e}")
        continue

# Cleanup
try:
    close_pool()
except Exception as e:
    logger.error(f"Error closing connection pool: {e}")

