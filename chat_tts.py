import uuid
import logging
from tts.xtts import tts_with_live_playback, speaker_wav
from memory import (
    init_db, close_pool, embed,
    search_similar_messages, format_memories_for_context,
    write_user_message, write_llm_message, update_metadata_from_response
)
from llm.chat import create_llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory configuration
MEMORY_CONFIG = {
    'top_k': 5,
    'priority_threshold': 0.3,
    'exclude_type': 'random_talk',
    'memory_window': 10  # Last N messages for short-term context
}


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

# Initialize LLM (configure provider in llm/chat.py)
try:
    chat = create_llm()
    print(f"LLM initialized: {chat.__class__.__name__}")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Please check your LLM configuration in llm/chat.py")
    exit(1)

# Generate conversation ID for this session
conversation_id = uuid.uuid4()
message_index = 0

# Short-term memory window (in-memory only, for context)
short_term_memory = []


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
        try:
            llm_response = chat.send_message(full_prompt, retry_on_parse_failure=True)
            assistant_message = llm_response['response']
            user_metadata = llm_response['user_message_metadata']
            prev_llm_metadata = llm_response['previous_llm_message_metadata']
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            assistant_message = "I apologize, but I'm having trouble responding right now."
            user_metadata = None
            prev_llm_metadata = None
        
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

