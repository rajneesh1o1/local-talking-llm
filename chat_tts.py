import os
import uuid
import json
import logging

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tts.xtts import tts_with_live_playback, speaker_wav
from memory import (
    init_db, close_pool, embed,
    search_similar_messages, format_memories_for_context,
    write_user_message, write_llm_message, update_metadata_from_response
)
from llm.chat import create_llm
from voice_to_text import voice_to_text_with_fallback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Memory configuration
MEMORY_CONFIG = {
    'top_k': 5,  # Number of memories to retrieve
    'priority_threshold': 0.3,  # Minimum priority to include
    'exclude_type': 'random_talk',  # Exclude low-value random talk
    'memory_window': 10,  # Last N messages for short-term context
    'cross_conversation': True,  # Search across all conversations for personalization
    'personal_info_boost': True  # Boost personal information (about_user type)
}


def get_memory_context(conversation_id: uuid.UUID, query_text: str) -> str:
    """
    Retrieve relevant memories and format for context.
    Uses semantic search with role, type, and priority filtering.
    """
    try:
        logger.info(f"📚 Retrieving memory context for query: '{query_text[:50]}...'")
        
        # Generate embedding for semantic search
        query_embedding = embed(query_text)
        logger.debug(f"   Generated embedding vector (dim={len(query_embedding)})")
        
        # First, try to get personal information (about_user type, high priority)
        personal_memories = search_similar_messages(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=3,  # Get top 3 personal memories
            priority_threshold=0.5,  # Higher threshold for personal info
            exclude_type=None,  # Don't exclude anything for personal search
            role_filter=None,  # Both user and assistant messages
            type_filter='about_user',  # Focus on personal information
            cross_conversation=True  # Search across all conversations
        )
        
        # Then get general informative context
        informative_memories = search_similar_messages(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=MEMORY_CONFIG['top_k'],
            priority_threshold=MEMORY_CONFIG['priority_threshold'],
            exclude_type=MEMORY_CONFIG['exclude_type'],
            role_filter=None,
            type_filter='informative',
            cross_conversation=True
        )
        
        # Get general relevant context (excluding random_talk)
        general_memories = search_similar_messages(
            query_embedding=query_embedding,
            conversation_id=conversation_id,
            top_k=MEMORY_CONFIG['top_k'],
            priority_threshold=MEMORY_CONFIG['priority_threshold'],
            exclude_type=MEMORY_CONFIG['exclude_type'],
            role_filter=None,
            type_filter=None,
            cross_conversation=True
        )
        
        # Combine and deduplicate (by message ID)
        all_memories = {}
        for mem in personal_memories + informative_memories + general_memories:
            mem_id = mem['id']
            if mem_id not in all_memories or mem['score'] > all_memories[mem_id]['score']:
                all_memories[mem_id] = mem
        
        # Sort by score and limit
        sorted_memories = sorted(all_memories.values(), key=lambda x: x['score'], reverse=True)
        final_memories = sorted_memories[:MEMORY_CONFIG['top_k']]
        
        logger.info(f"   Combined {len(final_memories)} unique memories from knowledge base")
        
        formatted_context = format_memories_for_context(final_memories)
        
        if formatted_context:
            logger.info(f"✅ Successfully retrieved and formatted memory context ({len(final_memories)} memories)")
        else:
            logger.info("   No relevant memories found in knowledge base")
        
        return formatted_context
    except Exception as e:
        logger.error(f"❌ Failed to retrieve memory context: {e}", exc_info=True)
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
print("Voice input enabled - speak your message")
print("Say 'quit' or 'exit' to stop, or press Ctrl+C\n")

while True:
    try:
        # Get voice input instead of keyboard input
        print("🎤 Listening for your voice... (speak clearly)")
        user_input = voice_to_text_with_fallback(timeout=3, phrase_time_limit=15).strip()
        
        if not user_input:
            print("⚠️ No speech detected or could not understand. Try again...\n")
            continue
        
        # Filter out common false positives from speech recognition
        false_positives = ["true", "false", "yes", "no", "okay", "ok", "uh", "um", "ah"]
        if user_input.lower() in false_positives:
            print(f"⚠️ Detected likely false positive: '{user_input}'. Please try again...\n")
            continue
        
        print(f"You: {user_input}\n")
        
        if user_input.lower() in ['quit', 'exit', 'stop', 'goodbye']:
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
        
        # Step 3: Retrieve relevant memories from PostgreSQL knowledge base
        logger.info("=" * 60)
        logger.info("🔍 MEMORY RETRIEVAL - Querying PostgreSQL knowledge base")
        logger.info("=" * 60)
        memory_context = get_memory_context(conversation_id, user_input)
        
        # Step 4: Build prompt with memory context
        if memory_context:
            logger.info("📝 Injecting memory context into LLM prompt")
            full_prompt = f"{memory_context}\n\nUser: {user_input}"
            logger.debug(f"   Full prompt length: {len(full_prompt)} characters")
        else:
            logger.info("   No memory context available, using user input only")
            full_prompt = user_input
        logger.info("=" * 60)
        
        # Step 5: Get LLM response (with retry on JSON parse failure)
        print("Thinking...")
        try:
            llm_response = chat.send_message(full_prompt, retry_on_parse_failure=True)
            assistant_message = llm_response['response']
            user_metadata = llm_response['user_message_metadata']
            prev_llm_metadata = llm_response['previous_llm_message_metadata']
            
            # Log the full LLM JSON response for analysis
            logger.info("=" * 60)
            logger.info("📋 LLM JSON RESPONSE ANALYSIS")
            logger.info("=" * 60)
            logger.info("Full LLM Response JSON:")
            response_json = {
                'response': assistant_message,
                'user_message_metadata': user_metadata,
                'previous_llm_message_metadata': prev_llm_metadata,
                'raw_response': llm_response.get('raw_response', 'N/A')
            }
            logger.info(json.dumps(response_json, indent=2, ensure_ascii=False))
            
            # Detailed analysis
            logger.info("\n📊 Response Analysis:")
            logger.info(f"   Response length: {len(assistant_message)} characters")
            logger.info(f"   Response preview: {assistant_message[:100]}...")
            
            if user_metadata:
                logger.info(f"\n   User Message Metadata:")
                logger.info(f"      Type: {user_metadata.get('type', 'N/A')}")
                logger.info(f"      Priority: {user_metadata.get('priority', 'N/A')}")
            else:
                logger.info(f"\n   User Message Metadata: None (not provided by LLM)")
            
            if prev_llm_metadata:
                logger.info(f"\n   Previous LLM Message Metadata:")
                logger.info(f"      Type: {prev_llm_metadata.get('type', 'N/A')}")
                logger.info(f"      Priority: {prev_llm_metadata.get('priority', 'N/A')}")
            else:
                logger.info(f"\n   Previous LLM Message Metadata: None (not provided by LLM)")
            
            logger.info("=" * 60)
            
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
        logger.info("=" * 60)
        logger.info("🔄 METADATA UPDATE - Updating message metadata in PostgreSQL")
        logger.info("=" * 60)
        if user_metadata:
            logger.info(f"   User message metadata to update: {user_metadata}")
        else:
            logger.warning("   ⚠️ No user message metadata provided by LLM")
        
        if prev_llm_metadata:
            logger.info(f"   Previous LLM message metadata to update: {prev_llm_metadata}")
        else:
            logger.warning("   ⚠️ No previous LLM message metadata provided by LLM")
        
        if user_metadata or prev_llm_metadata:
            try:
                success = update_metadata_from_response(
                    conversation_id=conversation_id,
                    user_message_metadata=user_metadata,
                    previous_llm_message_metadata=prev_llm_metadata
                )
                if success:
                    logger.info("✅ Metadata update completed successfully")
                else:
                    logger.warning("⚠️ Metadata update returned False (may have failed)")
            except Exception as e:
                logger.error(f"❌ Failed to update metadata: {e}", exc_info=True)
        else:
            logger.warning("⚠️ No metadata to update - LLM did not provide metadata")
        logger.info("=" * 60)
        
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

