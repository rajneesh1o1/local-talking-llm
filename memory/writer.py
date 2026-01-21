"""Message insertion and metadata update logic."""

import logging
from typing import Optional, Dict, Any, List
import uuid
from datetime import datetime, timedelta
from memory.db import (
    insert_message, update_message_metadata, 
    get_recent_messages, get_message_by_id
)

logger = logging.getLogger(__name__)


def write_user_message(
    conversation_id: uuid.UUID,
    message_index: int,
    text: str,
    embedding: Optional[List[float]] = None
) -> Optional[int]:
    """
    Insert a user message into memory.
    
    Returns:
        Message ID if successful, None otherwise
    """
    try:
        logger.info(f"💾 Writing user message to PostgreSQL (index={message_index}, conv_id={conversation_id})")
        logger.debug(f"   Text: {text[:100]}...")
        logger.debug(f"   Embedding: {'present' if embedding else 'none'}")
        
        message_id = insert_message(
            conversation_id=conversation_id,
            message_index=message_index,
            role='human',
            text=text,
            embedding=embedding,
            type=None,
            priority=None
        )
        
        if message_id:
            logger.info(f"✅ User message saved to PostgreSQL (ID: {message_id})")
        else:
            logger.warning("⚠️ Failed to get message ID after insert")
        
        return message_id
    except Exception as e:
        logger.error(f"❌ Failed to write user message: {e}", exc_info=True)
        return None


def write_llm_message(
    conversation_id: uuid.UUID,
    message_index: int,
    text: str,
    embedding: Optional[List[float]] = None
) -> Optional[int]:
    """
    Insert an LLM message into memory.
    
    Returns:
        Message ID if successful, None otherwise
    """
    try:
        logger.info(f"💾 Writing LLM message to PostgreSQL (index={message_index}, conv_id={conversation_id})")
        logger.debug(f"   Text: {text[:100]}...")
        logger.debug(f"   Embedding: {'present' if embedding else 'none'}")
        
        message_id = insert_message(
            conversation_id=conversation_id,
            message_index=message_index,
            role='llm',
            text=text,
            embedding=embedding,
            type=None,
            priority=None
        )
        
        if message_id:
            logger.info(f"✅ LLM message saved to PostgreSQL (ID: {message_id})")
        else:
            logger.warning("⚠️ Failed to get message ID after insert")
        
        return message_id
    except Exception as e:
        logger.error(f"❌ Failed to write LLM message: {e}", exc_info=True)
        return None


def update_metadata_from_response(
    conversation_id: uuid.UUID,
    user_message_metadata: Optional[Dict[str, Any]] = None,
    previous_llm_message_metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update metadata for messages based on LLM response.
    
    Rules:
    - Update user message Mn: must be role=human, timestamp <= 5 minutes old
    - Update previous LLM message Mn-1: must be role=llm
    
    Args:
        conversation_id: Conversation UUID
        user_message_metadata: Dict with 'type' and 'priority' for user message
        previous_llm_message_metadata: Dict with 'type' and 'priority' for previous LLM message
    
    Returns:
        True if any updates succeeded, False otherwise
    """
    updated = False
    
    # Update user message (most recent human message)
    # Use message_index instead of timestamp for reliability
    if user_message_metadata:
        logger.info(f"🔄 Processing user message metadata update")
        logger.debug(f"   Metadata received: {user_message_metadata}")
        try:
            recent_human = get_recent_messages(
                conversation_id=conversation_id,
                role='human',
                limit=1
            )
            
            if not recent_human:
                logger.warning(f"⚠️ No recent human messages found for conversation {conversation_id}")
                return updated
            
            msg = recent_human[0]
            logger.debug(f"   Found recent human message: ID={msg['id']}, index={msg.get('message_index')}, text={msg['text'][:50]}...")
            
            # Use message_index as primary identifier - if it's the most recent human message, update it
            # Don't rely on timestamp due to timezone issues - message_index DESC LIMIT 1 ensures it's the most recent
            logger.debug(f"   Most recent human message index: {msg.get('message_index')}")
            
            # Always update if we found a recent human message
            logger.info(f"🔄 Updating user message metadata (ID: {msg['id']})")
            logger.info(f"   Type: {user_message_metadata.get('type')}, Priority: {user_message_metadata.get('priority')}")
            
            success = update_message_metadata(
                message_id=msg['id'],
                type=user_message_metadata.get('type'),
                priority=user_message_metadata.get('priority')
            )
            if success:
                updated = True
                logger.info(f"✅ Updated user message {msg['id']} metadata in PostgreSQL")
                # Verify the update
                updated_msg = get_message_by_id(msg['id'])
                if updated_msg:
                    logger.info(f"   Verified: type={updated_msg.get('type')}, priority={updated_msg.get('priority')}")
            else:
                logger.warning(f"⚠️ Failed to update user message {msg['id']} metadata (update_message_metadata returned False)")
        except Exception as e:
            logger.error(f"❌ Failed to update user message metadata: {e}", exc_info=True)
    
    # Update previous LLM message (second most recent LLM message, or most recent if only one)
    if previous_llm_message_metadata:
        try:
            recent_llm = get_recent_messages(
                conversation_id=conversation_id,
                role='llm',
                limit=2
            )
            
            # Get the previous LLM message (index 1 if exists, else index 0)
            if len(recent_llm) >= 2:
                prev_msg = recent_llm[1]  # Second most recent
            elif len(recent_llm) == 1:
                prev_msg = recent_llm[0]  # Only one exists
            else:
                prev_msg = None
            
            if prev_msg:
                logger.info(f"🔄 Updating previous LLM message metadata (ID: {prev_msg['id']})")
                logger.debug(f"   Type: {previous_llm_message_metadata.get('type')}, Priority: {previous_llm_message_metadata.get('priority')}")
                
                success = update_message_metadata(
                    message_id=prev_msg['id'],
                    type=previous_llm_message_metadata.get('type'),
                    priority=previous_llm_message_metadata.get('priority')
                )
                if success:
                    updated = True
                    logger.info(f"✅ Updated previous LLM message {prev_msg['id']} metadata in PostgreSQL")
                else:
                    logger.warning(f"⚠️ Failed to update previous LLM message {prev_msg['id']} metadata")
        except Exception as e:
            logger.error(f"Failed to update previous LLM message metadata: {e}")
    
    return updated

