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
        message_id = insert_message(
            conversation_id=conversation_id,
            message_index=message_index,
            role='human',
            text=text,
            embedding=embedding,
            type=None,
            priority=None
        )
        return message_id
    except Exception as e:
        logger.error(f"Failed to write user message: {e}")
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
        message_id = insert_message(
            conversation_id=conversation_id,
            message_index=message_index,
            role='llm',
            text=text,
            embedding=embedding,
            type=None,
            priority=None
        )
        return message_id
    except Exception as e:
        logger.error(f"Failed to write LLM message: {e}")
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
    
    # Update user message (most recent human message, <= 5 minutes old)
    if user_message_metadata:
        try:
            recent_human = get_recent_messages(
                conversation_id=conversation_id,
                role='human',
                limit=1
            )
            
            if recent_human:
                msg = recent_human[0]
                created_at = msg['created_at']
                
                # Check if message is <= 5 minutes old
                if isinstance(created_at, str):
                    created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                elif isinstance(created_at, datetime):
                    pass
                else:
                    logger.warning(f"Unexpected created_at type: {type(created_at)}")
                    created_at = None
                
                if created_at:
                    age = datetime.now(created_at.tzinfo) - created_at if created_at.tzinfo else datetime.now() - created_at.replace(tzinfo=None)
                    
                    if age <= timedelta(minutes=5):
                        success = update_message_metadata(
                            message_id=msg['id'],
                            type=user_message_metadata.get('type'),
                            priority=user_message_metadata.get('priority')
                        )
                        if success:
                            updated = True
                            logger.info(f"Updated user message {msg['id']} metadata")
                    else:
                        logger.debug(f"User message too old ({age}), skipping metadata update")
        except Exception as e:
            logger.error(f"Failed to update user message metadata: {e}")
    
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
                success = update_message_metadata(
                    message_id=prev_msg['id'],
                    type=previous_llm_message_metadata.get('type'),
                    priority=previous_llm_message_metadata.get('priority')
                )
                if success:
                    updated = True
                    logger.info(f"Updated previous LLM message {prev_msg['id']} metadata")
        except Exception as e:
            logger.error(f"Failed to update previous LLM message metadata: {e}")
    
    return updated

