"""Vector search and memory retrieval."""

import logging
from typing import List, Dict, Any, Optional
import uuid
from psycopg2.extras import RealDictCursor
from memory.db import get_connection, return_connection

logger = logging.getLogger(__name__)


def search_similar_messages(
    query_embedding: List[float],
    conversation_id: Optional[uuid.UUID] = None,
    top_k: int = 5,
    priority_threshold: float = 0.3,
    exclude_type: Optional[str] = 'random_talk',
    role_filter: Optional[str] = None,
    type_filter: Optional[str] = None,
    cross_conversation: bool = True
) -> List[Dict[str, Any]]:
    """
    Search for similar messages using vector similarity with advanced filtering.
    
    Args:
        query_embedding: Embedding vector for the query
        conversation_id: Optional conversation ID to filter by (if cross_conversation=False)
        top_k: Number of results to return
        priority_threshold: Minimum priority to include
        exclude_type: Message type to exclude (e.g., 'random_talk')
        role_filter: Filter by role ('human' or 'llm'), None for all
        type_filter: Filter by specific type (e.g., 'about_user', 'informative')
        cross_conversation: If True, search across all conversations; if False, only current
    
    Returns:
        List of message dictionaries with similarity scores
    """
    conn = get_connection()
    if conn is None:
        logger.error("Cannot search: no connection")
        return []
    
    try:
        logger.info(f"🔍 Starting semantic search (top_k={top_k}, priority_threshold={priority_threshold})")
        if conversation_id:
            logger.info(f"   Conversation ID: {conversation_id}")
        logger.info(f"   Cross-conversation: {cross_conversation}")
        logger.info(f"   Role filter: {role_filter}, Type filter: {type_filter}, Exclude type: {exclude_type}")
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Build query with filters
            where_clauses = ["embedding IS NOT NULL"]
            params = []
            
            # Conversation filtering
            if not cross_conversation and conversation_id:
                where_clauses.append("conversation_id = %s::uuid")
                params.append(str(conversation_id))
                logger.debug(f"   Filtering by conversation_id: {conversation_id}")
            
            # Priority filtering
            if priority_threshold is not None:
                where_clauses.append("(priority IS NULL OR priority > %s)")
                params.append(priority_threshold)
                logger.debug(f"   Priority threshold: >{priority_threshold}")
            
            # Type filtering
            if exclude_type:
                where_clauses.append("(type IS NULL OR type != %s)")
                params.append(exclude_type)
                logger.debug(f"   Excluding type: {exclude_type}")
            
            if type_filter:
                where_clauses.append("type = %s")
                params.append(type_filter)
                logger.debug(f"   Including only type: {type_filter}")
            
            # Role filtering
            if role_filter:
                where_clauses.append("role = %s")
                params.append(role_filter)
                logger.debug(f"   Role filter: {role_filter}")
            
            where_sql = " AND ".join(where_clauses)
            
            # Convert embedding list to PostgreSQL vector format: '[1,2,3]'
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Vector similarity search using cosine distance
            # 1 - cosine_distance = cosine_similarity
            query_sql = f"""
                SELECT 
                    id, conversation_id, message_index, role, text, 
                    type, priority, created_at,
                    1 - (embedding <=> %s::vector) as similarity
                FROM conversation_memory
                WHERE {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            
            logger.debug(f"   Executing query with {len(params)} parameters")
            cur.execute(query_sql, [embedding_str] + params + [embedding_str, top_k])
            
            results = cur.fetchall()
            logger.info(f"   Found {len(results)} results from PostgreSQL")
            
            messages = []
            
            for row in results:
                msg_dict = dict(row)
                similarity = float(msg_dict['similarity'])
                priority = float(msg_dict['priority']) if msg_dict['priority'] else 0.0
                
                # Enhanced scoring: prioritize high-priority and about_user messages
                base_score = similarity
                
                # Boost score based on priority
                if priority > 0:
                    # Weighted combination: 70% similarity, 30% priority
                    score = (base_score * 0.7) + (priority * 0.3)
                else:
                    score = base_score
                
                # Additional boost for 'about_user' type (personal information)
                if msg_dict.get('type') == 'about_user':
                    score *= 1.2
                    logger.debug(f"   Boosted score for 'about_user' message ID {msg_dict['id']}")
                
                # Additional boost for high-priority informative messages
                if msg_dict.get('type') == 'informative' and priority > 0.7:
                    score *= 1.15
                
                msg_dict['score'] = score
                msg_dict['base_similarity'] = similarity
                messages.append(msg_dict)
            
            # Sort by combined score (similarity * priority)
            messages.sort(key=lambda x: x['score'], reverse=True)
            
            # Log retrieved memories
            if messages:
                logger.info(f"✅ Retrieved {len(messages)} memories from knowledge base:")
                for i, mem in enumerate(messages[:top_k], 1):
                    logger.info(f"   {i}. [ID:{mem['id']}] [{mem['role']}] "
                              f"Type:{mem.get('type', 'N/A')} Priority:{mem.get('priority', 'N/A')} "
                              f"Score:{mem['score']:.3f} Similarity:{mem['base_similarity']:.3f}")
                    logger.debug(f"      Text: {mem['text'][:100]}...")
            else:
                logger.info("   No relevant memories found")
            
            return messages[:top_k]
    except Exception as e:
        logger.error(f"❌ Failed to search similar messages: {e}", exc_info=True)
        return []
    finally:
        return_connection(conn)


def format_memories_for_context(memories: List[Dict[str, Any]]) -> str:
    """
    Format retrieved memories for injection into LLM context.
    Includes metadata to help personalize responses.
    """
    if not memories:
        return ""
    
    formatted = "Relevant past conversation snippets from knowledge base:\n"
    
    # Group by type for better organization
    about_user_memories = [m for m in memories if m.get('type') == 'about_user']
    informative_memories = [m for m in memories if m.get('type') == 'informative']
    other_memories = [m for m in memories if m.get('type') not in ['about_user', 'informative']]
    
    # Prioritize 'about_user' memories (personal information)
    if about_user_memories:
        formatted += "\n[Personal Information About User]\n"
        for i, mem in enumerate(about_user_memories, 1):
            role_label = "User" if mem['role'] == 'human' else "Assistant"
            text = mem['text'][:250]
            if len(mem['text']) > 250:
                text += "..."
            priority_info = f" (Priority: {mem.get('priority', 'N/A')})" if mem.get('priority') else ""
            formatted += f"{i}. [{role_label}] {text}{priority_info}\n"
    
    # Include informative memories
    if informative_memories:
        formatted += "\n[Informative Context]\n"
        for i, mem in enumerate(informative_memories, 1):
            role_label = "User" if mem['role'] == 'human' else "Assistant"
            text = mem['text'][:200]
            if len(mem['text']) > 200:
                text += "..."
            formatted += f"{i}. [{role_label}] {text}\n"
    
    # Include other relevant memories
    if other_memories:
        formatted += "\n[Other Relevant Context]\n"
        for i, mem in enumerate(other_memories, 1):
            role_label = "User" if mem['role'] == 'human' else "Assistant"
            text = mem['text'][:200]
            if len(mem['text']) > 200:
                text += "..."
            formatted += f"{i}. [{role_label}] {text}\n"
    
    formatted += "\nUse this context to personalize your response and maintain conversation continuity.\n"
    
    return formatted

