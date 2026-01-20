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
    exclude_type: Optional[str] = 'random_talk'
) -> List[Dict[str, Any]]:
    """
    Search for similar messages using vector similarity.
    
    Args:
        query_embedding: Embedding vector for the query
        conversation_id: Optional conversation ID to filter by
        top_k: Number of results to return
        priority_threshold: Minimum priority to include
        exclude_type: Message type to exclude (e.g., 'random_talk')
    
    Returns:
        List of message dictionaries with similarity scores
    """
    conn = get_connection()
    if conn is None:
        logger.error("Cannot search: no connection")
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Build query with filters
            where_clauses = ["embedding IS NOT NULL"]
            params = []
            
            if conversation_id:
                where_clauses.append("conversation_id = %s")
                params.append(conversation_id)
            
            if priority_threshold is not None:
                where_clauses.append("(priority IS NULL OR priority > %s)")
                params.append(priority_threshold)
            
            if exclude_type:
                where_clauses.append("(type IS NULL OR type != %s)")
                params.append(exclude_type)
            
            where_sql = " AND ".join(where_clauses)
            
            # Convert embedding list to PostgreSQL vector format: '[1,2,3]'
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Vector similarity search using cosine distance
            # 1 - cosine_distance = cosine_similarity
            cur.execute(f"""
                SELECT 
                    id, conversation_id, message_index, role, text, 
                    type, priority, created_at,
                    1 - (embedding <=> %s::vector) as similarity
                FROM conversation_memory
                WHERE {where_sql}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, [embedding_str] + params + [embedding_str, top_k])
            
            results = cur.fetchall()
            messages = []
            
            for row in results:
                msg_dict = dict(row)
                similarity = float(msg_dict['similarity'])
                priority = float(msg_dict['priority']) if msg_dict['priority'] else 0.0
                
                # Calculate combined score: similarity * priority
                # If priority is None, use similarity only
                score = similarity * priority if priority > 0 else similarity
                
                msg_dict['score'] = score
                messages.append(msg_dict)
            
            # Sort by combined score (similarity * priority)
            messages.sort(key=lambda x: x['score'], reverse=True)
            
            return messages[:top_k]
    except Exception as e:
        logger.error(f"Failed to search similar messages: {e}")
        return []
    finally:
        return_connection(conn)


def format_memories_for_context(memories: List[Dict[str, Any]]) -> str:
    """Format retrieved memories for injection into LLM context."""
    if not memories:
        return ""
    
    formatted = "Relevant past conversation snippets:\n"
    for i, mem in enumerate(memories, 1):
        role_label = "User" if mem['role'] == 'human' else "Assistant"
        text = mem['text'][:200]  # Truncate long messages
        if len(mem['text']) > 200:
            text += "..."
        formatted += f"{i}. [{role_label}] {text}\n"
    
    return formatted

