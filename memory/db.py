"""PostgreSQL database connection and operations for conversation memory."""

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool
import logging
from typing import Optional, List, Dict, Any
import uuid

logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'dbname': 'ai',
    'user': 'ai',
    'password': 'ai',
    'host': 'localhost',
    'port': 5532
}

# Connection pool
_pool: Optional[ThreadedConnectionPool] = None


def get_pool() -> ThreadedConnectionPool:
    """Get or create connection pool."""
    global _pool
    if _pool is None:
        try:
            _pool = ThreadedConnectionPool(
                minconn=1,
                maxconn=5,
                **DB_CONFIG
            )
        except Exception as e:
            logger.error(f"Failed to create connection pool: {e}")
            raise
    return _pool


def get_connection():
    """Get a connection from the pool."""
    try:
        pool = get_pool()
        return pool.getconn()
    except Exception as e:
        logger.error(f"Failed to get connection: {e}")
        return None


def return_connection(conn):
    """Return a connection to the pool."""
    try:
        pool = get_pool()
        if pool:
            pool.putconn(conn)
    except Exception as e:
        logger.error(f"Failed to return connection: {e}")


def init_db():
    """Initialize database schema."""
    conn = get_connection()
    if conn is None:
        logger.error("Cannot initialize DB: no connection")
        return False
    
    try:
        with conn.cursor() as cur:
            # Read and execute schema.sql line by line to handle errors gracefully
            schema_path = __file__.replace('db.py', 'schema.sql')
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Split by semicolons and execute each statement
            statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
            
            for statement in statements:
                try:
                    cur.execute(statement)
                except Exception as e:
                    # Log but continue - some statements might fail if already exists
                    logger.debug(f"Schema statement warning (may already exist): {e}")
            
            conn.commit()
            logger.info("Database schema initialized successfully")
            return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        conn.rollback()
        return False
    finally:
        return_connection(conn)


def insert_message(
    conversation_id: uuid.UUID,
    message_index: int,
    role: str,
    text: str,
    embedding: Optional[List[float]] = None,
    type: Optional[str] = None,
    priority: Optional[float] = None
) -> Optional[int]:
    """Insert a message into conversation_memory."""
    conn = get_connection()
    if conn is None:
        logger.error("Cannot insert message: no connection")
        return None
    
    try:
        with conn.cursor() as cur:
            # Convert embedding list to PostgreSQL vector format: '[1,2,3]'
            embedding_param = None
            if embedding:
                embedding_param = '[' + ','.join(map(str, embedding)) + ']'
            
            cur.execute("""
                INSERT INTO conversation_memory 
                (conversation_id, message_index, role, text, embedding, type, priority)
                VALUES (%s::uuid, %s, %s, %s, %s::vector, %s, %s)
                RETURNING id
            """, (
                str(conversation_id),
                message_index,
                role,
                text,
                embedding_param,
                type,
                priority
            ))
            result = cur.fetchone()
            conn.commit()
            return result[0] if result else None
    except Exception as e:
        logger.error(f"Failed to insert message: {e}")
        conn.rollback()
        return None
    finally:
        return_connection(conn)


def update_message_metadata(
    message_id: int,
    type: Optional[str] = None,
    priority: Optional[float] = None
) -> bool:
    """Update metadata for a message."""
    conn = get_connection()
    if conn is None:
        logger.error("Cannot update message: no connection")
        return False
    
    try:
        logger.debug(f"   Updating message {message_id} with type={type}, priority={priority}")
        
        with conn.cursor() as cur:
            updates = []
            params = []
            
            if type is not None:
                updates.append("type = %s")
                params.append(type)
                logger.debug(f"   Adding type update: {type}")
            
            if priority is not None:
                updates.append("priority = %s")
                params.append(priority)
                logger.debug(f"   Adding priority update: {priority}")
            
            if not updates:
                logger.warning(f"   No updates to apply for message {message_id}")
                return True  # Nothing to update
            
            params.append(message_id)
            update_sql = f"""
                UPDATE conversation_memory
                SET {', '.join(updates)}
                WHERE id = %s
            """
            logger.debug(f"   Executing SQL: {update_sql}")
            logger.debug(f"   Parameters: {params}")
            
            cur.execute(update_sql, params)
            rows_affected = cur.rowcount
            conn.commit()
            
            logger.debug(f"   Rows affected: {rows_affected}")
            
            if rows_affected > 0:
                # Verify the update
                cur.execute("SELECT type, priority FROM conversation_memory WHERE id = %s", (message_id,))
                result = cur.fetchone()
                if result:
                    logger.debug(f"   Verified update: type={result[0]}, priority={result[1]}")
                return True
            else:
                logger.warning(f"   No rows updated for message {message_id} - message may not exist")
                return False
    except Exception as e:
        logger.error(f"❌ Failed to update message metadata: {e}", exc_info=True)
        conn.rollback()
        return False
    finally:
        return_connection(conn)


def get_message_by_id(message_id: int) -> Optional[Dict[str, Any]]:
    """Get a message by ID."""
    conn = get_connection()
    if conn is None:
        return None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, conversation_id, message_index, role, text, 
                       embedding, type, priority, created_at
                FROM conversation_memory
                WHERE id = %s
            """, (message_id,))
            result = cur.fetchone()
            return dict(result) if result else None
    except Exception as e:
        logger.error(f"Failed to get message: {e}")
        return None
    finally:
        return_connection(conn)


def get_recent_messages(
    conversation_id: uuid.UUID,
    role: Optional[str] = None,
    limit: int = 1
) -> List[Dict[str, Any]]:
    """Get recent messages for a conversation, optionally filtered by role."""
    conn = get_connection()
    if conn is None:
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if role:
                cur.execute("""
                    SELECT id, conversation_id, message_index, role, text, 
                           embedding, type, priority, created_at
                    FROM conversation_memory
                    WHERE conversation_id = %s::uuid AND role = %s
                    ORDER BY message_index DESC
                    LIMIT %s
                """, (str(conversation_id), role, limit))
            else:
                cur.execute("""
                    SELECT id, conversation_id, message_index, role, text, 
                           embedding, type, priority, created_at
                    FROM conversation_memory
                    WHERE conversation_id = %s::uuid
                    ORDER BY message_index DESC
                    LIMIT %s
                """, (str(conversation_id), limit))
            
            results = cur.fetchall()
            return [dict(row) for row in results]
    except Exception as e:
        logger.error(f"Failed to get recent messages: {e}")
        return []
    finally:
        return_connection(conn)


def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None

