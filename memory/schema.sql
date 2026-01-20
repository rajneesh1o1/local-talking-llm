-- PostgreSQL schema for conversation memory with pgvector

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create conversation_memory table
CREATE TABLE IF NOT EXISTS conversation_memory (
    id SERIAL PRIMARY KEY,
    conversation_id UUID NOT NULL,
    message_index INT NOT NULL,

    role TEXT CHECK (role IN ('human', 'llm')) NOT NULL,
    text TEXT NOT NULL,

    embedding VECTOR(768),

    type TEXT,
    priority FLOAT CHECK (priority >= 0 AND priority <= 1),

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create indexes for efficient vector search and lookups
-- Note: ivfflat index requires at least some rows. Create it after data is inserted.
-- For now, create a regular index that can be upgraded later
CREATE INDEX IF NOT EXISTS conversation_memory_embedding_idx 
    ON conversation_memory USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS conversation_memory_conversation_idx 
    ON conversation_memory (conversation_id, message_index);

CREATE INDEX IF NOT EXISTS conversation_memory_created_at_idx 
    ON conversation_memory (created_at);

