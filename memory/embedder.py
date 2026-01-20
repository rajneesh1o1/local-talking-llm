"""Local embedding model wrapper."""

import logging
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Global embedding model
_embedding_model = None


def load_embedding_model():
    """Load the local embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Using a lightweight, fast model that produces 768-dim embeddings
            # all-MiniLM-L6-v2 produces 384-dim, so we'll use all-mpnet-base-v2 (768-dim)
            # or we can use a smaller one and pad/truncate
            # Let's use all-MiniLM-L6-v2 and pad to 768 for compatibility
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _embedding_model


def embed(text: str) -> List[float]:
    """
    Generate embedding for text.
    Returns a 768-dimensional vector (padded if needed).
    """
    if not text or not text.strip():
        # Return zero vector for empty text
        return [0.0] * 768
    
    try:
        model = load_embedding_model()
        # Generate embedding (all-MiniLM-L6-v2 produces 384-dim)
        embedding = model.encode(text, normalize_embeddings=True)
        
        # Convert to list
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
        
        # Pad or truncate to 768 dimensions
        if len(embedding_list) < 768:
            # Pad with zeros
            embedding_list.extend([0.0] * (768 - len(embedding_list)))
        elif len(embedding_list) > 768:
            # Truncate
            embedding_list = embedding_list[:768]
        
        return embedding_list
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        # Return zero vector on error
        return [0.0] * 768

