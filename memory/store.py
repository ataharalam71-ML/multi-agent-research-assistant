"""
memory/store.py — Shared memory store for inter-agent communication.
Uses Redis when REDIS_URL is set, otherwise falls back to a plain dict.
"""
from __future__ import annotations
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemoryStore:
    """
    Simple key-value store. 
    - set(key, value)  — store any JSON-serialisable value
    - get(key)         — retrieve or None
    - delete(key)
    - clear()          — wipe all keys for this session
    """

    def __init__(self, redis_url: str = "") -> None:
        self._redis = None
        self._local: dict[str, str] = {}

        if redis_url:
            try:
                import redis
                self._redis = redis.from_url(redis_url, decode_responses=True)
                self._redis.ping()
                logger.info("MemoryStore: connected to Redis at %s", redis_url)
            except Exception as exc:
                logger.warning("MemoryStore: Redis unavailable (%s), using in-memory fallback", exc)
                self._redis = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        serialised = json.dumps(value, default=str)
        if self._redis:
            self._redis.setex(key, ttl_seconds, serialised)
        else:
            self._local[key] = serialised

    def get(self, key: str) -> Any | None:
        raw = self._redis.get(key) if self._redis else self._local.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def delete(self, key: str) -> None:
        if self._redis:
            self._redis.delete(key)
        else:
            self._local.pop(key, None)

    def clear(self, prefix: str = "") -> None:
        if self._redis:
            keys = self._redis.keys(f"{prefix}*")
            if keys:
                self._redis.delete(*keys)
        else:
            if prefix:
                to_remove = [k for k in self._local if k.startswith(prefix)]
                for k in to_remove:
                    del self._local[k]
            else:
                self._local.clear()

    def append_to_list(self, key: str, item: Any) -> None:
        existing: list = self.get(key) or []
        existing.append(item)
        self.set(key, existing)


# ── Long-term vector memory ────────────────────────────────────────────────────

class VectorMemory:
    """
    Persists past research sessions as embeddings in ChromaDB.
    Allows the Searcher to retrieve semantically similar past results.
    """

    def __init__(self, persist_dir: str = "./data/chroma") -> None:
        try:
            import chromadb
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name="research_sessions",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("VectorMemory: ChromaDB at %s", persist_dir)
        except Exception as exc:
            logger.warning("VectorMemory: ChromaDB unavailable (%s)", exc)
            self._collection = None

    def save_session(self, query: str, answer: str, session_id: str) -> None:
        if self._collection is None:
            return
        try:
            self._collection.add(
                documents=[f"Q: {query}\nA: {answer}"],
                ids=[session_id],
                metadatas=[{"query": query}],
            )
        except Exception as exc:
            logger.warning("VectorMemory.save_session failed: %s", exc)

    def retrieve_similar(self, query: str, top_k: int = 3) -> list[dict]:
        if self._collection is None:
            return []
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(top_k, self._collection.count()),
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]
        except Exception as exc:
            logger.warning("VectorMemory.retrieve_similar failed: %s", exc)
            return []
