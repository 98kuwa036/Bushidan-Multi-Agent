"""
将軍システム v8.0 - Knowledge Base (RAG) System
知識基盤: 外部知識の保存・検索システム

Features:
- Qdrant vector database integration
- Sentence Transformers embeddings
- Multi-language support
- Auto-cleanup (30 days retention)
- Real-time search with scoring
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from sentence_transformers import SentenceTransformer
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Data source types for knowledge base entries"""
    OLLAMA_WEB_SEARCH = "ollama_web_search"
    MANUAL_DOCUMENT = "manual_document"  
    GITHUB_ISSUE = "github_issue"
    GITHUB_PR = "github_pr"
    NOTION_PAGE = "notion_page"


@dataclass
class KnowledgeEntry:
    """Knowledge base entry structure"""
    id: str
    title: str
    content: str
    source: DataSource
    url: Optional[str] = None
    metadata: Dict[str, Any] = None
    language: str = "ja"
    created_at: datetime = None
    relevance_score: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}


class KnowledgeBase:
    """
    将軍システム Knowledge Base (RAG) Implementation
    
    Provides semantic search capabilities for the Shogun system using:
    - Qdrant for vector storage
    - Sentence Transformers for embeddings
    - Automatic data lifecycle management
    """
    
    def __init__(
        self,
        host: str = "192.168.1.10",
        port: int = 6333,
        collection_name: str = "shogun_knowledge",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        retention_days: int = 30
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.retention_days = retention_days
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=host, port=port)
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize collection
        asyncio.create_task(self._initialize_collection())
        
        logger.info(f"KnowledgeBase initialized - Host: {host}:{port}, Collection: {collection_name}")
    
    async def _initialize_collection(self) -> None:
        """Initialize Qdrant collection if it doesn't exist"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def add_entry(self, entry: KnowledgeEntry) -> bool:
        """Add a knowledge entry to the database"""
        try:
            # Generate embedding
            combined_text = f"{entry.title} {entry.content}"
            embedding = self._generate_embedding(combined_text)
            
            # Prepare payload
            payload = {
                "id": entry.id,
                "title": entry.title,
                "content": entry.content,
                "source": entry.source.value,
                "url": entry.url,
                "metadata": entry.metadata,
                "language": entry.language,
                "created_at": entry.created_at.isoformat(),
                "text": combined_text  # For full-text search
            }
            
            # Insert into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=entry.id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            logger.info(f"Added knowledge entry: {entry.id} - {entry.title[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add entry {entry.id}: {e}")
            return False
    
    async def search(
        self,
        query: str,
        max_results: int = 5,
        score_threshold: float = 0.7,
        source_filter: Optional[List[DataSource]] = None,
        language_filter: Optional[str] = None
    ) -> List[KnowledgeEntry]:
        """
        Search knowledge base using semantic similarity
        
        Args:
            query: Search query text
            max_results: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            source_filter: Filter by data sources
            language_filter: Filter by language
        
        Returns:
            List of relevant knowledge entries with scores
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Prepare filters
            must_conditions = []
            
            if source_filter:
                source_values = [source.value for source in source_filter]
                must_conditions.append(
                    models.FieldCondition(
                        key="source",
                        match=models.MatchAny(any=source_values)
                    )
                )
            
            if language_filter:
                must_conditions.append(
                    models.FieldCondition(
                        key="language",
                        match=models.MatchValue(value=language_filter)
                    )
                )
            
            query_filter = None
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)
            
            # Perform search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=max_results,
                score_threshold=score_threshold
            )
            
            # Convert results to KnowledgeEntry objects
            entries = []
            for result in search_results:
                payload = result.payload
                entry = KnowledgeEntry(
                    id=payload["id"],
                    title=payload["title"],
                    content=payload["content"],
                    source=DataSource(payload["source"]),
                    url=payload.get("url"),
                    metadata=payload.get("metadata", {}),
                    language=payload["language"],
                    created_at=datetime.fromisoformat(payload["created_at"]),
                    relevance_score=result.score
                )
                entries.append(entry)
            
            logger.info(f"Search query: '{query}' - Found {len(entries)} results")
            return entries
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Retrieve a specific knowledge entry by ID"""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[entry_id]
            )
            
            if not result:
                return None
            
            payload = result[0].payload
            entry = KnowledgeEntry(
                id=payload["id"],
                title=payload["title"],
                content=payload["content"],
                source=DataSource(payload["source"]),
                url=payload.get("url"),
                metadata=payload.get("metadata", {}),
                language=payload["language"],
                created_at=datetime.fromisoformat(payload["created_at"])
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Failed to retrieve entry {entry_id}: {e}")
            return None
    
    async def update_entry(self, entry: KnowledgeEntry) -> bool:
        """Update an existing knowledge entry"""
        try:
            return await self.add_entry(entry)  # Upsert operation
        except Exception as e:
            logger.error(f"Failed to update entry {entry.id}: {e}")
            return False
    
    async def delete_entry(self, entry_id: str) -> bool:
        """Delete a knowledge entry"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[entry_id])
            )
            
            logger.info(f"Deleted knowledge entry: {entry_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete entry {entry_id}: {e}")
            return False
    
    async def cleanup_old_entries(self) -> int:
        """
        Remove entries older than retention_days
        Returns number of entries deleted
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            # Scroll through all entries to find old ones
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Process in batches
                with_payload=True
            )
            
            old_entry_ids = []
            
            while True:
                points = scroll_result[0]
                
                for point in points:
                    created_at = datetime.fromisoformat(point.payload["created_at"])
                    if created_at < cutoff_date:
                        old_entry_ids.append(point.id)
                
                # Check if there are more entries
                next_page_offset = scroll_result[1]
                if next_page_offset is None:
                    break
                
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=next_page_offset,
                    with_payload=True
                )
            
            # Delete old entries in batches
            if old_entry_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=models.PointIdsList(points=old_entry_ids)
                )
                
                logger.info(f"Cleanup: Deleted {len(old_entry_ids)} old entries (older than {self.retention_days} days)")
            else:
                logger.info("Cleanup: No old entries to delete")
            
            return len(old_entry_ids)
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get source distribution
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Large enough to get all entries for stats
                with_payload=True
            )
            
            source_counts = {}
            language_counts = {}
            total_entries = 0
            
            for point in search_result[0]:
                payload = point.payload
                source = payload.get("source", "unknown")
                language = payload.get("language", "unknown")
                
                source_counts[source] = source_counts.get(source, 0) + 1
                language_counts[language] = language_counts.get(language, 0) + 1
                total_entries += 1
            
            stats = {
                "total_entries": total_entries,
                "collection_status": collection_info.status.value,
                "vectors_count": collection_info.vectors_count or 0,
                "indexed_vectors_count": collection_info.indexed_vectors_count or 0,
                "source_distribution": source_counts,
                "language_distribution": language_counts,
                "retention_days": self.retention_days,
                "embedding_dimension": self.embedding_dimension
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the knowledge base"""
        try:
            # Check Qdrant connection
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            # Check embedding model
            test_embedding = self._generate_embedding("test")
            embedding_works = len(test_embedding) == self.embedding_dimension
            
            health_status = {
                "status": "healthy" if (collection_exists and embedding_works) else "unhealthy",
                "qdrant_connection": "ok" if collection_exists else "failed",
                "embedding_model": "ok" if embedding_works else "failed",
                "collection_exists": collection_exists,
                "host": self.host,
                "port": self.port,
                "collection_name": self.collection_name,
                "checked_at": datetime.utcnow().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "checked_at": datetime.utcnow().isoformat()
            }


# Factory function for easy instantiation
def create_knowledge_base(config: Dict[str, Any]) -> KnowledgeBase:
    """Create KnowledgeBase instance from configuration"""
    return KnowledgeBase(
        host=config.get("host", "192.168.1.10"),
        port=config.get("port", 6333),
        collection_name=config.get("collection_name", "shogun_knowledge"),
        embedding_model=config.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"),
        retention_days=config.get("retention_days", 30)
    )


# Example usage and testing
if __name__ == "__main__":
    async def test_knowledge_base():
        """Test the knowledge base functionality"""
        kb = KnowledgeBase()
        
        # Test entry
        test_entry = KnowledgeEntry(
            id="test_001",
            title="React 19.1 新機能",
            content="React 19.1では新しいコンカレント機能が追加されました。Suspenseの改善とServer Componentsの安定化が主要な変更点です。",
            source=DataSource.OLLAMA_WEB_SEARCH,
            url="https://react.dev/blog/react-19",
            metadata={"version": "19.1", "type": "feature_update"},
            language="ja"
        )
        
        # Add entry
        success = await kb.add_entry(test_entry)
        print(f"Add entry: {success}")
        
        # Search
        results = await kb.search("React 新機能", max_results=3)
        print(f"Search results: {len(results)}")
        for result in results:
            print(f"  - {result.title} (score: {result.relevance_score:.3f})")
        
        # Stats
        stats = await kb.get_stats()
        print(f"Stats: {stats}")
        
        # Health check
        health = await kb.health_check()
        print(f"Health: {health}")
    
    # Run test
    asyncio.run(test_knowledge_base())