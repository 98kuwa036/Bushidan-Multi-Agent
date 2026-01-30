"""
将軍システム v8.0 - Ollama Web Search Integration (10番足軽)
最新情報専門足軽: Ollama Web Search APIを使用した最新情報取得・RAG自動保存

Features:
- Ollama Web Search API integration
- Real-time web search capabilities
- Automatic RAG knowledge base integration
- Multi-language search (Japanese/English)
- Cost-effective (¥0 with free tier)
- Knowledge gap filling for R1 and Claude agents
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
import httpx
from urllib.parse import quote_plus

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.knowledge_base import KnowledgeBase, KnowledgeEntry, DataSource
except ImportError:
    # Fallback for standalone testing
    KnowledgeBase = None
    KnowledgeEntry = None
    DataSource = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    """Web search query structure"""
    id: str
    query: str
    language: str = "ja"
    max_results: int = 5
    requested_by: str = "unknown"  # Which agent requested this
    context: Optional[str] = None  # Additional context for the search
    
    def __post_init__(self):
        if not self.id:
            # Generate ID from query and timestamp
            query_hash = hashlib.md5(f"{self.query}{datetime.utcnow()}".encode()).hexdigest()[:8]
            self.id = f"search_{query_hash}"


@dataclass 
class SearchResult:
    """Individual search result"""
    title: str
    url: str
    snippet: str
    published_date: Optional[str] = None
    source_domain: Optional[str] = None
    relevance_score: Optional[float] = None


@dataclass
class SearchResponse:
    """Complete search response"""
    query_id: str
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: int
    language: str
    requested_by: str
    executed_at: datetime = None
    knowledge_entries_created: int = 0
    
    def __post_init__(self):
        if self.executed_at is None:
            self.executed_at = datetime.utcnow()
    
    @property
    def success(self) -> bool:
        return len(self.results) > 0


class OllamaWebSearch:
    """
    将軍システム Ollama Web Search Implementation (10番足軽)
    
    Provides real-time web search capabilities using Ollama's Web Search API:
    - Fill knowledge gaps for R1 (DeepSeek cutoff: 2024-11)
    - Provide latest information to Claude (Sonnet/Opus cutoff: 2025-01)  
    - Automatically save results to RAG knowledge base
    - Support multi-language searches
    - Cost-effective with free tier usage
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.ollama.com",
        knowledge_base: Optional[KnowledgeBase] = None,
        auto_rag_integration: bool = True,
        free_tier_limit: int = 1000
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.knowledge_base = knowledge_base
        self.auto_rag_integration = auto_rag_integration
        self.free_tier_limit = free_tier_limit
        
        # Search history for analysis
        self.search_history: List[SearchResponse] = []
        
        # Request statistics
        self.monthly_requests = 0
        self.last_reset_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        logger.info(f"OllamaWebSearch initialized - Auto RAG: {auto_rag_integration}")
    
    async def search(self, query: SearchQuery) -> SearchResponse:
        """
        Perform web search using Ollama Web Search API
        
        Args:
            query: Search query object
            
        Returns:
            SearchResponse with results and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Check free tier limits
            if not self._check_rate_limits():
                logger.warning(f"Monthly rate limit exceeded: {self.monthly_requests}/{self.free_tier_limit}")
                return SearchResponse(
                    query_id=query.id,
                    query=query.query,
                    results=[],
                    total_results=0,
                    search_time_ms=0,
                    language=query.language,
                    requested_by=query.requested_by
                )
            
            # Perform the search
            search_results = await self._execute_search(query)
            
            # Process results
            processed_results = self._process_results(search_results, query)
            
            # Calculate search time
            search_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Create response object
            response = SearchResponse(
                query_id=query.id,
                query=query.query,
                results=processed_results,
                total_results=len(processed_results),
                search_time_ms=search_time_ms,
                language=query.language,
                requested_by=query.requested_by
            )
            
            # Auto-save to RAG if enabled
            if self.auto_rag_integration and self.knowledge_base and processed_results:
                knowledge_entries_created = await self._save_to_knowledge_base(query, processed_results)
                response.knowledge_entries_created = knowledge_entries_created
            
            # Update statistics
            self.monthly_requests += 1
            self.search_history.append(response)
            
            # Keep only last 50 searches in memory
            if len(self.search_history) > 50:
                self.search_history = self.search_history[-50:]
            
            logger.info(f"Search completed: '{query.query}' - {len(processed_results)} results in {search_time_ms}ms")
            return response
            
        except Exception as e:
            logger.error(f"Search failed for query '{query.query}': {e}")
            return SearchResponse(
                query_id=query.id,
                query=query.query,
                results=[],
                total_results=0,
                search_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                language=query.language,
                requested_by=query.requested_by
            )
    
    async def _execute_search(self, query: SearchQuery) -> Dict[str, Any]:
        """Execute the actual search API call"""
        try:
            # Prepare search parameters
            search_params = {
                "query": query.query,
                "language": query.language,
                "num_results": query.max_results
            }
            
            # Add context if provided
            if query.context:
                search_params["context"] = query.context
            
            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "ShogunSystem-v8.0-Ashigaru10"
            }
            
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make the API request
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/web-search",
                    json=search_params,
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit exceeded
                    logger.warning("Rate limit exceeded")
                    return {"error": "rate_limit_exceeded", "results": []}
                elif response.status_code == 401:
                    # Authentication error
                    logger.error("Authentication failed - check API key")
                    return {"error": "authentication_failed", "results": []}
                else:
                    logger.error(f"API request failed: {response.status_code} - {response.text}")
                    return {"error": f"api_error_{response.status_code}", "results": []}
                    
        except httpx.TimeoutException:
            logger.error("Search request timed out")
            return {"error": "timeout", "results": []}
        except Exception as e:
            logger.error(f"Search request exception: {e}")
            return {"error": str(e), "results": []}
    
    def _process_results(self, api_response: Dict[str, Any], query: SearchQuery) -> List[SearchResult]:
        """Process raw API response into SearchResult objects"""
        try:
            if "error" in api_response:
                logger.error(f"API returned error: {api_response['error']}")
                return []
            
            raw_results = api_response.get("results", [])
            processed_results = []
            
            for idx, raw_result in enumerate(raw_results):
                try:
                    # Extract fields with fallbacks
                    title = raw_result.get("title", f"Search Result {idx + 1}")
                    url = raw_result.get("url", "")
                    snippet = raw_result.get("snippet", raw_result.get("description", ""))
                    
                    # Optional fields
                    published_date = raw_result.get("published_date")
                    source_domain = raw_result.get("source", raw_result.get("domain"))
                    relevance_score = raw_result.get("score", raw_result.get("relevance"))
                    
                    # Create SearchResult object
                    result = SearchResult(
                        title=title.strip(),
                        url=url.strip(),
                        snippet=snippet.strip(),
                        published_date=published_date,
                        source_domain=source_domain,
                        relevance_score=relevance_score
                    )
                    
                    processed_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to process search result {idx}: {e}")
                    continue
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to process search results: {e}")
            return []
    
    async def _save_to_knowledge_base(self, query: SearchQuery, results: List[SearchResult]) -> int:
        """Save search results to RAG knowledge base"""
        try:
            if not self.knowledge_base:
                return 0
            
            entries_created = 0
            
            for idx, result in enumerate(results):
                try:
                    # Create knowledge entry
                    entry_id = f"web_search_{query.id}_{idx}"
                    
                    # Combine title and snippet for content
                    content = f"{result.snippet}"
                    if result.published_date:
                        content += f"\n\n発行日: {result.published_date}"
                    if result.source_domain:
                        content += f"\n出典: {result.source_domain}"
                    
                    # Metadata
                    metadata = {
                        "search_query": query.query,
                        "requested_by": query.requested_by,
                        "search_context": query.context,
                        "relevance_score": result.relevance_score,
                        "source_domain": result.source_domain,
                        "published_date": result.published_date,
                        "search_timestamp": datetime.utcnow().isoformat()
                    }
                    
                    # Create knowledge entry
                    knowledge_entry = KnowledgeEntry(
                        id=entry_id,
                        title=result.title,
                        content=content,
                        source=DataSource.OLLAMA_WEB_SEARCH,
                        url=result.url,
                        metadata=metadata,
                        language=query.language
                    )
                    
                    # Add to knowledge base
                    success = await self.knowledge_base.add_entry(knowledge_entry)
                    if success:
                        entries_created += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to save search result {idx} to knowledge base: {e}")
                    continue
            
            logger.info(f"Saved {entries_created} search results to knowledge base")
            return entries_created
            
        except Exception as e:
            logger.error(f"Failed to save results to knowledge base: {e}")
            return 0
    
    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        current_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Reset counter if new month
        if current_date > self.last_reset_date:
            self.monthly_requests = 0
            self.last_reset_date = current_date
        
        return self.monthly_requests < self.free_tier_limit
    
    async def quick_search(
        self,
        query_text: str,
        language: str = "ja",
        requested_by: str = "unknown",
        max_results: int = 3
    ) -> List[str]:
        """
        Quick search that returns just the snippet texts
        Useful for simple information retrieval
        """
        query = SearchQuery(
            id="",  # Will be auto-generated
            query=query_text,
            language=language,
            max_results=max_results,
            requested_by=requested_by
        )
        
        response = await self.search(query)
        
        # Return just the snippets
        return [result.snippet for result in response.results]
    
    async def search_for_context(
        self,
        topic: str,
        context_query: str,
        language: str = "ja",
        requested_by: str = "taisho"
    ) -> Optional[str]:
        """
        Search for specific context about a topic
        Returns a condensed summary of relevant information
        """
        # Construct enhanced query
        enhanced_query = f"{topic} {context_query}"
        
        query = SearchQuery(
            id="",
            query=enhanced_query,
            language=language,
            max_results=5,
            requested_by=requested_by,
            context=f"Looking for information about {topic} specifically related to {context_query}"
        )
        
        response = await self.search(query)
        
        if not response.success:
            return None
        
        # Combine snippets into a coherent summary
        snippets = [result.snippet for result in response.results[:3]]  # Top 3 results
        combined_info = " ".join(snippets)
        
        # Simple deduplication
        sentences = combined_info.split('。')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen and len(sentence) > 10:
                unique_sentences.append(sentence)
                seen.add(sentence)
        
        if unique_sentences:
            return '。'.join(unique_sentences[:5]) + '。'  # Max 5 sentences
        else:
            return combined_info[:500] + "..." if len(combined_info) > 500 else combined_info
    
    async def get_latest_info(
        self,
        technology: str,
        info_type: str = "最新情報",
        language: str = "ja"
    ) -> Optional[str]:
        """
        Get latest information about a specific technology
        Useful for filling knowledge gaps in R1 and Claude
        """
        query_text = f"{technology} {info_type} 2025"  # Include current year
        
        return await self.search_for_context(
            topic=technology,
            context_query=info_type,
            language=language,
            requested_by="knowledge_gap_filling"
        )
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics"""
        try:
            if not self.search_history:
                return {"message": "No search history available"}
            
            total_searches = len(self.search_history)
            successful_searches = sum(1 for s in self.search_history if s.success)
            
            # Requester distribution
            requester_stats = {}
            language_stats = {}
            
            for search in self.search_history:
                # Requester stats
                requester = search.requested_by
                if requester not in requester_stats:
                    requester_stats[requester] = {"total": 0, "successful": 0}
                requester_stats[requester]["total"] += 1
                if search.success:
                    requester_stats[requester]["successful"] += 1
                
                # Language stats
                lang = search.language
                if lang not in language_stats:
                    language_stats[lang] = 0
                language_stats[lang] += 1
            
            # Average results and search time
            avg_results = sum(s.total_results for s in self.search_history) / total_searches if total_searches > 0 else 0
            avg_search_time = sum(s.search_time_ms for s in self.search_history) / total_searches if total_searches > 0 else 0
            
            # Knowledge base integration stats
            total_kb_entries = sum(s.knowledge_entries_created for s in self.search_history)
            
            stats = {
                "total_searches": total_searches,
                "successful_searches": successful_searches,
                "success_rate_percent": round(successful_searches / total_searches * 100, 2) if total_searches > 0 else 0,
                "monthly_requests_used": self.monthly_requests,
                "monthly_limit": self.free_tier_limit,
                "usage_percent": round(self.monthly_requests / self.free_tier_limit * 100, 2),
                "average_results_per_search": round(avg_results, 1),
                "average_search_time_ms": round(avg_search_time, 1),
                "requester_distribution": requester_stats,
                "language_distribution": language_stats,
                "knowledge_base_entries_created": total_kb_entries,
                "auto_rag_enabled": self.auto_rag_integration
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get usage statistics: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the web search service"""
        try:
            # Test search
            test_query = SearchQuery(
                id="health_check",
                query="test search",
                language="en",
                max_results=1,
                requested_by="health_check"
            )
            
            start_time = datetime.utcnow()
            response = await self.search(test_query)
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Knowledge base connectivity
            kb_status = "unknown"
            if self.knowledge_base:
                kb_health = await self.knowledge_base.health_check()
                kb_status = kb_health.get("status", "unknown")
            
            health_status = {
                "status": "healthy" if response.success else "unhealthy",
                "search_api_accessible": response.success or response.search_time_ms > 0,
                "response_time_ms": response_time_ms,
                "rate_limit_ok": self._check_rate_limits(),
                "monthly_usage": f"{self.monthly_requests}/{self.free_tier_limit}",
                "knowledge_base_status": kb_status,
                "auto_rag_integration": self.auto_rag_integration,
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


# Factory function
def create_ollama_web_search(config: Dict[str, Any], knowledge_base: Optional[KnowledgeBase] = None) -> OllamaWebSearch:
    """Create OllamaWebSearch instance from configuration"""
    return OllamaWebSearch(
        api_key=config.get("api_key"),
        base_url=config.get("base_url", "https://api.ollama.com"),
        knowledge_base=knowledge_base,
        auto_rag_integration=config.get("auto_rag_integration", True),
        free_tier_limit=config.get("free_tier_limit", 1000)
    )


# Example usage and testing
if __name__ == "__main__":
    async def test_ollama_web_search():
        """Test the Ollama web search functionality"""
        # Initialize without knowledge base for testing
        search_service = OllamaWebSearch(auto_rag_integration=False)
        
        # Test search
        query = SearchQuery(
            id="test_search",
            query="React 19 新機能",
            language="ja",
            max_results=3,
            requested_by="test",
            context="React.jsの最新バージョンの機能について調べています"
        )
        
        print(f"Performing search: '{query.query}'")
        response = await search_service.search(query)
        
        print(f"Search results: {response.total_results}")
        for i, result in enumerate(response.results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   Snippet: {result.snippet[:100]}...")
            print()
        
        # Test quick search
        print("Testing quick search...")
        snippets = await search_service.quick_search("Python 3.13 新機能", language="ja")
        print(f"Quick search returned {len(snippets)} snippets")
        
        # Usage statistics
        stats = search_service.get_usage_statistics()
        print(f"Usage statistics: {stats}")
        
        # Health check
        health = await search_service.health_check()
        print(f"Health status: {health}")
    
    # Run test
    asyncio.run(test_ollama_web_search())