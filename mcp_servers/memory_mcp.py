"""
Bushidan Multi-Agent System v9.1 - Memory MCP

JSONL-based memory system for the 3-layer memory architecture.
Layer 2: Medium-term memory with automatic expiration and search.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from utils.logger import get_logger


logger = get_logger(__name__)


class MemoryMCP:
    """
    Memory MCP - Medium-term memory system
    
    Features:
    - JSONL storage format (human-readable)
    - Git-compatible (version control)
    - Fast search with grep/jq
    - Automatic expiration (7 days for web search cache)
    - No external dependencies
    """
    
    def __init__(self, memory_file: str = "shogun_memory.jsonl"):
        self.memory_file = Path(memory_file)
        self.initialized = False
        
    async def initialize(self) -> None:
        """Initialize memory system"""
        logger.info("📝 Initializing Memory MCP...")
        
        # Ensure memory file exists
        if not self.memory_file.exists():
            self.memory_file.touch()
            logger.info(f"✅ Created memory file: {self.memory_file}")
        
        # Clean expired entries on startup
        await self._clean_expired_entries()
        
        self.initialized = True
        logger.info("✅ Memory MCP initialized")
    
    async def store(self, entry: Dict[str, Any]) -> None:
        """Store entry in memory"""
        
        if not self.initialized:
            await self.initialize()
        
        # Add timestamp if not present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
        
        # Write to JSONL file
        try:
            with open(self.memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            logger.info(f"📝 Stored memory entry: {entry.get('category', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store memory entry: {e}")
            raise
    
    async def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory entries"""
        
        if not self.initialized:
            await self.initialize()
        
        results = []
        
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                        
                        # Check if entry is expired
                        if self._is_expired(entry):
                            continue
                        
                        # Simple text search in content
                        content = json.dumps(entry, ensure_ascii=False).lower()
                        if query.lower() in content:
                            results.append(entry)
                            
                            if len(results) >= limit:
                                break
                                
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"🔍 Found {len(results)} memory entries for query: {query[:30]}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Memory search failed: {e}")
            return []
    
    async def store_web_search(self, query: str, results: str, source: str) -> None:
        """Store web search results with expiration"""
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "category": "web_search",
            "query": query,
            "content": results[:2000],  # Limit content size
            "source": source,
            "expires": (datetime.now() + timedelta(days=7)).isoformat()
        }
        
        await self.store(entry)
    
    async def get_web_search_cache(self, query: str) -> Optional[str]:
        """Get cached web search results"""
        
        results = await self.search(query)
        
        for entry in results:
            if (entry.get("category") == "web_search" and 
                entry.get("query", "").lower() == query.lower()):
                return entry.get("content")
        
        return None
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if entry is expired"""
        
        expires = entry.get("expires")
        if not expires:
            return False
        
        try:
            expiry_date = datetime.fromisoformat(expires.replace('Z', '+00:00'))
            return datetime.now() > expiry_date
        except (ValueError, TypeError):
            return False
    
    async def _clean_expired_entries(self) -> None:
        """Clean expired entries from memory file"""
        
        if not self.memory_file.exists():
            return
        
        valid_entries = []
        
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                        if not self._is_expired(entry):
                            valid_entries.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            # Rewrite file with valid entries only
            with open(self.memory_file, "w", encoding="utf-8") as f:
                for entry in valid_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            logger.info(f"🧹 Memory cleanup complete - {len(valid_entries)} entries retained")
            
        except Exception as e:
            logger.warning(f"⚠️ Memory cleanup failed: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        
        if not self.memory_file.exists():
            return {"total_entries": 0, "file_size": 0}
        
        total_entries = 0
        categories = {}
        
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        total_entries += 1
                        try:
                            entry = json.loads(line)
                            category = entry.get("category", "unknown")
                            categories[category] = categories.get(category, 0) + 1
                        except json.JSONDecodeError:
                            continue
            
            file_size = self.memory_file.stat().st_size
            
            return {
                "total_entries": total_entries,
                "categories": categories,
                "file_size": file_size,
                "file_path": str(self.memory_file)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    async def shutdown(self) -> None:
        """Shutdown memory MCP"""
        logger.info("📴 Memory MCP shutdown complete")