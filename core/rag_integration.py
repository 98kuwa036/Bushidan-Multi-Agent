"""RAG Integration for Shogun System

RAG (Retrieval-Augmented Generation) system for family precepts (家訓).

Features:
  - Vector embeddings for family precepts
  - Semantic search capabilities
  - Dual storage with Notion integration
  - Real-time knowledge retrieval during agent interactions

This works in tandem with NotionIntegration for comprehensive knowledge management.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import faiss
    import sentence_transformers
except ImportError:
    faiss = None
    sentence_transformers = None

logger = logging.getLogger("shogun.rag")


class RAGIntegration:
    """RAG system for family precepts and knowledge management."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.precepts_db = []
        self.embedding_dim = 384  # MiniLM dimension
        
        # Storage paths
        self.storage_dir = Path("/tmp/shogun_rag")
        self.storage_dir.mkdir(exist_ok=True)
        self.index_file = self.storage_dir / "precepts_index.faiss"
        self.db_file = self.storage_dir / "precepts_db.json"
        
        # Statistics
        self.stats = {
            "precepts_indexed": 0,
            "searches_performed": 0,
            "embeddings_generated": 0,
            "notion_syncs": 0,
        }
        
    async def initialize(self) -> None:
        """Initialize RAG system with sentence transformer."""
        if sentence_transformers is None:
            logger.warning("[RAG] sentence-transformers未インストール - pip install sentence-transformers")
            return
            
        if faiss is None:
            logger.warning("[RAG] faiss未インストール - pip install faiss-cpu")
            return
        
        try:
            # Load sentence transformer model
            self.encoder = sentence_transformers.SentenceTransformer(self.model_name)
            
            # Initialize FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            
            # Load existing data
            await self._load_existing_data()
            
            logger.info("[RAG] 初期化完了 - %d家訓インデックス済", len(self.precepts_db))
            
        except Exception as e:
            logger.error("[RAG] 初期化失敗: %s", e)
    
    async def _load_existing_data(self) -> None:
        """Load existing precepts and index."""
        try:
            # Load precepts database
            if self.db_file.exists():
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    self.precepts_db = json.load(f)
                
            # Load FAISS index
            if self.index_file.exists() and len(self.precepts_db) > 0:
                self.index = faiss.read_index(str(self.index_file))
                logger.info("[RAG] 既存インデックス読み込み: %d件", len(self.precepts_db))
            else:
                # Rebuild index if needed
                if self.precepts_db:
                    await self._rebuild_index()
                    
        except Exception as e:
            logger.warning("[RAG] データ読み込みエラー: %s", e)
    
    async def add_family_precepts(
        self, 
        precepts: List[str], 
        context: str = "",
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add family precepts to RAG system."""
        if not self.encoder or not precepts:
            return False
            
        try:
            new_entries = []
            embeddings = []
            
            for precept in precepts:
                if not precept.strip():
                    continue
                    
                # Create entry
                entry = {
                    "id": len(self.precepts_db) + len(new_entries),
                    "text": precept.strip(),
                    "context": context,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {},
                }
                
                # Generate embedding
                embedding = self.encoder.encode(precept, convert_to_tensor=False)
                
                new_entries.append(entry)
                embeddings.append(embedding)
                self.stats["embeddings_generated"] += 1
            
            if new_entries:
                # Add to database
                self.precepts_db.extend(new_entries)
                
                # Add to FAISS index
                embeddings_array = np.array(embeddings).astype('float32')
                # Normalize for cosine similarity
                faiss.normalize_L2(embeddings_array)
                self.index.add(embeddings_array)
                
                # Save to disk
                await self._save_data()
                
                self.stats["precepts_indexed"] += len(new_entries)
                logger.info("[RAG] 家訓追加: %d件", len(new_entries))
                
            return True
            
        except Exception as e:
            logger.error("[RAG] 家訓追加エラー: %s", e)
            return False
    
    async def search_precepts(
        self, 
        query: str, 
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for relevant family precepts using semantic similarity."""
        if not self.encoder or not query.strip() or not self.precepts_db:
            return []
            
        try:
            # Encode query
            query_embedding = self.encoder.encode(query, convert_to_tensor=False)
            query_array = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_array)
            
            # Search FAISS index
            scores, indices = self.index.search(query_array, min(top_k, len(self.precepts_db)))
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if score >= threshold and idx < len(self.precepts_db):
                    precept = self.precepts_db[idx].copy()
                    precept["similarity_score"] = float(score)
                    results.append(precept)
            
            self.stats["searches_performed"] += 1
            
            logger.debug("[RAG] 検索結果: %d件 (クエリ: '%s')", len(results), query[:50])
            return results
            
        except Exception as e:
            logger.error("[RAG] 検索エラー: %s", e)
            return []
    
    async def get_relevant_context(
        self, 
        user_prompt: str, 
        top_k: int = 3
    ) -> Tuple[List[str], str]:
        """Get relevant family precepts as context for agent prompts."""
        results = await self.search_precepts(user_prompt, top_k=top_k)
        
        if not results:
            return [], ""
        
        precepts = [result["text"] for result in results]
        
        # Format context
        context_lines = [
            "## 関連する家訓（過去の学び）:",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            score = result["similarity_score"]
            context_lines.extend([
                f"{i}. **{result['text']}**",
                f"   関連度: {score:.2f}",
                f"   記録日: {result['timestamp'][:10]}",
                ""
            ])
        
        context_text = "\n".join(context_lines)
        
        logger.info("[RAG] コンテキスト提供: %d家訓", len(precepts))
        return precepts, context_text
    
    async def sync_with_notion(self, notion_integration) -> bool:
        """Sync family precepts with Notion."""
        if not notion_integration:
            return False
            
        try:
            # Get recent precepts from Notion
            notion_precepts = await notion_integration.get_family_precepts(limit=100)
            
            # Check for new precepts not in RAG
            existing_texts = {entry["text"] for entry in self.precepts_db}
            new_precepts = [p for p in notion_precepts if p not in existing_texts]
            
            if new_precepts:
                await self.add_family_precepts(
                    new_precepts, 
                    context="Notion sync",
                    metadata={"source": "notion"}
                )
                
            self.stats["notion_syncs"] += 1
            logger.info("[RAG] Notion同期完了: %d新規家訓", len(new_precepts))
            
            return True
            
        except Exception as e:
            logger.error("[RAG] Notion同期エラー: %s", e)
            return False
    
    async def _rebuild_index(self) -> None:
        """Rebuild FAISS index from existing precepts."""
        if not self.encoder or not self.precepts_db:
            return
            
        try:
            embeddings = []
            for entry in self.precepts_db:
                embedding = self.encoder.encode(entry["text"], convert_to_tensor=False)
                embeddings.append(embedding)
                
            if embeddings:
                embeddings_array = np.array(embeddings).astype('float32')
                faiss.normalize_L2(embeddings_array)
                
                # Recreate index
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.index.add(embeddings_array)
                
                await self._save_data()
                logger.info("[RAG] インデックス再構築完了: %d件", len(embeddings))
                
        except Exception as e:
            logger.error("[RAG] インデックス再構築エラー: %s", e)
    
    async def _save_data(self) -> None:
        """Save precepts database and FAISS index."""
        try:
            # Save precepts database
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(self.precepts_db, f, ensure_ascii=False, indent=2)
            
            # Save FAISS index
            if self.index and self.index.ntotal > 0:
                faiss.write_index(self.index, str(self.index_file))
                
            logger.debug("[RAG] データ保存完了")
            
        except Exception as e:
            logger.error("[RAG] データ保存エラー: %s", e)
    
    async def export_precepts(self, format: str = "json") -> Optional[str]:
        """Export family precepts in various formats."""
        if not self.precepts_db:
            return None
            
        try:
            if format.lower() == "json":
                return json.dumps(self.precepts_db, ensure_ascii=False, indent=2)
                
            elif format.lower() == "markdown":
                lines = [
                    "# 家訓集",
                    f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"総数: {len(self.precepts_db)}件",
                    "",
                ]
                
                for i, entry in enumerate(self.precepts_db, 1):
                    lines.extend([
                        f"## {i}. {entry['text']}",
                        f"**記録日:** {entry['timestamp'][:10]}",
                        f"**コンテキスト:** {entry['context'] or 'なし'}",
                        "",
                    ])
                
                return "\n".join(lines)
                
            elif format.lower() == "txt":
                lines = [f"{i}. {entry['text']}" for i, entry in enumerate(self.precepts_db, 1)]
                return "\n".join(lines)
                
        except Exception as e:
            logger.error("[RAG] エクスポートエラー: %s", e)
            
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return {
            "initialized": self.encoder is not None,
            "model_name": self.model_name,
            "precepts_count": len(self.precepts_db),
            "index_size": self.index.ntotal if self.index else 0,
            "stats": dict(self.stats),
        }
    
    def show_stats(self) -> str:
        """Format stats for display."""
        s = self.stats
        
        lines = [
            "=" * 50,
            "🧠 RAG統合 統計",
            "=" * 50,
            f"家訓インデックス数: {len(self.precepts_db)}件",
            f"検索実行回数: {s['searches_performed']}回",
            f"埋め込み生成: {s['embeddings_generated']}回",
            f"Notion同期: {s['notion_syncs']}回",
            "",
            f"モデル: {self.model_name}",
            f"インデックス状態: {'OK' if self.index else 'NG'}",
            "=" * 50,
        ]
        return "\n".join(lines)


# Utility function for agent integration
async def get_precept_context(rag_system: RAGIntegration, prompt: str) -> str:
    """Get family precept context for agent prompts."""
    if not rag_system:
        return ""
        
    precepts, context = await rag_system.get_relevant_context(prompt)
    return context if precepts else ""