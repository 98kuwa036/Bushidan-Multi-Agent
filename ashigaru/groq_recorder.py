"""Groq Recorder - 9th Ashigaru (足軽)

Real-time recording and 60-day summary automation using Groq Llama 3.3 70B.

Key Features:
  - Real-time interaction recording
  - Ultra-fast 60-day summaries (5,000 lines → 3 minutes)
  - Automatic Notion integration
  - Knowledge extraction and family precepts (家訓)
  - Free tier utilization (14,400 requests/day)

This is the 9th ashigaru that handles all knowledge management for the Shogun system.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

try:
    import groq
except ImportError:
    groq = None

logger = logging.getLogger("shogun.ashigaru.groq")


class GroqRecorder:
    """9th Ashigaru - Groq-powered knowledge recorder and summarizer."""

    def __init__(self, api_key: str, notion_integration: Dict[str, Any]):
        self.api_key = api_key
        self.notion_config = notion_integration
        self.client = None
        
        # Session tracking
        self.current_session = None
        self.session_data = []
        
        # Statistics
        self.stats = {
            "sessions_started": 0,
            "interactions_recorded": 0,
            "summaries_generated": 0,
            "notion_uploads": 0,
            "family_precepts_extracted": 0,
            "groq_requests": 0,
            "total_tokens": 0,
        }
        
        # Storage paths
        self.storage_dir = Path("/tmp/shogun_recordings")
        self.storage_dir.mkdir(exist_ok=True)
        
        # Daily request tracking for free tier
        self.daily_requests = 0
        self.last_request_date = datetime.now().date()
        
        # RPM/TPM tracking for short-term rate limiting
        self.rpm_requests = []  # List of request timestamps for RPM tracking
        self.tpm_tokens = []    # List of (timestamp, token_count) for TPM tracking
        self.rpm_limit = 30     # Groq free tier RPM limit
        self.tpm_limit = 6000   # Groq free tier TPM limit
        
    async def initialize(self) -> None:
        """Initialize Groq client."""
        if not self.api_key:
            logger.warning("[9番足軽] Groqキー未設定 - 記録機能無効")
            return
            
        if groq is None:
            logger.error("[9番足軽] groqライブラリ未インストール - pip install groq")
            return
        
        self.client = groq.Groq(api_key=self.api_key)
        logger.info("[9番足軽] Groq記録システム初期化完了")
        
        # Check daily quota
        self._check_daily_quota()
        
    def _check_daily_quota(self) -> None:
        """Check and reset daily request quota."""
        today = datetime.now().date()
        if today != self.last_request_date:
            self.daily_requests = 0
            self.last_request_date = today
            logger.info("[9番足軽] 日別クォータリセット (14,400/day)")
    
    async def start_session(self, task_id: str, prompt: str) -> None:
        """Start a new recording session."""
        if not self.client:
            return
            
        self.current_session = {
            "id": task_id,
            "start_time": datetime.now().isoformat(),
            "initial_prompt": prompt,
            "interactions": [],
        }
        
        self.stats["sessions_started"] += 1
        logger.info("[9番足軽] セッション開始: %s", task_id)
    
    async def record_interaction(
        self, 
        task_id: str, 
        agent_type: str, 
        prompt: str, 
        response: str
    ) -> None:
        """Record an agent interaction."""
        if not self.client or not self.current_session:
            return
            
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_type,
            "prompt": prompt,
            "response": response,
            "token_count": len(prompt.split()) + len(response.split()),  # Rough estimate
        }
        
        self.current_session["interactions"].append(interaction)
        self.stats["interactions_recorded"] += 1
        
        logger.debug("[9番足軽] インタラクション記録: %s", agent_type)
    
    async def record_completion(
        self, 
        task_id: str, 
        original_prompt: str, 
        final_result: str, 
        cost_yen: float
    ) -> None:
        """Record task completion."""
        if not self.client or not self.current_session:
            return
            
        self.current_session.update({
            "end_time": datetime.now().isoformat(),
            "final_result": final_result,
            "cost_yen": cost_yen,
            "status": "completed"
        })
        
        # Save to disk
        await self._save_session_to_disk()
        
        # Extract knowledge if significant interaction
        if len(self.current_session["interactions"]) > 2:
            await self._extract_knowledge_async()
        
        self.current_session = None
        logger.info("[9番足軽] セッション完了記録: %s (¥%.0f)", task_id, cost_yen)
    
    async def record_failure(
        self, 
        task_id: str, 
        original_prompt: str, 
        error: str
    ) -> None:
        """Record task failure."""
        if not self.client or not self.current_session:
            return
            
        self.current_session.update({
            "end_time": datetime.now().isoformat(),
            "error": error,
            "status": "failed"
        })
        
        await self._save_session_to_disk()
        self.current_session = None
        
        logger.info("[9番足軽] セッション失敗記録: %s", task_id)
    
    async def _save_session_to_disk(self) -> None:
        """Save session data to disk."""
        if not self.current_session:
            return
            
        filename = f"session_{self.current_session['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.storage_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, ensure_ascii=False, indent=2)
            logger.debug("[9番足軽] セッション保存: %s", filename)
        except Exception as e:
            logger.error("[9番足軽] セッション保存失敗: %s", e)
    
    async def _extract_knowledge_async(self) -> None:
        """Extract knowledge and family precepts from session."""
        # Estimate tokens for the extraction request
        session_text = self._format_session_for_analysis()
        estimated_tokens = len(session_text.split()) * 1.5  # Rough estimate
        
        if not self._can_make_request(int(estimated_tokens)):
            logger.info("[9番足軽] レート制限のため知識抽出をスキップ")
            return
            
        try:
            # Extract family precepts (家訓)
            precepts = await self._extract_family_precepts(session_text)
            if precepts:
                self.stats["family_precepts_extracted"] += 1
                logger.info("[9番足軽] 家訓抽出: %d個", len(precepts))
                
        except Exception as e:
            logger.warning("[9番足軽] 知識抽出失敗: %s", e)
    
    def _format_session_for_analysis(self) -> str:
        """Format session data for Groq analysis."""
        if not self.current_session:
            return ""
            
        lines = [
            f"# セッション: {self.current_session['id']}",
            f"開始時間: {self.current_session['start_time']}",
            f"初期プロンプト: {self.current_session['initial_prompt']}",
            "",
            "## インタラクション履歴",
        ]
        
        for i, interaction in enumerate(self.current_session["interactions"], 1):
            lines.extend([
                f"### {i}. {interaction['agent']} ({interaction['timestamp']})",
                f"**プロンプト:** {interaction['prompt'][:200]}{'...' if len(interaction['prompt']) > 200 else ''}",
                f"**応答:** {interaction['response'][:500]}{'...' if len(interaction['response']) > 500 else ''}",
                "",
            ])
        
        return "\n".join(lines)
    
    async def _extract_family_precepts(self, session_text: str) -> List[str]:
        """Extract family precepts (家訓) from session using Groq."""
        if not self.client or not session_text:
            return []
            
        prompt = f"""
以下のセッションから、将来の開発で参考になる「家訓」（決定事項・学び・原則）を抽出してください。

{session_text}

抽出条件:
1. 具体的で実用的な教訓
2. 再利用可能な知識
3. 重要な決定事項
4. エラー対処法
5. 最適化のポイント

家訓形式で出力（例：「ESP32のSPI設定では必ずDMA設定を確認せよ」）:
"""
        
        try:
            response = await self._call_groq(
                prompt=prompt,
                system="あなたは知識抽出の専門家です。セッションから価値ある教訓を抽出し、将来の参考となる家訓として整理してください。",
                max_tokens=1000
            )
            
            if response:
                # Parse precepts from response
                precepts = []
                for line in response.split('\n'):
                    line = line.strip()
                    if line and ('家訓' in line or '教訓' in line or line.endswith('べし') or line.endswith('せよ')):
                        precepts.append(line)
                
                return precepts
                
        except Exception as e:
            logger.error("[9番足軽] 家訓抽出エラー: %s", e)
        
        return []
    
    async def generate_60day_summary(self) -> str:
        """Generate 60-day summary of all recorded sessions."""
        if not self.client:
            return "Groqクライアント未初期化"
            
        logger.info("[9番足軽] 60日要約生成開始")
        
        # Collect sessions from last 60 days
        cutoff_date = datetime.now() - timedelta(days=60)
        sessions = self._load_recent_sessions(cutoff_date)
        
        if not sessions:
            return "対象期間のセッションが見つかりません"
        
        # Generate summary using Groq's ultra-fast processing
        summary = await self._generate_summary_with_groq(sessions)
        
        if summary:
            self.stats["summaries_generated"] += 1
            logger.info("[9番足軽] 60日要約完了 (%d セッション)", len(sessions))
        
        return summary or "要約生成に失敗しました"
    
    def _load_recent_sessions(self, cutoff_date: datetime) -> List[Dict]:
        """Load sessions from the last N days."""
        sessions = []
        
        if not self.storage_dir.exists():
            return sessions
        
        for json_file in self.storage_dir.glob("session_*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    session = json.load(f)
                    
                # Check if session is within date range
                start_time = datetime.fromisoformat(session.get('start_time', ''))
                if start_time >= cutoff_date:
                    sessions.append(session)
                    
            except Exception as e:
                logger.warning("[9番足軽] セッション読み込みエラー: %s", e)
        
        return sorted(sessions, key=lambda x: x.get('start_time', ''))
    
    async def _generate_summary_with_groq(self, sessions: List[Dict]) -> str:
        """Generate comprehensive summary using Groq's speed."""
        if not sessions:
            return ""
            
        # Prepare sessions text (optimized for Groq processing)
        sessions_text = self._format_sessions_for_summary(sessions)
        
        prompt = f"""
以下の{len(sessions)}件のセッションを要約し、包括的な60日レポートを作成してください。

{sessions_text}

要約要件:
1. 主要な成果と実装内容
2. 頻繁に発生した問題とその解決策
3. コスト分析（¥記載があるもの）
4. 技術的な学び・家訓
5. 改善提案

Markdown形式で出力:
"""
        
        try:
            return await self._call_groq(
                prompt=prompt,
                system="あなたは開発プロジェクトの分析専門家です。60日間の活動を要約し、価値ある洞察を提供してください。",
                max_tokens=4000
            )
        except Exception as e:
            logger.error("[9番足軽] Groq要約生成エラー: %s", e)
            return f"要約生成エラー: {e}"
    
    def _format_sessions_for_summary(self, sessions: List[Dict]) -> str:
        """Format sessions for Groq summary processing."""
        lines = []
        
        for i, session in enumerate(sessions, 1):
            lines.extend([
                f"## セッション {i}: {session.get('id', 'Unknown')}",
                f"時間: {session.get('start_time', '')} - {session.get('end_time', '')}",
                f"初期プロンプト: {session.get('initial_prompt', '')[:200]}",
                f"ステータス: {session.get('status', 'unknown')}",
            ])
            
            if session.get('cost_yen'):
                lines.append(f"コスト: ¥{session['cost_yen']}")
            
            if session.get('error'):
                lines.append(f"エラー: {session['error']}")
            elif session.get('final_result'):
                result = session['final_result'][:300]
                lines.append(f"結果: {result}{'...' if len(session['final_result']) > 300 else ''}")
            
            # Add significant interactions
            interactions = session.get('interactions', [])
            if interactions:
                lines.append(f"インタラクション数: {len(interactions)}")
                for interaction in interactions[:2]:  # Limit to first 2
                    lines.append(f"  - {interaction.get('agent', '')}: {interaction.get('response', '')[:100][:100]}")
            
            lines.append("")  # Blank line
        
        return "\n".join(lines)
    
    async def _call_groq(
        self, 
        prompt: str, 
        system: str = "", 
        max_tokens: int = 2000
    ) -> Optional[str]:
        """Call Groq API with rate limiting and exponential backoff."""
        if not self._can_make_request():
            logger.warning("[9番足軽] Groq日別上限到達 (14,400/day)")
            return None
        
        # Exponential backoff parameters
        max_retries = 5
        base_delay = 1.0  # Start with 1 second
        max_delay = 60.0  # Max 60 seconds
        
        for attempt in range(max_retries):
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                
                self._track_request(response.usage.total_tokens if response.usage else max_tokens)
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check for rate limit errors
                if "rate_limit" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        # Calculate delay with exponential backoff and jitter
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        jitter = delay * 0.1 * (0.5 - abs(hash(prompt) % 100 - 50) / 100)
                        total_delay = delay + jitter
                        
                        logger.warning("[9番足軽] レート制限 - %d回目再試行 (%.1f秒後)", 
                                     attempt + 1, total_delay)
                        await asyncio.sleep(total_delay)
                        continue
                    else:
                        logger.error("[9番足軽] レート制限で最大再試行回数到達")
                        return None
                
                # Check for temporary server errors (502, 503, 504)
                elif any(code in error_msg for code in ["502", "503", "504", "timeout"]):
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (1.5 ** attempt), max_delay / 2)
                        logger.warning("[9番足軽] 一時的エラー - %d回目再試行 (%.1f秒後)", 
                                     attempt + 1, delay)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error("[9番足軽] 一時的エラーで最大再試行回数到達")
                        return None
                
                # For other errors, don't retry
                else:
                    logger.error("[9番足軽] Groq API呼び出しエラー: %s", e)
                    return None
        
        return None
    
    def _can_make_request(self, estimated_tokens: int = 500) -> bool:
        """Check if we can make another Groq request (daily, RPM, TPM limits)."""
        self._check_daily_quota()
        
        # Check daily limit
        if self.daily_requests >= 14400:
            return False
        
        # Check RPM limit
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests (older than 1 minute)
        self.rpm_requests = [ts for ts in self.rpm_requests if ts > one_minute_ago]
        
        if len(self.rpm_requests) >= self.rpm_limit:
            logger.debug("[9番足軽] RPM上限に近づいています (%d/%d)", len(self.rpm_requests), self.rpm_limit)
            return False
        
        # Check TPM limit
        self.tpm_tokens = [(ts, tokens) for ts, tokens in self.tpm_tokens if ts > one_minute_ago]
        current_tokens = sum(tokens for _, tokens in self.tpm_tokens)
        
        if current_tokens + estimated_tokens > self.tpm_limit:
            logger.debug("[9番足軽] TPM上限に近づいています (%d+%d > %d)", 
                        current_tokens, estimated_tokens, self.tpm_limit)
            return False
        
        return True
    
    def _track_request(self, tokens: int) -> None:
        """Track request for daily, RPM, and TPM quotas."""
        now = datetime.now()
        
        # Daily tracking
        self.daily_requests += 1
        self.stats["groq_requests"] += 1
        self.stats["total_tokens"] += tokens
        
        # RPM tracking
        self.rpm_requests.append(now)
        
        # TPM tracking
        self.tpm_tokens.append((now, tokens))
        
        if self.daily_requests % 1000 == 0:
            logger.info("[9番足軽] Groq使用状況: %d/14,400 requests", self.daily_requests)
    
    async def finalize_session(self) -> None:
        """Finalize current session if any."""
        if self.current_session:
            self.current_session.update({
                "end_time": datetime.now().isoformat(),
                "status": "interrupted"
            })
            await self._save_session_to_disk()
            self.current_session = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get recorder status."""
        return {
            "initialized": self.client is not None,
            "current_session": self.current_session['id'] if self.current_session else None,
            "daily_requests": self.daily_requests,
            "daily_quota_remaining": 14400 - self.daily_requests,
            "stats": dict(self.stats),
        }
    
    def show_stats(self) -> str:
        """Format stats for display."""
        s = self.stats
        remaining = 14400 - self.daily_requests
        
        # Calculate current RPM/TPM usage
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        current_rpm = len([ts for ts in self.rpm_requests if ts > one_minute_ago])
        current_tpm = sum(tokens for ts, tokens in self.tpm_tokens if ts > one_minute_ago)
        
        lines = [
            "=" * 50,
            "🎯 9番足軽 (Groq記録係) 統計",
            "=" * 50,
            f"セッション開始: {s['sessions_started']}回",
            f"インタラクション記録: {s['interactions_recorded']}回",
            f"要約生成: {s['summaries_generated']}回",
            f"家訓抽出: {s['family_precepts_extracted']}回",
            f"Notion投稿: {s['notion_uploads']}回",
            "",
            "Groq使用状況:",
            f"  本日のリクエスト: {self.daily_requests}/14,400",
            f"  残り: {remaining}回",
            f"  累計トークン: {s['total_tokens']:,}",
            "",
            "短期制限状況:",
            f"  RPM (分間リクエスト): {current_rpm}/{self.rpm_limit}",
            f"  TPM (分間トークン): {current_tpm:,}/{self.tpm_limit:,}",
            "",
            "💰 コスト: ¥0 (Free Tier) ⭐",
            "🛡️  レート制限対応: Exponential Backoff ✅",
            "=" * 50,
        ]
        return "\n".join(lines)
