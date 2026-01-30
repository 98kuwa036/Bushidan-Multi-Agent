#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
家訓ジェネレーター v8.1
=====================================

月次反省会から自動的に家訓を生成するシステム

機能:
- 家老による家訓候補生成
- 将軍による家訓精査・優先順位付け
- 既存家訓との重複チェック
- RAG統合による家訓管理
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from providers.anthropic_api import AnthropicAPI
from core.rag_integration import RAGIntegration
from core.hanseikai_manager import HanseiKaiManager, HanseiData


@dataclass
class KatunCandidate:
    """家訓候補データクラス"""
    content: str
    category: str  # 'technical', 'operational', 'monthly'
    priority: int  # 1-5 (5が最高)
    confidence_score: float  # 0.0-1.0
    reasoning: str
    source_insights: List[str]


@dataclass
class MonthlyKatun:
    """月次家訓データクラス"""
    period: str  # YYYY-MM
    katun_list: List[KatunCandidate]
    generated_at: datetime
    expires_at: datetime
    status: str  # 'active', 'expired', 'archived'


class KatunGenerator:
    """家訓ジェネレーター - v8.1新機能"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.karo_api = AnthropicAPI(config.get('karo', {}))  # 家老
        self.shogun_api = AnthropicAPI(config.get('shogun', {}))  # 将軍
        self.rag_integration = RAGIntegration(config.get('knowledge_base', {}))
        self.hansei_manager = HanseiKaiManager(config)
        self.logger = logging.getLogger(__name__)
    
    async def generate_monthly_katun(
        self, 
        year: int, 
        month: int
    ) -> MonthlyKatun:
        """
        月次家訓を自動生成
        
        Args:
            year: 対象年
            month: 対象月
            
        Returns:
            MonthlyKatun: 生成された月次家訓
        """
        self.logger.info(f"月次家訓生成開始: {year}年{month}月")
        
        # 1. 月次データ収集
        monthly_report = await self.hansei_manager.generate_monthly_summary_report(year, month)
        
        # 2. 家老による家訓候補生成
        katun_candidates = await self._karo_generate_katun_candidates(monthly_report)
        
        # 3. 将軍による精査・優先順位付け
        refined_katun = await self._shogun_refine_katun(katun_candidates, monthly_report)
        
        # 4. 重複チェック
        unique_katun = await self._check_katun_duplicates(refined_katun)
        
        # 5. 月次家訓オブジェクト作成
        period_str = f"{year}-{month:02d}"
        expires_at = datetime(year, month, 1) + timedelta(days=90)  # 3ヶ月有効
        
        monthly_katun = MonthlyKatun(
            period=period_str,
            katun_list=unique_katun,
            generated_at=datetime.now(),
            expires_at=expires_at,
            status='active'
        )
        
        # 6. RAGに登録
        await self._register_katun_to_rag(monthly_katun)
        
        self.logger.info(f"月次家訓生成完了: {len(unique_katun)}個の家訓を生成")
        return monthly_katun
    
    async def _karo_generate_katun_candidates(
        self, 
        monthly_report: Dict[str, Any]
    ) -> List[KatunCandidate]:
        """家老による家訓候補生成"""
        self.logger.info("家老による家訓候補生成開始")
        
        # 家老への詳細プロンプト作成
        karo_prompt = self._create_karo_generation_prompt(monthly_report)
        
        try:
            # 家老APIに家訓生成を依頼
            response = await self.karo_api.generate_completion(
                prompt=karo_prompt,
                max_tokens=2000,
                temperature=0.8  # 創造性を重視
            )
            
            # レスポンス解析
            katun_candidates = self._parse_karo_response(response)
            
            self.logger.info(f"家老が{len(katun_candidates)}個の家訓候補を生成")
            return katun_candidates
            
        except Exception as e:
            self.logger.error(f"家老による家訓生成エラー: {e}")
            return []
    
    def _create_karo_generation_prompt(self, monthly_report: Dict[str, Any]) -> str:
        """家老用家訓生成プロンプト作成"""
        hansei_data = monthly_report.get('hansei_data', {})
        activity_analysis = monthly_report.get('activity_analysis', {})
        sandbox_stats = monthly_report.get('sandbox_statistics', {})
        
        prompt = f"""# 家老の任務: 月次家訓生成

## 背景
将軍システムv8.1において、{monthly_report.get('period', '')}の反省会データから来月以降の開発で守るべき「月次家訓」を生成せよ。

## データ分析結果

### 反省会データ
- 成功事例: {hansei_data.get('successes_count', 0)}件
- 失敗事例: {hansei_data.get('failures_count', 0)}件
- 改善提案: {hansei_data.get('improvements_count', 0)}件
- 来月方針: {hansei_data.get('policies_count', 0)}件

### 活動記録分析
- 総活動数: {activity_analysis.get('total_activities', 0)}件
- 頻出エラー: {len(activity_analysis.get('frequent_errors', []))}種類
- 共通判断: {len(activity_analysis.get('common_decisions', []))}種類

### 演習場統計
- 実行成功率: {sandbox_stats.get('success_rate', 0)}%
- 失敗実行数: {sandbox_stats.get('failed_executions', 0)}件

## 重要なインサイト
{chr(10).join(['- ' + insight for insight in monthly_report.get('summary', {}).get('key_insights', [])])}

## 家訓生成指示

以下の形式で3-5個の家訓を生成せよ:

```json
{{
  "katun_candidates": [
    {{
      "content": "具体的で実践可能な家訓文",
      "category": "technical|operational|monthly",
      "priority": 1-5,
      "confidence_score": 0.0-1.0,
      "reasoning": "この家訓が必要な理由",
      "source_insights": ["根拠となるデータ/インサイト"]
    }}
  ]
}}
```

## 家訓作成の原則
1. **具体性**: 抽象的ではなく、実行可能な指示にする
2. **期間限定**: 来月から3ヶ月間有効な時限的な家訓
3. **問題解決**: 今月発見された問題を予防する内容
4. **測定可能**: 効果が測定できる形にする
5. **実践的**: 日々の開発で参照できる内容

## カテゴリ定義
- **technical**: コーディング規約、設計原則、技術的決定
- **operational**: プロジェクト運営、チーム協業、デプロイ手順
- **monthly**: 今月の反省を受けた期間限定の注意事項

家老として、チームの成長と失敗の再発防止を最優先に家訓を生成せよ。"""

        return prompt
    
    def _parse_karo_response(self, response: str) -> List[KatunCandidate]:
        """家老のレスポンスを解析して家訓候補を抽出"""
        try:
            # JSONブロックを探索
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group(1))
            else:
                # JSONブロックがない場合、全体をJSONとして解析試行
                json_data = json.loads(response)
            
            candidates = []
            for candidate_data in json_data.get('katun_candidates', []):
                candidate = KatunCandidate(
                    content=candidate_data.get('content', ''),
                    category=candidate_data.get('category', 'monthly'),
                    priority=candidate_data.get('priority', 3),
                    confidence_score=candidate_data.get('confidence_score', 0.7),
                    reasoning=candidate_data.get('reasoning', ''),
                    source_insights=candidate_data.get('source_insights', [])
                )
                candidates.append(candidate)
            
            return candidates
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"家老レスポンス解析エラー: {e}")
            # フォールバック: テキストから簡単な家訓を抽出
            return self._extract_katun_from_text(response)
    
    def _extract_katun_from_text(self, response: str) -> List[KatunCandidate]:
        """テキストから家訓を抽出（フォールバック処理）"""
        lines = response.split('\n')
        katun_list = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 20 and any(marker in line for marker in ['家訓', '：', 'は', 'する', 'べき']):
                katun_list.append(KatunCandidate(
                    content=line,
                    category='monthly',
                    priority=3,
                    confidence_score=0.5,
                    reasoning='テキスト解析により抽出',
                    source_insights=[]
                ))
        
        return katun_list[:5]  # 最大5個
    
    async def _shogun_refine_katun(
        self, 
        candidates: List[KatunCandidate], 
        monthly_report: Dict[str, Any]
    ) -> List[KatunCandidate]:
        """将軍による家訓精査・優先順位付け"""
        if not candidates:
            return []
        
        self.logger.info("将軍による家訓精査開始")
        
        # 将軍への精査プロンプト作成
        shogun_prompt = self._create_shogun_refinement_prompt(candidates, monthly_report)
        
        try:
            response = await self.shogun_api.generate_completion(
                prompt=shogun_prompt,
                max_tokens=1500,
                temperature=0.2  # 精査は厳密に
            )
            
            refined_katun = self._parse_shogun_response(response, candidates)
            
            self.logger.info(f"将軍が{len(refined_katun)}個の家訓に精査")
            return refined_katun
            
        except Exception as e:
            self.logger.error(f"将軍による精査エラー: {e}")
            # エラー時は元の候補をそのまま返す
            return candidates
    
    def _create_shogun_refinement_prompt(
        self, 
        candidates: List[KatunCandidate], 
        monthly_report: Dict[str, Any]
    ) -> str:
        """将軍用精査プロンプト作成"""
        candidates_text = "\n".join([
            f"{i+1}. {candidate.content} (優先度: {candidate.priority}, カテゴリ: {candidate.category})"
            for i, candidate in enumerate(candidates)
        ])
        
        return f"""# 将軍の任務: 家訓精査・最終判断

## 背景
家老が提案した{len(candidates)}個の月次家訓候補を精査し、実際に適用する家訓を決定せよ。

## 家老提案の家訓候補
{candidates_text}

## 精査基準
1. **実現可能性**: 現在の技術・体制で実行可能か
2. **効果の期待度**: 問題解決・品質向上に寄与するか
3. **重複回避**: 既存の家訓と重複していないか
4. **優先順位**: 今最も重要な家訓はどれか

## 指示
以下の形式で精査結果を出力せよ:

```json
{{
  "refined_katun": [
    {{
      "index": 元の候補のインデックス(0-{len(candidates)-1}),
      "approved": true|false,
      "revised_content": "修正された家訓内容（修正がある場合）",
      "revised_priority": 1-5,
      "shogun_reasoning": "将軍としての判断理由"
    }}
  ]
}}
```

将軍として、システム全体の品質向上と効率化を最優先に判断せよ。
不要・重複・非実用的な家訓は容赦なく却下し、
真に価値ある家訓のみを承認すること。"""

    def _parse_shogun_response(
        self, 
        response: str, 
        original_candidates: List[KatunCandidate]
    ) -> List[KatunCandidate]:
        """将軍のレスポンスを解析"""
        try:
            import re
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group(1))
            else:
                json_data = json.loads(response)
            
            refined_katun = []
            for refinement in json_data.get('refined_katun', []):
                if not refinement.get('approved', False):
                    continue
                
                index = refinement.get('index', 0)
                if 0 <= index < len(original_candidates):
                    original = original_candidates[index]
                    
                    # 修正があれば適用
                    content = refinement.get('revised_content', original.content)
                    priority = refinement.get('revised_priority', original.priority)
                    
                    refined_candidate = KatunCandidate(
                        content=content,
                        category=original.category,
                        priority=priority,
                        confidence_score=original.confidence_score,
                        reasoning=f"{original.reasoning} / 将軍判断: {refinement.get('shogun_reasoning', '')}",
                        source_insights=original.source_insights
                    )
                    refined_katun.append(refined_candidate)
            
            return refined_katun
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"将軍レスポンス解析エラー: {e}")
            # エラー時は優先度の高い候補を返す
            return sorted(original_candidates, key=lambda x: x.priority, reverse=True)[:3]
    
    async def _check_katun_duplicates(
        self, 
        candidates: List[KatunCandidate]
    ) -> List[KatunCandidate]:
        """既存家訓との重複チェック"""
        if not candidates:
            return []
        
        self.logger.info("家訓重複チェック開始")
        
        try:
            # RAGから既存の家訓を検索
            existing_katun_queries = [candidate.content for candidate in candidates]
            
            unique_candidates = []
            for candidate in candidates:
                # 類似家訓検索
                similar_results = await self.rag_integration.search_similar(
                    query=candidate.content,
                    category="katun",
                    score_threshold=0.8,  # 高い類似度で重複判定
                    max_results=3
                )
                
                if not similar_results:
                    # 類似家訓がない場合は採用
                    unique_candidates.append(candidate)
                else:
                    # 類似家訓がある場合はログ出力して除外
                    self.logger.info(f"重複家訓を除外: {candidate.content[:50]}...")
            
            self.logger.info(f"重複チェック完了: {len(unique_candidates)}個が残存")
            return unique_candidates
            
        except Exception as e:
            self.logger.error(f"重複チェックエラー: {e}")
            # エラー時は全候補を返す
            return candidates
    
    async def _register_katun_to_rag(self, monthly_katun: MonthlyKatun) -> None:
        """月次家訓をRAGシステムに登録"""
        self.logger.info(f"RAG登録開始: {monthly_katun.period}")
        
        try:
            for i, katun in enumerate(monthly_katun.katun_list):
                document = {
                    'content': katun.content,
                    'metadata': {
                        'type': 'monthly_katun',
                        'period': monthly_katun.period,
                        'category': katun.category,
                        'priority': katun.priority,
                        'confidence_score': katun.confidence_score,
                        'generated_at': monthly_katun.generated_at.isoformat(),
                        'expires_at': monthly_katun.expires_at.isoformat(),
                        'index': i
                    }
                }
                
                await self.rag_integration.add_document(document)
            
            self.logger.info(f"RAG登録完了: {len(monthly_katun.katun_list)}個の家訓")
            
        except Exception as e:
            self.logger.error(f"RAG登録エラー: {e}")
    
    async def get_active_monthly_katun(self) -> List[KatunCandidate]:
        """現在有効な月次家訓を取得"""
        try:
            current_time = datetime.now()
            
            # RAGから有効な月次家訓を検索
            results = await self.rag_integration.search_by_metadata(
                metadata_filter={
                    'type': 'monthly_katun',
                    'expires_at': f'>{current_time.isoformat()}'
                }
            )
            
            active_katun = []
            for result in results:
                metadata = result.get('metadata', {})
                katun = KatunCandidate(
                    content=result.get('content', ''),
                    category=metadata.get('category', 'monthly'),
                    priority=metadata.get('priority', 3),
                    confidence_score=metadata.get('confidence_score', 0.7),
                    reasoning=f"期間: {metadata.get('period', '')}",
                    source_insights=[]
                )
                active_katun.append(katun)
            
            # 優先度順にソート
            return sorted(active_katun, key=lambda x: x.priority, reverse=True)
            
        except Exception as e:
            self.logger.error(f"有効家訓取得エラー: {e}")
            return []
    
    async def cleanup_expired_katun(self) -> int:
        """期限切れ家訓のクリーンアップ"""
        try:
            current_time = datetime.now()
            
            # 期限切れ家訓を検索
            expired_results = await self.rag_integration.search_by_metadata(
                metadata_filter={
                    'type': 'monthly_katun',
                    'expires_at': f'<{current_time.isoformat()}'
                }
            )
            
            # RAGから削除
            for result in expired_results:
                document_id = result.get('id')
                if document_id:
                    await self.rag_integration.delete_document(document_id)
            
            self.logger.info(f"期限切れ家訓クリーンアップ: {len(expired_results)}個削除")
            return len(expired_results)
            
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {e}")
            return 0