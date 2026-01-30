#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反省会マネージャー v8.1
=====================================

月次反省会から家訓を自動生成するシステム

機能:
- 反省会チャンネル検出・データ収集
- 陣中日記の統計分析
- 演習場エラーログ集計
- 家訓生成のためのデータ準備
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

from integrations.slack_bot import SlackBot
from core.activity_memory import ActivityMemory


@dataclass
class HanseiData:
    """反省会データクラス"""
    month: str
    successes: List[str]
    failures: List[str]
    improvements: List[str]
    next_month_policies: List[str]
    raw_messages: List[Dict[str, Any]]


@dataclass
class ErrorStatistics:
    """エラー統計データクラス"""
    error_type: str
    frequency: int
    recent_occurrences: List[str]
    impact_level: str


class HanseiKaiManager:
    """反省会マネージャー - v8.1新機能"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.slack_bot = SlackBot(config.get('slack', {}))
        self.activity_memory = ActivityMemory(config.get('activity_memory', {}))
        self.logger = logging.getLogger(__name__)
    
    async def collect_monthly_hansei_data(
        self, 
        year: int, 
        month: int
    ) -> HanseiData:
        """
        月次反省会データを収集
        
        Args:
            year: 対象年
            month: 対象月
            
        Returns:
            HanseiData: 反省会データ
        """
        self.logger.info(f"反省会データ収集開始: {year}年{month}月")
        
        # 1. 反省会チャンネル検出
        channel_name = f"#反省会-{year}-{month:02d}"
        channel_data = await self._get_hansei_channel_data(channel_name)
        
        if not channel_data:
            self.logger.warning(f"反省会チャンネルが見つかりません: {channel_name}")
            return self._create_empty_hansei_data(f"{year}-{month:02d}")
        
        # 2. メッセージ分析
        hansei_data = await self._analyze_hansei_messages(
            channel_data, f"{year}-{month:02d}"
        )
        
        self.logger.info(f"反省会データ収集完了: {len(hansei_data.raw_messages)}件のメッセージ")
        return hansei_data
    
    async def _get_hansei_channel_data(self, channel_name: str) -> Optional[List[Dict[str, Any]]]:
        """反省会チャンネルのデータを取得"""
        try:
            # Slackチャンネル検索・メッセージ取得
            messages = await self.slack_bot.get_channel_messages(channel_name)
            return messages if messages else None
        except Exception as e:
            self.logger.error(f"チャンネルデータ取得エラー: {e}")
            return None
    
    async def _analyze_hansei_messages(
        self, 
        messages: List[Dict[str, Any]], 
        month_str: str
    ) -> HanseiData:
        """反省会メッセージを分析してカテゴリ分け"""
        successes = []
        failures = []
        improvements = []
        next_month_policies = []
        
        for message in messages:
            text = message.get('text', '').lower()
            content = message.get('text', '')
            
            # キーワードベースの分類
            if any(keyword in text for keyword in ['成功', '良かった', '効果的', 'うまくいった']):
                successes.append(content)
            elif any(keyword in text for keyword in ['失敗', '問題', 'エラー', 'ミス', 'うまくいかなかった']):
                failures.append(content)
            elif any(keyword in text for keyword in ['改善', '見直し', '修正', '変更']):
                improvements.append(content)
            elif any(keyword in text for keyword in ['来月', '次月', '方針', '計画']):
                next_month_policies.append(content)
        
        return HanseiData(
            month=month_str,
            successes=successes,
            failures=failures,
            improvements=improvements,
            next_month_policies=next_month_policies,
            raw_messages=messages
        )
    
    def _create_empty_hansei_data(self, month_str: str) -> HanseiData:
        """空の反省会データを作成"""
        return HanseiData(
            month=month_str,
            successes=[],
            failures=[],
            improvements=[],
            next_month_policies=[],
            raw_messages=[]
        )
    
    async def analyze_activity_memory_patterns(
        self, 
        year: int, 
        month: int
    ) -> Dict[str, Any]:
        """
        陣中日記から月次パターンを分析
        
        Args:
            year: 対象年
            month: 対象月
            
        Returns:
            分析結果辞書
        """
        self.logger.info(f"陣中日記分析開始: {year}年{month}月")
        
        # 対象期間の開始・終了日時
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(seconds=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(seconds=1)
        
        try:
            # 陣中日記から月次データを取得
            activities = await self.activity_memory.get_activities_in_period(
                start_date, end_date
            )
            
            # パターン分析
            frequent_errors = self._analyze_frequent_errors(activities)
            common_decisions = self._analyze_common_decisions(activities)
            time_consuming_tasks = self._analyze_time_consuming_tasks(activities)
            
            return {
                'frequent_errors': frequent_errors,
                'common_decisions': common_decisions,
                'time_consuming_tasks': time_consuming_tasks,
                'total_activities': len(activities),
                'analysis_period': f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"
            }
            
        except Exception as e:
            self.logger.error(f"陣中日記分析エラー: {e}")
            return {
                'frequent_errors': [],
                'common_decisions': [],
                'time_consuming_tasks': [],
                'total_activities': 0,
                'analysis_period': f"{year}-{month:02d}",
                'error': str(e)
            }
    
    def _analyze_frequent_errors(self, activities: List[Dict[str, Any]]) -> List[ErrorStatistics]:
        """頻出エラーパターンを分析"""
        error_counts = {}
        error_details = {}
        
        for activity in activities:
            # 失敗や問題があった活動を抽出
            reasoning = activity.get('reasoning', '')
            if any(keyword in reasoning.lower() for keyword in ['エラー', '失敗', '問題', 'error', 'failed']):
                # エラーの種類を特定（簡易版）
                error_type = self._categorize_error(reasoning)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                
                if error_type not in error_details:
                    error_details[error_type] = []
                error_details[error_type].append(reasoning[:200])  # 最初の200文字
        
        # 頻度順にソート
        frequent_errors = []
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            if count >= 2:  # 2回以上発生したエラーのみ
                impact = 'high' if count >= 5 else 'medium' if count >= 3 else 'low'
                frequent_errors.append(ErrorStatistics(
                    error_type=error_type,
                    frequency=count,
                    recent_occurrences=error_details[error_type][-3:],  # 最新3件
                    impact_level=impact
                ))
        
        return frequent_errors[:5]  # 上位5件
    
    def _categorize_error(self, reasoning: str) -> str:
        """エラーを分類"""
        reasoning_lower = reasoning.lower()
        
        if 'タイムアウト' in reasoning_lower or 'timeout' in reasoning_lower:
            return 'タイムアウトエラー'
        elif 'メモリ' in reasoning_lower or 'memory' in reasoning_lower:
            return 'メモリ不足'
        elif 'ネットワーク' in reasoning_lower or 'network' in reasoning_lower:
            return 'ネットワークエラー'
        elif 'api' in reasoning_lower:
            return 'API呼び出しエラー'
        elif 'ファイル' in reasoning_lower or 'file' in reasoning_lower:
            return 'ファイル操作エラー'
        elif 'データベース' in reasoning_lower or 'database' in reasoning_lower:
            return 'データベースエラー'
        else:
            return 'その他のエラー'
    
    def _analyze_common_decisions(self, activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """共通する判断パターンを分析"""
        decision_patterns = {}
        
        for activity in activities:
            decision = activity.get('final_decision', '')
            if decision and len(decision) > 10:  # 意味のある判断のみ
                # 判断の種類を分類（簡易版）
                decision_type = self._categorize_decision(decision)
                if decision_type not in decision_patterns:
                    decision_patterns[decision_type] = {
                        'count': 0,
                        'examples': [],
                        'avg_confidence': 0,
                        'confidences': []
                    }
                
                decision_patterns[decision_type]['count'] += 1
                decision_patterns[decision_type]['examples'].append(decision[:150])
                
                confidence = activity.get('confidence_score', 0.5)
                decision_patterns[decision_type]['confidences'].append(confidence)
        
        # 平均確信度を計算
        result = []
        for decision_type, data in decision_patterns.items():
            if data['count'] >= 2:  # 2回以上の判断のみ
                avg_confidence = sum(data['confidences']) / len(data['confidences'])
                result.append({
                    'decision_type': decision_type,
                    'frequency': data['count'],
                    'average_confidence': round(avg_confidence, 2),
                    'examples': data['examples'][-3:]  # 最新3件
                })
        
        return sorted(result, key=lambda x: x['frequency'], reverse=True)[:5]
    
    def _categorize_decision(self, decision: str) -> str:
        """判断を分類"""
        decision_lower = decision.lower()
        
        if any(word in decision_lower for word in ['実装', 'implement', 'develop']):
            return '実装判断'
        elif any(word in decision_lower for word in ['設計', 'design', 'architecture']):
            return '設計判断'
        elif any(word in decision_lower for word in ['修正', 'fix', 'bug']):
            return 'バグ修正判断'
        elif any(word in decision_lower for word in ['最適化', 'optimize', 'performance']):
            return 'パフォーマンス判断'
        elif any(word in decision_lower for word in ['テスト', 'test']):
            return 'テスト判断'
        else:
            return 'その他の判断'
    
    def _analyze_time_consuming_tasks(self, activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """時間のかかるタスクを分析"""
        time_consuming = []
        
        for activity in activities:
            processing_time = activity.get('processing_time_seconds', 0)
            if processing_time > 30:  # 30秒以上のタスク
                task_summary = activity.get('task_summary', '不明なタスク')
                time_consuming.append({
                    'task_summary': task_summary[:100],
                    'processing_time_seconds': processing_time,
                    'reasoning': activity.get('reasoning', '')[:100]
                })
        
        # 処理時間順にソート
        return sorted(time_consuming, key=lambda x: x['processing_time_seconds'], reverse=True)[:5]
    
    async def get_sandbox_error_statistics(
        self, 
        year: int, 
        month: int
    ) -> Dict[str, Any]:
        """
        演習場のエラー統計を取得
        
        Args:
            year: 対象年
            month: 対象月
            
        Returns:
            エラー統計辞書
        """
        self.logger.info(f"演習場エラー統計収集: {year}年{month}月")
        
        try:
            # 演習場のログを分析（実装は演習場システムに依存）
            # ここでは模擬データを返す
            return {
                'total_executions': 150,
                'failed_executions': 12,
                'success_rate': 92.0,
                'common_errors': [
                    {'type': 'ImportError', 'count': 5, 'examples': ['numpy not found', 'module missing']},
                    {'type': 'SyntaxError', 'count': 4, 'examples': ['invalid syntax', 'indentation error']},
                    {'type': 'TimeoutError', 'count': 3, 'examples': ['execution timeout', 'long running process']}
                ],
                'languages': {
                    'python': {'executions': 100, 'failures': 8},
                    'nodejs': {'executions': 35, 'failures': 3},
                    'rust': {'executions': 15, 'failures': 1}
                }
            }
        except Exception as e:
            self.logger.error(f"演習場統計取得エラー: {e}")
            return {
                'total_executions': 0,
                'failed_executions': 0,
                'success_rate': 0.0,
                'common_errors': [],
                'languages': {},
                'error': str(e)
            }
    
    async def generate_monthly_summary_report(
        self, 
        year: int, 
        month: int
    ) -> Dict[str, Any]:
        """
        月次総合レポートを生成
        
        Args:
            year: 対象年
            month: 対象月
            
        Returns:
            総合レポート辞書
        """
        self.logger.info(f"月次総合レポート生成: {year}年{month}月")
        
        # 並列でデータ収集
        hansei_task = asyncio.create_task(
            self.collect_monthly_hansei_data(year, month)
        )
        activity_task = asyncio.create_task(
            self.analyze_activity_memory_patterns(year, month)
        )
        sandbox_task = asyncio.create_task(
            self.get_sandbox_error_statistics(year, month)
        )
        
        hansei_data = await hansei_task
        activity_analysis = await activity_task
        sandbox_stats = await sandbox_task
        
        # 総合レポート作成
        report = {
            'period': f"{year}年{month}月",
            'generated_at': datetime.now().isoformat(),
            'hansei_data': {
                'successes_count': len(hansei_data.successes),
                'failures_count': len(hansei_data.failures),
                'improvements_count': len(hansei_data.improvements),
                'policies_count': len(hansei_data.next_month_policies),
                'raw_data': hansei_data
            },
            'activity_analysis': activity_analysis,
            'sandbox_statistics': sandbox_stats,
            'summary': {
                'key_insights': self._extract_key_insights(
                    hansei_data, activity_analysis, sandbox_stats
                ),
                'recommended_katun': []  # 家訓生成器で後で追加
            }
        }
        
        return report
    
    def _extract_key_insights(
        self, 
        hansei_data: HanseiData,
        activity_analysis: Dict[str, Any],
        sandbox_stats: Dict[str, Any]
    ) -> List[str]:
        """キーとなる洞察を抽出"""
        insights = []
        
        # 反省会データから
        if len(hansei_data.failures) > len(hansei_data.successes):
            insights.append("失敗事例が成功事例を上回っている - 改善策の強化が必要")
        
        # 活動記録から
        if activity_analysis.get('frequent_errors'):
            top_error = activity_analysis['frequent_errors'][0]
            insights.append(f"最頻出エラー: {top_error.error_type} ({top_error.frequency}回)")
        
        # 演習場統計から
        success_rate = sandbox_stats.get('success_rate', 0)
        if success_rate < 90:
            insights.append(f"演習場成功率が{success_rate}%と低い - コード品質の改善が必要")
        
        return insights