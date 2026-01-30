#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
月次家訓自動スケジューラー v8.1
=====================================

月末に自動実行される家訓生成・通知システム

機能:
- 月末自動実行（cron）
- 家訓生成プロセス管理
- Slack通知
- Notion記録
- エラーハンドリング・再試行
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from core.katun_generator import KatunGenerator, MonthlyKatun
from core.hanseikai_manager import HanseiKaiManager
from integrations.slack_bot import SlackBot
from integrations.notion_integration import NotionIntegration


class MonthlyKatunScheduler:
    """月次家訓自動スケジューラー - v8.1新機能"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.katun_generator = KatunGenerator(config)
        self.hansei_manager = HanseiKaiManager(config)
        self.slack_bot = SlackBot(config.get('slack', {}))
        self.notion_integration = NotionIntegration(config.get('notion', {}))
        self.logger = logging.getLogger(__name__)
        
        # ログファイルパス
        self.log_dir = Path(config.get('log_directory', '/var/log/shogun'))
        self.log_dir.mkdir(exist_ok=True)
    
    async def execute_monthly_katun_generation(
        self, 
        target_year: Optional[int] = None,
        target_month: Optional[int] = None,
        force_execution: bool = False
    ) -> Dict[str, Any]:
        """
        月次家訓生成の実行
        
        Args:
            target_year: 対象年（Noneの場合は前月）
            target_month: 対象月（Noneの場合は前月）
            force_execution: 強制実行フラグ
            
        Returns:
            実行結果辞書
        """
        execution_start = datetime.now()
        execution_id = f"katun-gen-{execution_start.strftime('%Y%m%d-%H%M%S')}"
        
        self.logger.info(f"月次家訓生成開始: {execution_id}")
        
        # 対象月の決定
        if target_year is None or target_month is None:
            # 前月を対象とする
            last_month = datetime.now().replace(day=1) - timedelta(days=1)
            target_year = last_month.year
            target_month = last_month.month
        
        result = {
            'execution_id': execution_id,
            'target_period': f"{target_year}-{target_month:02d}",
            'execution_start': execution_start.isoformat(),
            'status': 'running',
            'steps': {},
            'errors': [],
            'generated_katun': None,
            'notifications_sent': {}
        }
        
        try:
            # Step 1: 実行条件チェック
            if not force_execution:
                should_execute = await self._should_execute_generation(target_year, target_month)
                if not should_execute:
                    result['status'] = 'skipped'
                    result['skip_reason'] = '実行条件を満たしていません'
                    self.logger.info(f"家訓生成をスキップ: {result['skip_reason']}")
                    return result
            
            result['steps']['condition_check'] = {'status': 'completed', 'timestamp': datetime.now().isoformat()}
            
            # Step 2: 月次家訓生成
            self.logger.info(f"家訓生成開始: {target_year}年{target_month}月")
            monthly_katun = await self.katun_generator.generate_monthly_katun(target_year, target_month)
            
            result['steps']['generation'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'katun_count': len(monthly_katun.katun_list)
            }
            result['generated_katun'] = monthly_katun
            
            # Step 3: 期限切れ家訓のクリーンアップ
            cleaned_count = await self.katun_generator.cleanup_expired_katun()
            result['steps']['cleanup'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'cleaned_count': cleaned_count
            }
            
            # Step 4: Slack通知
            slack_success = await self._send_slack_notification(monthly_katun, execution_id)
            result['notifications_sent']['slack'] = slack_success
            result['steps']['slack_notification'] = {
                'status': 'completed' if slack_success else 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Step 5: Notion記録
            notion_success = await self._record_to_notion(monthly_katun, result)
            result['notifications_sent']['notion'] = notion_success
            result['steps']['notion_record'] = {
                'status': 'completed' if notion_success else 'failed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Step 6: 実行ログ保存
            await self._save_execution_log(result)
            result['steps']['log_save'] = {'status': 'completed', 'timestamp': datetime.now().isoformat()}
            
            result['status'] = 'completed'
            result['execution_end'] = datetime.now().isoformat()
            
            self.logger.info(f"月次家訓生成完了: {execution_id}, {len(monthly_katun.katun_list)}個の家訓生成")
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append({
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'step': self._get_current_step(result['steps'])
            })
            result['execution_end'] = datetime.now().isoformat()
            
            self.logger.error(f"月次家訓生成失敗: {execution_id}, エラー: {e}")
            
            # エラー通知
            await self._send_error_notification(result, str(e))
        
        return result
    
    async def _should_execute_generation(self, year: int, month: int) -> bool:
        """実行条件をチェック"""
        try:
            # 既に同じ月の家訓が生成済みかチェック
            period_str = f"{year}-{month:02d}"
            existing_katun = await self.katun_generator.rag_integration.search_by_metadata(
                metadata_filter={
                    'type': 'monthly_katun',
                    'period': period_str
                }
            )
            
            if existing_katun:
                self.logger.info(f"{period_str}の家訓は既に生成済み")
                return False
            
            # 反省会データの存在をチェック
            hansei_data = await self.hansei_manager.collect_monthly_hansei_data(year, month)
            if not hansei_data.raw_messages:
                self.logger.warning(f"{period_str}の反省会データが見つかりません")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"実行条件チェックエラー: {e}")
            return False
    
    def _get_current_step(self, steps: Dict[str, Any]) -> str:
        """現在のステップを特定"""
        step_order = ['condition_check', 'generation', 'cleanup', 'slack_notification', 'notion_record', 'log_save']
        
        for step in step_order:
            if step not in steps:
                return step
        
        return 'unknown'
    
    async def _send_slack_notification(self, monthly_katun: MonthlyKatun, execution_id: str) -> bool:
        """Slackに家訓を通知"""
        try:
            # 家訓リストをフォーマット
            katun_text = self._format_katun_for_slack(monthly_katun)
            
            message = f"""🏯 **月次家訓が生成されました** 

📅 **対象期間**: {monthly_katun.period}
⏰ **生成時刻**: {monthly_katun.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
📜 **家訓数**: {len(monthly_katun.katun_list)}個
⏳ **有効期限**: {monthly_katun.expires_at.strftime('%Y-%m-%d')}まで

{katun_text}

---
🤖 **実行ID**: `{execution_id}`
⚙️ 将軍システム v8.1 - 家訓自動生成強化"""
            
            # Slackチャンネルに投稿
            channel = self.config.get('slack', {}).get('katun_channel', '#shogun-katun')
            success = await self.slack_bot.post_message(channel, message)
            
            if success:
                self.logger.info(f"Slack通知成功: {channel}")
            else:
                self.logger.error(f"Slack通知失敗: {channel}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Slack通知エラー: {e}")
            return False
    
    def _format_katun_for_slack(self, monthly_katun: MonthlyKatun) -> str:
        """Slack用に家訓をフォーマット"""
        if not monthly_katun.katun_list:
            return "⚠️ 今月は新しい家訓が生成されませんでした。"
        
        formatted_lines = []
        for i, katun in enumerate(monthly_katun.katun_list, 1):
            priority_emoji = '🔥' * katun.priority
            category_emoji = {
                'technical': '⚙️',
                'operational': '📋',
                'monthly': '📆'
            }.get(katun.category, '📌')
            
            formatted_lines.append(
                f"{category_emoji} **{i}. {katun.content}** {priority_emoji}\n"
                f"   └ *{katun.category}* (優先度: {katun.priority})"
            )
        
        return "\n\n".join(formatted_lines)
    
    async def _record_to_notion(self, monthly_katun: MonthlyKatun, execution_result: Dict[str, Any]) -> bool:
        """Notionに家訓を記録"""
        try:
            # Notion記録用のデータ準備
            notion_data = {
                'title': f"月次家訓 {monthly_katun.period}",
                'period': monthly_katun.period,
                'generated_at': monthly_katun.generated_at.isoformat(),
                'expires_at': monthly_katun.expires_at.isoformat(),
                'katun_count': len(monthly_katun.katun_list),
                'katun_list': [
                    {
                        'content': katun.content,
                        'category': katun.category,
                        'priority': katun.priority,
                        'confidence_score': katun.confidence_score,
                        'reasoning': katun.reasoning
                    }
                    for katun in monthly_katun.katun_list
                ],
                'execution_summary': {
                    'execution_id': execution_result['execution_id'],
                    'steps_completed': len([s for s in execution_result['steps'].values() if s.get('status') == 'completed']),
                    'total_steps': len(execution_result['steps']),
                    'errors_count': len(execution_result['errors'])
                }
            }
            
            # Notionに記録
            success = await self.notion_integration.create_katun_record(notion_data)
            
            if success:
                self.logger.info(f"Notion記録成功: {monthly_katun.period}")
            else:
                self.logger.error(f"Notion記録失敗: {monthly_katun.period}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Notion記録エラー: {e}")
            return False
    
    async def _send_error_notification(self, execution_result: Dict[str, Any], error_message: str) -> None:
        """エラー通知を送信"""
        try:
            error_text = f"""🚨 **月次家訓生成エラー** 

📅 **対象期間**: {execution_result['target_period']}
⏰ **実行時刻**: {execution_result['execution_start']}
❌ **エラー**: {error_message}

🔍 **実行ID**: `{execution_result['execution_id']}`
📊 **ステップ状況**:
{self._format_steps_status(execution_result['steps'])}

⚙️ システム管理者にお知らせください。"""

            # エラー通知チャンネルに送信
            error_channel = self.config.get('slack', {}).get('error_channel', '#shogun-errors')
            await self.slack_bot.post_message(error_channel, error_text)
            
        except Exception as e:
            self.logger.error(f"エラー通知送信失敗: {e}")
    
    def _format_steps_status(self, steps: Dict[str, Any]) -> str:
        """ステップ状況をフォーマット"""
        status_lines = []
        for step_name, step_data in steps.items():
            status = step_data.get('status', 'unknown')
            emoji = '✅' if status == 'completed' else '❌' if status == 'failed' else '⏳'
            status_lines.append(f"{emoji} {step_name}: {status}")
        
        return "\n".join(status_lines)
    
    async def _save_execution_log(self, execution_result: Dict[str, Any]) -> None:
        """実行ログを保存"""
        try:
            log_filename = f"monthly_katun_{execution_result['execution_id']}.json"
            log_path = self.log_dir / log_filename
            
            # 実行結果をJSON形式で保存（MonthlyKatunオブジェクトは除外）
            log_data = {k: v for k, v in execution_result.items() if k != 'generated_katun'}
            if execution_result.get('generated_katun'):
                katun = execution_result['generated_katun']
                log_data['generated_katun_summary'] = {
                    'period': katun.period,
                    'katun_count': len(katun.katun_list),
                    'generated_at': katun.generated_at.isoformat(),
                    'expires_at': katun.expires_at.isoformat()
                }
            
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"実行ログ保存: {log_path}")
            
        except Exception as e:
            self.logger.error(f"ログ保存エラー: {e}")
    
    async def schedule_monthly_execution(self) -> None:
        """月末実行スケジュール（cron用エントリーポイント）"""
        """
        このメソッドはcronから呼び出される
        例: 0 23 28-31 * * /path/to/python -m core.monthly_katun_scheduler
        """
        try:
            # 今日が月末かチェック
            today = datetime.now()
            tomorrow = today + timedelta(days=1)
            
            if today.month != tomorrow.month:
                # 月末なので家訓生成を実行
                result = await self.execute_monthly_katun_generation()
                
                if result['status'] == 'completed':
                    self.logger.info(f"月末自動実行完了: {result['execution_id']}")
                else:
                    self.logger.error(f"月末自動実行失敗: {result['execution_id']}")
            else:
                self.logger.info("月末ではないため実行をスキップ")
                
        except Exception as e:
            self.logger.error(f"月末自動実行エラー: {e}")


# CLI実行用
async def main():
    """CLI実行エントリーポイント"""
    import sys
    import yaml
    
    # 設定ファイル読み込み
    config_path = Path(__file__).parent.parent / 'config' / 'settings.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    scheduler = MonthlyKatunScheduler(config)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        # 強制実行
        result = await scheduler.execute_monthly_katun_generation(force_execution=True)
        print(f"実行結果: {result['status']}")
    else:
        # 通常の月末実行
        await scheduler.schedule_monthly_execution()


if __name__ == "__main__":
    asyncio.run(main())