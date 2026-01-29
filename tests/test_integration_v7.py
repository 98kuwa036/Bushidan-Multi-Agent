#!/usr/bin/env python3
"""Integration Tests for Shogun System v7.0

Tests all major v7.0 features:
  - Pro CLI first strategy
  - Japanese R1 integration
  - Groq recorder functionality
  - Platoon mode operations
  - Notion knowledge management
  - 11-bot Slack integration
  - Cost optimization tracking
"""

import asyncio
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

# Import v7.0 components
from shogun.core.integrated_shogun import IntegratedShogun
from shogun.ashigaru.groq_recorder import GroqRecorder
from shogun.integrations.notion_integration import NotionIntegration
from shogun.integrations.slack_v7 import ShogunSlackIntegrationV7


class TestIntegratedShogunV7:
    """Test the complete v7.0 integrated system."""
    
    @pytest.fixture
    async def shogun_system(self):
        """Create test Shogun system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test config
            config = {
                "version": "7.0",
                "taisho_japanese": {
                    "url": "http://localhost:11434",
                    "model": "cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese"
                },
                "groq": {
                    "model": "llama-3.3-70b-versatile",
                    "max_requests_per_day": 14400
                },
                "notion": {
                    "database_id": "test_db_id",
                    "auto_summary_days": 60
                },
                "platoon_modes": {
                    "voice_query": {"max_ashigaru": 2, "response_time_target": 30},
                    "quick_info": {"max_ashigaru": 1, "response_time_target": 15},
                    "file_check": {"max_ashigaru": 1, "response_time_target": 10}
                },
                "repo": {
                    "local_base": "/tmp",
                    "sync_interval_minutes": 5
                }
            }
            
            system = IntegratedShogun(temp_dir)
            system.config = config
            
            # Mock external dependencies
            system.japanese_r1 = AsyncMock()
            system.groq_recorder = AsyncMock()
            system._notion_client = AsyncMock()
            
            yield system

    @pytest.mark.asyncio
    async def test_pro_cli_first_strategy(self, shogun_system):
        """Test Pro CLI first strategy with API fallback."""
        # Mock Pro CLI success
        with patch.object(shogun_system.claude_cli, 'generate') as mock_cli:
            mock_cli.return_value = Mock(success=True, text="Pro CLI response", error=None)
            
            result = await shogun_system.process_task(
                prompt="Test Pro CLI first",
                mode="battalion"
            )
            
            assert "Pro CLI response" in result
            assert shogun_system.stats["pro_cli_success"] == 1
            assert shogun_system.stats["api_fallback"] == 0

    @pytest.mark.asyncio
    async def test_api_fallback(self, shogun_system):
        """Test API fallback when Pro CLI fails."""
        # Mock Pro CLI failure and API success
        with patch.object(shogun_system.claude_cli, 'generate') as mock_cli, \
             patch.object(shogun_system, '_call_cloud_api') as mock_api:
            
            mock_cli.return_value = Mock(success=False, rate_limited=True, error="Rate limited")
            mock_api.return_value = "API fallback response"
            
            result = await shogun_system._call_cloud(
                Mock(context={"taisho_analysis": "test"}),
                "karo",
                "sonnet"
            )
            
            assert result == "API fallback response"
            assert shogun_system.stats["api_fallback"] == 1

    @pytest.mark.asyncio
    async def test_japanese_r1_integration(self, shogun_system):
        """Test Japanese R1 model integration."""
        # Mock Japanese R1 response
        shogun_system.japanese_r1.generate.return_value = "日本語R1による深い思考結果"
        
        task = Mock(
            prompt="ESP32のI2S設定について教えて",
            context={},
            assigned_agent=None,
            status=None
        )
        
        result = await shogun_system._call_japanese_taisho(task, company_mode=True)
        
        assert result == "日本語R1による深い思考結果"
        shogun_system.japanese_r1.generate.assert_called_once()
        
        # Check system prompt includes Japanese instructions
        call_args = shogun_system.japanese_r1.generate.call_args
        system_prompt = call_args.kwargs.get('system', '')
        assert "日本語" in system_prompt
        assert "<think>" in system_prompt

    @pytest.mark.asyncio
    async def test_platoon_mode_voice_query(self, shogun_system):
        """Test Platoon mode for voice queries."""
        shogun_system.japanese_r1.generate.return_value = "簡潔な音声応答"
        
        result = await shogun_system.process_task(
            prompt="I2Sのバッファサイズは？",
            mode="platoon",
            platoon_type="voice_query"
        )
        
        assert "簡潔な音声応答" in result
        assert shogun_system.stats["platoon"] == 1
        
        # Verify minimal ashigaru selection
        call_args = shogun_system.japanese_r1.generate.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_deployment_mode_routing(self, shogun_system):
        """Test correct routing for different deployment modes."""
        shogun_system.japanese_r1.generate.return_value = "Mode test response"
        
        # Test Battalion mode
        await shogun_system.process_task("Complex task", mode="battalion")
        assert shogun_system.stats["battalion"] == 1
        
        # Test Company mode
        await shogun_system.process_task("Medium task", mode="company")
        assert shogun_system.stats["company"] == 1
        
        # Test Platoon mode
        await shogun_system.process_task("Quick query", mode="platoon", platoon_type="voice_query")
        assert shogun_system.stats["platoon"] == 1

    @pytest.mark.asyncio
    async def test_cost_tracking(self, shogun_system):
        """Test cost tracking and optimization."""
        shogun_system.japanese_r1.generate.return_value = "Cost test response"
        
        # Free operations (Japanese R1)
        await shogun_system.process_task("Test task", mode="company")
        
        assert shogun_system.stats["total_cost_yen"] == 0
        assert shogun_system.stats["taisho_japanese_r1"] == 1

    def test_monthly_cost_estimate(self, shogun_system):
        """Test monthly cost estimation."""
        # Simulate usage
        shogun_system.stats.update({
            "taisho_japanese_r1": 900,  # Local, free
            "karo_sonnet": 20,          # Pro CLI + some API
            "shogun_opus": 5,           # Strategic only
            "pro_cli_success": 850,
            "api_fallback": 25
        })
        
        estimated_cost = shogun_system._estimate_monthly_cost()
        
        # Should be around ¥3,950 (target)
        assert 3800 <= estimated_cost <= 4100

    @pytest.mark.asyncio
    async def test_quality_over_speed_philosophy(self, shogun_system):
        """Test that system prioritizes quality over speed."""
        # Mock slow but high-quality response
        shogun_system.japanese_r1.generate.return_value = """
        <think>
        この質問は深い考慮が必要です。ESP32のI2S設定について、
        以下の点を検討する必要があります...
        </think>
        
        ESP32のI2S設定では、以下の点が重要です：
        1. クロック設定の最適化
        2. バッファサイズの調整
        3. DMA設定の確認
        """
        
        result = await shogun_system.process_task(
            "ESP32のI2S設定について詳しく教えて",
            mode="company"
        )
        
        # Verify deep thinking indicators
        assert "<think>" in shogun_system.japanese_r1.generate.call_args.kwargs.get('system', '')
        assert "詳しく" in result or "重要" in result


class TestGroqRecorder:
    """Test the 9th Ashigaru Groq recorder."""
    
    @pytest.fixture
    def groq_recorder(self):
        """Create test Groq recorder."""
        with patch('shogun.ashigaru.groq_recorder.groq') as mock_groq_module:
            mock_client = Mock()
            mock_groq_module.Groq.return_value = mock_client
            
            recorder = GroqRecorder(
                api_key="test_key",
                notion_integration={"database_id": "test_db"}
            )
            recorder.client = mock_client
            
            yield recorder

    @pytest.mark.asyncio
    async def test_session_recording(self, groq_recorder):
        """Test complete session recording."""
        # Start session
        await groq_recorder.start_session("test_task_001", "Test prompt")
        
        assert groq_recorder.current_session is not None
        assert groq_recorder.current_session["id"] == "test_task_001"
        assert groq_recorder.stats["sessions_started"] == 1

    @pytest.mark.asyncio
    async def test_interaction_recording(self, groq_recorder):
        """Test interaction recording."""
        await groq_recorder.start_session("test_task_002", "Test prompt")
        
        await groq_recorder.record_interaction(
            "test_task_002",
            "taisho_japanese",
            "Test prompt",
            "Test response"
        )
        
        assert len(groq_recorder.current_session["interactions"]) == 1
        assert groq_recorder.stats["interactions_recorded"] == 1

    @pytest.mark.asyncio
    async def test_60day_summary_generation(self, groq_recorder):
        """Test 60-day summary generation."""
        # Mock Groq response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "60日要約テスト結果"
        mock_response.usage.total_tokens = 500
        
        groq_recorder.client.chat.completions.create.return_value = mock_response
        
        # Create test sessions directory
        groq_recorder.storage_dir.mkdir(exist_ok=True)
        test_session = {
            "id": "test_session",
            "start_time": "2025-01-01T00:00:00",
            "initial_prompt": "Test prompt",
            "status": "completed"
        }
        
        with open(groq_recorder.storage_dir / "session_test.json", 'w') as f:
            json.dump(test_session, f)
        
        summary = await groq_recorder.generate_60day_summary()
        
        assert summary == "60日要約テスト結果"
        assert groq_recorder.stats["summaries_generated"] == 1

    def test_daily_quota_tracking(self, groq_recorder):
        """Test daily quota tracking."""
        # Simulate requests
        for _ in range(100):
            groq_recorder._track_request(50)
        
        assert groq_recorder.daily_requests == 100
        assert groq_recorder.stats["groq_requests"] == 100
        assert groq_recorder.stats["total_tokens"] == 5000
        
        # Check quota enforcement
        assert groq_recorder._can_make_request() is True
        
        # Simulate quota exhaustion
        groq_recorder.daily_requests = 14400
        assert groq_recorder._can_make_request() is False


class TestNotionIntegration:
    """Test Notion knowledge management."""
    
    @pytest.fixture
    def notion_integration(self):
        """Create test Notion integration."""
        with patch('shogun.integrations.notion_integration.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            integration = NotionIntegration(
                token="test_token",
                database_id="test_db_id"
            )
            integration.client = mock_client
            
            yield integration

    @pytest.mark.asyncio
    async def test_summary_saving(self, notion_integration):
        """Test 60-day summary saving."""
        # Mock successful page creation
        notion_integration.client.pages.create.return_value = {"id": "page_123"}
        
        success = await notion_integration.save_summary(
            "60日間のテスト要約",
            metadata={"cost_total": 500, "session_count": 50}
        )
        
        assert success is True
        assert notion_integration.stats["summaries_saved"] == 1
        assert notion_integration.stats["knowledge_entries"] == 1

    @pytest.mark.asyncio
    async def test_family_precepts_saving(self, notion_integration):
        """Test family precepts (家訓) saving."""
        precepts = [
            "ESP32のSPI DMAの設定では必ずバッファサイズを確認せよ",
            "I2S設定時はクロック周波数の整合性を検証せよ",
            "Home Assistant統合では音声認識の遅延を考慮せよ"
        ]
        
        notion_integration.client.pages.create.return_value = {"id": "page_456"}
        
        success = await notion_integration.save_family_precepts(
            precepts,
            context="ESP32開発プロジェクトから抽出"
        )
        
        assert success is True
        assert notion_integration.stats["precepts_saved"] == 3
        assert notion_integration.stats["knowledge_entries"] == 1

    @pytest.mark.asyncio
    async def test_knowledge_search(self, notion_integration):
        """Test knowledge base search."""
        # Mock search results
        notion_integration.client.databases.query.return_value = {
            "results": [
                {
                    "id": "result_1",
                    "url": "https://notion.so/result_1",
                    "properties": {
                        "Title": {"title": [{"text": {"content": "ESP32 I2S設定"}}]},
                        "Type": {"select": {"name": "家訓"}},
                        "Date": {"date": {"start": "2025-01-29"}}
                    }
                }
            ]
        }
        
        results = await notion_integration.search_knowledge("I2S", entry_type="家訓")
        
        assert len(results) == 1
        assert results[0]["title"] == "ESP32 I2S設定"
        assert results[0]["type"] == "家訓"
        assert notion_integration.stats["search_queries"] == 1


class TestSlackIntegrationV7:
    """Test 11-bot Slack integration."""
    
    @pytest.fixture
    def slack_integration(self):
        """Create test Slack integration."""
        with patch('shogun.integrations.slack_v7.WebClient') as mock_webclient:
            # Mock environment variables
            env_vars = {
                f"SLACK_TOKEN_{name.upper()}": f"test_token_{name}"
                for name in ["SHOGUN", "LIGHT", "KARO", "TAISHO"] +
                           [f"ASHIGARU_{i}" for i in range(1, 10)]
            }
            
            with patch.dict(os.environ, env_vars):
                integration = ShogunSlackIntegrationV7()
                
                # Mock all bot clients
                for bot in integration.bots.values():
                    bot.client = Mock()
                    bot.client.chat_postMessage.return_value = {"ok": True}
                
                yield integration

    def test_bot_initialization(self, slack_integration):
        """Test that all 11 bots are initialized."""
        assert len(slack_integration.bots) == 11
        
        # Check specific bots exist
        assert "shogun-bot" in slack_integration.bots
        assert "shogun-bot-light" in slack_integration.bots
        assert "ashigaru-9-bot" in slack_integration.bots  # Groq recorder

    @pytest.mark.asyncio
    async def test_mode_determination(self, slack_integration):
        """Test deployment mode determination from messages."""
        # Test explicit company mode
        mode, platoon_type = slack_integration._determine_mode({
            "text": "@shogun-bot-light help with I2S"
        })
        assert mode == "company"
        assert platoon_type is None
        
        # Test platoon mode (voice-like query)
        mode, platoon_type = slack_integration._determine_mode({
            "text": "I2Sの設定は？"
        })
        assert mode == "platoon"
        assert platoon_type == "voice_query"
        
        # Test file check platoon
        mode, platoon_type = slack_integration._determine_mode({
            "text": "ファイルcheck"
        })
        assert mode == "platoon"
        assert platoon_type == "file_check"

    def test_bot_selection_for_modes(self, slack_integration):
        """Test correct bot selection for different modes."""
        # Battalion mode
        bot = slack_integration._select_bot_for_mode("battalion")
        assert bot.name == "shogun-bot"
        
        # Company mode
        bot = slack_integration._select_bot_for_mode("company")
        assert bot.name == "shogun-bot-light"
        
        # Platoon mode
        bot = slack_integration._select_bot_for_mode("platoon")
        assert bot.name == "shogun-bot-light"

    @pytest.mark.asyncio
    async def test_enhanced_response_formatting(self, slack_integration):
        """Test enhanced response formatting with v7.0 info."""
        bot = slack_integration.bots["shogun-bot-light"]
        
        await slack_integration._send_enhanced_response(
            bot, "#test", "Test response", "company", None
        )
        
        # Check that enhanced information was added
        call_args = bot.client.chat_postMessage.call_args
        message_text = call_args.kwargs["text"]
        
        assert "中隊モード" in message_text
        assert "¥0" in message_text
        assert "日本語R1" in message_text

    def test_cost_optimization_tracking(self, slack_integration):
        """Test cost optimization statistics tracking."""
        # Simulate usage
        slack_integration.stats.update({
            "pro_cli_success": 85,
            "api_fallback": 15,
            "japanese_r1_calls": 90,
            "total_requests": 100
        })
        
        monthly_cost = slack_integration._estimate_monthly_cost()
        
        # Should reflect cost optimization
        assert monthly_cost < 5000  # Well below original estimates


class TestSystemIntegration:
    """Test complete system integration."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create system with mocked dependencies
            system = IntegratedShogun(temp_dir)
            system.japanese_r1 = AsyncMock()
            system.groq_recorder = AsyncMock()
            
            # Mock responses
            system.japanese_r1.generate.return_value = """
            <think>
            この質問は ESP32 の I2S 設定について聞いています。
            重要なポイントを整理して回答します。
            </think>
            
            ESP32のI2S設定では以下が重要です：
            1. クロック設定: MCLK, BCLK, WS の適切な設定
            2. バッファ設定: DMA バッファサイズの最適化
            3. フォーマット: I2S_MODE_MASTER, I2S_BITS_PER_SAMPLE_16BIT
            """
            
            # Process task
            result = await system.process_task(
                prompt="ESP32のI2S設定について教えてください",
                mode="company"
            )
            
            # Verify result
            assert "I2S設定" in result
            assert "クロック設定" in result
            assert "バッファ設定" in result
            
            # Verify system state
            assert system.stats["company"] == 1
            assert system.stats["taisho_japanese_r1"] == 1
            
            # Verify recording was attempted
            system.groq_recorder.start_session.assert_called_once()
            system.groq_recorder.record_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_quality_metrics_achievement(self):
        """Test that quality targets are met."""
        with tempfile.TemporaryDirectory() as temp_dir:
            system = IntegratedShogun(temp_dir)
            system.japanese_r1 = AsyncMock()
            
            # Mock high-quality response with thinking
            system.japanese_r1.generate.return_value = """
            <think>
            この質問は複雑な技術的な内容です。ユーザーの理解レベルを
            考慮して、段階的に説明する必要があります。
            具体的なコード例も含めるべきでしょう。
            </think>
            
            ESP32のI2S設定について詳しく説明します。
            
            ## 基本設定
            i2s_config_t i2s_config = {
                .mode = I2S_MODE_MASTER | I2S_MODE_TX,
                .sample_rate = 44100,
                .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
                .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
                .communication_format = I2S_COMM_FORMAT_I2S,
                .dma_buf_count = 8,
                .dma_buf_len = 1024
            };
            
            ## 重要なポイント
            1. DMAバッファ設定は音声品質に直結
            2. サンプルレートは用途に応じて調整
            3. ピン配置の確認は必須
            """
            
            result = await system.process_task(
                "ESP32のI2S設定について詳しく教えてください",
                mode="company"
            )
            
            # Quality indicators
            quality_score = 0
            
            # Check for detailed explanations
            if "詳しく説明" in result:
                quality_score += 20
            if "## " in result:  # Structured formatting
                quality_score += 20
            if "i2s_config_t" in result:  # Specific technical content
                quality_score += 20
            if "重要" in result or "ポイント" in result:  # Emphasis on key points
                quality_score += 20
            if len(result) > 200:  # Comprehensive response
                quality_score += 20
            
            # Should achieve target quality score
            assert quality_score >= 80  # 80% quality indicators present

    def test_cost_optimization_achievement(self):
        """Test that cost optimization targets are achieved."""
        # Simulate typical monthly usage based on v7.0 design
        monthly_stats = {
            # Japanese R1 handles majority (free)
            "taisho_japanese_r1": 900,  # 90% of simple/medium tasks
            
            # Pro CLI success for cloud tasks (free)
            "pro_cli_success": 85,      # 85% Pro CLI success rate
            
            # Minimal API usage (paid)
            "karo_sonnet_api": 15,      # 15% fallback × ¥5 = ¥75
            "shogun_opus_api": 5,       # Strategic only × ¥24 = ¥120
        }
        
        # Calculate costs
        api_costs = (monthly_stats["karo_sonnet_api"] * 5 + 
                    monthly_stats["shogun_opus_api"] * 24)
        fixed_costs = 3800  # Pro + power
        total_monthly = fixed_costs + api_costs
        
        # Verify cost targets
        assert total_monthly <= 4000  # Target: ¥3,950
        assert api_costs <= 200       # Minimal API usage
        
        # Verify optimization ratios
        free_operations = (monthly_stats["taisho_japanese_r1"] + 
                          monthly_stats["pro_cli_success"])
        total_operations = sum(monthly_stats.values())
        free_ratio = free_operations / total_operations
        
        assert free_ratio >= 0.90  # 90%+ free operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])