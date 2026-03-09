# Discord LLM統合ガイド

各LLMエージェントが個別のDiscordアカウントとしてスレッド内で会話する機能の統合ガイド

## 📋 セットアップ完了状況

### ✅ 完了項目

1. **7つのエージェント用ウェブフック作成完了**
   - 🎌 将軍 (Shogun) - Claude Sonnet 4.6
   - 📋 軍師 (Gunshi) - Qwen3-Coder-Next
   - 👔 家老 (Karo) - Gemini 3 Flash
   - ⚔️ 大将 (Taisho) - Local Qwen3
   - 🗡️ 傭兵 (Yohei) - Kimi K2.5
   - 👁️ 検校 (Kengyo) - Kimi Vision
   - ⚡ 足軽-Groq (Ashigaru) - Llama 3.3 70B

2. **設定ファイル作成完了**
   - `config/discord_llm_accounts.json`

3. **メッセンジャークラス作成完了**
   - `bushidan/discord_llm_messenger.py`

---

## 🚀 使い方

### 基本的な使用方法

```python
from bushidan.discord_llm_messenger import DiscordLLMMessenger

# 非同期関数内で使用
async def task_processing():
    async with DiscordLLMMessenger() as messenger:
        # 将軍からメッセージ
        await messenger.send_as_agent(
            "shogun",
            "🎌 将軍より：任務を受領しました"
        )

        # 軍師からステータス更新
        await messenger.send_status_update(
            "gunshi",
            "進行中",
            "タスク分解を実施中..."
        )
```

### ステータス更新

```python
# サポートされているステータス
await messenger.send_status_update("karo", "開始", "タスク開始")      # 🚀
await messenger.send_status_update("karo", "進行中", "処理中...")    # ⏳
await messenger.send_status_update("karo", "完了", "処理完了")       # ✅
await messenger.send_status_update("karo", "エラー", "エラー発生")   # ❌
await messenger.send_status_update("karo", "警告", "警告あり")       # ⚠️
```

### 色付きEmbed

```python
# 構造化された情報を送信
await messenger.send_with_color(
    "gunshi",
    title="PDCA分析結果",
    description="タスクを5つのサブタスクに分解しました",
    fields=[
        {"name": "サブタスク1", "value": "要件定義", "inline": True},
        {"name": "サブタスク2", "value": "設計", "inline": True},
        {"name": "サブタスク3", "value": "実装", "inline": True}
    ]
)
```

---

## 🔧 既存コードへの統合

### Option 1: discord_reporter.py に統合

既存の `bushidan/discord_reporter.py` に `DiscordLLMMessenger` を統合する方法:

```python
# bushidan/discord_reporter.py に追加

from bushidan.discord_llm_messenger import DiscordLLMMessenger

class DiscordReporter:
    def __init__(self, ...):
        # 既存の初期化
        ...
        # LLMメッセンジャーを追加
        self.llm_messenger = None

    async def initialize(self):
        # 既存の初期化
        ...
        # LLMメッセンジャー初期化
        self.llm_messenger = DiscordLLMMessenger()

    async def report_delegation(self, task_id, from_agent, to_agent, reason):
        """委譲報告（エージェント別アカウントで送信）"""
        if self.llm_messenger:
            # from_agent のアカウントで送信
            await self.llm_messenger.send_as_agent(
                from_agent.lower(),  # "shogun", "karo" など
                f"📤 {to_agent}へ委譲: {reason}"
            )

    async def report_progress(self, task_id, message, agent="shogun"):
        """進捗報告（エージェント別アカウントで送信）"""
        if self.llm_messenger:
            await self.llm_messenger.send_status_update(
                agent.lower(),
                "進行中",
                message
            )

    async def report_completion(self, task_id, result, agent="shogun"):
        """完了報告（エージェント別アカウントで送信）"""
        if self.llm_messenger:
            await self.llm_messenger.send_status_update(
                agent.lower(),
                "完了",
                f"処理完了: {result[:100]}..."
            )
```

### Option 2: Shogun/Karo/Taisho に直接統合

各エージェントクラスに直接統合する方法:

```python
# core/shogun.py

class Shogun:
    def __init__(self, orchestrator):
        # 既存の初期化
        ...
        self.llm_messenger = None

    async def initialize(self):
        # 既存の初期化
        ...
        # LLMメッセンジャー初期化
        from bushidan.discord_llm_messenger import DiscordLLMMessenger
        self.llm_messenger = DiscordLLMMessenger()

    async def process_task(self, task):
        # タスク受領通知
        if self.llm_messenger:
            await self.llm_messenger.send_as_agent(
                "shogun",
                f"🎌 将軍、任務受領: {task.content[:50]}..."
            )

        # 既存の処理
        result = await self._process_task_internal(task)

        # 完了通知
        if self.llm_messenger:
            await self.llm_messenger.send_status_update(
                "shogun",
                "完了",
                f"任務完了: {result['status']}"
            )

        return result
```

---

## 📊 エージェントIDとアカウントの対応

| エージェントID | 名前 | モデル | 絵文字 | 色 |
|---------------|------|--------|--------|-----|
| `shogun` | 将軍 (Shogun) | Claude Sonnet 4.6 | 🎌 | Purple |
| `gunshi` | 軍師 (Gunshi) | Qwen3-Coder-Next | 📋 | Blue |
| `karo` | 家老 (Karo) | Gemini 3 Flash | 👔 | Green |
| `taisho` | 大将 (Taisho) | Local Qwen3 | ⚔️ | Red |
| `yohei` | 傭兵 (Yohei) | Kimi K2.5 | 🗡️ | Orange |
| `kengyo` | 検校 (Kengyo) | Kimi Vision | 👁️ | Teal |
| `groq` | 足軽-Groq (Ashigaru) | Llama 3.3 70B | ⚡ | Gray |

---

## 🧪 テスト方法

### 1. アカウント一覧確認

```bash
python maintenance/setup_discord_llm_accounts.py --list
```

### 2. テストメッセージ送信

```bash
python maintenance/setup_discord_llm_accounts.py --test
```

### 3. カスタムテスト

```python
import asyncio
from bushidan.discord_llm_messenger import DiscordLLMMessenger

async def test():
    async with DiscordLLMMessenger() as messenger:
        # 将軍
        await messenger.send_as_agent("shogun", "🎌 テストメッセージ from 将軍")
        await asyncio.sleep(1)

        # 軍師
        await messenger.send_as_agent("gunshi", "📋 テストメッセージ from 軍師")
        await asyncio.sleep(1)

        # 家老
        await messenger.send_as_agent("karo", "👔 テストメッセージ from 家老")

asyncio.run(test())
```

---

## ⚙️ 追加設定（オプション）

### ウェブフックの再作成

別のチャンネルにウェブフックを作成したい場合:

```bash
# 新しいチャンネルIDを取得して実行
python maintenance/setup_discord_llm_accounts.py --create-webhooks <新しいチャンネルID>
```

### 設定ファイルのバックアップ

```bash
cp config/discord_llm_accounts.json config/discord_llm_accounts.backup.json
```

---

## 🔍 トラブルシューティング

### エラー: 設定ファイルが見つからない

```bash
# 設定ファイルの存在確認
ls -la config/discord_llm_accounts.json

# なければ再作成
python maintenance/setup_discord_llm_accounts.py --create-webhooks <チャンネルID>
```

### エラー: ウェブフック送信失敗

1. ウェブフックが削除されていないか確認
2. Botの権限を確認（「ウェブフックの管理」「メッセージを送信」）
3. チャンネルがアーカイブされていないか確認

### エラー: レート制限

メッセージ送信間隔を調整:

```python
await messenger.send_as_agent("shogun", "メッセージ1")
await asyncio.sleep(1)  # 1秒待機
await messenger.send_as_agent("karo", "メッセージ2")
```

---

## 📝 次のステップ

### 推奨される統合順序

1. **テスト送信** - `--test` で動作確認
2. **discord_reporter.py に統合** - 既存のレポート機能と統合
3. **各エージェントクラスに統合** - Shogun → Karo → Taisho の順
4. **本番環境で動作確認** - 実際のタスク処理で確認

### 将来の拡張案

- **スレッド自動作成** - タスクごとに専用スレッド
- **リアクション機能** - ユーザーがリアクションで承認/拒否
- **進捗バー表示** - Embed でビジュアル化
- **エラー通知** - 重要なエラーのみ @mention

---

## ✅ 完了チェックリスト

- [x] Discord LLMアカウント作成
- [x] 設定ファイル生成
- [x] メッセンジャークラス作成
- [ ] discord_reporter.py への統合
- [ ] Shogun クラスへの統合
- [ ] Karo クラスへの統合
- [ ] Taisho クラスへの統合
- [ ] 本番環境での動作確認

---

すべての準備が完了しています！あとは既存のコードに統合するだけです。
