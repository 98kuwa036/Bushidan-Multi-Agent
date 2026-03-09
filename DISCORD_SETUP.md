# Discord マルチエージェント報告システム セットアップガイド

## 概要

武士団システムでは、各エージェント（将軍、軍師、家老、大将、検校）が **個別のアカウント** としてDiscordスレッドに投稿し、**会話形式** でタスク処理の様子を報告します。

```
スレッド: 任務: ファイルを作成してください
├─ 🎌 将軍 (Shogun): ⚔️ 合戦開始
│  └─ 任務内容: ファイルを作成してください
│  └─ 戦略: LangGraph Router による最適ルーティング
│
├─ 🔗 LangGraph Router: 📊 タスク分析完了
│  └─ 複雑度: MEDIUM
│  └─ 信頼度: 85%
│
├─ 🎌 将軍: 🔄 任務委譲
│  └─ 🎌 将軍 → 👔 家老
│  └─ 委譲経路: 🎌 → 👔
│
├─ 👔 家老 (Karo): 👔 家老 - 戦術調整
│  └─ タスクを分析し、大将に指示を出します
│
├─ 👔 家老: 🔄 任務委譲
│  └─ 👔 家老 → ⚔️ 大将
│  └─ 委譲経路: 🎌 → 👔 → ⚔️
│
├─ ⚔️ 大将 (Taisho): ⚔️ 大将 - 実装実行
│  └─ 📍 ステップ: 実装準備
│  └─ タスク: ファイルを作成してください
│  └─ フォールバックチェーン: Kimi K2.5 → Qwen3 → Qwen3+ → Gemini 3
│
├─ ⚔️ 大将: 📍 ステップ: Tier 1: Kimi K2.5 傭兵
│  └─ 🥇 128Kコンテキスト、並列実行可能な傭兵で処理中...
│
├─ ⚔️ 大将: ✅ Tier 1 (Kimi K2.5) 成功
│
├─ ⚔️ 大将: 🔧 MCP: `filesystem` - `write_file`
│  └─ パラメータ: path=test.py
│  └─ 結果: ファイル作成完了
│
├─ ⚔️ 大将: 📄 成果物作成
│  └─ 担当: ⚔️ 大将
│  └─ パス: `test.py`
│  └─ ファイルを作成しました (245 文字)
│
├─ ⚔️ 大将: 🔧 MCP: `git` - `commit`
│  └─ パラメータ: files=1, message=Implement: ファイルを作成
│  └─ 結果: ✅ コミット成功
│
└─ 🎌 将軍: ✅ 任務完了
   └─ 処理時間: 12.5秒
   └─ 関与エージェント: 🎌 👔 ⚔️
```

## セットアップ方法

### オプション1: Bot権限を追加（推奨）

#### ステップ1: Discord Developer Portalで権限を設定

1. https://discord.com/developers/applications にアクセス
2. あなたのBotアプリケーションを選択
3. **OAuth2 → URL Generator** に移動
4. **SCOPES** で `bot` を選択
5. **BOT PERMISSIONS** で以下を選択:
   - ✅ **Send Messages**
   - ✅ **Embed Links**
   - ✅ **Create Public Threads**
   - ✅ **Send Messages in Threads**
   - ✅ **Manage Webhooks** ← **これが最重要！**

6. 生成されたURLをコピー（または以下の形式で作成）:

```
https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=309774522368&scope=bot
```

7. このURLでBotを **再招待** してください
   - 既存のBotがいても問題ありません
   - 権限が自動的に更新されます

#### ステップ2: サービスを再起動

```bash
sudo systemctl restart bushidan-discord.service
```

#### ステップ3: 動作確認

Discordで `@Bushidan ファイルを作成してください` とメンションすると、スレッドが作成され、各エージェントが順次報告を投稿します。

---

### オプション2: 手動でWebhookを作成

Bot権限を変更できない場合、手動でWebhookを作成できます。

#### ステップ1: DiscordでWebhookを作成

1. Discordサーバーの設定を開く
2. **連携サービス** → **ウェブフック** → **新しいウェブフック** をクリック
3. 名前を `Bushidan Multi-Agent` に設定
4. 使用するチャンネルを選択
5. **ウェブフックURLをコピー** をクリック

URLは以下のような形式です:
```
https://discord.com/api/webhooks/123456789012345678/abcdefghijklmnopqrstuvwxyz
```

#### ステップ2: 環境変数に設定

`.env` ファイルに以下を追加:

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_URL_HERE
```

#### ステップ3: サービスを再起動

```bash
sudo systemctl restart bushidan-discord.service
```

**注意**: この方法では、Webhookを作成したチャンネルでのみマルチエージェント報告が動作します。

---

## エージェント一覧

各エージェントは以下の役割で投稿します:

| エージェント | 絵文字 | 表示名 | 役割 | 色 |
|------------|--------|--------|------|-----|
| Shogun | 🎌 | 将軍 (Shogun) | 戦略的意思決定 | Dark Red |
| Gunshi | 🧠 | 軍師 (Gunshi) | PDCA作戦立案 | Indigo |
| Karo | 👔 | 家老 (Karo) | 戦術調整 | Dark Green |
| Taisho | ⚔️ | 大将 (Taisho) | 実装実行 | Orange Red |
| Kengyo | 👁️ | 検校 (Kengyo) | ビジュアル検証 | Dark Violet |
| Ashigaru | 👣 | 足軽 (Ashigaru) | MCP実行 | Gray |
| LangGraph | 🔗 | LangGraph Router | タスク分析・ルーティング | Dodger Blue |
| Gemini | 🤖 | Gemini | 自律実行 | Blue |
| Groq | ⚡ | Groq | 即応回答 | Yellow |

## 報告の種類

### 1. 合戦開始報告 (`report_battle_start`)

タスク開始時に将軍が投稿:
- 任務内容
- 戦略
- 武士団配置（5層階層）

### 2. タスク分析報告 (`report_routing_analysis`)

LangGraph Routerが投稿:
- 複雑度（SIMPLE/MEDIUM/COMPLEX/STRATEGIC）
- 信頼度スコア
- タスク種別（複数ステップ/アクション/Q&A）

### 3. 任務委譲報告 (`report_delegation`)

エージェント間の委譲時に投稿:
- 委譲元 → 委譲先
- 委譲理由
- 委譲経路（全体の流れ）

### 4. 詳細進捗報告 (`report_detailed_progress`)

各ステップの実行時に投稿:
- ステップ名
- 詳細説明
- プログレスバー（オプション）

### 5. MCP使用報告 (`report_mcp_usage`)

MCPツール使用時に投稿:
- MCP名（filesystem, git, sequential_thinking など）
- 操作名（write_file, commit など）
- パラメータ
- 結果

### 6. 成果物作成報告 (`report_artifact_created`)

ファイル・コミット作成時に投稿:
- 成果物タイプ（file, directory, commit など）
- パス
- 説明

### 7. 完了報告 (`report_complete`)

タスク完了時に投稿:
- 処理時間
- 関与エージェント
- 最終結果

---

## トラブルシューティング

### エラー: `403 Forbidden: Missing Permissions`

**原因**: BotにWebhook権限がありません

**解決策**:
1. オプション1を試す（Bot権限追加）
2. または オプション2（手動Webhook）を使用

### エラー: `No permission to create webhook in #channel`

**原因**: BotがチャンネルでのWebhook作成権限を持っていません

**解決策**:
1. Botの役割（Role）を確認し、「Webhookの管理」権限を追加
2. チャンネルの権限設定でBotに明示的に権限を付与

### Webhookが動作しない

**確認事項**:
1. `.env` の `DISCORD_WEBHOOK_URL` が正しいか
2. WebhookのURLが期限切れでないか
3. Webhookが削除されていないか

**ログ確認**:
```bash
journalctl -u bushidan-discord.service -f
```

### 報告が表示されない

**確認事項**:
1. サービスが正常に動作しているか確認:
   ```bash
   systemctl status bushidan-discord.service
   ```

2. ログでWebhook関連のエラーを確認:
   ```bash
   journalctl -u bushidan-discord.service | grep -i webhook
   ```

3. タスクスレッドが正しく作成されているか確認

---

## 実装の詳細

### アーキテクチャ

```
BushidanDiscordBot (メインBot)
  ↓
WebhookPoolManager (Webhook管理)
  ↓
DiscordAgentReporter (エージェント報告)
  ↓
各エージェント (Shogun, Gunshi, Karo, Taisho, ...)
  ↓
Discordスレッド（会話形式で表示）
```

### Webhook vs Bot メッセージ

- **将軍 (Shogun)**: 実際のBot アカウント（タスク受領・完了）
- **その他のエージェント**: Webhook（名前・アイコンを変更して投稿）

これにより、複数のユーザーが会話しているように見えます。

### スレッド管理

各タスクに専用スレッドを作成:
- スレッド名: `任務: {タスク内容の最初の50文字}`
- 自動アーカイブ: 1時間後
- 反応（リアクション）でステータス表示:
  - ⏳ 処理中
  - ✅ 成功
  - ❌ 失敗

---

## 今後の拡張

- [ ] エージェントごとのアバター画像（Webhookに画像URL設定）
- [ ] 進捗バーのリアルタイム更新（メッセージ編集）
- [ ] エージェント間の「会話」（意見交換）
- [ ] 並列タスクの視覚化
- [ ] タスク完了サマリーの自動生成

---

## 参考

- [Discord Webhook ドキュメント](https://discord.com/developers/docs/resources/webhook)
- [discord.py ドキュメント](https://discordpy.readthedocs.io/)
- [武士団マルチエージェントシステム](../README.md)
