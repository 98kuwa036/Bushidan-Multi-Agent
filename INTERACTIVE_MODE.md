# インタラクティブモード ガイド

武士団マルチエージェントシステムの**インタラクティブモード**では、各作業ステップでユーザーの承認を求め、軌道修正が可能な完全対話型のタスク実行を実現します。

## 📋 目次

1. [概要](#概要)
2. [機能](#機能)
3. [セットアップ](#セットアップ)
4. [使い方](#使い方)
5. [設定のカスタマイズ](#設定のカスタマイズ)
6. [仕組み](#仕組み)
7. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### インタラクティブモードとは

従来の武士団システムは、タスクを受け取った後は自動的に実行を進めていました。インタラクティブモードでは、**各重要な作業ステップの前にユーザーの承認を求め**、ユーザーがリアルタイムで指示を出せるようになります。

### 主な特徴

- **ステップごとの承認**: ファイル作成、Git コミット、エージェント委譲など、重要な作業の前に確認
- **リアクションベースの操作**: 👍（承認）、👎（却下）、🔄（やり直し）、✏️（修正）
- **自然言語での指示**: コメントで「test2.pyに変更して」などの軌道修正が可能
- **エージェントとの会話**: 各エージェントがユーザーの返信を読んで行動を調整
- **タイムアウト処理**: 応答がない場合も安全に処理（作業種別により自動承認/却下）
- **戦国武将風の会話**: 各エージェントが武将として振る舞い、格式ある口調で報告
- **詳細なMCP名表示**: 絵文字だけでなく、MCP名や操作内容を明示的に表示

---

## 機能

### 1. 承認が必要な作業

デフォルトで以下の作業に承認が必要です（`config/interactive_config.yaml`で変更可能）:

| 作業種別 | 説明 | デフォルト | タイムアウト動作 |
|---------|------|-----------|----------------|
| **delegation** | エージェント間の委譲 | ✅ 必要 | 自動承認 |
| **file_create** | ファイル作成 | ✅ 必要 | 自動承認 |
| **file_edit** | ファイル編集 | ✅ 必要 | 自動承認 |
| **file_delete** | ファイル削除（危険） | ✅ 必要 | **自動却下** |
| **git_commit** | Git コミット | ✅ 必要 | 自動承認 |
| **git_push** | Git プッシュ（危険） | ✅ 必要 | **自動却下** |
| **command_execution** | シェルコマンド実行 | ✅ 必要 | **自動却下** |
| **database_mutation** | DB変更（危険） | ✅ 必要 | **自動却下** |

### 2. 承認方法

#### リアクションボタン

各承認メッセージには自動的にリアクションボタンが追加されます:

- **👍 承認** - 作業を続行
- **👎 却下** - 作業を中止
- **🔄 やり直し** - 再試行
- **✏️ 修正** - 修正指示を出す（コメントで詳細を記述）

#### コメントによる指示

スレッドにコメントを書くことで、以下のような指示が可能です:

```
承認                    → 作業を続行
ok / yes / いいよ       → 承認と同義

却下 / やめて           → 作業を中止
no / stop / ダメ        → 却下と同義

やり直し / retry        → 再試行

test2.pyに変更して      → ファイル名を修正
コミットメッセージを変更 → Git コミットメッセージを修正
```

---

## セットアップ

### 1. 環境変数の設定

`.env` ファイルに以下を追加:

```bash
# インタラクティブモードを有効化
INTERACTIVE_MODE=true
```

### 2. Discord Bot の権限確認

インタラクティブモードには以下の Discord 権限が必要です:

- ✅ Send Messages
- ✅ Embed Links
- ✅ Create Public Threads
- ✅ Send Messages in Threads
- ✅ Manage Webhooks
- ✅ **Read Message History** ← 必須！
- ✅ **Add Reactions** ← 必須！

#### Bot を再招待する

権限を追加するため、以下のURLでBotを再招待してください:

```
https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=309774526528&scope=bot
```

※ `YOUR_CLIENT_ID` はあなたのBotのクライアントIDに置き換えてください
※ 既存のBotがいても問題ありません（権限が自動更新されます）

### 3. サービスの再起動

```bash
sudo systemctl restart bushidan-discord.service
```

---

## 使い方

### 基本的な流れ

#### 例: ファイル作成タスク

```
1. ユーザー: @Bushidan test.pyを作成してください

2. 🎌 将軍 (Shogun): ⚔️ 合戦開始
   我が名は将軍なり。
   → 任務内容、戦略、武士団配置を表示

3. 🔗 LangGraph Router: 📊 タスク分析完了
   LangGraph、起動いたしました。
   → 複雑度、信頼度、タスク種別を分析

4. 🔗 LangGraph Router: 承認待ち
   采配の判断を仰ぎまする。

   🤝 任務委譲: LangGraph Router → 👔 家老 (Karo)
   💭 申し送り: 家老の采配に委譲

   [👍][👎][🔄][✏️]

   ユーザー: 👍 をクリック

5. 🔗 LangGraph Router: ✅ かたじけない。直ちに取り掛かる所存にて候

6. 👔 家老 (Karo): 戦術調整
   家老、罷り出でました。
   → タスクを分析し、大将に指示

7. 👔 家老 (Karo): 承認待ち
   殿、御下命を仰ぎまする。

   🤝 任務委譲: 家老 (Karo) → ⚔️ 大将 (Taisho)
   💭 申し送り: 実装タスクを大将に委譲

   [👍][👎][🔄]

   ユーザー: 👍 をクリック

8. 👔 家老 (Karo): ✅ はっ、直ちに。

9. ⚔️ 大将 (Taisho): 実装実行
   大将、推参仕りました。
   → ファイル作成準備

10. ⚔️ 大将 (Taisho): 承認待ち
    ご下命を賜りとうございまする。

    📄 新規書状作成: `test.py`
    📊 文字数: 87文字

    ```python
    def main():
        print("Hello")

    if __name__ == "__main__":
        main()
    ```

    [👍][👎][✏️]

    ユーザー: 👍 をクリック

11. ⚔️ 大将 (Taisho): ✅ 御意。ただちに実行つかまつる

12. ⚔️ 大将 (Taisho): 📄 新規書状作成
    👤 担当武将: ⚔️ 大将 (Taisho)
    📂 所在: `test.py`
    ファイルを作成いたしました (87 文字)

13. ⚔️ 大将 (Taisho): 🔧 MCP使用
    🔧 使用MCP: `filesystem` (ファイルシステム)
    ⚙️ 実行操作: `write_file`
    📝 パラメータ:
      • `path`: test.py
      • `content`: def main()...
    ✅ 実行結果: ファイル書き込み成功

14. ⚔️ 大将 (Taisho): 承認待ち
    ご下命を賜りとうございまする。

    📜 記録を後世に: `Implement: test.pyを作成してください`
    📁 対象ファイル数: 1件
    📝 対象:
      • `test.py`

    [👍][👎]

    ユーザー: 👍 をクリック

15. ⚔️ 大将 (Taisho): ✅ 御意。ただちに実行つかまつる

16. ⚔️ 大将 (Taisho): 💾 記録作成
    Gitコミット完了

17. 🎌 将軍 (Shogun): ✅ 任務完了
    御意。早速進めて参りまする。
    → 処理時間: 15.2秒
    → 関与武将: 🎌 将軍 🔗 LangGraph 👔 家老 ⚔️ 大将
```

---

### 軌道修正の例

#### 例: ファイル名の変更

```
⚔️ 大将 (Taisho): ご下命を賜りとうございまする。
📄 新規書状作成: `test.py`
[👍][👎][✏️]

ユーザー: test2.pyに変更して

⚔️ 大将 (Taisho): ✏️ 御意。test2.py にて進めまする

⚔️ 大将 (Taisho): ご下命を賜りとうございまする。
📄 新規書状作成: `test2.py`
[👍][👎]

ユーザー: 👍

⚔️ 大将 (Taisho): ✅ 御意。ただちに実行つかまつる

⚔️ 大将 (Taisho): 📄 新規書状作成
test2.py を作成いたしました
```

#### 例: 作業の却下

```
⚔️ 大将 (Taisho): ご下命を賜りとうございまする。
🔥 書状破棄（危険）: `important.py`
[👍][👎]

ユーザー: 👎 または「やめて」

⚔️ 大将 (Taisho): ❌ 承知つかまつった。直ちに中止いたす

🎌 将軍 (Shogun): 任務を中止いたしました
```

---

## 戦国武将風の会話

### 概要

各エージェントが戦国武将として格式ある口調で会話します。これにより、武士団システムの世界観をより深く体験できます。

### エージェントごとの口調

| エージェント | 挨拶 | 承認リクエスト | 承認時 | 却下時 |
|------------|------|---------------|--------|--------|
| **🎌 将軍** | 我が名は将軍なり | 上様、お伺い申し上げまする | かたじけない。直ちに取り掛かる所存にて候 | 承知つかまつった。直ちに中止いたす |
| **🧠 軍師** | 軍師、参上仕りました | 殿、お伺い立てまする | 御意。早速進めて参りまする | 承知仕りました。別の策を練りまする |
| **👔 家老** | 家老、罷り出でました | 殿、御下命を仰ぎまする | はっ、直ちに | 承知つかまつりました |
| **⚔️ 大将** | 大将、推参仕りました | ご下命を賜りとうございまする | 御意。ただちに実行つかまつる | 承知。作業を取り止めまする |
| **👁️ 検校** | 検校、参上つかまつった | ご検分いかがいたしましょうか | かしこまりました。検証に入りまする | 承知。検証を見送りまする |

### MCP名の明示

従来の絵文字のみの表示から、**MCP名と操作内容を明示的に表示**するように改善しました。

#### 改善前
```
⚔️ 🔧 filesystem - write_file
```

#### 改善後
```
⚔️ 大将 (Taisho)

🔧 使用MCP: `filesystem` (ファイルシステム)
⚙️ 実行操作: `write_file`
📝 パラメータ:
  • `path`: test.py
  • `content`: def main()...
✅ 実行結果: ファイル書き込み成功
```

### 主なMCPの日本語名

| MCP名 | 日本語表示 |
|-------|-----------|
| `filesystem` | ファイルシステム |
| `git` | Git管理 |
| `sequential_thinking` | 段階的思考 |
| `brave_search` | Brave検索 |
| `fetch` | Web取得 |
| `memory` | 記憶管理 |
| `postgres` | PostgreSQL |
| `github` | GitHub |
| `slack` | Slack |
| `puppeteer` | Puppeteer（ブラウザ操作）|

### 戦国武将風の無効化

通常の会話に戻したい場合は、`config/interactive_config.yaml` で無効化できます:

```yaml
interactive_mode:
  speaking_style:
    enabled: false  # 戦国武将風を無効化
```

---

## 設定のカスタマイズ

### config/interactive_config.yaml

詳細な設定は `config/interactive_config.yaml` で変更できます:

```yaml
interactive_mode:
  # インタラクティブモード有効化
  enabled: true

  # 承認が必要な作業種別
  approval_required:
    delegation: true       # エージェント間の委譲
    file_create: true      # ファイル作成
    file_edit: true        # ファイル編集
    file_delete: true      # ファイル削除（危険）
    git_commit: true       # Gitコミット
    git_push: true         # Gitプッシュ（危険）
    command_execution: true # シェルコマンド実行
    database_mutation: true # データベース操作（危険）

  # タイムアウト設定（秒）
  approval_timeout: 300  # 5分

  # タイムアウト時の動作
  timeout_actions:
    # 安全な操作：自動承認
    file_create: "auto_approve"
    file_edit: "auto_approve"
    git_commit: "auto_approve"
    delegation: "auto_approve"

    # 危険な操作：自動却下
    file_delete: "auto_reject"
    git_push: "auto_reject"
    command_execution: "auto_reject"
    database_mutation: "auto_reject"

  # リアクション絵文字
  reaction_emojis:
    approve: "👍"
    reject: "👎"
    retry: "🔄"
    modify: "✏️"

  # キーワード認識
  keywords:
    approve:
      - "承認"
      - "ok"
      - "OK"
      - "approve"
      - "yes"
      - "いいよ"
      - "よし"
      - "go"
      - "進めて"

    reject:
      - "却下"
      - "no"
      - "NO"
      - "reject"
      - "ダメ"
      - "やめて"
      - "中止"
      - "stop"

    retry:
      - "やり直し"
      - "retry"
      - "再試行"
      - "もう一度"
      - "再度"

    modify:
      - "変更"
      - "修正"
      - "modify"
      - "change"
      - "編集"
```

### カスタマイズ例

#### 1. 委譲は自動で進める（承認不要）

```yaml
approval_required:
  delegation: false  # 自動で進む
```

#### 2. タイムアウト時間を10分に変更

```yaml
approval_timeout: 600  # 10分
```

#### 3. ファイル作成も自動承認

```yaml
approval_required:
  file_create: false  # 自動で作成
```

#### 4. カスタムキーワードを追加

```yaml
keywords:
  approve:
    - "いいね"
    - "よろしく"
    - "実行して"
```

---

## 仕組み

### アーキテクチャ

```
BushidanDiscordBot (メインBot)
  ↓ on_message / on_reaction_add
ApprovalManager (承認管理)
  ↓ request_approval()
DiscordAgentReporter (報告・承認インターフェース)
  ↓
各エージェント (Shogun, Gunshi, Karo, Taisho, ...)
  ↓ 作業前に承認リクエスト
Discordスレッド（ユーザーとの対話）
```

### 主要コンポーネント

#### 1. ApprovalManager (`bushidan/approval_manager.py`)

- 承認リクエストの管理
- リアクション・コメントの検出と解釈
- タイムアウト処理
- 承認結果の通知

#### 2. InstructionParser (`utils/instruction_parser.py`)

- ユーザーの自然言語指示を解析
- キーワードマッチング（「承認」「却下」など）
- 修正指示の抽出（「test2.pyに変更」→ `{"filename": "test2.py"}`）

#### 3. DiscordAgentReporter (`bushidan/discord_reporter.py`)

- `request_approval()`: 承認をリクエスト
- `wait_for_user_response()`: ユーザーの返信を待つ（会話機能）
- Webhookによるマルチエージェント報告

#### 4. 各エージェント

- **Shogun**: 委譲前に承認リクエスト
- **LangGraph Router**: ルーティング前に承認リクエスト
- **Karo**: 大将への委譲前に承認リクエスト
- **Taisho**: ファイル作成・Git コミット前に承認リクエスト

### 承認フロー

```
1. エージェントが作業前に reporter.request_approval() を呼び出し
2. ApprovalManager が承認メッセージを投稿
3. リアクションボタンを自動追加 [👍][👎][🔄][✏️]
4. asyncio.wait_for() でユーザーの応答を待機（タイムアウト付き）
5. リアクション or コメントを検出
6. InstructionParser が指示を解釈
7. 承認結果（approved/rejected/modified）を返す
8. エージェントが結果に応じて行動
```

---

## トラブルシューティング

### 1. 承認メッセージが表示されない

**確認事項**:

1. `.env` で `INTERACTIVE_MODE=true` になっているか
2. サービスが再起動されたか
   ```bash
   sudo systemctl restart bushidan-discord.service
   ```
3. ログを確認
   ```bash
   journalctl -u bushidan-discord.service -f | grep -i approval
   ```

### 2. リアクションが機能しない

**原因**: BotにReaction権限がない

**解決策**:
- Bot に "Add Reactions" 権限を追加
- 招待URLで再招待:
  ```
  https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=309774526528&scope=bot
  ```

### 3. コメントが認識されない

**原因**: "Read Message History" 権限がない、またはキーワードが一致しない

**解決策**:
1. Bot に "Read Message History" 権限を追加
2. `config/interactive_config.yaml` の `keywords` セクションを確認
3. ログでユーザーメッセージの検出を確認
   ```bash
   journalctl -u bushidan-discord.service | grep -i "thread message"
   ```

### 4. タイムアウトが頻発する

**原因**: タイムアウト時間が短すぎる

**解決策**:
`config/interactive_config.yaml` でタイムアウトを延長:
```yaml
approval_timeout: 600  # 10分に変更
```

### 5. 承認をスキップしたい作業がある

**解決策**:
`config/interactive_config.yaml` で該当作業を `false` に:
```yaml
approval_required:
  delegation: false  # 委譲は自動で進む
  git_commit: false  # コミットも自動で進む
```

### 6. ログの確認方法

#### リアルタイムログ
```bash
journalctl -u bushidan-discord.service -f
```

#### 承認関連のログのみ
```bash
journalctl -u bushidan-discord.service | grep -i approval
```

#### エラーログ
```bash
journalctl -u bushidan-discord.service | grep -i error
```

### 7. 設定の確認

Pythonコンソールで設定を確認:
```python
import yaml

with open('config/interactive_config.yaml') as f:
    config = yaml.safe_load(f)

print(config['interactive_mode']['enabled'])
print(config['interactive_mode']['approval_required'])
```

---

## よくある質問 (FAQ)

### Q1: インタラクティブモードを一時的に無効化したい

**A**: `.env` で `INTERACTIVE_MODE=false` に設定して再起動:
```bash
# .env
INTERACTIVE_MODE=false

# 再起動
sudo systemctl restart bushidan-discord.service
```

### Q2: 特定のタスクだけ承認をスキップできる？

**A**: 現在は作業種別ごとの設定のみ対応しています。タスクごとのスキップは今後の拡張予定です。

### Q3: 複数人で承認できる？

**A**: 現在は最初に応答したユーザーの指示が有効です。投票機能は今後の拡張予定です。

### Q4: エージェントのアバター画像を変更したい

**A**: `bushidan/discord_reporter.py` の `AGENT_CONFIG` で画像URLを設定できます（将来実装予定）。

### Q5: タイムアウト後の通知を無効化したい

**A**: `config/interactive_config.yaml` の `messages` セクションでメッセージをカスタマイズできます。

---

## 今後の拡張予定

- [ ] タスクごとの承認スキップ設定
- [ ] 投票機能（複数ユーザー対応）
- [ ] エージェントアバター画像のカスタマイズ
- [ ] 承認履歴の可視化
- [ ] エージェントの提案（A案/B案の選択）
- [ ] 音声入力対応（Discord Voice）
- [ ] カスタムリアクション絵文字

---

## 参考リンク

- [武士団マルチエージェントシステム](README.md)
- [Discord セットアップガイド](DISCORD_SETUP.md)
- [実装計画書](/home/claude/.claude/plans/goofy-inventing-diffie.md)

---

## ライセンス

武士団マルチエージェントシステムに準拠

---

**質問・バグ報告**: GitHub Issues または Discord の該当チャンネルまで
