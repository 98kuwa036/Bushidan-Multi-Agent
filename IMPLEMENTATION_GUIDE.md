# インタラクティブモード実装ガイド（引き継ぎ）

## 📊 現在の実装状況

### ✅ 完成したもの（Phase 1完了）

1. **config/interactive_config.yaml** ✅
   - 詳細な設定ファイル（承認設定、タイムアウト、キーワードなど）
   - 作業種別ごとのタイムアウト動作設定

2. **bushidan/approval_manager.py** ✅ (600行)
   - `ApprovalManager` クラス: 承認システムのコア
   - `ApprovalRequest` クラス: 承認リクエストの状態管理
   - `ApprovalResult` クラス: 承認結果
   - `InteractiveConfig` クラス: 設定読み込み
   - リアクション処理: `handle_reaction()`
   - コメント処理: `handle_comment()`
   - タイムアウト処理: 作業種別で自動承認/却下を切り替え

3. **utils/instruction_parser.py** ✅ (400行)
   - `InstructionParser` クラス: 自然言語指示の解釈
   - `ParsedInstruction` クラス: 解析結果
   - キーワードベース解析（承認/却下/やり直し/修正）
   - LLMベース解析（Gemini/Groq/Claude対応）
   - ファイル名変更などの修正内容抽出

4. **bushidan/discord_bot.py** ✅ (追加・修正)
   - `ApprovalManager` インスタンス作成
   - `thread_id_to_task_id` マッピング追加
   - `on_message()` 拡張: Webhookメッセージ無視、スレッドメッセージ処理
   - `_handle_thread_message()` 追加: スレッド内ユーザーメッセージを処理
   - `on_reaction_add()` 追加: リアクション監視
   - Thread ID マッピングの登録・削除

---

## ⚠️ 未実装（Phase 2-3）

残り **5つの大規模ファイル** の修正が必要です。合計 **約2500行** の追加・修正が必要です。

### 1. bushidan/discord_reporter.py（必須）

**修正箇所:**

#### TaskThread クラスの拡張
```python
@dataclass
class TaskThread:
    # 既存フィールド
    ...

    # 新規追加
    approval_manager: Optional[ApprovalManager] = None  # 承認マネージャー
    paused: bool = False  # 一時停止フラグ
    pending_approval: Optional[ApprovalRequest] = None  # 現在の承認リクエスト
    user_messages: List[discord.Message] = field(default_factory=list)  # ユーザーメッセージ履歴
```

#### DiscordAgentReporter クラスに新しいメソッド追加

**`request_approval()` メソッド:**
```python
async def request_approval(
    self,
    task_id: str,
    action_type: str,
    action_details: Dict[str, Any],
    timeout: Optional[int] = None
) -> ApprovalResult:
    """
    作業前に承認をリクエスト

    Args:
        task_id: タスクID
        action_type: 作業種別（file_create, git_commit, delegation など）
        action_details: 作業詳細
        timeout: タイムアウト時間（秒）

    Returns:
        ApprovalResult: 承認結果
    """
    thread_data = self.bot.task_threads.get(task_id)
    if not thread_data:
        return ApprovalResult(status=ApprovalStatus.APPROVED, message="スレッドなし")

    # ApprovalManagerを使用
    result = await self.bot.approval_manager.request_approval(
        task_id=task_id,
        agent_name=thread_data.active_agent,
        action_type=action_type,
        action_details=action_details,
        thread=thread_data.thread,
        timeout=timeout
    )

    return result
```

**`wait_for_user_response()` メソッド:**
```python
async def wait_for_user_response(
    self,
    task_id: str,
    prompt_message: str,
    timeout: int = 300
) -> Optional[str]:
    """
    エージェントがユーザーの返信を待つ

    Args:
        task_id: タスクID
        prompt_message: プロンプトメッセージ
        timeout: タイムアウト時間（秒）

    Returns:
        ユーザーの返信内容、またはNone
    """
    thread_data = self.bot.task_threads.get(task_id)
    if not thread_data:
        return None

    # プロンプトメッセージを投稿
    await self.webhook_manager.send_as_agent(
        channel=thread_data.thread.parent,
        agent_name=thread_data.active_agent,
        message=prompt_message,
        thread=thread_data.thread
    )

    # ユーザーの返信を待つ（ApprovalManagerを使用）
    result = await self.bot.approval_manager.request_approval(
        task_id=task_id,
        agent_name=thread_data.active_agent,
        action_type="clarify",
        action_details={"prompt": prompt_message},
        thread=thread_data.thread,
        timeout=timeout
    )

    if result.status == ApprovalStatus.MODIFIED:
        return result.user_instruction.get("raw_message", "")

    return None
```

---

### 2. core/taisho.py（重要）

**ファイル作成時の承認統合（line ~1010付近）:**

```python
# ファイル作成前（_save_files_from_implementation メソッド内）
if content:
    # 承認リクエスト
    reporter = self.orchestrator.get_reporter()
    task_id = self.orchestrator.get_task_id()

    if reporter and task_id:
        approval = await reporter.request_approval(
            task_id,
            action_type="file_create",
            action_details={
                "filename": filename,
                "size": len(content),
                "preview": content[:200] + "..." if len(content) > 200 else content
            }
        )

        if approval.status == ApprovalStatus.REJECTED:
            logger.info(f"❌ ユーザーがファイル作成を却下: {filename}")
            continue  # このファイルをスキップ

        if approval.status == ApprovalStatus.MODIFIED:
            # ファイル名変更指示を解釈
            if approval.user_instruction and "filename" in approval.user_instruction:
                filename = approval.user_instruction["filename"]
                logger.info(f"✏️ ファイル名を変更: {filename}")

    # ファイル作成実行
    await self.filesystem_mcp.write_file(filename, content)
    files_created.append(filename)
    logger.info(f"📄 Created file: {filename}")

    # 成果物報告（既存コード）
    if reporter and task_id:
        await reporter.report_artifact_created(
            task_id,
            "file",
            filename,
            f"ファイルを作成しました ({len(content)} 文字)"
        )
```

**Gitコミット時の承認統合（line ~1137付近）:**

```python
# コミット前（_commit_changes メソッド内）
async def _commit_changes(self, task: ImplementationTask, result: Dict[str, Any]) -> None:
    """Commit changes using Git MCP"""

    if not self.git_mcp:
        return

    try:
        files_created = result.get("files_created", [])
        if not files_created:
            return

        commit_message = f"Implement: {task.content[:50]}\n\nGenerated by Taisho v{self.VERSION}"

        # 承認リクエスト
        reporter = self.orchestrator.get_reporter()
        task_id = self.orchestrator.get_task_id()

        if reporter and task_id:
            approval = await reporter.request_approval(
                task_id,
                action_type="git_commit",
                action_details={
                    "files": len(files_created),
                    "message": commit_message,
                    "files_list": ", ".join(files_created[:5])
                }
            )

            if approval.status == ApprovalStatus.REJECTED:
                logger.info("❌ ユーザーがGitコミットを却下")
                return

            if approval.status == ApprovalStatus.MODIFIED:
                # コミットメッセージ変更指示
                if approval.user_instruction and "message" in approval.user_instruction:
                    commit_message = approval.user_instruction["message"]
                    logger.info(f"✏️ コミットメッセージを変更")

        # コミット実行
        await self.git_mcp.add(files_created)
        await self.git_mcp.commit(commit_message)

        logger.info(f"✅ Changes committed: {len(files_created)} files")

        # MCP使用報告とコミット報告（既存コード）
        if reporter and task_id:
            await reporter.report_artifact_created(...)
            await reporter.report_mcp_usage(...)

    except Exception as e:
        logger.warning(f"⚠️ Git commit failed: {e}")
```

---

### 3. core/langgraph_router.py（重要）

**ルーティング決定時の承認（_route_decision メソッド内）:**

```python
async def _route_decision(self, state: TaskState) -> str:
    """Decide routing based on task analysis"""

    # 既存のルーティングロジック
    analysis = state.get("analysis", {})
    complexity = analysis.get("complexity", "medium")
    is_multi_step = analysis.get("is_multi_step", False)
    is_action_task = analysis.get("is_action_task", False)
    is_simple_qa = analysis.get("is_simple_qa", False)

    # ルート決定
    if is_simple_qa and not is_action_task:
        route = "groq_qa"
        target_agent = "groq"
        reason = "シンプルなQ&Aタスク、Groqで高速回答"
    elif is_multi_step and not is_action_task:
        route = "gemini_autonomous"
        target_agent = "gemini"
        reason = "複数ステップタスク、Gemini自律実行"
    elif is_action_task and not is_multi_step:
        route = "taisho_action"
        target_agent = "taisho"
        reason = "単一アクションタスク、大将が直接実行"
    else:
        route = "karo_default"
        target_agent = "karo"
        reason = "標準的な複雑度、家老が采配"

    # 承認リクエスト（インタラクティブモード）
    reporter = self.orchestrator.get_reporter()
    task_id = state.get("task_id", "")

    if reporter and task_id:
        approval = await reporter.request_approval(
            task_id,
            action_type="delegation",
            action_details={
                "from_agent": "langgraph",
                "to_agent": target_agent,
                "route": route,
                "reason": reason,
                "complexity": complexity
            }
        )

        if approval.status == ApprovalStatus.REJECTED:
            logger.info("❌ ユーザーがルーティングを却下")
            # フォールバックルート
            route = "karo_default"
            target_agent = "karo"

    logger.info(f"🔀 Route決定: {route} → {target_agent}")

    return route
```

---

### 4. core/karo.py（中優先度）

**大将への委譲時の承認（process_task メソッド内）:**

```python
async def process_task(self, task: Task) -> Dict[str, Any]:
    """タスク処理のメインメソッド"""

    # ... 既存の処理 ...

    # 大将に委譲する前
    reporter = self.orchestrator.get_reporter()
    task_id = self.orchestrator.get_task_id()

    if reporter and task_id:
        approval = await reporter.request_approval(
            task_id,
            action_type="delegation",
            action_details={
                "from_agent": "karo",
                "to_agent": "taisho",
                "reason": "実装タスクを大将に委譲"
            }
        )

        if approval.status == ApprovalStatus.REJECTED:
            logger.info("❌ ユーザーが委譲を却下")
            return {"status": "rejected", "message": "ユーザーによる却下"}

    # 大将に委譲
    result = await self.taisho.execute_implementation(impl_task)

    return result
```

---

### 5. core/shogun.py（中優先度）

**家老/軍師への委譲時の承認（process_task メソッド内）:**

```python
async def process_task(self, task: Task) -> Dict[str, Any]:
    """タスク処理のメインメソッド"""

    # ... 既存のルーティングロジック ...

    # 家老に委譲する前
    reporter = self.orchestrator.get_reporter()
    task_id = self.orchestrator.get_task_id()

    if reporter and task_id:
        approval = await reporter.request_approval(
            task_id,
            action_type="delegation",
            action_details={
                "from_agent": "shogun",
                "to_agent": "karo",
                "reason": "戦術レベルのタスク、家老に委譲"
            }
        )

        if approval.status == ApprovalStatus.REJECTED:
            logger.info("❌ ユーザーが委譲を却下")
            return {"status": "rejected", "message": "ユーザーによる却下"}

    # 家老に委譲
    result = await self.karo.process_task(task)

    return result
```

---

## 🚀 実装の順序（推奨）

### ステップ1: discord_reporter.py（最優先）
1. TaskThread クラス拡張
2. `request_approval()` メソッド追加
3. `wait_for_user_response()` メソッド追加

### ステップ2: core/taisho.py
1. ファイル作成前の承認
2. Gitコミット前の承認

### ステップ3: core/langgraph_router.py
1. ルーティング決定前の承認

### ステップ4: core/karo.py, core/shogun.py
1. 委譲時の承認

---

## 🧪 テスト方法

### 1. 基本動作確認

```bash
# サービスを再起動
sudo systemctl restart bushidan-discord.service

# ログ確認
journalctl -u bushidan-discord.service -f
```

### 2. Discord Botの権限更新

**必要な権限:**
- Send Messages
- Embed Links
- Create Public Threads
- Send Messages in Threads
- Manage Webhooks
- **Read Message History** ← 新規追加！
- **Add Reactions** ← 新規追加！

**招待URL（権限値: 309774526528）:**
```
https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=309774526528&scope=bot
```

### 3. テストシナリオ

#### シナリオ1: リアクションベース承認
```
1. Discordで: @Bushidan test.pyを作成して
2. スレッドが作成される
3. 大将が「test.pyを作成します」とメッセージ
4. リアクションボタン [👍][👎][🔄][✏️] が表示される
5. ユーザーが 👍 をクリック
6. 大将がファイル作成
7. 大将が「Gitコミットしますか？」とメッセージ
8. ユーザーが 👍 をクリック
9. コミット実行
```

#### シナリオ2: コメントによる修正
```
1. @Bushidan ファイルを作成して
2. 大将「example.pyを作成します」
3. ユーザーがコメント: "test2.pyに変更して"
4. 大将「承知しました。test2.pyで作成します」
5. ユーザーが 👍
6. ファイル作成
```

#### シナリオ3: 却下
```
1. @Bushidan ファイルを削除して
2. 大将「important.pyを削除します（危険な操作）」
3. ユーザーが 👎 をクリック
4. 大将「削除を中止しました」
```

---

## 🔧 設定ファイル

### .env に追加

```bash
# インタラクティブモード有効化
INTERACTIVE_MODE=true

# Discord Bot権限更新後のトークン（必要に応じて）
DISCORD_BOT_TOKEN=your_token_here
```

### config/interactive_config.yaml（既存）

デフォルト設定で動作しますが、カスタマイズ可能：

```yaml
interactive_mode:
  enabled: true

  approval_required:
    delegation: true      # 委譲時に承認
    file_create: true     # ファイル作成時に承認
    file_edit: true       # ファイル編集時に承認
    file_delete: true     # ファイル削除時に承認（危険）
    git_commit: true      # Gitコミット時に承認
    git_push: true        # Git Push時に承認（危険）

  approval_timeout: 300   # 5分

  timeout_actions:
    file_create: "auto_approve"   # 安全な操作は自動承認
    file_delete: "auto_reject"    # 危険な操作は自動却下
    git_push: "auto_reject"
```

---

## 📝 デバッグのヒント

### ログの確認

```bash
# リアルタイムログ
journalctl -u bushidan-discord.service -f

# 承認関連のみ
journalctl -u bushidan-discord.service | grep -i approval

# ユーザーメッセージ監視
journalctl -u bushidan-discord.service | grep -i "thread message"

# リアクション監視
journalctl -u bushidan-discord.service | grep -i reaction
```

### よくあるエラー

**エラー1: `403 Forbidden: Missing Permissions`**
- 原因: BotにWebhook/Reaction権限がない
- 解決: Bot権限を更新して再招待

**エラー2: `ApprovalManager not found`**
- 原因: approval_manager.py が読み込まれていない
- 解決: import文を確認、パスを確認

**エラー3: リアクションが動作しない**
- 原因: `on_reaction_add` イベントハンドラーが登録されていない
- 解決: discord_bot.py の実装を確認

---

## 📚 参考資料

- **計画書:** `/home/claude/.claude/plans/goofy-inventing-diffie.md`
- **Discord設定:** `DISCORD_SETUP.md`
- **既存実装:**
  - `bushidan/approval_manager.py` - 承認システムコア
  - `utils/instruction_parser.py` - 自然言語解釈
  - `bushidan/discord_bot.py` - メッセージ・リアクション監視

---

## ✅ 実装完了チェックリスト

### Phase 1（完了）
- [x] config/interactive_config.yaml
- [x] bushidan/approval_manager.py
- [x] utils/instruction_parser.py
- [x] bushidan/discord_bot.py（監視機能）

### Phase 2（未実装）
- [ ] bushidan/discord_reporter.py
  - [ ] TaskThread 拡張
  - [ ] request_approval() メソッド
  - [ ] wait_for_user_response() メソッド

### Phase 3（未実装）
- [ ] core/taisho.py
  - [ ] ファイル作成前の承認
  - [ ] Gitコミット前の承認
- [ ] core/langgraph_router.py
  - [ ] ルーティング決定前の承認
- [ ] core/karo.py
  - [ ] 大将への委譲前の承認
- [ ] core/shogun.py
  - [ ] 家老への委譲前の承認

### Phase 4（未実装）
- [ ] .env に INTERACTIVE_MODE=true 追加
- [ ] Discord Bot権限更新
- [ ] サービス再起動
- [ ] テスト実行

### ドキュメント（未実装）
- [ ] INTERACTIVE_MODE.md 作成
- [ ] DISCORD_SETUP.md 更新

---

## 🎯 次のセッションで実装すべきこと

**最優先:**
1. `bushidan/discord_reporter.py` の修正（request_approval メソッド追加）
2. `core/taisho.py` の修正（ファイル作成・コミット時の承認）

**その後:**
3. `core/langgraph_router.py` の修正（ルーティング時の承認）
4. `core/karo.py`, `core/shogun.py` の修正（委譲時の承認）
5. テスト実行

---

これで次のセッションでスムーズに実装を継続できます！
