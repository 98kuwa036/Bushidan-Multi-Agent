# 武士団マルチエージェント - Proxmox → TrueNAS SCALE LXC 移行スクリプト

## 概要

このディレクトリは、Proxmox ホストで稼働する Ubuntu 22.04 LXC コンテナ (`bushidan-honin`) を
TrueNAS SCALE の LXC コンテナに移行するための自動スクリプト群です。

### 特徴
- **自動化**: 手作業を最小化（Bushidan系ディレクトリのコピーのみ手動）
- **段階的実行**: 各フェーズが独立しており、中断・再開が可能
- **検証機能**: 移行後の疎通確認をPythonで自動実施
- **ロギング**: すべての操作をログに記録

---

## スクリプト一覧

| スクリプト | 実行場所 | 役割 | 実行ユーザー |
|-----------|---------|------|-----------|
| `config.sh` | 全スクリプト | 共通設定変数 + ユーティリティ関数 | N/A |
| `01_vzdump_export.sh` | **Proxmox ホスト** | LXC アーカイブ化 → TrueNAS 転送 | root |
| `02_extract_files.sh` | 任意ホスト (中継) | rootfs 展開 → 移行ファイル抽出 | root |
| `03_create_lxc.sh` | **TrueNAS ホスト** | incus で新 LXC 作成 → ファイル転送 | root |
| `04_setup_environment.sh` | **LXC コンテナ内** | パッケージ・設定・サービス構築 | root |
| `05_verify.py` | **LXC コンテナ内** | 移行後確認（systemd・port・package等） | root |

---

## 事前準備

### 必須確認

1. **Proxmox側**
   ```bash
   # LXC ID の確認
   pct list
   # 例: ID 100 = bushidan-honin
   ```

2. **TrueNAS側**
   - TrueNAS SCALE 24.10 (Electric Eel) 以降
   - `incus` または `lxc` コマンドが利用可能か確認
   - SSH キーベース認証が設定済みか確認

3. **ネットワーク設定確認**
   - Proxmox IP: 例 `192.168.11.200`
   - TrueNAS IP: 例 `192.168.11.240` ← **config.sh で指定**
   - 新 LXC IP: `192.168.11.231` (予定)

### config.sh の編集

```bash
# migration/config.sh を編集
nano migration/config.sh
```

**必須変更項目:**
```bash
PROXMOX_LXC_ID=100              # pct list で確認した ID
PROXMOX_HOST="192.168.11.200"   # Proxmox ホスト IP
TRUENAS_IP="192.168.11.240"     # TrueNAS ホスト IP ← 重要！
```

**その他オプション:**
```bash
LXC_IP="192.168.11.231"         # 新 LXC IP（変更不要なら同じ値）
LXC_CPU_CORES=4                 # 割り当てCPU
LXC_RAM_GB=4                    # 割り当てRAM
LXC_DISK_GB=50                  # 割り当てディスク
```

---

## 実行手順

### ステップ 1: Proxmox ホストでエクスポート

```bash
# Proxmox ホストにログイン
ssh root@192.168.11.200

# スクリプトをコピー（Bushidan-Multi-Agent から）
scp -r claude@192.168.11.231:/home/claude/Bushidan-Multi-Agent/migration /tmp/

# config.sh 編集（必要に応じて）
nano /tmp/migration/config.sh

# エクスポート実行
bash /tmp/migration/01_vzdump_export.sh
```

**出力例:**
```
[2026-03-18 10:00:00] Proxmox LXC エクスポート開始
[2026-03-18 10:05:00] バックアップ完成: /var/lib/vz/dump/bushidan-honin-100-backup.tar.zst (サイズ: 1.5GB)
[2026-03-18 10:10:00] 転送完了 (所要時間: 300秒)
[2026-03-18 10:10:01] チェックサム検証OK
```

### ステップ 2: rootfs から移行ファイル抽出

```bash
# TrueNAS ホストにログイン
ssh root@192.168.11.240

# スクリプトをコピー
mkdir -p /tmp/migration
scp -r /tmp/migration/* root@192.168.11.240:/tmp/migration/

# ターゲットディレクトリを指定して実行
bash /tmp/migration/02_extract_files.sh /tmp/bushidan_migration
```

**結果:**
```
/tmp/bushidan_migration/migration_files.tar.gz (数100MB)
```

### ステップ 3: TrueNAS ホストで新 LXC 作成

```bash
# TrueNAS ホスト上で（ステップ 2 の続き）
bash /tmp/migration/03_create_lxc.sh
```

**出力:**
- incus でコンテナ `bushidan-honin` を作成
- 静的 IP `192.168.11.231` を割り当て
- セットアップスクリプトをコンテナ内に配置

### ステップ 4: LXC 内でセットアップ

```bash
# LXC コンテナ内で実行（自動化も可能）
incus exec bushidan-honin -- bash /tmp/04_setup_environment.sh
```

**処理内容:**
1. claude ユーザー作成
2. Node.js v20, Python 3.10 インストール
3. PM2, Claude CLI インストール
4. 移行ファイル展開（.gitconfig, .npmrc など）
5. systemd サービス設定
6. Python venv 再構築

### ステップ 5: 移行後確認

```bash
# LXC コンテナ内で
incus exec bushidan-honin -- python3 /tmp/05_verify.py
```

**レポート例:**
```
【Users】
  claude ユーザー          ✓ PASS           uid=1000(claude) gid=1000(claude) groups=1000(claude),27(sudo)
  sudoers                  ✓ PASS           /etc/sudoers.d/claude 存在

【Node.js】
  Node.js                  ✓ PASS           v20.20.0
  npm                      ✓ PASS           10.8.2

【Python】
  Python3                  ✓ PASS           Python 3.10.12
  pip3                     ✓ PASS           pip 24.0 from /usr/lib/python3.10/...

総合結果: PASS: 24 | WARN: 3 | FAIL: 0
[判定] 移行完了（警告あり）
```

---

## 手動処理（このスクリプト範囲外）

### 1. Bushidan ディレクトリのコピー

```bash
# Proxmox → TrueNAS への rsync
# （LXC IP が 192.168.11.231 に割り当てられた後）

rsync -avz --delete \
  /home/claude/Bushidan-Multi-Agent \
  root@192.168.11.231:/home/claude/

rsync -avz --delete \
  /home/claude/Bushidan \
  root@192.168.11.231:/home/claude/
```

### 2. .env ファイルのコピー

```bash
# APIキーを含むため、セキュアにコピー
scp /home/claude/Bushidan-Multi-Agent/.env \
    root@192.168.11.231:/home/claude/Bushidan-Multi-Agent/
```

### 3. 所有者修正

```bash
# LXC 内で実行
incus exec bushidan-honin -- chown -R claude:claude /home/claude
```

---

## トラブルシューティング

### 1. Proxmox で SSH 接続エラー

```
✗ TrueNAS (192.168.11.xxx) への SSH 接続に失敗しました
```

**対処:**
```bash
# Proxmox ホストから SSH 接続テスト
ssh -v root@192.168.11.240 echo "OK"

# SSH キー設定（必要な場合）
ssh-copy-id root@192.168.11.240
```

### 2. LXC 起動タイムアウト

```
✗ コンテナの起動がタイムアウトしました
```

**対処:**
```bash
# TrueNAS ホストで状態確認
incus list
incus exec bushidan-honin -- ip addr

# ネットワーク問題の場合
incus restart bushidan-honin
```

### 3. Python パッケージ不足エラー

```
⚠ パッケージ langgraph: 見つかりません
```

**対処:**
```bash
# Bushidan-Multi-Agent/.venv が存在することを確認
incus exec bushidan-honin -- ls -la /home/claude/Bushidan-Multi-Agent/.venv

# requirements.txt から手動インストール
incus exec bushidan-honin -- bash -c \
  'cd /home/claude/Bushidan-Multi-Agent && pip install -r requirements.txt'
```

### 4. サービスが起動しない

```bash
# LXC 内でログ確認
incus exec bushidan-honin -- journalctl -u bushidan-main -n 50

# 手動起動
incus exec bushidan-honin -- systemctl start bushidan-main
```

---

## ロギング・デバッグ

### ログファイル確認

```bash
# ログディレクトリ
/tmp/bushidan_migration/logs/

# 最新ログ
tail -f /tmp/bushidan_migration/logs/migration_*.log
```

### verbose モードでの実行

```bash
# 各スクリプト内の set -x を有効化
bash -x /tmp/migration/04_setup_environment.sh
```

---

## ロールバック・キャンセル

### LXC コンテナの削除

```bash
# TrueNAS ホストで
incus delete -f bushidan-honin
```

### 元の Proxmox LXC への戻し

```bash
# Proxmox ホストで
pct start 100
```

---

## FAQ

**Q: Bushidan-Multi-Agent は別途コピーが必要ですか？**
A: はい。スクリプトはシステムファイル・設定のみを自動化しており、アプリケーションコード自体は
rsync で別途コピーしてください。これにより、アプリコード更新時の差分同期が容易になります。

**Q: 既存の PM2 プロセスは復元されますか？**
A: はい。`.pm2/dump.pm2` を移行し、LXC 内で `pm2 start ecosystem.config.cjs` を実行すれば、
定義済みプロセスが復元されます。

**Q: .env ファイルは自動的にコピーされますか？**
A: いいえ。API キーを含むため、セキュリティ上の理由から手動コピーです。
`scp` または安全なファイル転送で行ってください。

**Q: 新 LXC の IP を変更したいです。**
A: config.sh の `LXC_IP` 変数を変更してください。ステップ 3 実行前に編集してください。

---

## サポート・質問

各スクリプト内のコメント（`# ─── `で始まる行）で処理内容を記述しています。
詳細は各スクリプトを参照してください。

---

**更新日:** 2026-03-18
**バージョン:** 1.0
**対応環境:** Proxmox 8.x + TrueNAS SCALE 24.10+
