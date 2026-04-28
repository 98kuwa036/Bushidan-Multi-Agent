# CLAUDE.md — 武士団 Multi-Agent System

## パス対応表

| 環境 | Bushidan | Bushidan-Multi-Agent |
|---|---|---|
| **pct237 (Claude作業環境)** | `/mnt/Bushidan` | `/mnt/Bushidan-Multi-Agent` |
| **pct100 (実行環境・実体)** | `/home/claude/Bushidan` | `/home/claude/Bushidan-Multi-Agent` |
| **pct100 シンボリックリンク** | `/mnt/Bushidan` → 実体 | `/mnt/Bushidan-Multi-Agent` → 実体 |

**どちらのパスも pct100 上で有効。** ファイル編集は `/mnt/...` で行い、SSH 実行コマンドも `/mnt/...` を使ってよい。

`/Bushidan` および `/Bushidan-Multi-Agent` はクラウド LLM が触れる唯一のローカルディレクトリ。

## インフラ構成

| PCT | IP | 役割 |
|---|---|---|
| pct100 | 192.168.11.231 | bushidan-honjin — Bushidan アプリ本体 |
| pct105 | 192.168.11.236 | PostgreSQL 17 (LangGraph PostgresSaver) |
| pct237 | 192.168.11.237 | Claude Server — Claude Pro CLI + このファイルの作業場所 |
| LLM機 | 192.168.11.239 | ローカルLLM (Gemma4 27B + Nemotron) :8082 |

※ pct103 (Matrix Synapse) は廃止済み。Matrix / Mattermost 連携は撤去。

## 主要サービス

| サービス | ホスト | ポート |
|---|---|---|
| Bushidan コンソール (FastAPI) | pct100 | 8067 |
| Claude API Server | pct237 | 8070 |
| Local LLM Server | 192.168.11.239 | 8082 |
| JupyterLab | pct100 | 8888 |

## systemd サービス (pct100 ユーザーサービス)

```bash
systemctl --user --no-pager status bushidan-evolution.service
systemctl --user --no-pager restart bushidan-evolution.service
systemctl --user --no-pager status bushidan-console.service
```

## 開発ルール

- ファイル編集: pct237 の `/mnt/Bushidan-Multi-Agent/` で行う (バインドマウント経由で即反映)
- SSH コマンド: `ssh 192.168.11.231 "cd /mnt/Bushidan-Multi-Agent && ..."` で統一
- Python 実行: `ssh 192.168.11.231 "cd /mnt/Bushidan-Multi-Agent && ~/.venv/bin/python ..."`
- 環境変数: `/home/claude/Bushidan-Multi-Agent/.env` (= `/mnt/Bushidan-Multi-Agent/.env`)
