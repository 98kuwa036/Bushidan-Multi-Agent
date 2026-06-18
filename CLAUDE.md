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
| Proxmox Host | 192.168.11.230 | pve — ハイパーバイザー本体 |
| pct100 | 192.168.11.231 | bushidan-honjin — Bushidan アプリ本体 |
| pct101 | Dawarich | 位置情報トラッキング |
| pct102 | Seafile + MariaDB | ファイルクラウド |
| pct104 | 192.168.11.235 | Immich (写真管理) |
| pct105 | 192.168.11.236 | PostgreSQL 17 (LangGraph PostgresSaver) |
| pct106 | Claude Server | Claude Pro CLI |
| pct108 | 192.168.11.238 | Psono + Keycloak (Docker) |
| pct237 | 192.168.11.237 | Claude作業環境 — このファイルの作業場所 |
| LLM機(旧) | 192.168.11.240 | 旧ローカルLLM機 (移行前: Gemma4 27B + Nemotron) |
| LLM機(新) | 192.168.11.239 | Hyper-V Ubuntu VM — 新Gemma4推論サーバー（予定） |

※ pct103 (Matrix Synapse) は廃止済み。Matrix / Mattermost 連携は撤去。

## 主要サービス

| サービス | ホスト | ポート |
|---|---|---|
| Bushidan コンソール (FastAPI) | pct100 | 8067 |
| Claude API Server | pct106 | 8070 |
| Local LLM Server (新) | 192.168.11.239 | 8082 |
| Psono | pct108 (192.168.11.238) | 10200 |
| Keycloak | pct108 (192.168.11.238) | 8081 |
| Seafile | pct108 (192.168.11.238) | 10400 |
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
