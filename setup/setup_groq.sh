#!/bin/bash
# ============================================================================
# 武士団マルチエージェントシステム v10.1 - Groq 記録係セットアップ
# 9番足軽 (Groq Recorder) Setup
# ============================================================================
#
# 【コア要件】必ず達成すべきこと:
#   1. Groq Python SDK のインストール
#   2. API キーの設定 (.env ファイル)
#
# 【最終結果】このスクリプト完了後の状態:
#   - groq Python パッケージがインストール済み
#   - GROQ_API_KEY が .env に設定済み
#   - groq-status / groq-benchmark コマンドが使用可能 (オプション)
#
# 【動作保証】
#   - python3, pip があれば異常終了しない
#   - API キー未設定でもスクリプトは完了 (警告表示)
#   - --auto オプションで対話的プロンプトをスキップ
#
# Groq 仕様:
#   - Model: Llama 3.3 70B Versatile
#   - Speed: 300-500 tokens/second
#   - Quota: 14,400 requests/day (FREE)
#
# Usage:
#   bash setup_groq.sh              # 対話モード
#   bash setup_groq.sh --auto       # 自動モード (APIキー設定済み前提)
#   bash setup_groq.sh --test       # API テストのみ
#
# ============================================================================

# エラー時に即座に終了しない
# set -e は使わない

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_DIR}/.env"

# 実行モード
AUTO_MODE=0

# ============================================================================
# 結果トラッキング
# ============================================================================

declare -A RESULTS
CORE_FAILED=0

mark_success() {
    RESULTS["$1"]="OK"
}

mark_failed() {
    RESULTS["$1"]="FAILED"
    if [[ "$2" == "core" ]]; then
        CORE_FAILED=1
    fi
}

mark_skipped() {
    RESULTS["$1"]="SKIPPED"
}

# ============================================================================
# ユーティリティ関数
# ============================================================================

log_info() {
    echo "[INFO] $1"
}

log_success() {
    echo "[OK] $1"
}

log_warning() {
    echo "[WARN] $1"
}

log_error() {
    echo "[ERROR] $1"
}

# ネットワーク操作のリトライ
retry_network() {
    local cmd="$1"
    local max_attempts=4
    local delay=2

    for ((i=1; i<=max_attempts; i++)); do
        if eval "$cmd"; then
            return 0
        fi
        if [ $i -lt $max_attempts ]; then
            log_warning "リトライ $i/$max_attempts (${delay}秒後)..."
            sleep $delay
            delay=$((delay * 2))
        fi
    done
    return 1
}

# ============================================================================
# 前提条件チェック
# ============================================================================

check_dependencies() {
    log_info "依存関係を確認中..."

    local missing=()

    for cmd in python3 pip3; do
        if ! command -v $cmd &>/dev/null; then
            # pip3 がなければ pip を試す
            if [ "$cmd" = "pip3" ] && command -v pip &>/dev/null; then
                continue
            fi
            missing+=($cmd)
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "必須コマンドがありません: ${missing[*]}"
        log_info "setup/install.sh を先に実行してください"
        return 1
    fi

    mark_success "dependencies"
    log_success "依存関係OK"
    return 0
}

# ============================================================================
# Groq SDK インストール
# ============================================================================

install_groq_sdk() {
    log_info "Groq SDK をインストール中..."

    # pip コマンド決定
    local PIP_CMD="pip3"
    if ! command -v pip3 &>/dev/null; then
        PIP_CMD="pip"
    fi

    # 仮想環境があれば使用
    if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
        source "${PROJECT_DIR}/.venv/bin/activate"
        PIP_CMD="pip"
    fi

    # Groq SDK インストール
    if retry_network "$PIP_CMD install groq requests python-dateutil 2>/dev/null"; then
        mark_success "groq_sdk"
        log_success "Groq SDK インストール完了"
    else
        log_error "Groq SDK インストール失敗"
        mark_failed "groq_sdk" "core"
        return 1
    fi

    return 0
}

# ============================================================================
# API キー設定
# ============================================================================

configure_api_key() {
    log_info "API キーを設定中..."

    # 既存キーチェック
    local existing_key=""
    if [ -f "$ENV_FILE" ] && grep -q "GROQ_API_KEY" "$ENV_FILE"; then
        existing_key=$(grep "GROQ_API_KEY" "$ENV_FILE" | cut -d'=' -f2)
        if [[ "$existing_key" =~ ^gsk_ ]]; then
            log_success "GROQ_API_KEY 既に設定済み"
            mark_success "api_key"
            export GROQ_API_KEY="$existing_key"
            return 0
        fi
    fi

    # 環境変数チェック
    if [ -n "$GROQ_API_KEY" ] && [[ "$GROQ_API_KEY" =~ ^gsk_ ]]; then
        log_success "GROQ_API_KEY 環境変数から取得"
        # .env に保存
        if [ -f "$ENV_FILE" ]; then
            if grep -q "GROQ_API_KEY" "$ENV_FILE"; then
                sed -i "s/GROQ_API_KEY=.*/GROQ_API_KEY=$GROQ_API_KEY/" "$ENV_FILE"
            else
                echo "GROQ_API_KEY=$GROQ_API_KEY" >> "$ENV_FILE"
            fi
        else
            echo "GROQ_API_KEY=$GROQ_API_KEY" > "$ENV_FILE"
        fi
        mark_success "api_key"
        return 0
    fi

    # 自動モードでキーがない場合
    if [ $AUTO_MODE -eq 1 ]; then
        log_warning "GROQ_API_KEY が設定されていません"
        log_info "手動で .env に追加してください:"
        echo "  echo 'GROQ_API_KEY=gsk_xxx' >> $ENV_FILE"
        mark_skipped "api_key"
        return 0
    fi

    # 対話的プロンプト
    echo ""
    echo "============================================="
    echo "  Groq API キー設定"
    echo "============================================="
    echo ""
    echo "1. https://console.groq.com/keys にアクセス"
    echo "2. アカウント作成 (無料)"
    echo "3. API キーを生成"
    echo ""

    local api_key
    while true; do
        read -p "Groq API キー (gsk_...): " api_key

        if [[ "$api_key" =~ ^gsk_[a-zA-Z0-9]{20,}$ ]]; then
            break
        elif [ -z "$api_key" ]; then
            log_warning "スキップします (後で設定可能)"
            mark_skipped "api_key"
            return 0
        else
            log_error "無効な形式です。Groq API キーは 'gsk_' で始まります"
        fi
    done

    # .env に保存
    if [ -f "$ENV_FILE" ]; then
        if grep -q "GROQ_API_KEY" "$ENV_FILE"; then
            sed -i "s/GROQ_API_KEY=.*/GROQ_API_KEY=$api_key/" "$ENV_FILE"
        else
            echo "GROQ_API_KEY=$api_key" >> "$ENV_FILE"
        fi
    else
        echo "GROQ_API_KEY=$api_key" > "$ENV_FILE"
    fi

    export GROQ_API_KEY="$api_key"
    mark_success "api_key"
    log_success "API キー保存完了: $ENV_FILE"

    return 0
}

# ============================================================================
# API テスト
# ============================================================================

test_api() {
    log_info "Groq API をテスト中..."

    # API キー確認
    if [ -z "$GROQ_API_KEY" ]; then
        if [ -f "$ENV_FILE" ] && grep -q "GROQ_API_KEY" "$ENV_FILE"; then
            export GROQ_API_KEY=$(grep "GROQ_API_KEY" "$ENV_FILE" | cut -d'=' -f2)
        fi
    fi

    if [ -z "$GROQ_API_KEY" ] || [[ ! "$GROQ_API_KEY" =~ ^gsk_ ]]; then
        log_warning "GROQ_API_KEY が設定されていません。テストをスキップ"
        mark_skipped "api_test"
        return 0
    fi

    # 仮想環境
    if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
        source "${PROJECT_DIR}/.venv/bin/activate"
    fi

    # Python テスト
    local test_result
    test_result=$(python3 << 'PYTHON' 2>&1
import os
from groq import Groq

try:
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Respond in one word."},
            {"role": "user", "content": "Say 'OK' if you're working."}
        ],
        max_tokens=10,
        temperature=0.1,
    )
    if response.choices and response.choices[0].message.content:
        print(f"SUCCESS:{response.usage.total_tokens}")
    else:
        print("ERROR:Empty response")
except Exception as e:
    print(f"ERROR:{e}")
PYTHON
)

    if [[ "$test_result" == SUCCESS:* ]]; then
        local tokens="${test_result#SUCCESS:}"
        mark_success "api_test"
        log_success "API テスト成功 (${tokens} tokens)"
    else
        local error="${test_result#ERROR:}"
        mark_failed "api_test"
        log_error "API テスト失敗: $error"
    fi
}

# ============================================================================
# 管理ツール作成
# ============================================================================

create_tools() {
    log_info "管理ツールを作成中..."

    # ツールディレクトリ
    local TOOLS_DIR="${PROJECT_DIR}/scripts"
    mkdir -p "$TOOLS_DIR"

    # groq-status スクリプト
    cat > "${TOOLS_DIR}/groq-status.sh" << 'BASH'
#!/bin/bash
# Groq 9番足軽 ステータス確認

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_DIR}/.env"

echo "============================================="
echo "  9番足軽 (Groq記録係) ステータス"
echo "============================================="

# API キー確認
if [ -f "$ENV_FILE" ] && grep -q "GROQ_API_KEY" "$ENV_FILE"; then
    echo "  [OK] API Key: 設定済み"
else
    echo "  [--] API Key: 未設定"
fi

# 仮想環境
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# .env を読み込み
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

# API テスト
echo ""
echo "  API テスト中..."
python3 -c "
import os
from groq import Groq
try:
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'user', 'content': 'Test'}],
        max_tokens=5
    )
    print('  [OK] API 応答正常')
except Exception as e:
    print(f'  [ERROR] {e}')
" 2>/dev/null || echo "  [ERROR] Python 環境エラー"

echo ""
echo "============================================="
BASH

    chmod +x "${TOOLS_DIR}/groq-status.sh" 2>/dev/null || true

    # groq-benchmark スクリプト
    cat > "${TOOLS_DIR}/groq-benchmark.sh" << 'BASH'
#!/bin/bash
# Groq パフォーマンスベンチマーク

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_DIR}/.env"

echo "============================================="
echo "  Groq パフォーマンスベンチマーク"
echo "============================================="

# 仮想環境
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

# .env を読み込み
if [ -f "$ENV_FILE" ]; then
    export $(grep -v '^#' "$ENV_FILE" | xargs)
fi

python3 << 'PYTHON'
import os
import time
from groq import Groq

try:
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

    prompt = """以下の5項目について簡潔に要約してください:
1. ESP32-P4のI2S設定
2. Home Assistantの音声認識
3. Spotify API連携
4. 消費電力測定
5. 自動テスト生成"""

    print(f"  プロンプト: {len(prompt)} 文字")
    print("  生成中...")

    start = time.time()
    response = client.chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[
            {'role': 'system', 'content': '簡潔に日本語で回答してください。'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=500,
        temperature=0.3,
    )
    duration = time.time() - start

    if response.usage:
        total = response.usage.total_tokens
        completion = response.usage.completion_tokens
        speed = total / duration

        print(f"  応答時間: {duration:.2f}秒")
        print(f"  総トークン: {total}")
        print(f"  出力トークン: {completion}")
        print(f"  速度: {speed:.0f} tok/s")

        if speed > 200:
            print("  評価: Excellent (>200 tok/s)")
        elif speed > 100:
            print("  評価: Good (>100 tok/s)")
        else:
            print("  評価: Below expected")

except Exception as e:
    print(f"  [ERROR] {e}")
PYTHON

echo ""
echo "============================================="
BASH

    chmod +x "${TOOLS_DIR}/groq-benchmark.sh" 2>/dev/null || true

    mark_success "tools"
    log_success "管理ツール作成完了"
    log_info "  scripts/groq-status.sh    - ステータス確認"
    log_info "  scripts/groq-benchmark.sh - パフォーマンステスト"
}

# ============================================================================
# 最終レポート
# ============================================================================

print_final_report() {
    echo ""
    echo "============================================="
    echo "  Groq 記録係 - セットアップ完了レポート"
    echo "============================================="
    echo ""

    echo "【処理結果】"
    for key in "${!RESULTS[@]}"; do
        local status="${RESULTS[$key]}"
        case "$status" in
            OK)      echo "  [OK]      $key" ;;
            FAILED)  echo "  [FAILED]  $key" ;;
            SKIPPED) echo "  [SKIP]    $key" ;;
        esac
    done

    echo ""

    if [ $CORE_FAILED -eq 1 ]; then
        echo "【状態】コア機能に失敗があります"
    else
        echo "【状態】正常完了"
        echo ""
        echo "【Groq 仕様】"
        echo "  Model: Llama 3.3 70B Versatile"
        echo "  Speed: 300-500 tokens/second"
        echo "  Quota: 14,400 requests/day (FREE)"
        echo ""
        echo "【管理コマンド】"
        echo "  ./scripts/groq-status.sh     # ステータス確認"
        echo "  ./scripts/groq-benchmark.sh  # パフォーマンステスト"
    fi

    echo ""
    echo "============================================="
}

# ============================================================================
# メイン処理
# ============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --auto    自動モード (対話プロンプトなし)"
    echo "  --test    API テストのみ"
    echo "  --help    このヘルプを表示"
    echo ""
    echo "【コア要件】"
    echo "  1. Groq SDK インストール"
    echo "  2. API キー設定"
    echo ""
    echo "【最終結果】"
    echo "  - groq Python パッケージ利用可能"
    echo "  - GROQ_API_KEY で API 呼び出し可能"
}

run_full_setup() {
    echo ""
    echo "============================================="
    echo "  武士団 v10.1 - Groq 記録係セットアップ"
    echo "  9番足軽 (Llama 3.3 70B Versatile)"
    echo "============================================="
    echo ""
    echo "  Speed: 300-500 tokens/second"
    echo "  Quota: 14,400 requests/day (FREE)"
    echo ""

    if ! check_dependencies; then
        CORE_FAILED=1
        print_final_report
        exit 1
    fi

    install_groq_sdk
    configure_api_key
    test_api
    create_tools

    print_final_report
}

# ============================================================================
# エントリーポイント
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            AUTO_MODE=1
            shift
            ;;
        --test)
            test_api
            print_final_report
            exit $CORE_FAILED
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            log_error "不明なオプション: $1"
            show_usage
            exit 1
            ;;
    esac
done

run_full_setup

exit $CORE_FAILED
