#!/bin/bash
# ============================================================================
# 武士団マルチエージェントシステム v11.4 - 隠密の練兵 (HP ProDesk 600)
# Nemotron-3-Nano-30B Setup Script
# ============================================================================
#
# 【概要】
#   NVIDIA Nemotron-3-Nano-30B-A3B を HP ProDesk 600 (32GB RAM) で動かすための
#   llama.cpp セットアップスクリプト。v11.4 より Qwen3 から移行（脱中国企業）。
#
# 【コア要件】必ず達成すべきこと:
#   1. llama.cpp のビルド (llama-server バイナリ生成)
#   2. Nemotron-3-Nano Q4_K_M モデルのダウンロード (~21GB)
#   3. 起動スクリプト (start_nemotron.sh) の作成
#   4. systemd サービス設定 (自動起動)
#
# 【最終結果】このスクリプト完了後の状態:
#   - ~/llama.cpp/build/bin/llama-server が実行可能
#   - models/nemotron/Nemotron-3-Nano-Q4_K_M.gguf が配置 (~21GB)
#   - scripts/start_nemotron.sh で手動起動可能
#   - systemd サービス: bushidan-nemotron で自動起動
#   - http://192.168.11.232:8080 で OpenAI 互換 API が稼働
#
# 【ハードウェア要件】
#   - HP ProDesk 600 G4
#   - Intel Core i5-8100 (4C/4T)
#   - RAM: 32GB DDR4 (必須: Q4_K_M モデル ~21GB 使用)
#   - ディスク: 25GB 以上の空き容量
#
# 【v11.4 変更点】
#   - Qwen3-Coder-30B (中国企業) → Nemotron-3-Nano (NVIDIA) に移行
#   - 脱中国企業・信頼性の向上
#
# Usage:
#   ./setup/setup_nemotron.sh           # 対話モード
#   ./setup/setup_nemotron.sh --auto    # 自動モード (全実行)
#   ./setup/setup_nemotron.sh --build   # llama.cpp ビルドのみ
#   ./setup/setup_nemotron.sh --model   # モデルダウンロードのみ
#   ./setup/setup_nemotron.sh --verify  # 検証のみ
#
# ============================================================================

# ============================================================================
# 設定
# ============================================================================

LLAMA_CPP_DIR="${HOME}/llama.cpp"
MODEL_DIR="${HOME}/Bushidan-Multi-Agent/models/nemotron"
MODEL_NAME="Nemotron-3-Nano-Q4_K_M.gguf"

# NVIDIA Nemotron-3-Nano-30B-A3B (Q4_K_M) - HuggingFace
# 公式: nvidia/Nemotron-Mini-4B-Instruct もしくは nemotron-3 系
MODEL_HF_REPO="nvidia/Nemotron-Mini-4B-Instruct-GGUF"
MODEL_URL="https://huggingface.co/bartowski/Nemotron-3-Nano-30B-A3B-Instruct-GGUF/resolve/main/Nemotron-3-Nano-30B-A3B-Instruct-Q4_K_M.gguf"

PROJECT_DIR="${HOME}/Bushidan-Multi-Agent"

# HP ProDesk 600 CPU 最適化設定
CPU_THREADS=4          # i5-8500: 6コア (HT無し)
CONTEXT_SIZE=8192      # 隠密: 長文対応（メモリに余裕あれば増やす）
BATCH_SIZE=512         # CPU 最適バッチサイズ
PARALLEL_REQUESTS=1    # CPU 安定性のため単一リクエスト
PORT=8080
HOST="0.0.0.0"

# エンドポイント（EliteDesk 本陣からのアクセス用）
PRODESDK_IP="192.168.11.239"

# 実行モード
AUTO_MODE=0

# ============================================================================
# 結果トラッキング
# ============================================================================

declare -A RESULTS
CORE_FAILED=0

mark_success() { RESULTS["$1"]="OK"; }
mark_failed()  { RESULTS["$1"]="FAILED"; [[ "$2" == "core" ]] && CORE_FAILED=1; }
mark_skipped() { RESULTS["$1"]="SKIPPED"; }

# ============================================================================
# ユーティリティ関数
# ============================================================================

log_info()    { echo "[INFO]    $1"; }
log_success() { echo "[OK]      $1"; }
log_warning() { echo "[WARN]    $1"; }
log_error()   { echo "[ERROR]   $1"; }

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

confirm() {
    local prompt="$1"
    local default="${2:-n}"

    if [ $AUTO_MODE -eq 1 ]; then
        return 0
    fi

    local yn
    if [ "$default" = "y" ]; then
        read -p "$prompt [Y/n]: " -n 1 -r yn
    else
        read -p "$prompt [y/N]: " -n 1 -r yn
    fi
    echo
    yn="${yn:-$default}"
    [[ "$yn" =~ ^[Yy]$ ]]
}

# ============================================================================
# システム情報・前提条件
# ============================================================================

show_system_info() {
    log_info "システム情報:"

    local cpu_model=$(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name:\s*//' || echo "不明")
    local cpu_cores=$(nproc 2>/dev/null || echo "不明")
    local total_mem=$(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo "不明")
    local free_disk=$(df -h . 2>/dev/null | tail -1 | awk '{print $4}' || echo "不明")

    echo "  CPU: $cpu_model"
    echo "  コア数: $cpu_cores"
    echo "  メモリ: $total_mem"
    echo "  空きディスク: $free_disk"
    echo ""

    # 32GB RAM チェック
    local total_mem_gb=$(free -g 2>/dev/null | grep Mem | awk '{print $2}' || echo "0")
    if [ "$total_mem_gb" -lt 28 ]; then
        log_warning "⚠️  RAM ${total_mem_gb}GB 検出 - Nemotron Q4_K_M (~21GB) には 32GB 推奨"
        log_warning "   実行は可能ですが、スワップが発生し速度が大幅低下します"
    else
        log_success "RAM ${total_mem_gb}GB: Nemotron Q4_K_M (~21GB) の動作に十分です ✅"
    fi
}

check_dependencies() {
    log_info "依存関係を確認中..."

    local missing=()
    local required_cmds="git cmake make gcc g++ wget"

    for cmd in $required_cmds; do
        if ! command -v $cmd &>/dev/null; then
            missing+=($cmd)
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "必須コマンドがありません: ${missing[*]}"
        log_info "インストール: sudo apt install build-essential cmake git wget"
        mark_failed "dependencies" "core"
        return 1
    fi

    mark_success "dependencies"
    log_success "依存関係OK"
    return 0
}

# ============================================================================
# llama.cpp ビルド
# ============================================================================

build_llama_cpp() {
    log_info "llama.cpp をビルド中..."

    if [ -d "$LLAMA_CPP_DIR" ]; then
        log_info "既存の llama.cpp を更新中..."
        cd "$LLAMA_CPP_DIR"
        retry_network "git pull origin master 2>/dev/null" || log_warning "更新失敗 (既存コードを使用)"
    else
        log_info "llama.cpp をクローン中..."
        if ! retry_network "git clone https://github.com/ggerganov/llama.cpp.git '$LLAMA_CPP_DIR'"; then
            log_error "llama.cpp のクローンに失敗"
            mark_failed "llama_clone" "core"
            return 1
        fi
        cd "$LLAMA_CPP_DIR"
    fi
    mark_success "llama_clone"

    mkdir -p build
    cd build

    log_info "CMake 設定中 (CPU最適化: AVX2, FMA)..."
    if cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_NATIVE=ON \
        -DLLAMA_AVX2=ON \
        -DLLAMA_FMA=ON \
        -DLLAMA_F16C=ON \
        -DLLAMA_BUILD_SERVER=ON 2>/dev/null; then
        mark_success "cmake_config"
        log_success "CMake 設定完了"
    else
        log_error "CMake 設定失敗"
        mark_failed "cmake_config" "core"
        return 1
    fi

    local cores=$(nproc 2>/dev/null || echo "4")
    log_info "ビルド中 (${cores}コア使用)..."
    if make -j$cores 2>/dev/null; then
        mark_success "llama_build"
        log_success "llama.cpp ビルド完了"
    else
        log_error "llama.cpp ビルド失敗"
        mark_failed "llama_build" "core"
        return 1
    fi

    return 0
}

# ============================================================================
# Nemotron-3-Nano モデルダウンロード
# ============================================================================

download_model() {
    log_info "Nemotron-3-Nano-30B Q4_K_M モデルをダウンロード中..."

    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"

    if [ -f "$MODEL_NAME" ]; then
        local size=$(du -h "$MODEL_NAME" 2>/dev/null | cut -f1 || echo "不明")
        log_warning "モデルファイルは既に存在します: $MODEL_NAME ($size)"

        if ! confirm "再ダウンロードしますか?"; then
            log_info "既存のモデルを使用します"
            mark_success "model_download"
            return 0
        fi
    fi

    log_info "モデルサイズ: 約21GB (Q4_K_M)"
    log_info "ダウンロードには20-60分かかります..."
    log_info "ダウンロード元: HuggingFace (bartowski/Nemotron-3-Nano-30B-A3B-Instruct-GGUF)"

    # huggingface-cli が利用可能なら使用
    if command -v huggingface-cli &>/dev/null; then
        log_info "huggingface-cli でダウンロード中..."
        if huggingface-cli download \
            "bartowski/Nemotron-3-Nano-30B-A3B-Instruct-GGUF" \
            "Nemotron-3-Nano-30B-A3B-Instruct-Q4_K_M.gguf" \
            --local-dir "$MODEL_DIR" \
            --local-dir-use-symlinks False 2>/dev/null; then

            # ファイル名を標準化
            if [ -f "${MODEL_DIR}/Nemotron-3-Nano-30B-A3B-Instruct-Q4_K_M.gguf" ] && [ "$MODEL_NAME" != "Nemotron-3-Nano-30B-A3B-Instruct-Q4_K_M.gguf" ]; then
                mv "${MODEL_DIR}/Nemotron-3-Nano-30B-A3B-Instruct-Q4_K_M.gguf" "${MODEL_DIR}/${MODEL_NAME}"
            fi

            mark_success "model_download"
            log_success "Nemotron モデルダウンロード完了"
            return 0
        fi
    fi

    # wget でダウンロード (再開可能)
    if wget -c "$MODEL_URL" -O "$MODEL_NAME" 2>/dev/null; then
        mark_success "model_download"
        log_success "Nemotron モデルダウンロード完了"
    else
        log_error "モデルダウンロード失敗"
        log_info ""
        log_info "手動でダウンロードしてください:"
        echo "  # 方法1: wget"
        echo "  mkdir -p ${MODEL_DIR}"
        echo "  wget -c '${MODEL_URL}' -O '${MODEL_DIR}/${MODEL_NAME}'"
        echo ""
        echo "  # 方法2: huggingface-cli"
        echo "  pip install huggingface_hub"
        echo "  huggingface-cli download bartowski/Nemotron-3-Nano-30B-A3B-Instruct-GGUF \\"
        echo "      Nemotron-3-Nano-30B-A3B-Instruct-Q4_K_M.gguf \\"
        echo "      --local-dir ${MODEL_DIR}"
        echo ""
        mark_failed "model_download" "core"
        return 1
    fi

    return 0
}

# ============================================================================
# 起動スクリプト作成
# ============================================================================

create_start_script() {
    log_info "起動スクリプトを作成中..."

    local SCRIPT_PATH="${PROJECT_DIR}/scripts/start_nemotron.sh"

    local SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"
    if [ ! -f "$SERVER_PATH" ]; then
        SERVER_PATH="$LLAMA_CPP_DIR/build/llama-server"
    fi

    mkdir -p "${PROJECT_DIR}/scripts"

    cat > "$SCRIPT_PATH" << STARTSCRIPT
#!/bin/bash
# ============================================================================
# 武士団 v11.4 - 隠密 Nemotron-3-Nano 起動スクリプト
# HP ProDesk 600 G4 (i5-8500, 32GB) CPU 最適化
# ============================================================================

SERVER_PATH="${SERVER_PATH}"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

THREADS=${CPU_THREADS}
CONTEXT=${CONTEXT_SIZE}
BATCH=${BATCH_SIZE}
PORT=${PORT}
HOST="${HOST}"

# バイナリ確認
if [ ! -f "\$SERVER_PATH" ]; then
    echo "[ERROR] llama-server が見つかりません: \$SERVER_PATH"
    echo "  ./setup/setup_nemotron.sh --build を実行してください"
    exit 1
fi

# モデル確認
if [ ! -f "\$MODEL_PATH" ]; then
    echo "[ERROR] モデルが見つかりません: \$MODEL_PATH"
    echo "  ./setup/setup_nemotron.sh --model を実行してください"
    exit 1
fi

# メモリロック制限を解除 (21GB モデル用)
ulimit -l unlimited

echo "============================================="
echo "  🥷 武士団 隠密 - Nemotron-3-Nano 起動"
echo "============================================="
echo "  Model: Nemotron-3-Nano-30B-A3B (Q4_K_M)"
echo "  Threads: \$THREADS"
echo "  Context: \$CONTEXT"
echo "  Port: \$PORT"
echo "  Endpoint: http://${PRODESDK_IP}:\$PORT"
echo "  MemLock: unlimited"
echo "============================================="

\$SERVER_PATH \\
    -m "\$MODEL_PATH" \\
    -c \$CONTEXT \\
    -t \$THREADS \\
    -b \$BATCH \\
    --parallel 1 \\
    --host \$HOST \\
    --port \$PORT \\
    --mlock \\
    --mmap

STARTSCRIPT

    if chmod +x "$SCRIPT_PATH"; then
        mark_success "start_script"
        log_success "起動スクリプト作成: $SCRIPT_PATH"
    else
        log_error "起動スクリプト作成失敗"
        mark_failed "start_script"
    fi
}

# ============================================================================
# systemd サービス作成
# ============================================================================

create_systemd_service() {
    log_info "systemd サービスを作成中..."

    local SERVICE_NAME="bushidan-nemotron"
    local SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

    local SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"
    if [ ! -f "$SERVER_PATH" ]; then
        SERVER_PATH="$LLAMA_CPP_DIR/build/llama-server"
    fi

    if [ ! -f "$SERVER_PATH" ]; then
        log_warning "llama-server が見つかりません。サービス作成をスキップ"
        mark_skipped "systemd_service"
        return 0
    fi

    local SERVICE_CONTENT="[Unit]
Description=Bushidan v11.4 - Onmitsu Nemotron-3-Nano Server
Documentation=https://github.com/98kuwa036/Bushidan-Multi-Agent
After=network.target
Wants=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${HOME}
ExecStart=${SERVER_PATH} \\
    -m ${MODEL_DIR}/${MODEL_NAME} \\
    -c ${CONTEXT_SIZE} \\
    -t ${CPU_THREADS} \\
    -b ${BATCH_SIZE} \\
    --parallel 1 \\
    --host ${HOST} \\
    --port ${PORT} \\
    --mlock \\
    --mmap
Restart=on-failure
RestartSec=10
MemoryMax=25G
LimitMEMLOCK=infinity

# 隠密: ローカル完結・セキュリティ設定
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
"

    if [ "$(id -u)" -eq 0 ]; then
        echo "$SERVICE_CONTENT" > "$SERVICE_FILE"
        systemctl daemon-reload 2>/dev/null || true
        mark_success "systemd_service"
        log_success "systemd サービス作成: ${SERVICE_NAME}.service"
    elif sudo -n true 2>/dev/null; then
        echo "$SERVICE_CONTENT" | sudo tee "$SERVICE_FILE" > /dev/null
        sudo systemctl daemon-reload 2>/dev/null || true
        mark_success "systemd_service"
        log_success "systemd サービス作成: ${SERVICE_NAME}.service"
    else
        log_warning "root 権限なし。手動でサービスを作成してください"
        log_info "以下の内容を ${SERVICE_FILE} に保存:"
        echo "---"
        echo "$SERVICE_CONTENT"
        echo "---"
        mark_skipped "systemd_service"
    fi
}

# ============================================================================
# 動作検証
# ============================================================================

verify_setup() {
    log_info "セットアップを検証中..."

    local errors=0

    # llama-server 確認
    local SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"
    if [ ! -f "$SERVER_PATH" ]; then
        SERVER_PATH="$LLAMA_CPP_DIR/build/llama-server"
    fi

    if [ -f "$SERVER_PATH" ]; then
        mark_success "verify_server"
        log_success "llama-server: OK ($SERVER_PATH)"
    else
        mark_failed "verify_server"
        log_error "llama-server: 見つかりません"
        ((errors++))
    fi

    # モデル確認
    if [ -f "${MODEL_DIR}/${MODEL_NAME}" ]; then
        local size=$(du -h "${MODEL_DIR}/${MODEL_NAME}" 2>/dev/null | cut -f1 || echo "不明")
        mark_success "verify_model"
        log_success "Nemotron モデル: OK ($size)"
    else
        mark_failed "verify_model"
        log_error "Nemotron モデル: 見つかりません (${MODEL_DIR}/${MODEL_NAME})"
        ((errors++))
    fi

    # 起動スクリプト確認
    if [ -x "${PROJECT_DIR}/scripts/start_nemotron.sh" ]; then
        mark_success "verify_script"
        log_success "起動スクリプト: OK"
    else
        mark_failed "verify_script"
        log_error "起動スクリプト: 見つかりません"
        ((errors++))
    fi

    # メモリ確認
    local total_mem_gb=$(free -g 2>/dev/null | grep Mem | awk '{print $2}' || echo "0")
    if [ "$total_mem_gb" -ge 28 ]; then
        mark_success "verify_memory"
        log_success "メモリ: ${total_mem_gb}GB (Nemotron Q4_K_M ~21GB に十分)"
    elif [ "$total_mem_gb" -ge 16 ]; then
        mark_skipped "verify_memory"
        log_warning "メモリ: ${total_mem_gb}GB (32GB 推奨。スワップ発生の可能性)"
    else
        mark_failed "verify_memory"
        log_error "メモリ: ${total_mem_gb}GB (不足。32GB 必須)"
        ((errors++))
    fi

    # サーバー起動テスト（オプション）
    if command -v curl &>/dev/null; then
        if curl -s --max-time 3 "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            mark_success "verify_server_live"
            log_success "llama.cpp サーバー: 稼働中 ✅ (http://localhost:${PORT})"
        else
            mark_skipped "verify_server_live"
            log_info "llama.cpp サーバー: 未起動 (手動起動が必要)"
        fi
    fi

    if [ $errors -eq 0 ]; then
        log_success "検証完了: すべてOK"
        return 0
    else
        log_error "検証問題: $errors 件のエラー"
        return 1
    fi
}

# ============================================================================
# 最終レポート
# ============================================================================

print_final_report() {
    echo ""
    echo "============================================="
    echo "  🥷 隠密の練兵 (Nemotron-3-Nano) - 完了レポート"
    echo "============================================="
    echo ""

    echo "【処理結果】"
    for key in "${!RESULTS[@]}"; do
        local status="${RESULTS[$key]}"
        case "$status" in
            OK)      echo "  ✅ $key" ;;
            FAILED)  echo "  ❌ $key" ;;
            SKIPPED) echo "  ⏭️  $key" ;;
        esac
    done

    echo ""

    if [ $CORE_FAILED -eq 1 ]; then
        echo "【状態】⚠️  コア機能に失敗があります"
        echo ""
        echo "  エラーを確認し、再実行してください:"
        echo "  ./setup/setup_nemotron.sh --auto"
    else
        echo "【状態】✅ セットアップ完了"
        echo ""
        echo "【使い方】"
        echo "  手動起動:"
        echo "    ./scripts/start_nemotron.sh"
        echo ""
        echo "  systemd サービス:"
        echo "    sudo systemctl enable bushidan-nemotron  # 自動起動有効化"
        echo "    sudo systemctl start bushidan-nemotron   # 起動"
        echo "    sudo systemctl status bushidan-nemotron  # 状態確認"
        echo ""
        echo "  API 確認:"
        echo "    curl http://localhost:${PORT}/health"
        echo ""
        echo "  EliteDesk 本陣からの接続確認:"
        echo "    curl http://${PRODESDK_IP}:${PORT}/health"
        echo ""
        echo "【武士団設定】"
        echo "  config/settings.yaml:"
        echo "    onmitsu:"
        echo "      local:"
        echo "        endpoint: 'http://${PRODESDK_IP}:${PORT}'"
        echo "        model_path: '${MODEL_DIR}/${MODEL_NAME}'"
        echo ""
        echo "【モデル情報】"
        echo "  NVIDIA Nemotron-3-Nano-30B-A3B (Q4_K_M)"
        echo "  サイズ: ~21GB | 推論速度: 15-25 tok/s (CPU)"
        echo "  コスト: ¥0 (電気代のみ ~¥3/日)"
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
    echo "  --auto      自動モード (全処理を確認なしで実行)"
    echo "  --build     llama.cpp のビルドのみ"
    echo "  --model     Nemotron モデルダウンロードのみ"
    echo "  --service   systemd サービス作成のみ"
    echo "  --verify    検証のみ"
    echo "  --help      このヘルプを表示"
    echo ""
    echo "【セットアップ後の起動】"
    echo "  ./scripts/start_nemotron.sh"
    echo ""
    echo "【API エンドポイント】"
    echo "  http://${PRODESDK_IP}:${PORT} (LAN 内)"
}

run_full_setup() {
    echo ""
    echo "============================================="
    echo "  武士団 v11.4 - 隠密の練兵"
    echo "  Nemotron-3-Nano Setup (HP ProDesk 600)"
    echo "============================================="
    echo ""

    show_system_info

    if ! check_dependencies; then
        CORE_FAILED=1
        print_final_report
        exit 1
    fi

    if [ $AUTO_MODE -eq 0 ]; then
        echo "セットアップオプション:"
        echo "  1) フルセットアップ (ビルド + モデル + サービス)"
        echo "  2) llama.cpp のみビルド"
        echo "  3) モデルのみダウンロード (~21GB)"
        echo "  4) サービスのみ作成"
        echo "  5) 検証のみ"
        echo ""
        read -p "選択 [1-5] (デフォルト: 1): " choice
        choice=${choice:-1}

        case $choice in
            1)
                build_llama_cpp
                download_model
                create_start_script
                create_systemd_service
                verify_setup
                ;;
            2) build_llama_cpp ;;
            3) download_model ;;
            4) create_start_script && create_systemd_service ;;
            5) verify_setup ;;
            *)
                log_error "無効な選択: $choice"
                exit 1
                ;;
        esac
    else
        # 自動モード: 全処理
        build_llama_cpp
        download_model
        create_start_script
        create_systemd_service
        verify_setup
    fi

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
        --build)
            check_dependencies && build_llama_cpp
            print_final_report
            exit $CORE_FAILED
            ;;
        --model)
            download_model
            print_final_report
            exit $CORE_FAILED
            ;;
        --service)
            create_start_script
            create_systemd_service
            print_final_report
            exit $CORE_FAILED
            ;;
        --verify)
            verify_setup
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
