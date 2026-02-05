#!/bin/bash
# ============================================================================
# 武士団マルチエージェントシステム v9.4 - 侍大将の練兵 (CT 101)
# llama.cpp Setup for HP ProDesk 600
# ============================================================================
#
# 【コア要件】必ず達成すべきこと:
#   1. llama.cpp のビルド (llama-server バイナリ生成)
#   2. Qwen3-Coder-30B モデルのダウンロード
#   3. 起動スクリプト (start_llamacpp.sh) の作成
#
# 【最終結果】このスクリプト完了後の状態:
#   - ~/llama.cpp/build/bin/llama-server が実行可能
#   - models/qwen3/Qwen3-Coder-30B-Q4_K_M.gguf が配置
#   - scripts/start_llamacpp.sh で手動起動可能
#   - systemd サービスでバックグラウンド起動可能 (オプション)
#
# 【動作保証】
#   - 必須プログラム (git, cmake, make, gcc, g++, wget) があれば異常終了しない
#   - 対話的プロンプトを --auto オプションでスキップ可能
#   - ネットワークエラーは自動リトライ
#
# HP ProDesk 600 スペック想定:
#   - Intel Core i5-8500 (6C/6T, HT無し)
#   - 16-32GB DDR4 RAM
#   - ディスクリートGPU無し (Intel UHD Graphics)
#
# Usage:
#   ./scripts/setup_llamacpp_prodesck600.sh           # 対話モード
#   ./scripts/setup_llamacpp_prodesck600.sh --auto    # 自動モード (全実行)
#   ./scripts/setup_llamacpp_prodesck600.sh --build   # ビルドのみ
#   ./scripts/setup_llamacpp_prodesck600.sh --model   # モデルのみ
#   ./scripts/setup_llamacpp_prodesck600.sh --verify  # 検証のみ
#
# ============================================================================

# エラー時に即座に終了しない
# set -e は使わない

# ============================================================================
# 設定
# ============================================================================

LLAMA_CPP_DIR="${HOME}/llama.cpp"
MODEL_DIR="${HOME}/Bushidan-Multi-Agent/models/qwen3"
MODEL_NAME="Qwen3-Coder-30B-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf?download=true"
PROJECT_DIR="${HOME}/Bushidan-Multi-Agent"

# HP ProDesk 600 CPU最適化設定
CPU_THREADS=6          # i5-8500は6コア (HT無し)
CONTEXT_SIZE=4096      # 速度最適化のため縮小
BATCH_SIZE=512         # CPU最適バッチサイズ
PARALLEL_REQUESTS=1    # CPUの安定性のため単一リクエスト
PORT=8080

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

# ネットワーク操作のリトライ (最大4回、指数バックオフ)
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

# 対話的確認 (--auto モードではスキップ)
confirm() {
    local prompt="$1"
    local default="${2:-n}"

    if [ $AUTO_MODE -eq 1 ]; then
        return 0  # 自動モードでは常にYes
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
# 前提条件チェック
# ============================================================================

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
        log_info "Ubuntu/Debian: sudo apt install ${missing[*]}"
        return 1
    fi

    mark_success "dependencies"
    log_success "依存関係OK"
    return 0
}

# ============================================================================
# システム情報表示
# ============================================================================

show_system_info() {
    log_info "システム情報:"

    # CPU情報
    local cpu_model=$(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name:\s*//' || echo "不明")
    local cpu_cores=$(nproc 2>/dev/null || echo "不明")

    # メモリ情報
    local total_mem=$(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo "不明")
    local free_mem=$(free -h 2>/dev/null | grep Mem | awk '{print $4}' || echo "不明")

    # ディスク情報
    local free_disk=$(df -h . 2>/dev/null | tail -1 | awk '{print $4}' || echo "不明")

    echo "  CPU: $cpu_model"
    echo "  コア数: $cpu_cores"
    echo "  メモリ: $total_mem (空き: $free_mem)"
    echo "  空きディスク: $free_disk"
    echo ""
}

# ============================================================================
# llama.cpp ビルド
# ============================================================================

build_llama_cpp() {
    log_info "llama.cpp をビルド中..."

    # クローンまたは更新
    if [ -d "$LLAMA_CPP_DIR" ]; then
        log_info "既存のllama.cppを更新中..."
        cd "$LLAMA_CPP_DIR"
        if retry_network "git pull origin master 2>/dev/null"; then
            log_success "リポジトリ更新完了"
        else
            log_warning "リポジトリ更新失敗 (既存コードを使用)"
        fi
    else
        log_info "llama.cppをクローン中..."
        if retry_network "git clone https://github.com/ggerganov/llama.cpp.git '$LLAMA_CPP_DIR'"; then
            log_success "クローン完了"
        else
            log_error "llama.cpp のクローンに失敗"
            mark_failed "llama_clone" "core"
            return 1
        fi
        cd "$LLAMA_CPP_DIR"
    fi
    mark_success "llama_clone"

    # ビルドディレクトリ作成
    mkdir -p build
    cd build

    # CMake設定
    log_info "CMake設定中 (CPU最適化)..."
    if cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_NATIVE=ON \
        -DLLAMA_AVX2=ON \
        -DLLAMA_FMA=ON \
        -DLLAMA_F16C=ON \
        -DLLAMA_BUILD_SERVER=ON 2>/dev/null; then
        mark_success "cmake_config"
        log_success "CMake設定完了"
    else
        log_error "CMake設定失敗"
        mark_failed "cmake_config" "core"
        return 1
    fi

    # ビルド
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
# モデルダウンロード
# ============================================================================

download_model() {
    log_info "Qwen3-Coder-30B モデルをダウンロード中..."

    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"

    # 既存ファイルチェック
    if [ -f "$MODEL_NAME" ]; then
        local size=$(du -h "$MODEL_NAME" 2>/dev/null | cut -f1 || echo "不明")
        log_warning "モデルファイルは既に存在します: $MODEL_NAME ($size)"

        if ! confirm "再ダウンロードしますか?"; then
            log_info "既存のモデルを使用します"
            mark_success "model_download"
            return 0
        fi
    fi

    log_info "モデルサイズ: 約17GB"
    log_info "ダウンロードには時間がかかります..."

    # wget でダウンロード (再開可能)
    if wget -c "$MODEL_URL" -O "$MODEL_NAME" 2>/dev/null; then
        mark_success "model_download"
        log_success "モデルダウンロード完了"
    else
        log_error "モデルダウンロード失敗"
        log_info "手動でダウンロードしてください:"
        echo "  wget -c '$MODEL_URL' -O '$MODEL_DIR/$MODEL_NAME'"
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

    local SCRIPT_PATH="${PROJECT_DIR}/scripts/start_llamacpp.sh"

    # サーバーパス検出
    local SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"
    if [ ! -f "$SERVER_PATH" ]; then
        SERVER_PATH="$LLAMA_CPP_DIR/build/llama-server"
    fi

    cat > "$SCRIPT_PATH" << EOF
#!/bin/bash
# ============================================================================
# Bushidan v9.4 - llama.cpp Server 起動スクリプト
# HP ProDesk 600 CPU最適化
# ============================================================================

SERVER_PATH="$SERVER_PATH"
MODEL_PATH="$MODEL_DIR/$MODEL_NAME"

# CPU設定
THREADS=$CPU_THREADS
CONTEXT=$CONTEXT_SIZE
BATCH=$BATCH_SIZE
PARALLEL=$PARALLEL_REQUESTS
PORT=$PORT

# サーバーバイナリ確認
if [ ! -f "\$SERVER_PATH" ]; then
    echo "[ERROR] llama-server が見つかりません: \$SERVER_PATH"
    echo "  ./scripts/setup_llamacpp_prodesck600.sh --build を実行してください"
    exit 1
fi

# モデル確認
if [ ! -f "\$MODEL_PATH" ]; then
    echo "[ERROR] モデルが見つかりません: \$MODEL_PATH"
    echo "  ./scripts/setup_llamacpp_prodesck600.sh --model を実行してください"
    exit 1
fi

echo "============================================="
echo "  Bushidan llama.cpp Server 起動"
echo "============================================="
echo "  Model: \$MODEL_PATH"
echo "  Threads: \$THREADS"
echo "  Context: \$CONTEXT"
echo "  Port: \$PORT"
echo "============================================="

\$SERVER_PATH \\
    -m "\$MODEL_PATH" \\
    -c \$CONTEXT \\
    -t \$THREADS \\
    -b \$BATCH \\
    --parallel \$PARALLEL \\
    --host 127.0.0.1 \\
    --port \$PORT \\
    --mlock \\
    --mmap

EOF

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

    local SERVICE_FILE="/etc/systemd/system/bushidan-llamacpp.service"

    # サーバーパス検出
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
Description=Bushidan llama.cpp Server (Qwen3-Coder-30B)
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME
ExecStart=$SERVER_PATH \\
    -m $MODEL_DIR/$MODEL_NAME \\
    -c $CONTEXT_SIZE \\
    -t $CPU_THREADS \\
    -b $BATCH_SIZE \\
    --parallel $PARALLEL_REQUESTS \\
    --host 127.0.0.1 \\
    --port $PORT \\
    --mlock \\
    --mmap
Restart=on-failure
RestartSec=10
MemoryMax=20G

[Install]
WantedBy=multi-user.target
"

    # root 権限チェック
    if [ "$(id -u)" -eq 0 ]; then
        echo "$SERVICE_CONTENT" > "$SERVICE_FILE"
        systemctl daemon-reload 2>/dev/null || true
        mark_success "systemd_service"
        log_success "systemd サービス作成: bushidan-llamacpp.service"
    else
        # sudo が使えるか確認
        if sudo -n true 2>/dev/null; then
            echo "$SERVICE_CONTENT" | sudo tee "$SERVICE_FILE" > /dev/null
            sudo systemctl daemon-reload 2>/dev/null || true
            mark_success "systemd_service"
            log_success "systemd サービス作成: bushidan-llamacpp.service"
        else
            log_warning "root 権限がありません。手動でサービスを作成してください"
            log_info "以下の内容を $SERVICE_FILE に保存:"
            echo "$SERVICE_CONTENT"
            mark_skipped "systemd_service"
        fi
    fi
}

# ============================================================================
# 環境検証
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
        log_success "llama-server: OK"
    else
        mark_failed "verify_server"
        log_error "llama-server: 見つかりません"
        ((errors++))
    fi

    # モデル確認
    if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
        local size=$(du -h "$MODEL_DIR/$MODEL_NAME" 2>/dev/null | cut -f1 || echo "不明")
        mark_success "verify_model"
        log_success "モデル: OK ($size)"
    else
        mark_failed "verify_model"
        log_error "モデル: 見つかりません"
        ((errors++))
    fi

    # 起動スクリプト確認
    if [ -x "${PROJECT_DIR}/scripts/start_llamacpp.sh" ]; then
        mark_success "verify_script"
        log_success "起動スクリプト: OK"
    else
        mark_failed "verify_script"
        log_error "起動スクリプト: 見つかりません"
        ((errors++))
    fi

    # メモリ確認
    local total_mem=$(free -g 2>/dev/null | grep Mem | awk '{print $2}' || echo "0")
    if [ "$total_mem" -ge 16 ]; then
        mark_success "verify_memory"
        log_success "メモリ: OK (${total_mem}GB)"
    else
        mark_skipped "verify_memory"
        log_warning "メモリ: ${total_mem}GB (16GB以上推奨)"
    fi

    if [ $errors -eq 0 ]; then
        log_success "検証完了: すべてOK"
        return 0
    else
        log_error "検証に問題があります: $errors 件のエラー"
        return 1
    fi
}

# ============================================================================
# 最終レポート
# ============================================================================

print_final_report() {
    echo ""
    echo "============================================="
    echo "  侍大将の練兵 - 完了レポート"
    echo "============================================="
    echo ""

    # 結果一覧
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

    # コア機能チェック
    if [ $CORE_FAILED -eq 1 ]; then
        echo "【状態】コア機能に失敗があります"
        echo ""
        echo "  必須コンポーネントが正しくインストールされていません。"
        echo "  エラーメッセージを確認し、再実行してください。"
    else
        echo "【状態】正常完了"
        echo ""
        echo "【使い方】"
        echo "  手動起動:"
        echo "    ./scripts/start_llamacpp.sh"
        echo ""
        echo "  systemd サービス:"
        echo "    sudo systemctl enable bushidan-llamacpp  # 自動起動有効化"
        echo "    sudo systemctl start bushidan-llamacpp   # 起動"
        echo "    sudo systemctl status bushidan-llamacpp  # 状態確認"
        echo ""
        echo "  API 確認:"
        echo "    curl http://127.0.0.1:$PORT/health"
        echo ""
        echo "  本陣から接続:"
        echo "    LLAMACPP_ENDPOINT=http://<CT101_IP>:$PORT"
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
    echo "  --model     モデルダウンロードのみ"
    echo "  --service   サービス作成のみ"
    echo "  --verify    検証のみ"
    echo "  --help      このヘルプを表示"
    echo ""
    echo "【コア要件】"
    echo "  1. llama.cpp のビルド"
    echo "  2. Qwen3-Coder-30B モデル"
    echo "  3. 起動スクリプト"
    echo ""
    echo "【最終結果】"
    echo "  - llama-server で推論サーバー起動可能"
    echo "  - systemd で自動起動可能"
}

run_full_setup() {
    echo ""
    echo "============================================="
    echo "  武士団 v9.4 - 侍大将の練兵"
    echo "  llama.cpp + Qwen3-Coder-30B Setup"
    echo "============================================="
    echo ""

    show_system_info

    if ! check_dependencies; then
        CORE_FAILED=1
        print_final_report
        exit 1
    fi

    # 対話的メニュー (--auto モードではスキップ)
    if [ $AUTO_MODE -eq 0 ]; then
        echo "セットアップオプション:"
        echo "  1) フルセットアップ (ビルド + モデル + サービス)"
        echo "  2) llama.cpp のみビルド"
        echo "  3) モデルのみダウンロード"
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
            2)
                build_llama_cpp
                ;;
            3)
                download_model
                ;;
            4)
                create_start_script
                create_systemd_service
                ;;
            5)
                verify_setup
                ;;
            *)
                log_error "無効な選択: $choice"
                exit 1
                ;;
        esac
    else
        # 自動モード: 全処理実行
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

# 引数解析
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

# 終了コード: コア機能が失敗した場合のみ 1
exit $CORE_FAILED
