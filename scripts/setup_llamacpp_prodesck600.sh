#!/bin/bash
# ============================================================================
# Bushidan Multi-Agent System v9.4 - llama.cpp Setup for HP ProDesk 600
# ============================================================================
#
# HP ProDesk 600 スペック想定:
# - Intel Core i5-10500 または i7-10700 (6-8コア)
# - 16-32GB DDR4 RAM
# - ディスクリートGPU無し（Intel UHD Graphics）
#
# このスクリプトは以下を行います:
# 1. llama.cppをビルド（CPU最適化）
# 2. Qwen3-Coder-30Bモデルをダウンロード
# 3. systemdサービスをセットアップ
# 4. 環境検証
#
# 使用方法:
#   chmod +x scripts/setup_llamacpp_prodesck600.sh
#   ./scripts/setup_llamacpp_prodesck600.sh
#
# ============================================================================

set -e  # エラー時停止

# 色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# 設定
# ============================================================================

LLAMA_CPP_DIR="${HOME}/llama.cpp"
MODEL_DIR="${HOME}/Bushidan-Multi-Agent/models/qwen3"
MODEL_NAME="Qwen3-Coder-30B-Q4_K_M.gguf"
MODEL_URL="https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf?download=true"

# HP ProDesk 600 CPU最適化設定
CPU_THREADS=8          # i7-10700は8コア
CONTEXT_SIZE=4096      # 速度最適化のため縮小
BATCH_SIZE=512         # CPU最適バッチサイズ
PARALLEL_REQUESTS=1    # CPUの安定性のため単一リクエスト
PORT=8080

# ============================================================================
# 依存関係チェック
# ============================================================================

check_dependencies() {
    log_info "依存関係を確認中..."

    local missing=()

    # 必須コマンド
    for cmd in git cmake make gcc g++ curl wget; do
        if ! command -v $cmd &> /dev/null; then
            missing+=($cmd)
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "以下のパッケージをインストールしてください: ${missing[*]}"
        log_info "Ubuntu/Debian: sudo apt install ${missing[*]}"
        exit 1
    fi

    log_success "依存関係OK"
}

# ============================================================================
# システム情報表示
# ============================================================================

show_system_info() {
    log_info "システム情報:"
    echo "  CPU: $(lscpu | grep 'Model name' | sed 's/Model name:\s*//')"
    echo "  コア数: $(nproc)"
    echo "  メモリ: $(free -h | grep Mem | awk '{print $2}')"
    echo "  空きメモリ: $(free -h | grep Mem | awk '{print $4}')"
    echo "  空きディスク: $(df -h . | tail -1 | awk '{print $4}')"
    echo ""
}

# ============================================================================
# llama.cppビルド
# ============================================================================

build_llama_cpp() {
    log_info "llama.cppをビルド中..."

    # クローンまたは更新
    if [ -d "$LLAMA_CPP_DIR" ]; then
        log_info "既存のllama.cppを更新中..."
        cd "$LLAMA_CPP_DIR"
        git pull origin master
    else
        log_info "llama.cppをクローン中..."
        git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
        cd "$LLAMA_CPP_DIR"
    fi

    # ビルドディレクトリ作成
    mkdir -p build
    cd build

    # CMake設定（CPU最適化）
    log_info "CMake設定中（CPU最適化）..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_NATIVE=ON \
        -DLLAMA_AVX2=ON \
        -DLLAMA_FMA=ON \
        -DLLAMA_F16C=ON \
        -DLLAMA_BUILD_SERVER=ON

    # ビルド（全コア使用）
    log_info "ビルド中（$(nproc)コア使用）..."
    make -j$(nproc)

    log_success "llama.cppビルド完了"
}

# ============================================================================
# モデルダウンロード
# ============================================================================

download_model() {
    log_info "Qwen3-Coder-30Bモデルをダウンロード中..."

    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"

    if [ -f "$MODEL_NAME" ]; then
        log_warning "モデルファイルは既に存在します: $MODEL_NAME"
        read -p "再ダウンロードしますか? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "既存のモデルを使用します"
            return
        fi
    fi

    log_info "モデルサイズ: 約17GB"
    log_info "ダウンロードには時間がかかります..."

    # wgetでダウンロード（再開可能）
    wget -c "$MODEL_URL" -O "$MODEL_NAME"

    log_success "モデルダウンロード完了"
}

# ============================================================================
# systemdサービス作成
# ============================================================================

create_systemd_service() {
    log_info "systemdサービスを作成中..."

    local SERVICE_FILE="/etc/systemd/system/bushidan-llamacpp.service"
    local SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"

    # サーバーバイナリ確認
    if [ ! -f "$SERVER_PATH" ]; then
        SERVER_PATH="$LLAMA_CPP_DIR/build/llama-server"
    fi

    if [ ! -f "$SERVER_PATH" ]; then
        log_error "llama-serverが見つかりません"
        log_info "手動でサーバーを起動してください"
        return
    fi

    # サービスファイル内容
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

# メモリ制限（RAM 16GBの場合は20GB、32GBの場合は24GBに調整）
MemoryMax=20G

[Install]
WantedBy=multi-user.target
"

    # rootでサービスファイル作成
    echo "$SERVICE_CONTENT" | sudo tee "$SERVICE_FILE" > /dev/null

    # systemd再読み込み
    sudo systemctl daemon-reload

    log_success "systemdサービス作成完了: bushidan-llamacpp.service"
    log_info "使用方法:"
    echo "  sudo systemctl enable bushidan-llamacpp  # 自動起動有効化"
    echo "  sudo systemctl start bushidan-llamacpp   # 起動"
    echo "  sudo systemctl status bushidan-llamacpp  # 状態確認"
    echo "  sudo systemctl stop bushidan-llamacpp    # 停止"
}

# ============================================================================
# 起動スクリプト作成
# ============================================================================

create_start_script() {
    log_info "起動スクリプトを作成中..."

    local SCRIPT_PATH="$HOME/Bushidan-Multi-Agent/scripts/start_llamacpp.sh"

    local SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"
    if [ ! -f "$SERVER_PATH" ]; then
        SERVER_PATH="$LLAMA_CPP_DIR/build/llama-server"
    fi

    cat > "$SCRIPT_PATH" << EOF
#!/bin/bash
# ============================================================================
# Bushidan v9.4 - llama.cpp Server起動スクリプト
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

echo "🏯 Bushidan llama.cpp Server 起動中..."
echo "  Model: \$MODEL_PATH"
echo "  Threads: \$THREADS"
echo "  Context: \$CONTEXT"
echo "  Port: \$PORT"

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

    chmod +x "$SCRIPT_PATH"
    log_success "起動スクリプト作成完了: $SCRIPT_PATH"
}

# ============================================================================
# 環境検証
# ============================================================================

verify_setup() {
    log_info "セットアップを検証中..."

    local errors=0

    # llama-server確認
    local SERVER_PATH="$LLAMA_CPP_DIR/build/bin/llama-server"
    if [ ! -f "$SERVER_PATH" ]; then
        SERVER_PATH="$LLAMA_CPP_DIR/build/llama-server"
    fi

    if [ -f "$SERVER_PATH" ]; then
        log_success "llama-server: OK"
    else
        log_error "llama-server: 見つかりません"
        ((errors++))
    fi

    # モデル確認
    if [ -f "$MODEL_DIR/$MODEL_NAME" ]; then
        local size=$(du -h "$MODEL_DIR/$MODEL_NAME" | cut -f1)
        log_success "モデル: OK ($size)"
    else
        log_error "モデル: 見つかりません"
        ((errors++))
    fi

    # メモリ確認
    local total_mem=$(free -g | grep Mem | awk '{print $2}')
    if [ "$total_mem" -ge 16 ]; then
        log_success "メモリ: OK (${total_mem}GB)"
    else
        log_warning "メモリ: ${total_mem}GB (16GB以上推奨)"
    fi

    if [ $errors -eq 0 ]; then
        log_success "セットアップ検証完了: すべてOK"
    else
        log_error "セットアップに問題があります: $errors 件のエラー"
    fi
}

# ============================================================================
# 使用方法表示
# ============================================================================

show_usage() {
    echo ""
    echo "============================================================================"
    echo " Bushidan v9.4 - llama.cpp セットアップ完了"
    echo "============================================================================"
    echo ""
    echo "【手動起動】"
    echo "  ./scripts/start_llamacpp.sh"
    echo ""
    echo "【systemdサービス】"
    echo "  sudo systemctl enable bushidan-llamacpp  # 自動起動有効化"
    echo "  sudo systemctl start bushidan-llamacpp   # 起動"
    echo "  sudo systemctl status bushidan-llamacpp  # 状態確認"
    echo ""
    echo "【API確認】"
    echo "  curl http://127.0.0.1:$PORT/health"
    echo ""
    echo "【Bushidan起動】"
    echo "  python main.py"
    echo ""
    echo "============================================================================"
}

# ============================================================================
# メイン処理
# ============================================================================

main() {
    echo ""
    echo "============================================================================"
    echo " Bushidan v9.4 - llama.cpp Setup for HP ProDesk 600"
    echo " CPU最適化ローカルLLM環境構築"
    echo "============================================================================"
    echo ""

    show_system_info
    check_dependencies

    # ステップ選択
    echo "セットアップオプション:"
    echo "  1) フルセットアップ（ビルド + モデル + サービス）"
    echo "  2) llama.cppのみビルド"
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

    show_usage
}

# 実行
main "$@"
