#!/bin/bash
# ============================================================================
# 武士団マルチエージェントシステム v11.5 - ローカルLLM総合セットアップ
# ============================================================================
#
# 【対象マシン】
#   HP EliteDesk (192.168.11.239)
#   Intel Core i5-8100 (4C/4T), 32GB DDR4
#
# 【セットアップ対象モデル】
#   port 8080 - Nemotron-3-Nano-30B (常駐・機密処理)    ~22GB Q4_K_M
#   port 8081 - Llama-3-ELYZA-JP-8B (オンデマンド・日本語)  ~5GB Q4_K_M
#   port 8081 - ELYZA-japanese-Llama-2-7b (フォールバック)   ~4.5GB Q4_K_M
#
# 【メモリ配分】
#   Nemotron 22GB + ELYZA-8B 5GB + OS 5GB ≈ 32GB (ギリギリ共存可能)
#
# Usage:
#   ./setup/setup_local_llm.sh              # 対話モード
#   ./setup/setup_local_llm.sh --auto       # 全自動 (Nemotron + ELYZA Llama-3)
#   ./setup/setup_local_llm.sh --nemotron   # Nemotronのみ
#   ./setup/setup_local_llm.sh --elyza      # ELYZA Llama-3-8Bのみ
#   ./setup/setup_local_llm.sh --elyza-fb   # ELYZA Llama-2-7b フォールバックのみ
#   ./setup/setup_local_llm.sh --build      # llama.cpp ビルドのみ
#   ./setup/setup_local_llm.sh --verify     # 検証のみ
#   ./setup/setup_local_llm.sh --service    # systemd サービス作成のみ
#
# ============================================================================

# ============================================================================
# 設定
# ============================================================================

LLAMA_CPP_DIR="${HOME}/llama.cpp"
PROJECT_DIR="${HOME}/Bushidan-Multi-Agent"
MODEL_BASE_DIR="${PROJECT_DIR}/models"

# ── Nemotron-3-Nano-30B (port 8080・常駐) ─────────────────────────────────
NEMOTRON_DIR="${MODEL_BASE_DIR}/nemotron"
NEMOTRON_FILE="Nemotron-3-Nano-30B-A3B-Instruct-Q4_K_M.gguf"
NEMOTRON_HF_REPO="bartowski/Nemotron-3-Nano-30B-A3B-Instruct-GGUF"
NEMOTRON_URL="https://huggingface.co/bartowski/Nemotron-3-Nano-30B-A3B-Instruct-GGUF/resolve/main/Nemotron-3-Nano-30B-A3B-Instruct-Q4_K_M.gguf"
NEMOTRON_PORT=8080

# ── ELYZA Llama-3-8B (port 8081・オンデマンド・推奨) ──────────────────────
ELYZA3_DIR="${MODEL_BASE_DIR}/elyza-llama3"
ELYZA3_FILE="Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
ELYZA3_HF_REPO="elyza/Llama-3-ELYZA-JP-8B-GGUF"
ELYZA3_URL="https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B-GGUF/resolve/main/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
ELYZA3_PORT=8081

# ── ELYZA Llama-2-7b (port 8081・フォールバック) ──────────────────────────
ELYZA2_DIR="${MODEL_BASE_DIR}/elyza-llama2"
ELYZA2_FILE="ELYZA-japanese-Llama-2-7b-instruct-q4_k_m.gguf"
# GGUF変換済みコミュニティ版 (mmnga)
ELYZA2_HF_REPO="mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf"
ELYZA2_URL="https://huggingface.co/mmnga/ELYZA-japanese-Llama-2-7b-instruct-gguf/resolve/main/ELYZA-japanese-Llama-2-7b-instruct-q4_K_M.gguf"
ELYZA2_PORT=8081

# ── CPU設定 (i5-8100: 4C/4T) ──────────────────────────────────────────────
CPU_THREADS=4
HOST="0.0.0.0"
LOCAL_IP="192.168.11.239"

# ── 実行モード ──────────────────────────────────────────────────────────────
AUTO_MODE=0
RUN_NEMOTRON=0
RUN_ELYZA3=0
RUN_ELYZA2=0

# ============================================================================
# 結果トラッキング
# ============================================================================

declare -A RESULTS
CORE_FAILED=0

mark_success() { RESULTS["$1"]="OK"; }
mark_failed()  { RESULTS["$1"]="FAILED"; [[ "$2" == "core" ]] && CORE_FAILED=1; }
mark_skipped() { RESULTS["$1"]="SKIPPED"; }

# ============================================================================
# ユーティリティ
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
        if eval "$cmd"; then return 0; fi
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
    [ $AUTO_MODE -eq 1 ] && return 0
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
# システム情報
# ============================================================================

show_system_info() {
    echo ""
    echo "============================================="
    echo "  🏯 武士団 v11.5 - ローカルLLM セットアップ"
    echo "  EliteDesk (${LOCAL_IP})"
    echo "============================================="
    echo ""

    local cpu_model total_mem free_disk total_gb
    cpu_model=$(lscpu 2>/dev/null | grep 'Model name' | sed 's/Model name:\s*//' || echo "不明")
    total_mem=$(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo "不明")
    free_disk=$(df -h "${PROJECT_DIR}" 2>/dev/null | tail -1 | awk '{print $4}' || echo "不明")
    total_gb=$(free -g 2>/dev/null | grep Mem | awk '{print $2}' || echo "0")

    echo "  CPU:       $cpu_model"
    echo "  メモリ:    $total_mem"
    echo "  空きDisk:  $free_disk"
    echo ""

    echo "  【メモリ計算】"
    echo "    Nemotron-30B  : ~22GB (常駐)"
    echo "    ELYZA-8B      : ~5GB  (オンデマンド)"
    echo "    OS/その他     : ~5GB"
    echo "    合計          : ~32GB (${total_gb}GB実装)"
    echo ""

    if [ "$total_gb" -lt 28 ]; then
        log_warning "RAM ${total_gb}GB - 32GB 推奨。スワップ発生の可能性"
    elif [ "$total_gb" -ge 30 ]; then
        log_success "RAM ${total_gb}GB: Nemotron+ELYZA 共存可能 ✅"
    else
        log_warning "RAM ${total_gb}GB: ギリギリ。Nemotron単体は問題なし"
    fi
    echo ""
}

check_dependencies() {
    log_info "依存関係確認中..."
    local missing=()
    for cmd in git cmake make gcc g++ wget curl; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "必須コマンドがありません: ${missing[*]}"
        log_info "インストール: sudo apt install build-essential cmake git wget curl"
        mark_failed "dependencies" "core"
        return 1
    fi
    mark_success "dependencies"
    log_success "依存関係OK"
}

# ============================================================================
# llama.cpp ビルド (共通)
# ============================================================================

build_llama_cpp() {
    log_info "llama.cpp ビルド中..."

    if [ -d "$LLAMA_CPP_DIR" ]; then
        log_info "既存の llama.cpp を更新中..."
        cd "$LLAMA_CPP_DIR"
        retry_network "git pull origin master 2>/dev/null" || log_warning "更新失敗 (既存コードを使用)"
    else
        log_info "llama.cpp をクローン中..."
        if ! retry_network "git clone https://github.com/ggerganov/llama.cpp.git '$LLAMA_CPP_DIR'"; then
            log_error "llama.cpp のクローンに失敗"
            mark_failed "llama_clone" "core"; return 1
        fi
        cd "$LLAMA_CPP_DIR"
    fi
    mark_success "llama_clone"

    mkdir -p build && cd build

    log_info "CMake 設定中 (CPU最適化: AVX2, FMA / i5-8100)..."
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
        mark_failed "cmake_config" "core"; return 1
    fi

    local cores
    cores=$(nproc 2>/dev/null || echo "4")
    log_info "ビルド中 (${cores}コア)..."
    if make -j"$cores" 2>/dev/null; then
        mark_success "llama_build"
        log_success "llama.cpp ビルド完了"
    else
        log_error "llama.cpp ビルド失敗"
        mark_failed "llama_build" "core"; return 1
    fi
}

# llama-server バイナリのパスを返す
find_llama_server() {
    local p1="$LLAMA_CPP_DIR/build/bin/llama-server"
    local p2="$LLAMA_CPP_DIR/build/llama-server"
    [ -f "$p1" ] && echo "$p1" && return
    [ -f "$p2" ] && echo "$p2" && return
    echo ""
}

# ============================================================================
# モデルダウンロード (共通)
# ============================================================================

download_model() {
    local label="$1"
    local model_dir="$2"
    local model_file="$3"
    local hf_repo="$4"
    local direct_url="$5"
    local size_hint="$6"

    log_info "${label} モデルダウンロード中... (${size_hint})"
    mkdir -p "$model_dir"
    cd "$model_dir"

    if [ -f "$model_file" ]; then
        local sz
        sz=$(du -h "$model_file" 2>/dev/null | cut -f1 || echo "不明")
        log_warning "既存ファイルあり: $model_file ($sz)"
        confirm "再ダウンロードしますか?" || { mark_success "dl_${label}"; return 0; }
    fi

    # huggingface-cli 優先
    if command -v huggingface-cli &>/dev/null && [ -n "$hf_repo" ]; then
        log_info "huggingface-cli でダウンロード中..."
        if huggingface-cli download "$hf_repo" "$model_file" \
            --local-dir "$model_dir" \
            --local-dir-use-symlinks False 2>/dev/null; then
            mark_success "dl_${label}"
            log_success "${label} ダウンロード完了"
            return 0
        fi
        log_warning "huggingface-cli 失敗 → wget にフォールバック"
    fi

    # wget (再開可能)
    log_info "wget でダウンロード中... (途中から再開可能)"
    if wget -c "$direct_url" -O "$model_file" 2>/dev/null; then
        mark_success "dl_${label}"
        log_success "${label} ダウンロード完了"
    else
        log_error "${label} ダウンロード失敗"
        echo ""
        echo "  手動ダウンロード:"
        echo "    mkdir -p ${model_dir}"
        echo "    # huggingface-cli 方法:"
        echo "    pip install huggingface_hub"
        echo "    huggingface-cli download ${hf_repo} ${model_file} --local-dir ${model_dir}"
        echo "    # wget 方法:"
        echo "    wget -c '${direct_url}' -O '${model_dir}/${model_file}'"
        echo ""
        mark_failed "dl_${label}" "core"
        return 1
    fi
}

# ============================================================================
# 起動スクリプト作成
# ============================================================================

create_start_script_nemotron() {
    log_info "Nemotron 起動スクリプト作成中..."
    local server_path
    server_path=$(find_llama_server)
    [ -z "$server_path" ] && server_path="$LLAMA_CPP_DIR/build/bin/llama-server"

    mkdir -p "${PROJECT_DIR}/scripts"
    local script="${PROJECT_DIR}/scripts/start_nemotron.sh"

    cat > "$script" << SCRIPT
#!/bin/bash
# 武士団 v11.5 - 隠密 Nemotron-3-Nano 起動スクリプト
# port ${NEMOTRON_PORT} | 常駐 | 機密・オフライン専用

SERVER="${server_path}"
MODEL="${NEMOTRON_DIR}/${NEMOTRON_FILE}"

[ ! -f "\$SERVER" ] && echo "[ERROR] llama-server not found: \$SERVER" && exit 1
[ ! -f "\$MODEL"  ] && echo "[ERROR] Model not found: \$MODEL" && exit 1

ulimit -l unlimited

echo "====================================="
echo "  🥷 Nemotron-3-Nano (port ${NEMOTRON_PORT})"
echo "  Endpoint: http://${LOCAL_IP}:${NEMOTRON_PORT}"
echo "====================================="

\$SERVER \\
    -m "\$MODEL" \\
    -c 8192 \\
    -t ${CPU_THREADS} \\
    -b 512 \\
    --parallel 1 \\
    --host ${HOST} \\
    --port ${NEMOTRON_PORT} \\
    --mlock \\
    --mmap
SCRIPT

    chmod +x "$script"
    mark_success "script_nemotron"
    log_success "起動スクリプト: $script"
}

create_start_script_elyza3() {
    log_info "ELYZA Llama-3-8B 起動スクリプト作成中..."
    local server_path
    server_path=$(find_llama_server)
    [ -z "$server_path" ] && server_path="$LLAMA_CPP_DIR/build/bin/llama-server"

    mkdir -p "${PROJECT_DIR}/scripts"
    local script="${PROJECT_DIR}/scripts/start_elyza_llama3.sh"

    cat > "$script" << SCRIPT
#!/bin/bash
# 武士団 v11.5 - ELYZA Llama-3-8B 起動スクリプト
# port ${ELYZA3_PORT} | オンデマンド | 日本語特化

SERVER="${server_path}"
MODEL="${ELYZA3_DIR}/${ELYZA3_FILE}"

[ ! -f "\$SERVER" ] && echo "[ERROR] llama-server not found: \$SERVER" && exit 1
[ ! -f "\$MODEL"  ] && echo "[ERROR] Model not found: \$MODEL" && exit 1

ulimit -l unlimited

echo "====================================="
echo "  🎌 ELYZA Llama-3-JP-8B (port ${ELYZA3_PORT})"
echo "  Endpoint: http://${LOCAL_IP}:${ELYZA3_PORT}"
echo "  ⚠️  Nemotron (port ${NEMOTRON_PORT}) と同時起動でメモリ使用 ~27GB"
echo "====================================="

\$SERVER \\
    -m "\$MODEL" \\
    -c 4096 \\
    -t ${CPU_THREADS} \\
    -b 256 \\
    --parallel 1 \\
    --host ${HOST} \\
    --port ${ELYZA3_PORT} \\
    --chat-template llama3 \\
    --mlock \\
    --mmap
SCRIPT

    chmod +x "$script"
    mark_success "script_elyza3"
    log_success "起動スクリプト: $script"
}

create_start_script_elyza2() {
    log_info "ELYZA Llama-2-7b 起動スクリプト作成中..."
    local server_path
    server_path=$(find_llama_server)
    [ -z "$server_path" ] && server_path="$LLAMA_CPP_DIR/build/bin/llama-server"

    mkdir -p "${PROJECT_DIR}/scripts"
    local script="${PROJECT_DIR}/scripts/start_elyza_llama2.sh"

    cat > "$script" << SCRIPT
#!/bin/bash
# 武士団 v11.5 - ELYZA Llama-2-7b 起動スクリプト
# port ${ELYZA2_PORT} | オンデマンド | 日本語フォールバック

SERVER="${server_path}"
MODEL="${ELYZA2_DIR}/${ELYZA2_FILE}"

[ ! -f "\$SERVER" ] && echo "[ERROR] llama-server not found: \$SERVER" && exit 1
[ ! -f "\$MODEL"  ] && echo "[ERROR] Model not found: \$MODEL" && exit 1

ulimit -l unlimited

echo "====================================="
echo "  🎌 ELYZA Llama-2-7b (port ${ELYZA2_PORT})"
echo "  Endpoint: http://${LOCAL_IP}:${ELYZA2_PORT}"
echo "====================================="

\$SERVER \\
    -m "\$MODEL" \\
    -c 4096 \\
    -t ${CPU_THREADS} \\
    -b 256 \\
    --parallel 1 \\
    --host ${HOST} \\
    --port ${ELYZA2_PORT} \\
    --mlock \\
    --mmap
SCRIPT

    chmod +x "$script"
    mark_success "script_elyza2"
    log_success "起動スクリプト: $script"
}

# ============================================================================
# systemd サービス作成
# ============================================================================

_write_service() {
    local service_name="$1"
    local content="$2"
    local service_file="/etc/systemd/system/${service_name}.service"

    if [ "$(id -u)" -eq 0 ]; then
        echo "$content" > "$service_file"
        systemctl daemon-reload 2>/dev/null || true
        log_success "systemd サービス作成: ${service_name}"
        return 0
    elif sudo -n true 2>/dev/null; then
        echo "$content" | sudo tee "$service_file" > /dev/null
        sudo systemctl daemon-reload 2>/dev/null || true
        log_success "systemd サービス作成: ${service_name}"
        return 0
    else
        log_warning "root権限なし。手動で以下を ${service_file} に保存:"
        echo "---"; echo "$content"; echo "---"
        return 1
    fi
}

create_systemd_services() {
    log_info "systemd サービス作成中..."
    local server_path
    server_path=$(find_llama_server)

    if [ -z "$server_path" ]; then
        log_warning "llama-server 未ビルド。サービス作成スキップ"
        mark_skipped "systemd"; return 0
    fi

    # Nemotron (常駐)
    if [ $RUN_NEMOTRON -eq 1 ] || [ $AUTO_MODE -eq 1 ]; then
        if _write_service "bushidan-nemotron" "[Unit]
Description=Bushidan v11.5 - Onmitsu Nemotron-3-Nano (port ${NEMOTRON_PORT})
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${HOME}
ExecStart=${server_path} -m ${NEMOTRON_DIR}/${NEMOTRON_FILE} -c 8192 -t ${CPU_THREADS} -b 512 --parallel 1 --host ${HOST} --port ${NEMOTRON_PORT} --mlock --mmap
Restart=on-failure
RestartSec=10
MemoryMax=24G
LimitMEMLOCK=infinity
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target"; then
            mark_success "svc_nemotron"
        else
            mark_skipped "svc_nemotron"
        fi
    fi

    # ELYZA Llama-3 (オンデマンド: 自動起動はしない)
    if [ $RUN_ELYZA3 -eq 1 ] || [ $AUTO_MODE -eq 1 ]; then
        if _write_service "bushidan-elyza-llama3" "[Unit]
Description=Bushidan v11.5 - ELYZA Llama-3-JP-8B (port ${ELYZA3_PORT}, on-demand)
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${HOME}
ExecStart=${server_path} -m ${ELYZA3_DIR}/${ELYZA3_FILE} -c 4096 -t ${CPU_THREADS} -b 256 --parallel 1 --host ${HOST} --port ${ELYZA3_PORT} --mlock --mmap
Restart=on-failure
RestartSec=10
MemoryMax=6G
LimitMEMLOCK=infinity
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target"; then
            mark_success "svc_elyza3"
        else
            mark_skipped "svc_elyza3"
        fi
    fi

    # ELYZA Llama-2 (フォールバック: 自動起動はしない)
    if [ $RUN_ELYZA2 -eq 1 ]; then
        if _write_service "bushidan-elyza-llama2" "[Unit]
Description=Bushidan v11.5 - ELYZA Llama-2-7b (port ${ELYZA2_PORT}, fallback)
After=network.target

[Service]
Type=simple
User=${USER}
WorkingDirectory=${HOME}
ExecStart=${server_path} -m ${ELYZA2_DIR}/${ELYZA2_FILE} -c 4096 -t ${CPU_THREADS} -b 256 --parallel 1 --host ${HOST} --port ${ELYZA2_PORT} --mlock --mmap
Restart=on-failure
RestartSec=10
MemoryMax=6G
LimitMEMLOCK=infinity
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target"; then
            mark_success "svc_elyza2"
        else
            mark_skipped "svc_elyza2"
        fi
    fi
}

# ============================================================================
# 検証
# ============================================================================

verify_setup() {
    log_info "セットアップ検証中..."

    local server_path
    server_path=$(find_llama_server)

    if [ -n "$server_path" ]; then
        log_success "llama-server: OK ($server_path)"
        mark_success "verify_server"
    else
        log_error "llama-server: 見つかりません (先に --build を実行)"
        mark_failed "verify_server"
    fi

    # Nemotron
    if [ -f "${NEMOTRON_DIR}/${NEMOTRON_FILE}" ]; then
        local sz
        sz=$(du -h "${NEMOTRON_DIR}/${NEMOTRON_FILE}" | cut -f1)
        log_success "Nemotron モデル: OK ($sz)"
        mark_success "verify_nemotron_model"
    else
        log_warning "Nemotron モデル: なし (${NEMOTRON_DIR}/${NEMOTRON_FILE})"
        mark_skipped "verify_nemotron_model"
    fi

    # ELYZA Llama-3
    if [ -f "${ELYZA3_DIR}/${ELYZA3_FILE}" ]; then
        local sz
        sz=$(du -h "${ELYZA3_DIR}/${ELYZA3_FILE}" | cut -f1)
        log_success "ELYZA Llama-3-8B: OK ($sz)"
        mark_success "verify_elyza3_model"
    else
        log_warning "ELYZA Llama-3-8B: なし (${ELYZA3_DIR}/${ELYZA3_FILE})"
        mark_skipped "verify_elyza3_model"
    fi

    # ELYZA Llama-2
    if [ -f "${ELYZA2_DIR}/${ELYZA2_FILE}" ]; then
        local sz
        sz=$(du -h "${ELYZA2_DIR}/${ELYZA2_FILE}" | cut -f1)
        log_success "ELYZA Llama-2-7b: OK ($sz)"
        mark_success "verify_elyza2_model"
    else
        log_info "ELYZA Llama-2-7b: なし (フォールバック用・任意)"
        mark_skipped "verify_elyza2_model"
    fi

    # メモリ
    local total_gb
    total_gb=$(free -g 2>/dev/null | grep Mem | awk '{print $2}' || echo "0")
    if [ "$total_gb" -ge 30 ]; then
        log_success "メモリ: ${total_gb}GB (Nemotron+ELYZA 共存可能)"
        mark_success "verify_memory"
    elif [ "$total_gb" -ge 24 ]; then
        log_warning "メモリ: ${total_gb}GB (Nemotron単体は可。ELYZA共存は要注意)"
        mark_skipped "verify_memory"
    else
        log_error "メモリ: ${total_gb}GB (不足)"
        mark_failed "verify_memory"
    fi

    # サーバー稼働確認
    if curl -s --max-time 3 "http://localhost:${NEMOTRON_PORT}/health" > /dev/null 2>&1; then
        log_success "Nemotron サーバー: 稼働中 ✅ (port ${NEMOTRON_PORT})"
        mark_success "verify_nemotron_live"
    else
        log_info "Nemotron サーバー: 未起動"
        mark_skipped "verify_nemotron_live"
    fi

    if curl -s --max-time 3 "http://localhost:${ELYZA3_PORT}/health" > /dev/null 2>&1; then
        log_success "ELYZA サーバー: 稼働中 ✅ (port ${ELYZA3_PORT})"
        mark_success "verify_elyza_live"
    else
        log_info "ELYZA サーバー: 未起動 (オンデマンド)"
        mark_skipped "verify_elyza_live"
    fi
}

# ============================================================================
# 最終レポート
# ============================================================================

print_final_report() {
    echo ""
    echo "============================================="
    echo "  🏯 武士団 v11.5 - セットアップ完了レポート"
    echo "============================================="
    echo ""

    echo "【処理結果】"
    for key in "${!RESULTS[@]}"; do
        case "${RESULTS[$key]}" in
            OK)      echo "  ✅ $key" ;;
            FAILED)  echo "  ❌ $key" ;;
            SKIPPED) echo "  ⏭️  $key" ;;
        esac
    done
    echo ""

    if [ $CORE_FAILED -eq 1 ]; then
        echo "【状態】⚠️  エラーがあります。ログを確認して再実行してください。"
        return
    fi

    echo "【起動方法】"
    echo ""
    echo "  # Nemotron (常駐・機密処理)"
    echo "  ./scripts/start_nemotron.sh"
    echo "  # または systemd:"
    echo "  sudo systemctl enable --now bushidan-nemotron"
    echo ""
    echo "  # ELYZA Llama-3-8B (オンデマンド・日本語)"
    echo "  ./scripts/start_elyza_llama3.sh"
    echo "  # または systemd (自動起動なし):"
    echo "  sudo systemctl start bushidan-elyza-llama3"
    echo ""
    echo "【ヘルスチェック】"
    echo "  curl http://${LOCAL_IP}:${NEMOTRON_PORT}/health  # Nemotron"
    echo "  curl http://${LOCAL_IP}:${ELYZA3_PORT}/health    # ELYZA"
    echo ""
    echo "【メモリ状況確認】"
    echo "  free -h"
    echo "  # Nemotron: ~22GB, ELYZA同時: ~27GB"
    echo ""
    echo "【武士団の設定確認】"
    echo "  config/settings.yaml → onmitsu セクション"
    echo "    nemotron.endpoint: http://${LOCAL_IP}:${NEMOTRON_PORT}"
    echo "    elyza.primary.endpoint: http://${LOCAL_IP}:${ELYZA3_PORT}"
    echo ""
    echo "============================================="
}

# ============================================================================
# フルセットアップ
# ============================================================================

run_setup() {
    local do_nemotron=$1
    local do_elyza3=$2
    local do_elyza2=$3

    show_system_info
    check_dependencies || { print_final_report; exit 1; }

    build_llama_cpp || { print_final_report; exit 1; }

    [ $do_nemotron -eq 1 ] && download_model "Nemotron-30B" \
        "$NEMOTRON_DIR" "$NEMOTRON_FILE" "$NEMOTRON_HF_REPO" "$NEMOTRON_URL" "~21GB"

    [ $do_elyza3 -eq 1 ] && download_model "ELYZA-Llama3-8B" \
        "$ELYZA3_DIR" "$ELYZA3_FILE" "$ELYZA3_HF_REPO" "$ELYZA3_URL" "~5GB"

    [ $do_elyza2 -eq 1 ] && download_model "ELYZA-Llama2-7b" \
        "$ELYZA2_DIR" "$ELYZA2_FILE" "$ELYZA2_HF_REPO" "$ELYZA2_URL" "~4.5GB"

    [ $do_nemotron -eq 1 ] && create_start_script_nemotron
    [ $do_elyza3 -eq 1 ]   && create_start_script_elyza3
    [ $do_elyza2 -eq 1 ]   && create_start_script_elyza2

    create_systemd_services
    verify_setup
    print_final_report
}

# ============================================================================
# 対話モード
# ============================================================================

run_interactive() {
    show_system_info
    echo "セットアップするモデルを選択してください:"
    echo "  1) 全部 (Nemotron + ELYZA Llama-3) ← 推奨"
    echo "  2) Nemotron のみ (常駐・機密処理)"
    echo "  3) ELYZA Llama-3-8B のみ (日本語・オンデマンド)"
    echo "  4) ELYZA Llama-2-7b フォールバックのみ"
    echo "  5) 全部 (Nemotron + ELYZA Llama-3 + Llama-2フォールバック)"
    echo "  6) llama.cpp ビルドのみ"
    echo "  7) 起動スクリプト/サービスのみ (再作成)"
    echo "  8) 検証のみ"
    echo ""
    read -p "選択 [1-8] (デフォルト: 1): " choice
    choice=${choice:-1}

    case $choice in
        1) RUN_NEMOTRON=1; RUN_ELYZA3=1; run_setup 1 1 0 ;;
        2) RUN_NEMOTRON=1; run_setup 1 0 0 ;;
        3) RUN_ELYZA3=1;   run_setup 0 1 0 ;;
        4) RUN_ELYZA2=1;   run_setup 0 0 1 ;;
        5) RUN_NEMOTRON=1; RUN_ELYZA3=1; RUN_ELYZA2=1; run_setup 1 1 1 ;;
        6) check_dependencies && build_llama_cpp; print_final_report ;;
        7)
            create_start_script_nemotron
            create_start_script_elyza3
            create_start_script_elyza2
            create_systemd_services
            print_final_report
            ;;
        8) verify_setup; print_final_report ;;
        *) log_error "無効な選択: $choice"; exit 1 ;;
    esac
}

# ============================================================================
# エントリーポイント
# ============================================================================

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "  --auto        全自動 (Nemotron + ELYZA Llama-3)"
    echo "  --nemotron    Nemotronのみ (port ${NEMOTRON_PORT})"
    echo "  --elyza       ELYZA Llama-3-8Bのみ (port ${ELYZA3_PORT})"
    echo "  --elyza-fb    ELYZA Llama-2-7b フォールバックのみ"
    echo "  --build       llama.cpp ビルドのみ"
    echo "  --service     起動スクリプト/systemd サービスのみ"
    echo "  --verify      検証のみ"
    echo "  --help        このヘルプ"
}

if [[ $# -eq 0 ]]; then
    run_interactive
    exit $CORE_FAILED
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --auto)
            AUTO_MODE=1; RUN_NEMOTRON=1; RUN_ELYZA3=1
            run_setup 1 1 0
            exit $CORE_FAILED
            ;;
        --nemotron)
            RUN_NEMOTRON=1
            run_setup 1 0 0
            exit $CORE_FAILED
            ;;
        --elyza)
            RUN_ELYZA3=1
            run_setup 0 1 0
            exit $CORE_FAILED
            ;;
        --elyza-fb)
            RUN_ELYZA2=1
            run_setup 0 0 1
            exit $CORE_FAILED
            ;;
        --build)
            check_dependencies && build_llama_cpp
            print_final_report
            exit $CORE_FAILED
            ;;
        --service)
            RUN_NEMOTRON=1; RUN_ELYZA3=1; RUN_ELYZA2=1
            create_start_script_nemotron
            create_start_script_elyza3
            create_start_script_elyza2
            create_systemd_services
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
    shift
done
