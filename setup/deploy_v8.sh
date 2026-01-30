#!/bin/bash

# =============================================================================
# 将軍システム v8.0 - Deployment Script
# =============================================================================
# Complete v8.0 system deployment with all components
# =============================================================================

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHOGUN_ROOT="$(dirname "$SCRIPT_DIR")"
INSTALL_LOG="/tmp/shogun_v8_install.log"
PYTHON_VERSION="3.11"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$INSTALL_LOG"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✅ $1${NC}" | tee -a "$INSTALL_LOG"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}" | tee -a "$INSTALL_LOG"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ❌ $1${NC}" | tee -a "$INSTALL_LOG"
}

# Header
echo -e "${BLUE}"
cat << "EOF"
🏯 ============================================================
   将軍システム v8.0 - Deployment Script
   完全自律型AI開発環境の構築
============================================================ 🏯
EOF
echo -e "${NC}"

log "Starting Shogun System v8.0 deployment..."

# =============================================================================
# Phase 1: System Prerequisites Check
# =============================================================================

log "Phase 1: Checking system prerequisites..."

# Check if running on supported system
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    log_error "This script is designed for Linux systems only"
    exit 1
fi

# Check if running as root for system-level operations
if [[ $EUID -eq 0 ]]; then
    log_warning "Running as root - some operations may need adjustment"
    IS_ROOT=true
else
    IS_ROOT=false
fi

# Check available memory
AVAILABLE_MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [[ $AVAILABLE_MEMORY_GB -lt 20 ]]; then
    log_error "Insufficient memory: ${AVAILABLE_MEMORY_GB}GB (minimum 20GB required)"
    exit 1
fi

log_success "Memory check passed: ${AVAILABLE_MEMORY_GB}GB available"

# Check disk space
AVAILABLE_DISK_GB=$(df -BG "$SHOGUN_ROOT" | awk 'NR==2 {print $4}' | tr -d 'G')
if [[ $AVAILABLE_DISK_GB -lt 10 ]]; then
    log_error "Insufficient disk space: ${AVAILABLE_DISK_GB}GB (minimum 10GB required)"
    exit 1
fi

log_success "Disk space check passed: ${AVAILABLE_DISK_GB}GB available"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_INSTALLED_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log "Python version found: $PYTHON_INSTALLED_VERSION"
else
    log_error "Python 3 not found - please install Python 3.11+"
    exit 1
fi

# =============================================================================
# Phase 2: Install System Dependencies
# =============================================================================

log "Phase 2: Installing system dependencies..."

# Update package lists
if command -v apt-get &> /dev/null; then
    log "Updating package lists (apt)..."
    sudo apt-get update || log_warning "Failed to update package lists"
    
    # Install required packages
    PACKAGES=(
        "python3-pip"
        "python3-venv"
        "python3-dev"
        "build-essential"
        "curl"
        "wget"
        "git"
        "sqlite3"
        "docker.io"
        "docker-compose"
        "htop"
        "jq"
    )
    
    for package in "${PACKAGES[@]}"; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            log "Installing $package..."
            sudo apt-get install -y "$package" || log_warning "Failed to install $package"
        else
            log "$package already installed"
        fi
    done
    
elif command -v yum &> /dev/null; then
    log "Using yum package manager..."
    sudo yum update -y || log_warning "Failed to update packages"
    
    PACKAGES=(
        "python3-pip"
        "python3-devel"
        "gcc"
        "gcc-c++"
        "make"
        "curl"
        "wget"
        "git"
        "sqlite"
        "docker"
        "docker-compose"
        "htop"
        "jq"
    )
    
    for package in "${PACKAGES[@]}"; do
        sudo yum install -y "$package" || log_warning "Failed to install $package"
    done
    
else
    log_error "Unsupported package manager - please install dependencies manually"
    exit 1
fi

log_success "System dependencies installed"

# =============================================================================
# Phase 3: Docker Setup (for Qdrant)
# =============================================================================

log "Phase 3: Setting up Docker for Qdrant..."

# Start Docker service
if ! systemctl is-active --quiet docker; then
    log "Starting Docker service..."
    sudo systemctl start docker || log_error "Failed to start Docker"
    sudo systemctl enable docker || log_warning "Failed to enable Docker auto-start"
fi

# Add user to docker group if not root
if [[ $IS_ROOT == false ]]; then
    sudo usermod -aG docker "$USER" || log_warning "Failed to add user to docker group"
    log_warning "You may need to log out and back in for Docker permissions to take effect"
fi

log_success "Docker setup completed"

# =============================================================================
# Phase 4: Python Environment Setup
# =============================================================================

log "Phase 4: Setting up Python environment..."

cd "$SHOGUN_ROOT"

# Create virtual environment
if [[ ! -d "venv" ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv venv || {
        log_error "Failed to create virtual environment"
        exit 1
    }
fi

# Activate virtual environment
source venv/bin/activate || {
    log_error "Failed to activate virtual environment"
    exit 1
}

# Upgrade pip
log "Upgrading pip..."
pip install --upgrade pip || log_warning "Failed to upgrade pip"

# Install v8.0 dependencies
if [[ -f "requirements_v8.txt" ]]; then
    log "Installing v8.0 Python dependencies..."
    pip install -r requirements_v8.txt || {
        log_error "Failed to install Python dependencies"
        exit 1
    }
else
    log_error "requirements_v8.txt not found"
    exit 1
fi

log_success "Python environment setup completed"

# =============================================================================
# Phase 5: Qdrant (Knowledge Base) Setup
# =============================================================================

log "Phase 5: Setting up Qdrant vector database..."

# Create Qdrant data directory
QDRANT_DATA_DIR="/opt/shogun/qdrant_data"
sudo mkdir -p "$QDRANT_DATA_DIR" || log_error "Failed to create Qdrant data directory"
sudo chown -R "$USER:$USER" "$QDRANT_DATA_DIR" 2>/dev/null || log_warning "Could not change ownership of Qdrant directory"

# Create Qdrant Docker Compose file
cat > docker-compose.qdrant.yml << EOF
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.7.0
    container_name: shogun_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ${QDRANT_DATA_DIR}:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF

# Start Qdrant
log "Starting Qdrant container..."
docker-compose -f docker-compose.qdrant.yml up -d || {
    log_error "Failed to start Qdrant"
    exit 1
}

# Wait for Qdrant to be ready
log "Waiting for Qdrant to start..."
for i in {1..30}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        break
    fi
    sleep 2
    if [[ $i -eq 30 ]]; then
        log_error "Qdrant failed to start within 60 seconds"
        exit 1
    fi
done

log_success "Qdrant vector database started"

# =============================================================================
# Phase 6: Activity Memory (SQLite) Setup
# =============================================================================

log "Phase 6: Setting up Activity Memory database..."

# Create database directory
ACTIVITY_DB_DIR="/opt/shogun"
sudo mkdir -p "$ACTIVITY_DB_DIR" || log_error "Failed to create activity database directory"
sudo chown -R "$USER:$USER" "$ACTIVITY_DB_DIR" 2>/dev/null || log_warning "Could not change ownership of activity database directory"

# Initialize SQLite database
ACTIVITY_DB_PATH="/opt/shogun/taisho_activity.db"
if [[ ! -f "$ACTIVITY_DB_PATH" ]]; then
    log "Initializing Activity Memory database..."
    
    # Create empty database file
    touch "$ACTIVITY_DB_PATH"
    chmod 644 "$ACTIVITY_DB_PATH"
    
    # Database will be initialized by the application on first run
fi

log_success "Activity Memory database setup completed"

# =============================================================================
# Phase 7: Configuration Setup
# =============================================================================

log "Phase 7: Setting up configuration..."

# Copy configuration template if settings_v8.yaml doesn't exist
if [[ ! -f "config/settings_v8.yaml" ]]; then
    if [[ -f "config/settings.yaml" ]]; then
        log "Creating v8.0 configuration from template..."
        cp "config/settings.yaml" "config/settings_v8.yaml"
    else
        log_error "No configuration template found"
        exit 1
    fi
fi

# Update configuration paths
log "Updating configuration paths..."
sed -i "s|/home/claude|$HOME|g" config/settings_v8.yaml 2>/dev/null || log_warning "Could not update home directory in config"
sed -i "s|/opt/shogun/taisho_activity.db|$ACTIVITY_DB_PATH|g" config/settings_v8.yaml 2>/dev/null || log_warning "Could not update database path in config"

log_success "Configuration setup completed"

# =============================================================================
# Phase 8: Environment Variables Setup
# =============================================================================

log "Phase 8: Setting up environment variables..."

# Create .env file template if it doesn't exist
if [[ ! -f ".env" ]]; then
    log "Creating environment variables template..."
    cat > .env << EOF
# 将軍システム v8.0 Environment Variables

# API Keys (fill in as needed)
ANTHROPIC_API_KEY=
GROQ_API_KEY=
OLLAMA_API_KEY=
BRAVE_API_KEY=

# Slack Integration
SLACK_TOKEN=
SLACK_SIGNING_SECRET=

# Notion Integration
NOTION_TOKEN=
NOTION_DATABASE_ID=

# GitHub Integration
GITHUB_TOKEN=

# System Configuration
SHOGUN_CONFIG_PATH=config/settings_v8.yaml
SHOGUN_LOG_LEVEL=INFO

# Database Paths
QDRANT_URL=http://localhost:6333
ACTIVITY_DB_PATH=$ACTIVITY_DB_PATH
EOF
    
    chmod 600 .env
    log_warning "Please edit .env file with your API keys"
else
    log "Environment file already exists"
fi

log_success "Environment variables setup completed"

# =============================================================================
# Phase 9: System Services Setup (Optional)
# =============================================================================

log "Phase 9: Setting up system services..."

# Create systemd service file for v8.0
if [[ $IS_ROOT == true ]] || groups "$USER" | grep -q sudo; then
    log "Creating systemd service..."
    
    SERVICE_FILE="/etc/systemd/system/shogun-v8.service"
    sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Shogun System v8.0 - Complete AI Development Environment
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$SHOGUN_ROOT
Environment=PATH=$SHOGUN_ROOT/venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$SHOGUN_ROOT/venv/bin/python main_v8.py --mode server --host 0.0.0.0 --port 8080
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload || log_warning "Failed to reload systemd"
    sudo systemctl enable shogun-v8.service || log_warning "Failed to enable Shogun service"
    
    log_success "Systemd service created and enabled"
else
    log_warning "Cannot create systemd service (requires sudo privileges)"
fi

# =============================================================================
# Phase 10: Validation and Testing
# =============================================================================

log "Phase 10: Validating installation..."

# Test Python imports
log "Testing Python module imports..."
python -c "
try:
    import fastapi, uvicorn, httpx, yaml, sqlite3, sentence_transformers, qdrant_client
    print('✅ All required Python modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
" || {
    log_error "Python module validation failed"
    exit 1
}

# Test Qdrant connectivity
log "Testing Qdrant connectivity..."
if curl -s http://localhost:6333/health | grep -q "ok"; then
    log_success "Qdrant connectivity test passed"
else
    log_error "Qdrant connectivity test failed"
    exit 1
fi

# Test database creation
log "Testing Activity Memory database..."
python -c "
import sqlite3
import os
db_path = '$ACTIVITY_DB_PATH'
try:
    conn = sqlite3.connect(db_path)
    conn.execute('CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY)')
    conn.execute('DROP TABLE test_table')
    conn.close()
    print('✅ SQLite database test passed')
except Exception as e:
    print(f'❌ Database test failed: {e}')
    exit(1)
" || {
    log_error "Database validation failed"
    exit 1
}

log_success "All validation tests passed"

# =============================================================================
# Deployment Complete
# =============================================================================

echo -e "${GREEN}"
cat << "EOF"
🎌 ============================================================
   将軍システム v8.0 - Deployment Complete! 
============================================================ 🎌
EOF
echo -e "${NC}"

log_success "Shogun System v8.0 deployment completed successfully!"

echo ""
log "📋 Next Steps:"
echo "1. Edit .env file with your API keys:"
echo "   nano $SHOGUN_ROOT/.env"
echo ""
echo "2. Start the system:"
echo "   cd $SHOGUN_ROOT"
echo "   source venv/bin/activate"
echo "   python main_v8.py --mode server"
echo ""
echo "3. Or start as system service:"
echo "   sudo systemctl start shogun-v8"
echo ""
echo "4. Access the API documentation:"
echo "   http://localhost:8080/docs"
echo ""
echo "5. Check system status:"
echo "   http://localhost:8080/status"

echo ""
log "📚 Important Files:"
echo "   Configuration: config/settings_v8.yaml"
echo "   Environment:   .env" 
echo "   Logs:          $INSTALL_LOG"
echo "   Service:       /etc/systemd/system/shogun-v8.service"
echo "   Qdrant Data:   $QDRANT_DATA_DIR"
echo "   Activity DB:   $ACTIVITY_DB_PATH"

echo ""
log "🔧 System Components:"
echo "   ✅ Knowledge Base (Qdrant): http://localhost:6333"
echo "   ✅ Activity Memory (SQLite): $ACTIVITY_DB_PATH"
echo "   ✅ Python Environment: $SHOGUN_ROOT/venv"
echo "   ✅ Docker Services: Qdrant"

echo ""
echo -e "${BLUE}🏯 Welcome to the Shogun System v8.0! 🏯${NC}"
echo ""

exit 0