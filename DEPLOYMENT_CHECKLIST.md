# Claude API Server Deployment Checklist

## Overview
Deploy Flask-based Claude API Server on `claude-dedicated` LXC (192.168.11.237:8070) to separate Claude processing from bushidan-honjin and solve memory pressure issues.

---

## Phase 1: Prepare claude-dedicated LXC

### Step 1.1: Copy server files
On claude-dedicated LXC:
```bash
# Create /opt directory if needed
mkdir -p /opt

# Option A: Copy from bushidan-honjin (if accessible)
scp /home/claude/Bushidan-Multi-Agent/claude_api_server.py root@192.168.11.237:/opt/

# Option B: Create manually
# Copy the contents of claude_api_server.py to /opt/claude_api_server.py
```

### Step 1.2: Verify Python environment
```bash
# Verify Python 3.10+ is installed
python3 --version

# Create virtual environment (if not already created)
cd /opt
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install flask httpx anthropic
```

### Step 1.3: Configure environment variables
```bash
# Edit or create /opt/.env
cat > /opt/.env << 'EOF'
# Anthropic API key for fallback (required)
ANTHROPIC_API_KEY=sk-ant-...

# Server port (optional, defaults to 8070)
CLAUDE_API_PORT=8070
EOF

# Or export directly
export ANTHROPIC_API_KEY="sk-ant-..."
export CLAUDE_API_PORT=8070
```

---

## Phase 2: Test server locally

### Step 2.1: Start the server
```bash
cd /opt
source venv/bin/activate
python3 claude_api_server.py
```

### Step 2.2: Verify health check (from same LXC)
```bash
# In another terminal
curl http://localhost:8070/health

# Expected response:
# {"status": "ok", "service": "claude-api-server"}
```

### Step 2.3: Check API status
```bash
curl http://localhost:8070/api/status

# Expected response:
# {
#   "status": "ok",
#   "claude_cli_available": true,
#   "anthropic_api_available": true,
#   "cli_path": "/home/claude/.local/bin/claude"
# }
```

### Step 2.4: Test Claude call (local)
```bash
curl -X POST http://localhost:8070/api/claude \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, Claude!",
    "system": "You are a helpful assistant.",
    "max_tokens": 100
  }'

# Expected response:
# {
#   "content": "Hello! I'm Claude, an AI assistant...",
#   "model": "claude-pro-cli",
#   "source": "cli",
#   "error": null
# }
```

---

## Phase 3: Configure systemd service (optional but recommended)

### Step 3.1: Create systemd service file
```bash
sudo tee /etc/systemd/system/claude-api-server.service > /dev/null << 'EOF'
[Unit]
Description=Claude API Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt
Environment="ANTHROPIC_API_KEY=sk-ant-..."
Environment="CLAUDE_API_PORT=8070"
ExecStart=/opt/venv/bin/python3 /opt/claude_api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
```

### Step 3.2: Enable and start service
```bash
sudo systemctl daemon-reload
sudo systemctl enable claude-api-server
sudo systemctl start claude-api-server

# Check status
sudo systemctl status claude-api-server

# View logs
sudo journalctl -u claude-api-server -f
```

---

## Phase 4: Test from bushidan-honjin

### Step 4.1: Health check from bushidan-honjin
```bash
# On bushidan-honjin
curl http://192.168.11.237:8070/health

# Expected response:
# {"status": "ok", "service": "claude-api-server"}
```

### Step 4.2: Test API call from bushidan-honjin
```bash
curl -X POST http://192.168.11.237:8070/api/claude \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is 2+2?",
    "max_tokens": 50
  }'
```

### Step 4.3: Verify Python integration
```python
# In bushidan-honjin Python environment
import asyncio
from utils.claude_api_client import call_claude_api

async def test():
    result = await call_claude_api(
        prompt="Hello, Claude!",
        system="You are helpful."
    )
    print(result)

asyncio.run(test())
```

---

## Phase 5: Verify bushidan-honjin integration

### Step 5.1: Check .env configuration
```bash
# Verify CLAUDE_API_SERVER_URL is set
grep CLAUDE_API_SERVER_URL /home/claude/Bushidan-Multi-Agent/.env

# Should output:
# CLAUDE_API_SERVER_URL=http://192.168.11.237:8070
```

### Step 5.2: Test three-tier fallback chain
```bash
# Start bushidan-honjin with logging
LOG_LEVEL=DEBUG python3 main.py

# Should see logs:
# ✅ リモート Claude API 成功 (source=cli, model=claude-pro-cli)
# OR
# ⚠️ リモート Claude API HTTP 503 → フォールバック
# → ✅ claude CLI 成功 (Proプラン使用, model=...)
```

---

## Phase 6: Monitor and troubleshoot

### Step 6.1: Network connectivity
```bash
# From bushidan-honjin to claude-dedicated
ping 192.168.11.237
netstat -tuln | grep 8070  # On claude-dedicated
```

### Step 6.2: Check logs
```bash
# On claude-dedicated (if using systemd)
sudo journalctl -u claude-api-server -f

# Or if running in terminal
# Watch the process output directly
```

### Step 6.3: API endpoint inspection
```bash
curl -v http://192.168.11.237:8070/api/status | jq .

# Check CLI availability
# Check API key configuration
# Verify timeout settings
```

---

## Success Criteria

✅ Health check responds within 1 second
✅ API call completes within 60 seconds
✅ Claude Pro CLI优先 used (source=cli in response)
✅ Fallback to Anthropic API works (source=api)
✅ bushidan-honjin memory usage stable (no spikes)
✅ All agents still responsive via Mattermost

---

## Rollback Plan (if needed)

If the remote API causes issues:

1. Stop the claude-api-server service:
   ```bash
   sudo systemctl stop claude-api-server
   ```

2. Remove CLAUDE_API_SERVER_URL from bushidan-honjin .env:
   ```bash
   sed -i '/CLAUDE_API_SERVER_URL/d' .env
   ```

3. Restart bushidan-honjin:
   ```bash
   # Will fall back to local Claude CLI → Anthropic API
   ```

---

## Questions & Support

Refer to:
- `/home/claude/Bushidan-Multi-Agent/docs/CLAUDE_API_SERVER_SETUP.md` - Detailed setup guide
- `claude_api_server.py` - Source code with inline documentation
- `/home/claude/Bushidan-Multi-Agent/utils/claude_api_client.py` - Client library
