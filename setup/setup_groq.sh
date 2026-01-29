#!/bin/bash
# setup_groq.sh - Groq Setup for Shogun System v7.0
#
# Sets up Groq API access for 9th Ashigaru (足軽) recorder
# Ultra-fast summarization with Llama 3.3 70B
#
# Usage: Run on CT 100 (本陣)
# Requirements: Groq API key, internet connection

set -e

echo "=========================================="
echo "🚀 9番足軽 - Groq記録係 セットアップ"
echo "=========================================="
echo "Model: Llama 3.3 70B Versatile"
echo "Purpose: Real-time recording & 60-day summaries"
echo "Speed: 300-500 tok/s (10x faster than GPU)"
echo "Cost: FREE (14,400 requests/day)"
echo "=========================================="

# Check if running on correct system
if [ ! -f "/etc/hostname" ] || ! grep -q "honmaru-control\|本陣" /etc/hostname 2>/dev/null; then
    echo "⚠️ Warning: This script should be run on CT 100 (本陣)"
    echo "Current hostname: $(hostname)"
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "[1/5] Groq SDK installation"

# Install Groq Python SDK
pip3 install --break-system-packages groq requests python-dateutil

echo "✅ Groq SDK installed"

# Check for existing API key
echo "[2/5] API key configuration"

if [ -f "/root/.env" ] && grep -q "GROQ_API_KEY" /root/.env; then
    echo "✅ GROQ_API_KEY found in .env file"
    GROQ_API_KEY=$(grep "GROQ_API_KEY" /root/.env | cut -d'=' -f2)
else
    echo ""
    echo "🔑 Groq API key setup required"
    echo "1. Visit: https://console.groq.com/keys"
    echo "2. Create account (free)"
    echo "3. Generate API key"
    echo ""
    
    # Prompt for API key
    while true; do
        echo -n "Enter your Groq API key (gsk_...): "
        read -r GROQ_API_KEY
        
        if [[ $GROQ_API_KEY =~ ^gsk_[a-zA-Z0-9]{50,}$ ]]; then
            break
        else
            echo "❌ Invalid format. Groq API keys start with 'gsk_'"
        fi
    done
    
    # Save to .env
    echo "GROQ_API_KEY=$GROQ_API_KEY" >> /root/.env
    echo "✅ API key saved to /root/.env"
fi

# Test API connection
echo "[3/5] API connection test"

cat > /tmp/groq_test.py << 'PYTHON'
#!/usr/bin/env python3
import os
import sys
from groq import Groq

try:
    # Initialize client
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    
    # Test simple request
    print("🧪 Testing Groq API connection...")
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Respond briefly."},
            {"role": "user", "content": "Hello, can you confirm you're working?"}
        ],
        max_tokens=50,
        temperature=0.1,
    )
    
    if response.choices and response.choices[0].message.content:
        print("✅ API connection successful")
        print(f"Response: {response.choices[0].message.content}")
        print(f"Usage: {response.usage.total_tokens} tokens")
        sys.exit(0)
    else:
        print("❌ Empty response from API")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ API test failed: {e}")
    sys.exit(1)
PYTHON

# Run test with environment
export GROQ_API_KEY="$GROQ_API_KEY"
python3 /tmp/groq_test.py

if [ $? -eq 0 ]; then
    echo "✅ Groq API connection verified"
else
    echo "❌ Groq API test failed"
    exit 1
fi

# Create usage tracking
echo "[4/5] Usage tracking setup"

mkdir -p /var/log/shogun/groq
cat > /var/log/shogun/groq/usage.json << 'JSON'
{
    "daily_requests": 0,
    "last_reset_date": "",
    "total_tokens": 0,
    "sessions_recorded": 0,
    "summaries_generated": 0
}
JSON

# Create daily reset cron job
cat > /etc/cron.d/groq-quota-reset << 'CRON'
# Reset Groq daily quota at midnight
0 0 * * * root /bin/bash -c 'echo "{\"daily_requests\": 0, \"last_reset_date\": \"$(date -I)\", \"total_tokens\": 0, \"sessions_recorded\": 0, \"summaries_generated\": 0}" > /var/log/shogun/groq/usage.json'
CRON

echo "✅ Usage tracking configured"

# Create monitoring script
echo "[5/5] Monitoring tools setup"

cat > /usr/local/bin/groq-status << 'BASH'
#!/bin/bash
# Groq 9番足軽 status checker

echo "========================================"
echo "🚀 9番足軽 (Groq記録係) ステータス"
echo "========================================"

# Check API key
if [ -n "$GROQ_API_KEY" ] || grep -q "GROQ_API_KEY" /root/.env; then
    echo "✅ API Key: Configured"
else
    echo "❌ API Key: Not configured"
fi

# Check usage stats
if [ -f "/var/log/shogun/groq/usage.json" ]; then
    python3 -c "
import json
try:
    with open('/var/log/shogun/groq/usage.json', 'r') as f:
        data = json.load(f)
    daily = data.get('daily_requests', 0)
    total_tokens = data.get('total_tokens', 0)
    remaining = 14400 - daily
    
    print(f'📊 Usage Today: {daily}/14,400 requests')
    print(f'📊 Remaining: {remaining} requests')
    print(f'📊 Total Tokens: {total_tokens:,}')
    print(f'📊 Sessions: {data.get(\"sessions_recorded\", 0)}')
    print(f'📊 Summaries: {data.get(\"summaries_generated\", 0)}')
    
    if remaining < 1000:
        print('⚠️ Warning: Less than 1000 requests remaining today')
    elif remaining < 100:
        print('🚫 Critical: Less than 100 requests remaining today')
    
except Exception as e:
    print(f'❌ Error reading usage stats: {e}')
"
else
    echo "❌ Usage tracking not initialized"
fi

# Test API
echo ""
echo "🧪 Quick API Test:"
source /root/.env 2>/dev/null || true
export GROQ_API_KEY
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
    print('✅ API responding normally')
except Exception as e:
    print(f'❌ API error: {e}')
"

echo "========================================"
BASH

chmod +x /usr/local/bin/groq-status

# Create performance test script
cat > /usr/local/bin/groq-benchmark << 'BASH'
#!/bin/bash
# Groq performance benchmark

echo "🏃‍♂️ Groq Performance Benchmark"
echo "Testing Llama 3.3 70B inference speed..."

source /root/.env 2>/dev/null || true
export GROQ_API_KEY

python3 -c "
import os
import time
from groq import Groq

client = Groq(api_key=os.environ.get('GROQ_API_KEY'))

# Test prompt
prompt = '''
以下のタスクを分析し、60日間の要約レポートを作成してください：

1. ESP32-P4のI2S設定最適化
2. Home Assistantの音声認識統合
3. Spotify API連携実装
4. 消費電力測定システム構築
5. 自動テストケース生成

各項目について、実装内容、発生した問題、解決策、今後の改善点をまとめてください。
'''

print(f'Prompt length: {len(prompt)} characters')
print('Generating response...')

start_time = time.time()

response = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    messages=[
        {'role': 'system', 'content': '効率的で包括的な要約レポートを日本語で作成してください。'},
        {'role': 'user', 'content': prompt}
    ],
    max_tokens=1000,
    temperature=0.3,
)

end_time = time.time()
duration = end_time - start_time

if response.usage:
    total_tokens = response.usage.total_tokens
    completion_tokens = response.usage.completion_tokens
    
    print(f'⏱️ Response time: {duration:.2f} seconds')
    print(f'📊 Total tokens: {total_tokens}')
    print(f'📊 Completion tokens: {completion_tokens}')
    print(f'🚀 Speed: {total_tokens/duration:.1f} tokens/second')
    print(f'🚀 Output speed: {completion_tokens/duration:.1f} tokens/second')
    
    if total_tokens/duration > 200:
        print('✅ Performance: Excellent (>200 tok/s)')
    elif total_tokens/duration > 100:
        print('✅ Performance: Good (>100 tok/s)')
    else:
        print('⚠️ Performance: Below expected (<100 tok/s)')
        
    # Show first 200 chars of response
    print('')
    print('📝 Sample output:')
    print(response.choices[0].message.content[:200] + '...')
    
else:
    print('❌ No usage information returned')
"
BASH

chmod +x /usr/local/bin/groq-benchmark

# Final verification
echo ""
echo "🔍 Final verification..."

# Run status check
/usr/local/bin/groq-status

echo ""
echo "========================================"
echo "🎉 9番足軽 (Groq記録係) セットアップ完了!"
echo "========================================"
echo "Model: Llama 3.3 70B Versatile"
echo "Daily Quota: 14,400 requests (FREE)"
echo "Speed: 300-500 tokens/second"
echo ""
echo "管理コマンド:"
echo "  groq-status      - ステータス確認"
echo "  groq-benchmark   - 性能テスト"
echo ""
echo "設定ファイル:"
echo "  /root/.env       - API key"
echo "  /var/log/shogun/groq/usage.json - 使用量追跡"
echo ""
echo "統合方法:"
echo "  - GroqRecorder クラスが自動的に使用"
echo "  - リアルタイム記録"
echo "  - 60日要約自動生成"
echo "  - Notion自動転送"
echo ""
echo "🚀 超高速記録・要約システム稼働準備完了!"
echo "========================================"

# Clean up
rm -f /tmp/groq_test.py