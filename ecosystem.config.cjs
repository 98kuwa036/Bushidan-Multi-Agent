/**
 * PM2 Ecosystem Configuration v12
 * 武士団マルチエージェントシステム v12
 * EliteDesk (192.168.11.230)
 *
 * API キーは /home/claude/Bushidan-Multi-Agent/.env で一元管理。
 * このファイルは .env を読み込んで各プロセスに渡す。
 */

const fs = require('fs');
const path = require('path');

const BUSHIDAN_DIR = '/home/claude/Bushidan-Multi-Agent';
const VENV_PYTHON  = `${BUSHIDAN_DIR}/.venv/bin/python`;
const ENV_FILE     = `${BUSHIDAN_DIR}/.env`;

// ─── .env ファイルパーサー ────────────────────────────────────────────────────
function loadEnvFile(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const env = {};
    for (const line of content.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const idx = trimmed.indexOf('=');
      if (idx === -1) continue;
      const key = trimmed.slice(0, idx).trim();
      let value = trimmed.slice(idx + 1).trim();
      // インラインコメント除去 (スペース2個以上 + #)
      const commentMatch = value.match(/\s{2,}#.*/);
      if (commentMatch) value = value.slice(0, commentMatch.index).trim();
      // クォート除去
      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      env[key] = value;
    }
    return env;
  } catch (e) {
    console.warn(`[ecosystem] .env 読み込み失敗: ${e.message}`);
    return {};
  }
}

const e = loadEnvFile(ENV_FILE);

// ─── 共通環境変数 ─────────────────────────────────────────────────────────────
const commonEnv = {
  // Anthropic (将軍・大元帥)
  CLAUDE_API_KEY:    e.CLAUDE_API_KEY    || e.ANTHROPIC_API_KEY || '',
  ANTHROPIC_API_KEY: e.ANTHROPIC_API_KEY || e.CLAUDE_API_KEY    || '',
  // Google (検校)
  GEMINI_API_KEY:    e.GEMINI_API_KEY    || '',
  // OpenAI (軍師 o3-mini)
  OPENAI_API_KEY:    e.OPENAI_API_KEY    || '',
  // Cohere (受付 Command R / 外事 Command R+) — v12
  COHERE_API_KEY:    e.COHERE_API_KEY    || '',
  // Mistral AI (参謀 Mistral Large 3)
  MISTRAL_API_KEY:   e.MISTRAL_API_KEY   || '',
  // Groq (斥候 Llama 3.3)
  GROQ_API_KEY:      e.GROQ_API_KEY      || '',
  // Web 検索
  TAVILY_API_KEY:    e.TAVILY_API_KEY    || '',
  SMITHERY_API_KEY:  e.SMITHERY_API_KEY  || '',
  // Notion (長期記憶)
  NOTION_API_KEY:    e.NOTION_API_KEY    || '',
  // GitHub
  GITHUB_TOKEN:      e.GITHUB_TOKEN      || '',
  // Discord
  DISCORD_BOT_TOKEN: e.DISCORD_BOT_TOKEN || '',
  // OpenRouter (汎用フォールバック)
  OPENROUTER_API_KEY: e.OPENROUTER_API_KEY || '',
  // ローカルLLM (EliteDesk 192.168.11.239)
  LLAMACPP_ENDPOINT: e.LLAMACPP_ENDPOINT || 'http://192.168.11.239:8080',
  ELYZA_HOST:        e.ELYZA_HOST        || '192.168.11.239',
  ELYZA_PORT:        e.ELYZA_PORT        || '8081',
  // システム設定
  SYSTEM_MODE:       e.SYSTEM_MODE       || 'battalion',
  INTERACTIVE_MODE:  e.INTERACTIVE_MODE  || 'true',
  LOG_LEVEL:         e.LOG_LEVEL         || 'INFO',
};

module.exports = {
  apps: [
    // ─────────────────────────────────────────────
    // 武士団 Mattermost Bot
    // ─────────────────────────────────────────────
    {
      name: 'bushidan-mattermost',
      script: VENV_PYTHON,
      args: '-m bushidan.mattermost_bot',
      cwd: BUSHIDAN_DIR,
      interpreter: 'none',
      env: {
        ...commonEnv,
        PYTHONPATH: BUSHIDAN_DIR,
        PYTHONUNBUFFERED: '1',
        MATTERMOST_URL:           e.MATTERMOST_URL           || '192.168.11.234',
        MATTERMOST_PORT:          e.MATTERMOST_PORT          || '8065',
        MATTERMOST_SCHEME:        e.MATTERMOST_SCHEME        || 'http',
        MATTERMOST_TOKEN:         'qdiubp1fdpb3ub9ysxz3768p5a',  // bushidan-bot トークン
        MATTERMOST_TEAM_NAME:     'bushidan',
        MATTERMOST_CHANNEL:       'twauoqakufbumpq7fp668apeer',
        MATTERMOST_CALLBACK_HOST: '192.168.11.230',
        MATTERMOST_CALLBACK_PORT: '8066',
      },
      watch: false,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      error_file: '/home/claude/.pm2/logs/bushidan-mattermost-error.log',
      out_file:   '/home/claude/.pm2/logs/bushidan-mattermost-out.log',
    },

    // ─────────────────────────────────────────────
    // 武士団 Discord Bot
    // ─────────────────────────────────────────────
    {
      name: 'bushidan-discord',
      script: VENV_PYTHON,
      args: '-m bushidan.discord_bot',
      cwd: BUSHIDAN_DIR,
      interpreter: 'none',
      env: {
        ...commonEnv,
        PYTHONPATH: BUSHIDAN_DIR,
        PYTHONUNBUFFERED: '1',
      },
      watch: false,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      error_file: '/home/claude/.pm2/logs/bushidan-discord-error.log',
      out_file:   '/home/claude/.pm2/logs/bushidan-discord-out.log',
    },
  ],
};
