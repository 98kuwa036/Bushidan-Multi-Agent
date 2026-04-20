#!/usr/bin/env python3
"""
local_llm_server.py — ローカルLLMサーバー v18
192.168.11.239 で稼働。Gemma4 MoE（常時）と Nemotron（排他）を管理。

起動:
  python local_llm_server.py

環境変数:
  LOCAL_LLM_PORT:      ポート番号 (デフォルト: 8082)
  GEMMA_MODEL_PATH:    Gemma4 MoE GGUFパス
  NEMOTRON_MODEL_PATH: Nemotron GGUFパス
  LLM_N_GPU_LAYERS:    GPU レイヤー数 (デフォルト: 0 = CPU専用)
  LLM_N_THREADS:       CPUスレッド数 (デフォルト: 6 = i5-8500 物理コア数)
  LLM_N_CTX:           コンテキスト長 (デフォルト: 2048)
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("local_llm_server")

# ── 設定 ──────────────────────────────────────────────────────────────────

PORT         = int(os.environ.get("LOCAL_LLM_PORT", "8082"))
N_GPU_LAYERS = int(os.environ.get("LLM_N_GPU_LAYERS", "0"))   # CPU専用
N_THREADS    = int(os.environ.get("LLM_N_THREADS", "6"))       # i5-8500: 6物理コア
N_CTX        = int(os.environ.get("LLM_N_CTX", "2048"))

GEMMA_PATH = os.environ.get(
    "GEMMA_MODEL_PATH",
    os.path.expanduser(
        "~/Bushidan-Multi-Agent/models/grapeV-ai/"
        "gemma-4-26B-A4B-it-MXFP4_MOE.gguf"
    ),
)
NEMOTRON_PATH = os.environ.get(
    "NEMOTRON_MODEL_PATH",
    os.path.expanduser(
        "~/Bushidan-Multi-Agent/models/nemotron/"
        "Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf"
    ),
)

# ── モデル状態 ────────────────────────────────────────────────────────────

_gemma_model:    Optional[object] = None
_nemotron_model: Optional[object] = None
_active_model:   Optional[str]    = None   # "gemma" | "nemotron" | None
_model_lock      = asyncio.Lock()

# ── lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Local LLM Server v18 starting...")
    logger.info("   Gemma4 MoE: %s", GEMMA_PATH)
    logger.info("   Nemotron:   %s", NEMOTRON_PATH)
    logger.info("   n_threads=%d  n_gpu_layers=%d  n_ctx=%d", N_THREADS, N_GPU_LAYERS, N_CTX)
    success = await _load_gemma()
    if not success:
        logger.warning("⚠️  Gemma4 load failed. Will retry on first request.")
    yield
    logger.info("🛑 Server shutting down")

# ── FastAPI ───────────────────────────────────────────────────────────────

app = FastAPI(title="Bushidan Local LLM Server", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── リクエストスキーマ ────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    system: str = ""

class StructuredRequest(BaseModel):
    prompt: str
    grammar: str           # GBNF 文法文字列
    max_tokens: int = 256
    temperature: float = 0.1   # 文法強制時は低温が安定

class SwitchResponse(BaseModel):
    success: bool
    active_model: Optional[str]
    message: str

# ── ヘルパー ──────────────────────────────────────────────────────────────

import re as _re
# Gemma 4 の内部思考トークンをレスポンスから除去
_THINKING_PAT = _re.compile(r'<\|channel\>.*?<channel\|>', _re.DOTALL)

def _strip_thinking(text: str) -> str:
    """Gemma 4 の <|channel>thought...<channel|> タグを除去"""
    return _THINKING_PAT.sub('', text).strip()

def _free_memory() -> None:
    gc.collect()

async def _load_gemma() -> bool:
    global _gemma_model, _active_model

    if _gemma_model is not None:
        return True
    if not Path(GEMMA_PATH).exists():
        logger.error("❌ Gemma4 not found: %s", GEMMA_PATH)
        return False

    try:
        from llama_cpp import Llama
        logger.info("🎌 Loading Gemma4 MoE: %s", GEMMA_PATH)
        t0 = time.time()
        loop = asyncio.get_event_loop()
        _gemma_model = await loop.run_in_executor(None, lambda: Llama(
            model_path=GEMMA_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=512,
            use_mlock=True,
            chat_format="gemma",
            verbose=False,
        ))
        _active_model = "gemma"
        logger.info("✅ Gemma4 MoE ready (%.1fs)", time.time() - t0)
        return True
    except Exception as e:
        logger.error("❌ Failed to load Gemma4: %s", e)
        return False

async def _unload_gemma() -> None:
    global _gemma_model, _active_model
    if _gemma_model is None:
        return
    logger.info("🗑️  Unloading Gemma4...")
    _gemma_model = None
    _active_model = None
    _free_memory()
    await asyncio.sleep(0.1)
    logger.info("✅ Gemma4 unloaded")

async def _load_nemotron() -> bool:
    global _nemotron_model, _active_model
    if _nemotron_model is not None:
        return True
    if not Path(NEMOTRON_PATH).exists():
        logger.error("❌ Nemotron not found: %s", NEMOTRON_PATH)
        return False

    try:
        from llama_cpp import Llama
        logger.info("🐉 Loading Nemotron: %s", NEMOTRON_PATH)
        t0 = time.time()
        loop = asyncio.get_event_loop()
        _nemotron_model = await loop.run_in_executor(None, lambda: Llama(
            model_path=NEMOTRON_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            n_threads=N_THREADS,
            n_batch=256,
            use_mlock=True,
            chat_format="chatml",
            verbose=False,
        ))
        _active_model = "nemotron"
        logger.info("✅ Nemotron ready (%.1fs)", time.time() - t0)
        return True
    except Exception as e:
        logger.error("❌ Failed to load Nemotron: %s", e)
        return False

async def _unload_nemotron() -> None:
    global _nemotron_model, _active_model
    if _nemotron_model is None:
        return
    logger.info("🗑️  Unloading Nemotron...")
    _nemotron_model = None
    _active_model = None
    _free_memory()
    await asyncio.sleep(0.1)
    logger.info("✅ Nemotron unloaded")

# ── エンドポイント ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "active_model": _active_model,
        "gemma_loaded": _gemma_model is not None,
        "nemotron_loaded": _nemotron_model is not None,
    }

@app.get("/status")
def status():
    return {
        "active_model": _active_model,
        "n_threads": N_THREADS,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_ctx": N_CTX,
        "gemma": {
            "loaded": _gemma_model is not None,
            "path": GEMMA_PATH,
        },
        "nemotron": {
            "loaded": _nemotron_model is not None,
            "path": NEMOTRON_PATH,
        },
    }

@app.post("/generate/gemma")
async def generate_gemma(req: GenerateRequest):
    """Gemma4 MoE で推論（チャットテンプレート適用）"""
    async with _model_lock:
        if _active_model == "nemotron":
            raise HTTPException(503, "Nemotron active. Call /switch/gemma first.")
        if _gemma_model is None:
            if not await _load_gemma():
                raise HTTPException(503, "Gemma4 unavailable")
        try:
            t0 = time.time()
            messages = []
            if req.system:
                messages.append({"role": "system", "content": req.system})
            messages.append({"role": "user", "content": req.prompt})
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: _gemma_model.create_chat_completion(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ))
            choices = result.get("choices", [])
            text = _strip_thinking(choices[0]["message"]["content"] if choices else "")
            elapsed_ms = (time.time() - t0) * 1000
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            return {
                "content": text,
                "model": "gemma4-26b-moe",
                "elapsed_ms": round(elapsed_ms),
                "tokens": tokens,
                "tok_per_sec": round(tokens / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0,
            }
        except Exception as e:
            logger.error("Gemma4 generation failed: %s", e)
            raise HTTPException(500, str(e))

@app.post("/generate/nemotron")
async def generate_nemotron(req: GenerateRequest):
    """Nemotron で推論（ChatML テンプレート適用）"""
    async with _model_lock:
        if _nemotron_model is None:
            raise HTTPException(503, "Nemotron not loaded. Call /switch/nemotron first.")
        try:
            t0 = time.time()
            messages = []
            if req.system:
                messages.append({"role": "system", "content": req.system})
            messages.append({"role": "user", "content": req.prompt})
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: _nemotron_model.create_chat_completion(
                messages=messages,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            ))
            choices = result.get("choices", [])
            text = choices[0]["message"]["content"] if choices else ""
            elapsed_ms = (time.time() - t0) * 1000
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            return {
                "content": text,
                "model": "nemotron-3-30b",
                "elapsed_ms": round(elapsed_ms),
                "tokens": tokens,
                "tok_per_sec": round(tokens / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else 0,
            }
        except Exception as e:
            logger.error("Nemotron generation failed: %s", e)
            raise HTTPException(500, str(e))

@app.post("/generate/structured")
async def generate_structured(req: StructuredRequest):
    """GBNF 文法制約付き構造化 JSON 生成（Uchu/Karasu 分析向け）"""
    async with _model_lock:
        if _active_model == "nemotron":
            raise HTTPException(503, "Nemotron active. Call /switch/gemma first.")
        if _gemma_model is None:
            if not await _load_gemma():
                raise HTTPException(503, "Gemma4 unavailable")
        try:
            from llama_cpp import LlamaGrammar
            grammar = LlamaGrammar.from_string(req.grammar)
            t0 = time.time()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: _gemma_model.create_completion(
                prompt=req.prompt,
                grammar=grammar,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            ))
            choices = result.get("choices", [])
            text = choices[0]["text"] if choices else ""
            elapsed_ms = (time.time() - t0) * 1000
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            return {
                "content": text,
                "model": "gemma4-structured",
                "elapsed_ms": round(elapsed_ms),
                "tokens": tokens,
            }
        except Exception as e:
            logger.error("Structured generation failed: %s", e)
            raise HTTPException(500, str(e))

@app.post("/switch/nemotron", response_model=SwitchResponse)
async def switch_to_nemotron():
    """Gemma4 → Nemotron 排他切り替え"""
    async with _model_lock:
        if _active_model == "nemotron":
            return SwitchResponse(success=True, active_model="nemotron", message="Already on Nemotron")
        try:
            await _unload_gemma()
            success = await _load_nemotron()
            return SwitchResponse(
                success=success, active_model=_active_model,
                message="Switched to Nemotron" if success else "Switch failed",
            )
        except Exception as e:
            logger.error("Switch to Nemotron failed: %s", e)
            return SwitchResponse(success=False, active_model=_active_model, message=str(e))

@app.post("/switch/gemma", response_model=SwitchResponse)
async def switch_to_gemma():
    """Nemotron → Gemma4 MoE 排他切り替え"""
    async with _model_lock:
        if _active_model == "gemma":
            return SwitchResponse(success=True, active_model="gemma", message="Already on Gemma4 MoE")
        try:
            await _unload_nemotron()
            success = await _load_gemma()
            return SwitchResponse(
                success=success, active_model=_active_model,
                message="Switched to Gemma4 MoE" if success else "Switch failed",
            )
        except Exception as e:
            logger.error("Switch to Gemma4 failed: %s", e)
            return SwitchResponse(success=False, active_model=_active_model, message=str(e))

@app.post("/benchmark")
async def benchmark():
    """推論速度ベンチマーク（50トークン生成）"""
    async with _model_lock:
        if _gemma_model is None:
            raise HTTPException(503, "Gemma4 not loaded")
        try:
            t0 = time.time()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: _gemma_model.create_chat_completion(
                messages=[{"role": "user", "content": "日本語で1から10まで数えてください。"}],
                max_tokens=50,
                temperature=0.1,
            ))
            elapsed_ms = (time.time() - t0) * 1000
            choices = result.get("choices", [])
            text = _strip_thinking(choices[0]["message"]["content"] if choices else "")
            tokens = result.get("usage", {}).get("completion_tokens", 0)
            return {
                "elapsed_ms": round(elapsed_ms),
                "tokens": tokens,
                "tok_per_sec": round(tokens / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else 0,
                "text": text,
                "model": "gemma4-26b-moe",
            }
        except Exception as e:
            raise HTTPException(500, str(e))

# ── メイン ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
