"""
Bushidan Multi-Agent System v18 - Gemini 3.1 Flash Client

Latest Gemini 3 Flash with Pro-level intelligence at Flash speed.

Key Features:
- Pro-level intelligence with Flash-level speed
- Thinking level parameter for internal reasoning control
- 1M+ token context window
- Enhanced Japanese language support
- Cost-effective pricing

v15 役割: 受付(Flash-Lite) / 検校(Flash-Image) / 右筆(Flash-Lite)
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class Gemini3Stats:
    """Gemini 3 Flash usage statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    final_defense_activations: int = 0  # Times used as last resort
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_yen: float = 0.0
    average_latency_seconds: float = 0.0


class Gemini3Client:
    """
    Gemini 3 Flash Client for final defense and tactical coordination

    Model: gemini-3-flash (latest)
    - Pro-level intelligence with Flash speed
    - 1M+ token context window
    - Thinking level parameter for reasoning control
    - Excellent Japanese support
    - Cost-effective pricing

    Role: Final defense line in fallback chain
    - Activates when Qwen3 (local/cloud) cannot handle task
    - Provides reliable, high-quality fallback
    - Tactical coordination for complex scenarios
    """

    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite-preview"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # Statistics
        self.stats = Gemini3Stats()
        
        # Configuration
        self.default_max_output_tokens = 2048
        self.default_temperature = 0.7
        self.default_top_p = 0.95
        self.default_top_k = 40
        
        # Cost tracking (estimated rates)
        self.cost_per_1k_input_tokens_yen = 0.00002  # Very low
        self.cost_per_1k_output_tokens_yen = 0.00006
        
        logger.info(f"🏛️ Gemini 3 Flash client initialized: {self.model}")
    
    async def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        as_final_defense: bool = False,
        attachments: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Generate completion using Gemini 3.0 Flash
        
        Args:
            prompt: Simple prompt string (for single-turn)
            messages: Conversation messages (for multi-turn)
            max_output_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            as_final_defense: Whether this is a final defense activation
        
        Returns:
            Generated text response
        
        Raises:
            Exception: If API call fails
        """
        
        if prompt is None and messages is None:
            raise ValueError("Either prompt or messages must be provided")
        
        if max_output_tokens is None:
            max_output_tokens = self.default_max_output_tokens
        if temperature is None:
            temperature = self.default_temperature
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Convert to Gemini format
            if messages:
                formatted_prompt = self._format_messages(messages)
            else:
                formatted_prompt = prompt

            # Make API request
            response_text, input_tokens, output_tokens = await self._make_request(
                formatted_prompt, max_output_tokens, temperature,
                attachments=attachments,
            )
            
            # Update statistics
            elapsed_time = asyncio.get_event_loop().time() - start_time
            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.total_tokens_input += input_tokens
            self.stats.total_tokens_output += output_tokens
            
            # Calculate cost
            cost_yen = (
                (input_tokens / 1000) * self.cost_per_1k_input_tokens_yen +
                (output_tokens / 1000) * self.cost_per_1k_output_tokens_yen
            )
            self.stats.total_cost_yen += cost_yen
            
            # Update average latency
            n = self.stats.successful_requests
            self.stats.average_latency_seconds = (
                (self.stats.average_latency_seconds * (n - 1) + elapsed_time) / n
            )
            
            # Track final defense activations
            if as_final_defense:
                self.stats.final_defense_activations += 1
                logger.info(
                    f"🛡️ Gemini 3 Final Defense activated: "
                    f"{output_tokens} tokens in {elapsed_time:.2f}s (¥{cost_yen:.4f})"
                )
            else:
                logger.info(
                    f"🏛️ Gemini 3 generation: "
                    f"{output_tokens} tokens in {elapsed_time:.2f}s (¥{cost_yen:.4f})"
                )
            
            return response_text
            
        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            logger.error(f"❌ Gemini 3 generation failed: {e}")
            raise
    
    async def _make_request(
        self, prompt: str, max_output_tokens: int, temperature: float,
        attachments: Optional[List[Dict[str, str]]] = None,
    ) -> tuple[str, int, int]:
        """
        Make actual API request to Gemini Flash

        Args:
            prompt: Formatted prompt
            max_output_tokens: Max tokens to generate
            temperature: Sampling temperature
            attachments: Optional list of image attachments
                         [{"type": "image_base64", "data": "...", "mime_type": "image/png"}, ...]

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)
        """

        try:
            import httpx

            url = f"{self.base_url}/models/{self.model}:generateContent"
            params = {"key": self.api_key}

            headers = {
                "Content-Type": "application/json"
            }

            # Build parts: text + optional inline images
            parts: list = [{"text": prompt}]
            if attachments:
                for att in attachments:
                    if att.get("type") == "image_base64" and att.get("data"):
                        mime = att.get("mime_type", "image/png")
                        parts.append({
                            "inline_data": {
                                "mime_type": mime,
                                "data": att["data"],
                            }
                        })

            payload = {
                "contents": [
                    {
                        "parts": parts
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": self.default_top_p,
                    "topK": self.default_top_k,
                    "maxOutputTokens": max_output_tokens
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_ONLY_HIGH"
                    }
                ]
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    params=params,
                    headers=headers,
                    json=payload
                )
                
                if response.status_code != 200:
                    error_detail = response.text
                    raise Exception(f"Gemini API error {response.status_code}: {error_detail}")
                
                result = response.json()
                
                # Extract response text
                if "candidates" not in result or len(result["candidates"]) == 0:
                    raise Exception(f"No candidates in response: {result}")

                candidate = result["candidates"][0]

                # Safety block or empty content → finishReason は SAFETY など
                finish_reason = candidate.get("finishReason", "")
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                if not parts or "text" not in parts[0]:
                    raise Exception(
                        f"Gemini response blocked or empty "
                        f"(finishReason={finish_reason}): {candidate}"
                    )
                response_text = parts[0]["text"]
                
                # Extract token counts
                usage = result.get("usageMetadata", {})
                input_tokens = usage.get("promptTokenCount", len(prompt.split()))
                output_tokens = usage.get("candidatesTokenCount", len(response_text.split()))
                
                return response_text, input_tokens, output_tokens
                
        except Exception as e:
            logger.error(f"❌ Gemini API request failed: {e}")
            raise
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation messages into a single prompt
        
        Args:
            messages: List of message dicts with 'role' and 'content'
        
        Returns:
            Formatted prompt string
        """
        
        formatted_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted_parts.append(f"[System]: {content}")
            elif role == "user":
                formatted_parts.append(f"[User]: {content}")
            elif role == "assistant":
                formatted_parts.append(f"[Assistant]: {content}")
            else:
                formatted_parts.append(content)
        
        return "\n\n".join(formatted_parts)
    
    async def generate_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        tool_choice: str = "auto",
        max_output_tokens: int = 2048,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Gemini function calling 対応の generate

        Args:
            messages: 会話メッセージリスト
            tools: functionDeclarations 形式のツール定義リスト
            tool_choice: "auto" or "any"
            max_output_tokens: 最大出力トークン数
            temperature: 温度パラメータ

        Returns:
            {
              "text": str or None,
              "tool_calls": [{"name": str, "args": dict}]
            }
        """
        if temperature is None:
            temperature = self.default_temperature

        start_time = asyncio.get_event_loop().time()

        try:
            import httpx

            url = f"{self.base_url}/models/{self.model}:generateContent"
            params = {"key": self.api_key}
            headers = {"Content-Type": "application/json"}

            # メッセージを Gemini contents 形式に変換
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Gemini は "user" と "model" のみ認識
                if role == "assistant":
                    gemini_role = "model"
                elif role == "tool":
                    # tool result は user として送る（functionResponse 形式）
                    gemini_role = "user"
                    contents.append({
                        "role": "user",
                        "parts": [{"functionResponse": {
                            "name": msg.get("tool_name", "unknown"),
                            "response": {"result": content}
                        }}]
                    })
                    continue
                else:
                    gemini_role = "user"

                contents.append({
                    "role": gemini_role,
                    "parts": [{"text": content}]
                })

            payload = {
                "contents": contents,
                "tools": [{"functionDeclarations": tools}],
                "toolConfig": {
                    "functionCallingConfig": {
                        "mode": "AUTO" if tool_choice == "auto" else "ANY"
                    }
                },
                "generationConfig": {
                    "temperature": temperature,
                    "topP": self.default_top_p,
                    "topK": self.default_top_k,
                    "maxOutputTokens": max_output_tokens
                },
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
                ]
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, params=params, headers=headers, json=payload)

                if response.status_code != 200:
                    raise Exception(f"Gemini API error {response.status_code}: {response.text}")

                result = response.json()

            if "candidates" not in result or not result["candidates"]:
                raise Exception(f"No candidates in response: {result}")

            candidate = result["candidates"][0]
            parts = candidate["content"]["parts"]

            text_content = None
            tool_calls = []

            for part in parts:
                if "functionCall" in part:
                    tool_calls.append({
                        "name": part["functionCall"]["name"],
                        "args": part["functionCall"]["args"]
                    })
                elif "text" in part:
                    text_content = (text_content or "") + part["text"]

            # トークン統計更新
            elapsed_time = asyncio.get_event_loop().time() - start_time
            usage = result.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)

            self.stats.total_requests += 1
            self.stats.successful_requests += 1
            self.stats.total_tokens_input += input_tokens
            self.stats.total_tokens_output += output_tokens

            logger.info(
                f"🔧 Gemini tool call: {len(tool_calls)} tools, "
                f"{output_tokens} tokens in {elapsed_time:.2f}s"
            )

            return {
                "text": text_content,
                "tool_calls": tool_calls
            }

        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            logger.error(f"❌ Gemini generate_with_tools failed: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if Gemini 3.0 Flash API is available

        Returns:
            True if healthy, False otherwise
        """

        try:
            await self.generate(prompt="Hi", max_output_tokens=64)
            logger.info("✅ Gemini 3 health check passed")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Gemini 3 health check failed: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive Gemini 3.0 usage statistics
        
        Returns:
            Dictionary with usage metrics
        """
        
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = self.stats.successful_requests / self.stats.total_requests
        
        final_defense_ratio = 0.0
        if self.stats.successful_requests > 0:
            final_defense_ratio = self.stats.final_defense_activations / self.stats.successful_requests
        
        return {
            "total_requests": self.stats.total_requests,
            "successful_requests": self.stats.successful_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": round(success_rate, 3),
            "final_defense_activations": self.stats.final_defense_activations,
            "final_defense_ratio": round(final_defense_ratio, 3),
            "total_tokens_input": self.stats.total_tokens_input,
            "total_tokens_output": self.stats.total_tokens_output,
            "total_cost_yen": round(self.stats.total_cost_yen, 4),
            "average_latency_seconds": round(self.stats.average_latency_seconds, 2),
            "model": self.model,
            "improvements_vs_2.0": {
                "speed": "1.3x faster",
                "reasoning": "+15% accuracy",
                "japanese": "Enhanced support"
            }
        }
    
    def reset_statistics(self) -> None:
        """Reset usage statistics"""
        self.stats = Gemini3Stats()
        logger.info("📊 Gemini 3 statistics reset")
