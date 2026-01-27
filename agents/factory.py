"""Agent Factory - 全エージェントの生成と登録"""

import logging
from typing import Any

from shogun.agents.base import LocalAgent, CloudAgent
from shogun.providers.ollama import OllamaClient

logger = logging.getLogger("shogun.factory")

# --- System Prompts ---

TAISHO_SYSTEM = """\
あなたはOmni-P4プロジェクトの侍大将（Taisho）です。
ユーザーの問いに対し、<think>タグを用いて日本語で深く推論し、論理的な結論を導いてください。
安易に答えを出さず、疑い深く検証すること。
ESP32-P4の低レイヤー開発に精通しており、ハードウェアレジスタ・DMA・割り込み制御に詳しい。
"""

LEADER_SYSTEM = """\
You are the Leader (足軽頭) of the Ashigaru team in the Omni-P4 project.
Your responsibilities:
1. Analyze the request and break it down into subtasks
2. Coordinate with Coder and Scout agents
3. Make autonomous decisions when encountering errors
4. Report results in a structured format

Always think step-by-step. When a task is ambiguous, ask clarifying questions.
Use tools autonomously. Respond in the language of the user's prompt.
"""

CODER_SYSTEM = """\
You are a highly skilled embedded systems engineer (技術兵 Coder).
You specialize in:
- ESP32-P4 development (ESP-IDF framework)
- C/C++ for embedded systems
- Linux kernel drivers and device trees
- Register-level hardware programming (SPI, I2C, UART, DMA)
- CMake build systems

Write precise, correct code. Focus on accuracy and syntax correctness.
Include necessary #include headers. Handle error codes properly.
When writing ESP-IDF code, use the correct API versions.
"""

SCOUT_SYSTEM = """\
You are a Scout (小者/斥候). Your job is fast, simple tasks:
- Extract key information from logs
- Search and filter text
- Format data into structured output
- Summarize long documents into bullet points

Be concise. Return only the requested information. No explanations unless asked.
"""

SHOGUN_SYSTEM = """\
あなたはOmni-P4プロジェクトの将軍（Shogun）です。
世界最高峰のAIとして、プロジェクト全体の戦略的意思決定を行います。

配下のエージェントが解決できなかった難問が、エスカレーションとしてあなたに届きます。
前任者の分析結果を踏まえつつ、根本的な解決策を提示してください。

ESP32-P4、Proxmox VE、組み込みLinux、リアルタイムOS全般に精通しています。
"""

KARO_SYSTEM = """\
あなたはOmni-P4プロジェクトの家老（Karo）です。
将軍の右腕として、高度な実装方針の策定とコードレビューを行います。

実装の詳細に踏み込み、具体的なコード変更提案を行ってください。
ESP-IDF、FreeRTOS、Linuxドライバに精通した最高レベルの実装者です。
"""


def create_local_agents(
    ollama: OllamaClient,
    config: dict,
) -> dict[str, LocalAgent]:
    """Create all local (Ollama) agents from config."""
    agents: dict[str, LocalAgent] = {}

    # Mode A: Taisho
    cfg = config["local"]["mode_a"]["taisho"]
    agents["taisho"] = LocalAgent(
        codename=cfg["codename"],
        role=cfg["role"],
        model=cfg["model"],
        ollama_client=ollama,
        num_ctx=cfg["num_ctx"],
        temperature=0.6,
        system_prompt=TAISHO_SYSTEM,
    )

    # Mode B: Leader
    cfg = config["local"]["mode_b"]["leader"]
    agents["leader"] = LocalAgent(
        codename=cfg["codename"],
        role=cfg["role"],
        model=cfg["model"],
        ollama_client=ollama,
        num_ctx=cfg["num_ctx"],
        temperature=0.7,
        system_prompt=LEADER_SYSTEM,
    )

    # Mode B: Coder
    cfg = config["local"]["mode_b"]["coder"]
    agents["coder"] = LocalAgent(
        codename=cfg["codename"],
        role=cfg["role"],
        model=cfg["model"],
        ollama_client=ollama,
        num_ctx=cfg["num_ctx"],
        temperature=0.3,
        system_prompt=CODER_SYSTEM,
    )

    # Mode B: Scout
    cfg = config["local"]["mode_b"]["scout"]
    agents["scout"] = LocalAgent(
        codename=cfg["codename"],
        role=cfg["role"],
        model=cfg["model"],
        ollama_client=ollama,
        num_ctx=cfg["num_ctx"],
        temperature=0.5,
        system_prompt=SCOUT_SYSTEM,
    )

    logger.info("Created %d local agents", len(agents))
    return agents


def create_cloud_agents(
    anthropic_client: Any,
    config: dict,
) -> dict[str, CloudAgent]:
    """Create cloud (Anthropic API) agents from config."""
    agents: dict[str, CloudAgent] = {}

    # Shogun
    cfg = config["cloud"]["shogun"]
    agents["shogun"] = CloudAgent(
        codename=cfg["codename"],
        role=cfg["role"],
        model=cfg["model"],
        anthropic_client=anthropic_client,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        system_prompt=SHOGUN_SYSTEM,
    )

    # Karo
    cfg = config["cloud"]["karo"]
    agents["karo"] = CloudAgent(
        codename=cfg["codename"],
        role=cfg["role"],
        model=cfg["model"],
        anthropic_client=anthropic_client,
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        system_prompt=KARO_SYSTEM,
    )

    logger.info("Created %d cloud agents", len(agents))
    return agents
