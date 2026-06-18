"""
utils/semantic_router.py — セマンティックルーター v18

LLM を使わず、ローカル埋め込みモデルによるコサイン類似度でルーティングを行う。
analyze_intent (Gemma) の前段に置き、明確なケースは直接ルーティングして
5〜13秒のレイテンシを削減する。

閾値:
  >= CONFIDENT_THRESHOLD  → 直接ルーティング
  >= CANDIDATE_THRESHOLD  → ヒント付きで analyze_intent へ
  <  CANDIDATE_THRESHOLD  → 従来通り analyze_intent へ
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# コサイン類似度閾値
CONFIDENT_THRESHOLD = 0.72   # これ以上なら直接アサイン
CANDIDATE_THRESHOLD = 0.55   # これ以上なら候補ヒントとして渡す

_KNOWLEDGE_PATH = Path(__file__).parent.parent / "config" / "semantic_knowledge.json"
_MAX_LEARNED_EXAMPLES = 20  # ロールごとの学習例文上限


def _load_role_descriptions() -> dict[str, list[str]]:
    """semantic_knowledge.json からロール例文を読み込む。JSONがなければハードコードにフォールバック。"""
    try:
        data = json.loads(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        result = {}
        for role, info in data.get("roles", {}).items():
            examples = list(info.get("examples", []))
            learned = list(info.get("learned_examples", []))
            result[role] = examples + learned
        if result:
            logger.debug("SemanticRouter: %s からロール説明読み込み (%d ロール)", _KNOWLEDGE_PATH.name, len(result))
            return result
    except Exception as e:
        logger.warning("semantic_knowledge.json 読み込み失敗、ハードコードにフォールバック: %s", e)
    return _ROLE_DESCRIPTIONS_FALLBACK


# フォールバック用ハードコード（JSONが読めない場合のみ使用）
_ROLE_DESCRIPTIONS_FALLBACK: dict[str, list[str]] = {
    "uketuke_qa": [
        "今日の天気は？ 今何時？ 計算して。 雑談したい。 クイズ出して。",
        "簡単な質問に答えて。 事実を教えて。 ちょっと聞きたいことがある。",
        "What time is it? Quick question. Tell me a fact. Let's chat.",
    ],
    "yuhitsu_jp": [
        "日本語に翻訳して。 英語に翻訳して。 文章を清書して。 敬語に直して。",
        "この文章を要約して。 文体を変えて。 メールの文章を書いて。 校正して。",
        "Translate to Japanese. Rewrite this text. Proofread my writing.",
    ],
    "metsuke_proc": [
        "技術的に説明して。 比較して教えて。 レビューして。 ドキュメントを書いて。",
        "わかりやすく解説して。 メリットとデメリットを教えて。 報告書を作って。",
        "Explain technically. Compare these. Write documentation. Give a review.",
    ],
    "gunshi_haiku": [
        "アーキテクチャを設計して。 戦略を立てて。 深く分析して。 ビジネス計画を作って。",
        "複雑な問題を解決して。 システム設計の相談。 多角的に考えて。 推論して。",
        "Design the architecture. Plan a strategy. Deep analysis. Business logic.",
    ],
    "gaiji_rag": [
        "最新ニュースを検索して。 今の情報を教えて。 ウェブで調べて。 最近どうなっている？",
        "インターネットで検索して。 外部情報を調べて。 ニュース記事を探して。",
        "Search the web. Find latest news. Look up current information. Browse internet.",
    ],
    "sanbo_mcp": [
        "Pythonコードを実行して。 ファイルを操作して。 Gitコマンドを実行して。 コードを書いて。",
        "スクリプトを実行して。 ファイルを読んで。 GitHubにプッシュして。 コードを動かして。",
        "Run Python code. Execute script. File operation. Git commit. Write and run code.",
    ],
    "kengyo_vision": [
        "この画像を解析して。 スクリーンショットを見て。 画像に何が写っている？ 図を読み取って。",
        "写真を説明して。 グラフを解析して。 画像認識して。 ビジュアルを分析して。",
        "Analyze this image. Read the screenshot. What is in this picture? Describe the chart.",
    ],
    "onmitsu_local": [
        "機密情報を処理して。 プライベートなデータ。 社外秘の内容。 ローカルで処理して。",
        "外部に送らないで。 秘密の情報。 個人情報を扱って。 オフラインで処理して。",
        "Process confidential data. Private information. Secret content. Local only processing.",
    ],
    "shogun_plan": [
        "大規模プロジェクトを計画して。 ロードマップを作って。 長期計画を立てて。",
        "システム開発全体を管理して。 プロジェクト計画書を作って。 工程表を作成して。",
        "Plan a large project. Create roadmap. Long-term planning. Full software development plan.",
    ],
}


class SemanticRouter:
    """
    文埋め込みモデルを使ったローカルルーター。
    """

    _instance: Optional["SemanticRouter"] = None
    _available: bool = False
    
    # クラスレベルで共有 (プロセス内1回のみロード)
    _shared_model = None
    _shared_role_vectors: dict[str, list] = {}
    _shared_ready = False
    _loading_lock = asyncio.Lock()  # ロード重複防止用

    def __init__(self):
        pass

    @classmethod
    def get(cls) -> "SemanticRouter":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self) -> bool:
        """モデルをロードしてロールベクトルを計算する。非同期ではないため注意。"""
        if SemanticRouter._shared_ready:
            return True
        
        # 起動直後のリクエストで同時に initialize が呼ばれた場合に
        # 二重ロードされるのを防ぐ簡易チェック
        if SemanticRouter._shared_model is not None:
            return False

        try:
            from sentence_transformers import SentenceTransformer
            t0 = time.time()
            
            logger.info("⏳ SemanticRouter: モデルロード開始 (sentence-transformers)...")
            # このロードに時間がかかる
            SemanticRouter._shared_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            # ロール説明（例文リスト）をベクトル化し、ロールごとに平均ベクトルを計算
            import numpy as np
            role_descs = _load_role_descriptions()
            roles = list(role_descs.keys())
            
            new_vectors = {}
            for role in roles:
                examples = role_descs[role]
                vecs = SemanticRouter._shared_model.encode(examples, normalize_embeddings=True)
                avg = vecs.mean(axis=0)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                new_vectors[role] = avg.tolist()
            
            SemanticRouter._shared_role_vectors = new_vectors
            SemanticRouter._shared_ready = True
            SemanticRouter._available = True
            logger.info("✅ SemanticRouter 初期化完了 (%.1fs, %d ロール)", time.time() - t0, len(roles))
            return True
        except ImportError:
            logger.warning("⚠️ sentence-transformers 未インストール — SemanticRouter 無効")
            return False
        except Exception as e:
            logger.warning("⚠️ SemanticRouter 初期化失敗: %s", e)
            return False

    def route(self, message: str) -> tuple[Optional[str], float]:
        """
        メッセージを最も近いロールにルーティングする。
        """
        if not SemanticRouter._shared_ready or not SemanticRouter._shared_model:
            return None, 0.0

        try:
            import numpy as np
            vec = SemanticRouter._shared_model.encode([message], normalize_embeddings=True)[0]
            best_role = None
            best_score = -1.0
            for role, role_vec in SemanticRouter._shared_role_vectors.items():
                score = float(np.dot(vec, role_vec))
                if score > best_score:
                    best_score = score
                    best_role = role

            if best_score >= CONFIDENT_THRESHOLD:
                logger.info(
                    "🧭 SemanticRouter: %s (%.3f ≥ %.2f) → 直接ルーティング",
                    best_role, best_score, CONFIDENT_THRESHOLD,
                )
                return best_role, best_score
            elif best_score >= CANDIDATE_THRESHOLD:
                logger.debug(
                    "🧭 SemanticRouter: %s (%.3f) → 候補ヒント",
                    best_role, best_score,
                )
                return best_role, best_score
            else:
                logger.debug(
                    "🧭 SemanticRouter: スコア低 (%.3f) → analyze_intent へ", best_score
                )
                return None, best_score
        except Exception as e:
            logger.warning("⚠️ SemanticRouter.route 失敗: %s", e)
            return None, 0.0

    def add_example(self, role: str, text: str) -> bool:
        """成功した入力例をロールの学習例文に追加し、JSONとベクトルを更新する。"""
        try:
            data = json.loads(_KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return False

        role_data = data.get("roles", {}).get(role)
        if role_data is None:
            return False

        learned = role_data.setdefault("learned_examples", [])
        if text in learned or text in role_data.get("examples", []):
            return False  # 重複スキップ
        learned.append(text)

        if len(learned) > _MAX_LEARNED_EXAMPLES:
            learned[:] = learned[-_MAX_LEARNED_EXAMPLES:]

        import datetime
        data["updated_at"] = datetime.date.today().isoformat()
        try:
            _KNOWLEDGE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("semantic_knowledge.json 書き込み失敗: %s", e)
            return False

        if SemanticRouter._shared_ready and SemanticRouter._shared_model:
            try:
                import numpy as np
                role_descs = _load_role_descriptions()
                examples = role_descs.get(role, [])
                if examples:
                    vecs = SemanticRouter._shared_model.encode(examples, normalize_embeddings=True)
                    avg = vecs.mean(axis=0)
                    norm = np.linalg.norm(avg)
                    if norm > 0:
                        avg = avg / norm
                    SemanticRouter._shared_role_vectors[role] = avg.tolist()
                    logger.info("🧭 SemanticRouter: %s のベクトル更新 (例文数=%d)", role, len(examples))
            except Exception as e:
                logger.warning("SemanticRouter add_example ベクトル更新失敗: %s", e)

        return True

    def reload(self) -> bool:
        """JSONから再読み込みして全ベクトルを再計算する。"""
        SemanticRouter._shared_ready = False
        return self.initialize()

    @property
    def is_ready(self) -> bool:
        return SemanticRouter._shared_ready
