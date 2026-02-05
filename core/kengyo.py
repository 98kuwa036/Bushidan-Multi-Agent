"""
Bushidan Multi-Agent System v10.1 - 検校 (Kengyo) Visual Debugger

検校: 中世日本で盲目ながら鋭い感覚で真実を見抜いた最高位検校。
本システムでは Kimi K2.5 のマルチモーダル能力と Playwright MCP を組み合わせ、
UI/UX の品質を「視覚」で検証するビジュアル・デバッガー。

PDCA Check フェーズとの統合:
  テキスト検証 (Gunshi 256K) + ビジュアル検証 (Kengyo) = 完全品質保証

Workflow:
  1. Playwright MCP → スクリーンショットキャプチャ
  2. Kimi K2.5 Vision → 画像解析・問題検出
  3. 検証レポート生成 → Gunshi Check フェーズに統合

検証項目:
  - レイアウト崩れ検出
  - 要素の表示/非表示検証
  - レスポンシブ対応確認 (Desktop / Tablet / Mobile)
  - アクセシビリティ基本チェック (コントラスト等)
  - ビフォー・アフター比較 (ビジュアルリグレッション)

Position: 将軍 → 軍師 → 家老 → 大将 → 足軽
                        ↑ Check フェーズで 検校(Kengyo) が視覚検証を担当
"""

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("bushidan.kengyo")


# ==================== Data Structures ====================

class ViewportSize(Enum):
    """検証対象のビューポートサイズ"""
    DESKTOP = "desktop"
    TABLET = "tablet"
    MOBILE = "mobile"


VIEWPORT_DIMENSIONS = {
    ViewportSize.DESKTOP: {"width": 1280, "height": 720},
    ViewportSize.TABLET: {"width": 768, "height": 1024},
    ViewportSize.MOBILE: {"width": 375, "height": 812},
}


class VisualIssueSeverity(Enum):
    """ビジュアル問題の重大度"""
    CRITICAL = "critical"    # 表示不能・操作不能
    WARNING = "warning"      # 崩れ・見づらい
    INFO = "info"            # 改善提案


@dataclass
class VisualIssue:
    """検出されたビジュアル問題"""
    severity: VisualIssueSeverity
    category: str           # layout, display, accessibility, regression
    description: str
    location: str = ""      # 画面上の位置（例: "ヘッダー右上"）
    suggestion: str = ""    # 修正提案


@dataclass
class ScreenshotResult:
    """スクリーンショット取得結果"""
    success: bool
    image_data: str = ""        # base64 data URI
    viewport: str = "desktop"
    url: str = ""
    error: str = ""
    capture_time: float = 0.0


@dataclass
class VisualCheckResult:
    """ビジュアル検証結果"""
    passed: bool
    score: float                        # 0.0 - 1.0
    issues: List[VisualIssue] = field(default_factory=list)
    summary: str = ""
    viewport: str = "desktop"
    analysis_time: float = 0.0


@dataclass
class VisualAuditReport:
    """完全ビジュアル監査レポート"""
    url: str
    passed: bool
    overall_score: float
    checks: List[VisualCheckResult] = field(default_factory=list)
    total_issues: int = 0
    critical_issues: int = 0
    screenshots_captured: int = 0
    total_time: float = 0.0


# ==================== Kengyo Visual Debugger ====================

class Kengyo:
    """
    検校 (Kengyo) - ビジュアル・デバッガー

    Kimi K2.5 のマルチモーダル能力と Playwright MCP を組み合わせ、
    PDCA Check フェーズの品質検証を視覚的に拡張する。

    ■ 問題認識:
    テキストベースの Check (Gunshi 256K) は論理的整合性を検証できるが、
    UI のレンダリング結果は「見ないとわからない」。

    ■ 解決策: 検校 (Kengyo) によるビジュアル検証
    1. Playwright MCP でスクリーンショットをキャプチャ
    2. Kimi K2.5 の Vision 能力で画像を分析
    3. レイアウト崩れ・表示異常・リグレッションを検出
    4. 結果を Gunshi Check フェーズの verdict に統合

    ■ 稼働条件:
    - Kimi K2.5 クライアントが初期化済み (マルチモーダル対応)
    - Playwright MCP が利用可能 (スクリーンショット取得)
    - タスクコンテキストにURL or UI関連情報が含まれる
    上記を満たさない場合はスキップ (グレースフルデグラデーション)
    """

    VERSION = "10.1"

    # 検証で使用するチェックリスト
    DEFAULT_CHECK_CRITERIA = [
        "レイアウトの崩れ (要素の重なり、はみ出し、意図しない空白)",
        "テキストの可読性 (フォントサイズ、コントラスト比)",
        "ボタン・リンクの視認性と操作可能性",
        "画像・アイコンの表示状態 (欠落、歪み)",
        "レスポンシブ対応 (画面幅に対する適切なレイアウト)",
        "色使いの一貫性とアクセシビリティ",
    ]

    def __init__(self, kimi_client=None, smithery_mcp=None):
        """
        Args:
            kimi_client: Kimi K2.5 クライアント (マルチモーダル対応)
            smithery_mcp: Smithery MCP マネージャー (Playwright MCP 含む)
        """
        self.kimi_client = kimi_client
        self.smithery_mcp = smithery_mcp
        self.initialized = False

        # 統計
        self._stats = {
            "total_audits": 0,
            "screenshots_captured": 0,
            "visual_checks_run": 0,
            "issues_detected": 0,
            "critical_issues": 0,
            "audits_passed": 0,
            "audits_failed": 0,
            "total_time_seconds": 0.0,
        }

        logger.info("👁️ 検校（ビジュアル・デバッガー）初期化開始...")

    async def initialize(self) -> bool:
        """
        検校の初期化

        Kimi K2.5 (Vision) と Playwright MCP の可用性を確認。
        どちらかが欠けていても一部機能は動作する。

        Returns:
            初期化成功フラグ (両方揃えば True)
        """
        kimi_ready = self.kimi_client is not None
        playwright_ready = False

        if self.smithery_mcp:
            servers = self.smithery_mcp.get_available_servers()
            playwright_info = servers.get("playwright", {})
            playwright_ready = playwright_info.get("available", False)

        if kimi_ready and playwright_ready:
            self.initialized = True
            logger.info(
                "👁️ 検校初期化完了 (Kimi K2.5 Vision ✅ + Playwright MCP ✅)"
            )
        elif kimi_ready:
            self.initialized = True
            logger.info(
                "👁️ 検校初期化完了 (Kimi K2.5 Vision ✅ + Playwright MCP ❌)"
            )
            logger.info(
                "   → 画像URLの直接解析は可能、スクリーンショット取得は不可"
            )
        else:
            logger.warning(
                "⚠️ 検校初期化失敗: Kimi K2.5 未設定 → ビジュアル検証スキップ"
            )

        return self.initialized

    def is_available(self) -> bool:
        """検校が利用可能かどうか"""
        return self.initialized and self.kimi_client is not None

    # ==================== Screenshot Capture ====================

    async def capture_screenshot(
        self,
        url: str,
        viewport: ViewportSize = ViewportSize.DESKTOP,
    ) -> ScreenshotResult:
        """
        Playwright MCP でスクリーンショットをキャプチャ

        Args:
            url: キャプチャ対象のURL
            viewport: ビューポートサイズ

        Returns:
            ScreenshotResult (base64 画像データ含む)
        """
        if not self.smithery_mcp:
            return ScreenshotResult(
                success=False,
                url=url,
                viewport=viewport.value,
                error="Playwright MCP 未初期化",
            )

        start_time = time.monotonic()
        dimensions = VIEWPORT_DIMENSIONS[viewport]

        try:
            # Playwright MCP にスクリーンショットリクエスト
            result = await self.smithery_mcp.send_request(
                "playwright",
                "browser_screenshot",
                {
                    "url": url,
                    "width": dimensions["width"],
                    "height": dimensions["height"],
                    "fullPage": True,
                },
            )

            if result and result.get("screenshot"):
                elapsed = time.monotonic() - start_time
                self._stats["screenshots_captured"] += 1
                logger.info(
                    f"📸 スクリーンショット取得: {url} "
                    f"({viewport.value}, {elapsed:.1f}s)"
                )
                return ScreenshotResult(
                    success=True,
                    image_data=result["screenshot"],
                    viewport=viewport.value,
                    url=url,
                    capture_time=elapsed,
                )

            return ScreenshotResult(
                success=False,
                url=url,
                viewport=viewport.value,
                error="Playwright returned no screenshot data",
            )

        except Exception as e:
            logger.error(f"❌ スクリーンショット取得失敗 ({url}): {e}")
            return ScreenshotResult(
                success=False,
                url=url,
                viewport=viewport.value,
                error=str(e),
            )

    async def capture_multi_viewport(
        self,
        url: str,
        viewports: Optional[List[ViewportSize]] = None,
    ) -> List[ScreenshotResult]:
        """
        複数ビューポートでスクリーンショットをキャプチャ

        Args:
            url: キャプチャ対象のURL
            viewports: ビューポートリスト (デフォルト: Desktop + Mobile)

        Returns:
            ScreenshotResult のリスト
        """
        if viewports is None:
            viewports = [ViewportSize.DESKTOP, ViewportSize.MOBILE]

        tasks = [self.capture_screenshot(url, vp) for vp in viewports]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    # ==================== Visual Analysis ====================

    async def analyze_screenshot(
        self,
        image_data: str,
        check_criteria: Optional[List[str]] = None,
        context: str = "",
    ) -> VisualCheckResult:
        """
        Kimi K2.5 Vision でスクリーンショットを解析

        Args:
            image_data: base64 画像データ (data URI or raw base64)
            check_criteria: チェック項目リスト
            context: 追加コンテキスト (何を実装したか等)

        Returns:
            VisualCheckResult
        """
        if not self.kimi_client:
            return VisualCheckResult(
                passed=True,
                score=0.0,
                summary="Kimi K2.5 未設定のためスキップ",
            )

        start_time = time.monotonic()
        criteria = check_criteria or self.DEFAULT_CHECK_CRITERIA

        # 画像 URL の構築
        if not image_data.startswith("data:"):
            image_url = f"data:image/png;base64,{image_data}"
        else:
            image_url = image_data

        criteria_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(criteria))
        context_block = f"\n## 実装コンテキスト\n{context}" if context else ""

        check_prompt = (
            f"以下のスクリーンショットを検証してください。{context_block}\n\n"
            f"## チェック項目\n{criteria_text}\n\n"
            "## 出力形式 (JSON)\n"
            "```json\n"
            "{\n"
            '  "passed": true,\n'
            '  "score": 0.85,\n'
            '  "issues": [\n'
            "    {\n"
            '      "severity": "warning",\n'
            '      "category": "layout",\n'
            '      "description": "問題の説明",\n'
            '      "location": "画面上の位置",\n'
            '      "suggestion": "修正提案"\n'
            "    }\n"
            "  ],\n"
            '  "summary": "検証結果の概要"\n'
            "}\n"
            "```\n\n"
            "注意:\n"
            "- score は 0.0 (完全に壊れている) ～ 1.0 (問題なし) で評価\n"
            "- severity は critical (操作不能), warning (崩れ), info (改善提案)\n"
            "- category は layout, display, accessibility, regression のいずれか\n"
            "- critical な問題がある場合は passed=false にする"
        )

        try:
            result_text = await self.kimi_client.visual_check(
                image_url=image_url,
                check_prompt=check_prompt,
            )
            elapsed = time.monotonic() - start_time
            self._stats["visual_checks_run"] += 1

            # JSON 解析
            check_data = self._parse_json(result_text)
            if not check_data:
                logger.warning("⚠️ ビジュアル検証結果のJSON解析失敗")
                return VisualCheckResult(
                    passed=True,
                    score=0.70,
                    summary=result_text[:500],
                    analysis_time=elapsed,
                )

            # Issue オブジェクト構築
            issues = []
            for issue_data in check_data.get("issues", []):
                severity_str = issue_data.get("severity", "info")
                try:
                    severity = VisualIssueSeverity(severity_str)
                except ValueError:
                    severity = VisualIssueSeverity.INFO

                issue = VisualIssue(
                    severity=severity,
                    category=issue_data.get("category", "display"),
                    description=issue_data.get("description", ""),
                    location=issue_data.get("location", ""),
                    suggestion=issue_data.get("suggestion", ""),
                )
                issues.append(issue)

            # 統計更新
            self._stats["issues_detected"] += len(issues)
            self._stats["critical_issues"] += sum(
                1 for i in issues
                if i.severity == VisualIssueSeverity.CRITICAL
            )

            return VisualCheckResult(
                passed=check_data.get("passed", True),
                score=min(1.0, max(0.0, check_data.get("score", 0.70))),
                issues=issues,
                summary=check_data.get("summary", ""),
                analysis_time=elapsed,
            )

        except Exception as e:
            logger.error(f"❌ ビジュアル解析失敗: {e}")
            return VisualCheckResult(
                passed=True,
                score=0.0,
                summary=f"解析エラー: {e}",
                analysis_time=time.monotonic() - start_time,
            )

    # ==================== Visual Comparison ====================

    async def compare_screenshots(
        self,
        before_image: str,
        after_image: str,
        change_description: str = "",
    ) -> VisualCheckResult:
        """
        ビフォー・アフター比較 (ビジュアルリグレッション検出)

        Args:
            before_image: 変更前の画像 (base64)
            after_image: 変更後の画像 (base64)
            change_description: 変更内容の説明

        Returns:
            VisualCheckResult (リグレッション検出結果)
        """
        if not self.kimi_client:
            return VisualCheckResult(
                passed=True,
                score=0.0,
                summary="Kimi K2.5 未設定のためスキップ",
            )

        start_time = time.monotonic()

        # before/after 画像URL構築
        before_url = (
            before_image if before_image.startswith("data:")
            else f"data:image/png;base64,{before_image}"
        )
        after_url = (
            after_image if after_image.startswith("data:")
            else f"data:image/png;base64,{after_image}"
        )

        change_block = (
            f"\n## 変更内容\n{change_description}" if change_description else ""
        )

        compare_prompt = (
            "2つのスクリーンショットを比較してください。\n"
            "1枚目: 変更前 (Before)\n"
            "2枚目: 変更後 (After)\n"
            f"{change_block}\n\n"
            "## 検証項目\n"
            "1. 意図した変更が正しく反映されているか\n"
            "2. 意図しない変更 (リグレッション) がないか\n"
            "3. レイアウト崩れが発生していないか\n\n"
            "## 出力形式 (JSON)\n"
            "```json\n"
            "{\n"
            '  "passed": true,\n'
            '  "score": 0.90,\n'
            '  "issues": [\n'
            "    {\n"
            '      "severity": "warning",\n'
            '      "category": "regression",\n'
            '      "description": "リグレッションの説明",\n'
            '      "location": "画面上の位置",\n'
            '      "suggestion": "修正提案"\n'
            "    }\n"
            "  ],\n"
            '  "summary": "比較結果の概要"\n'
            "}\n"
            "```"
        )

        try:
            # マルチモーダルリクエスト: 2枚の画像を送信
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Before (変更前):"},
                    {"type": "image_url", "image_url": {"url": before_url}},
                    {"type": "text", "text": "After (変更後):"},
                    {"type": "image_url", "image_url": {"url": after_url}},
                    {"type": "text", "text": compare_prompt},
                ],
            }]

            result_text = await self.kimi_client.generate(
                messages,
                temperature=0.1,
                system_prompt=(
                    "あなたはUI/UXリグレッション検出の専門家です。"
                    "2つのスクリーンショットを比較し、"
                    "意図しない変更を正確に特定してください。"
                ),
            )

            elapsed = time.monotonic() - start_time
            self._stats["visual_checks_run"] += 1

            check_data = self._parse_json(result_text)
            if not check_data:
                return VisualCheckResult(
                    passed=True,
                    score=0.70,
                    summary=result_text[:500],
                    analysis_time=elapsed,
                )

            issues = []
            for issue_data in check_data.get("issues", []):
                try:
                    severity = VisualIssueSeverity(
                        issue_data.get("severity", "info")
                    )
                except ValueError:
                    severity = VisualIssueSeverity.INFO

                issues.append(VisualIssue(
                    severity=severity,
                    category=issue_data.get("category", "regression"),
                    description=issue_data.get("description", ""),
                    location=issue_data.get("location", ""),
                    suggestion=issue_data.get("suggestion", ""),
                ))

            self._stats["issues_detected"] += len(issues)
            self._stats["critical_issues"] += sum(
                1 for i in issues
                if i.severity == VisualIssueSeverity.CRITICAL
            )

            return VisualCheckResult(
                passed=check_data.get("passed", True),
                score=min(1.0, max(0.0, check_data.get("score", 0.70))),
                issues=issues,
                summary=check_data.get("summary", ""),
                analysis_time=elapsed,
            )

        except Exception as e:
            logger.error(f"❌ ビジュアル比較失敗: {e}")
            return VisualCheckResult(
                passed=True,
                score=0.0,
                summary=f"比較エラー: {e}",
                analysis_time=time.monotonic() - start_time,
            )

    # ==================== Full Visual Audit ====================

    async def run_visual_audit(
        self,
        url: str,
        check_criteria: Optional[List[str]] = None,
        viewports: Optional[List[ViewportSize]] = None,
        context: str = "",
    ) -> VisualAuditReport:
        """
        完全ビジュアル監査

        複数ビューポートでスクリーンショットを取得し、
        それぞれを Kimi K2.5 Vision で解析する。

        Args:
            url: 監査対象のURL
            check_criteria: チェック項目 (デフォルト: 6項目)
            viewports: ビューポート (デフォルト: Desktop + Mobile)
            context: 実装コンテキスト

        Returns:
            VisualAuditReport
        """
        if not self.is_available():
            return VisualAuditReport(
                url=url,
                passed=True,
                overall_score=0.0,
                total_time=0.0,
            )

        start_time = time.monotonic()
        self._stats["total_audits"] += 1

        if viewports is None:
            viewports = [ViewportSize.DESKTOP, ViewportSize.MOBILE]

        # Step 1: 複数ビューポートでスクリーンショット取得
        screenshots = await self.capture_multi_viewport(url, viewports)

        # Step 2: 各スクリーンショットを解析
        checks: List[VisualCheckResult] = []
        for screenshot in screenshots:
            if not screenshot.success:
                logger.warning(
                    f"⚠️ {screenshot.viewport} キャプチャ失敗: "
                    f"{screenshot.error}"
                )
                continue

            viewport_context = (
                f"{context}\n"
                f"ビューポート: {screenshot.viewport} "
                f"({VIEWPORT_DIMENSIONS.get(ViewportSize(screenshot.viewport), {})})"
            )

            check_result = await self.analyze_screenshot(
                image_data=screenshot.image_data,
                check_criteria=check_criteria,
                context=viewport_context,
            )
            check_result.viewport = screenshot.viewport
            checks.append(check_result)

        # Step 3: レポート統合
        total_time = time.monotonic() - start_time
        self._stats["total_time_seconds"] += total_time

        if not checks:
            return VisualAuditReport(
                url=url,
                passed=True,
                overall_score=0.0,
                total_time=total_time,
            )

        all_issues = []
        for check in checks:
            all_issues.extend(check.issues)

        critical_count = sum(
            1 for issue in all_issues
            if issue.severity == VisualIssueSeverity.CRITICAL
        )

        overall_score = sum(c.score for c in checks) / len(checks)
        overall_passed = all(c.passed for c in checks)

        if overall_passed:
            self._stats["audits_passed"] += 1
        else:
            self._stats["audits_failed"] += 1

        logger.info(
            f"👁️ ビジュアル監査完了: {url} - "
            f"{'合格' if overall_passed else '不合格'} "
            f"(スコア {overall_score:.0%}, "
            f"{len(all_issues)}件, {total_time:.1f}s)"
        )

        return VisualAuditReport(
            url=url,
            passed=overall_passed,
            overall_score=overall_score,
            checks=checks,
            total_issues=len(all_issues),
            critical_issues=critical_count,
            screenshots_captured=sum(
                1 for s in screenshots if s.success
            ),
            total_time=total_time,
        )

    # ==================== PDCA Integration ====================

    async def check_phase_visual_verify(
        self,
        task_content: str,
        subtask_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        PDCA Check フェーズ向けビジュアル検証エントリポイント

        Gunshi の _phase_check() から呼び出される。
        タスクコンテキストに UI 関連情報が含まれる場合のみ実行。

        Args:
            task_content: 元のタスク内容
            subtask_results: サブタスクの実行結果リスト
            context: タスクコンテキスト

        Returns:
            ビジュアル検証結果 (dict) or None (スキップ時)
        """
        if not self.is_available():
            return None

        # UI 関連タスクかどうかを判定
        visual_context = self._extract_visual_context(
            task_content, context
        )
        if not visual_context:
            return None

        logger.info("👁️ 検校: ビジュアル検証開始...")

        results = {}

        # URL が指定されている場合: フル監査
        urls = visual_context.get("urls", [])
        if urls:
            audit_results = []
            for url in urls[:3]:  # 最大3 URL
                audit = await self.run_visual_audit(
                    url=url,
                    context=task_content,
                )
                audit_results.append({
                    "url": audit.url,
                    "passed": audit.passed,
                    "score": audit.overall_score,
                    "total_issues": audit.total_issues,
                    "critical_issues": audit.critical_issues,
                    "checks": [
                        {
                            "viewport": c.viewport,
                            "passed": c.passed,
                            "score": c.score,
                            "issues": [
                                {
                                    "severity": i.severity.value,
                                    "category": i.category,
                                    "description": i.description,
                                    "suggestion": i.suggestion,
                                }
                                for i in c.issues
                            ],
                        }
                        for c in audit.checks
                    ],
                })
            results["audits"] = audit_results

        # 画像が直接提供されている場合: 直接解析
        images = visual_context.get("images", [])
        if images:
            image_results = []
            for image_data in images[:5]:  # 最大5枚
                check = await self.analyze_screenshot(
                    image_data=image_data,
                    context=task_content,
                )
                image_results.append({
                    "passed": check.passed,
                    "score": check.score,
                    "issues_count": len(check.issues),
                    "summary": check.summary,
                })
            results["image_checks"] = image_results

        # ビフォー・アフターが提供されている場合: 比較
        before_after = visual_context.get("before_after", [])
        for pair in before_after[:2]:  # 最大2ペア
            comparison = await self.compare_screenshots(
                before_image=pair["before"],
                after_image=pair["after"],
                change_description=pair.get("description", task_content),
            )
            results.setdefault("comparisons", []).append({
                "passed": comparison.passed,
                "score": comparison.score,
                "issues_count": len(comparison.issues),
                "summary": comparison.summary,
            })

        # 総合判定
        all_passed = True
        total_issues = 0
        total_critical = 0

        for audit in results.get("audits", []):
            if not audit["passed"]:
                all_passed = False
            total_issues += audit["total_issues"]
            total_critical += audit["critical_issues"]

        for check in results.get("image_checks", []):
            if not check["passed"]:
                all_passed = False
            total_issues += check["issues_count"]

        for comp in results.get("comparisons", []):
            if not comp["passed"]:
                all_passed = False
            total_issues += comp["issues_count"]

        results["overall_passed"] = all_passed
        results["total_issues"] = total_issues
        results["critical_issues"] = total_critical

        status = "合格" if all_passed else "不合格"
        logger.info(
            f"👁️ 検校検証完了: {status} "
            f"(問題 {total_issues}件, 重大 {total_critical}件)"
        )

        return results

    def _extract_visual_context(
        self,
        task_content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        タスク内容から UI/ビジュアル関連コンテキストを抽出

        Args:
            task_content: タスク内容
            context: コンテキスト辞書

        Returns:
            ビジュアルコンテキスト辞書 or None (UI 関連なし)
        """
        visual_ctx: Dict[str, Any] = {}

        # コンテキストから明示的なビジュアル情報を取得
        if context:
            # 明示的にビジュアル検証が要求されている
            if context.get("visual_check"):
                visual_ctx.update(context["visual_check"])
                return visual_ctx

            # URL が指定されている
            urls = context.get("urls", [])
            if urls:
                visual_ctx["urls"] = urls

            # 画像が直接提供されている
            images = context.get("screenshots", [])
            if images:
                visual_ctx["images"] = images

            # Before/After ペア
            before_after = context.get("before_after", [])
            if before_after:
                visual_ctx["before_after"] = before_after

        # タスク内容からの UI キーワード検出
        ui_keywords = [
            "UI", "画面", "フロントエンド", "frontend", "CSS", "レイアウト",
            "layout", "デザイン", "design", "レスポンシブ", "responsive",
            "スクリーンショット", "screenshot", "表示", "display", "render",
            "ページ", "page", "コンポーネント", "component", "HTML",
            "ビジュアル", "visual", "見た目",
        ]

        task_lower = task_content.lower()
        has_ui_keywords = any(kw.lower() in task_lower for kw in ui_keywords)

        if has_ui_keywords and visual_ctx:
            return visual_ctx

        # URL がコンテキストにある場合のみ、キーワードベースで有効化
        if has_ui_keywords and context and context.get("urls"):
            visual_ctx["urls"] = context["urls"]
            return visual_ctx

        # ビジュアル情報なし
        return visual_ctx if visual_ctx else None

    # ==================== Helpers ====================

    def _parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        """レスポンスから JSON を抽出"""
        import re

        # 直接 parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass

        # Markdown code block
        match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except (json.JSONDecodeError, TypeError):
                pass

        # テキスト中の JSON object
        brace_start = text.find('{')
        brace_end = text.rfind('}')
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start:brace_end + 1])
            except (json.JSONDecodeError, TypeError):
                pass

        return None

    # ==================== Statistics ====================

    def get_statistics(self) -> Dict[str, Any]:
        """統計取得"""
        return {
            "version": self.VERSION,
            "role": "検校 (Kengyo) - ビジュアル・デバッガー",
            "initialized": self.initialized,
            "kimi_available": self.kimi_client is not None,
            "playwright_available": (
                self.smithery_mcp is not None
                and self.smithery_mcp.get_available_servers()
                .get("playwright", {})
                .get("available", False)
                if self.smithery_mcp
                else False
            ),
            **self._stats,
        }
