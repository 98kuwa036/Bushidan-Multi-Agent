"""武士団 JupyterLab 共通ユーティリティ v18"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# プロジェクトルートを sys.path に追加（project モジュールの import 用）
PROJECT_ROOT = Path('/mnt/Bushidan-Multi-Agent')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# .env ロード（POSTGRES_URL 等）
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env', override=False)

import psycopg
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401  日本語フォント自動設定

# ── matplotlib スタイル ────────────────────────────────────────────
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    plt.style.use('ggplot')

matplotlib.rcParams.update({
    'figure.dpi': 110,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFAFA',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

# ── 接続 ────────────────────────────────────────────────────────────
POSTGRES_URL: str = os.environ.get('POSTGRES_URL', '')

def get_conn() -> psycopg.Connection:
    if not POSTGRES_URL:
        raise RuntimeError('POSTGRES_URL が .env に設定されていません')
    return psycopg.connect(POSTGRES_URL, autocommit=True)

def qdf(sql: str, params=None) -> pd.DataFrame:
    """SQL を実行して DataFrame を返す。エラー時は空 DataFrame。"""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d.name for d in cur.description]
                rows = cur.fetchall()
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        print(f'⚠️  クエリエラー: {e}')
        return pd.DataFrame()

def scalar(sql: str, params=None, default=0):
    """単一値を返す SQL ヘルパー。"""
    df = qdf(sql, params)
    return df.iloc[0, 0] if len(df) > 0 else default

# ── ロール定義 (v18 11役職) ──────────────────────────────────────────
ROLES = [
    'uketuke', 'gaiji', 'gunshi', 'sanbo',
    'shogun', 'daigensui', 'metsuke', 'kengyo', 'yuhitsu', 'onmitsu',
]

ROLE_JA = {
    'uketuke':   '受付',
    'gaiji':     '外事',
    'gunshi':    '軍師',
    'sanbo':     '参謀',
    'shogun':    '将軍',
    'daigensui': '大元帥',
    'metsuke':   '目付',
    'kengyo':    '検校',
    'yuhitsu':   '右筆',
    'onmitsu':   '隠密',
}

ROLE_MODEL = {
    'uketuke':   'Llama 3.3 70B (Groq)',
    'gaiji':     'Cohere Command R',
    'gunshi':    'Cohere Command A',
    'sanbo':     'Gemini Flash Preview',
    'shogun':    'Claude Sonnet 4.6',
    'daigensui': 'Claude Opus 4.6',
    'metsuke':   'Mistral Small',
    'kengyo':    'Gemini Flash Image',
    'yuhitsu':   'Gemma4 / Gemini FL',
    'onmitsu':   'Nemotron / Gemma4',
}

# ── カラーパレット（武士団ベージュテーマ） ────────────────────────────
PALETTE = [
    '#D4A574', '#5B8DB8', '#5A9E6F', '#9B6EC8', '#C8503A',
    '#E8B84B', '#6BAA5E', '#C87DB8', '#4AABB8', '#D86B6B', '#8B8070',
]
ROLE_COLOR = dict(zip(ROLES, PALETTE))

def role_label(key: str) -> str:
    """'shogun' → '将軍 (shogun)'"""
    ja = ROLE_JA.get(key, key)
    return f'{ja}\n({key})'

# 日本語名 → 英語キー（DBに日本語で入っているケースを正規化）
_JA_TO_KEY = {v: k for k, v in ROLE_JA.items()}

def normalize_role(r: str) -> str:
    """'右筆' → 'yuhitsu'、'shogun' → 'shogun' のように正規化"""
    return _JA_TO_KEY.get(r, r)

def ja_role(r: str) -> str:
    """英語キーか日本語名かに関わらず日本語表示名を返す"""
    if r in ROLE_JA:
        return ROLE_JA[r]
    if r in _JA_TO_KEY:
        return r  # 既に日本語
    return r

# ── グラフユーティリティ ─────────────────────────────────────────────
def bar_h(series: pd.Series, title: str, xlabel: str, ax=None, color: str | None = None):
    """横棒グラフをシンプルに描画。"""
    _ax = ax or plt.gca()
    series.plot(kind='barh', ax=_ax, color=color or PALETTE[0], edgecolor='white', linewidth=0.5)
    _ax.set_title(title)
    _ax.set_xlabel(xlabel)
    _ax.set_ylabel('')
    for bar in _ax.patches:
        w = bar.get_width()
        _ax.text(w + max(series.max() * 0.01, 0.1), bar.get_y() + bar.get_height() / 2,
                 f'{w:.0f}' if w >= 1 else f'{w:.2f}', va='center', fontsize=9)
    return _ax

def check_data(df: pd.DataFrame, name: str = 'データ') -> bool:
    """DataFrame が空なら案内メッセージを表示して False を返す。"""
    if len(df) == 0:
        print(f'ℹ️  {name} がまだありません。チャットを利用すると自動で記録されます。')
        return False
    return True

print('✅ 武士団 utils ロード完了')
print(f'   DB: {POSTGRES_URL.split("@")[-1] if "@" in POSTGRES_URL else "未設定"}')
