"""
notebook_setup.py — 모든 노트북 공통 설정

사용법: 노트북 첫 번째 셀에 아래 한 줄만 추가
    from notebook_setup import *
"""

import platform
import pathlib
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.stats import kurtosis, skew

# ── 한글 폰트 & 그래프 기본 설정 ────────────────────────────
_OS = platform.system()
if _OS == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif _OS == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False   # 마이너스(-) 기호 깨짐 방지
plt.rcParams['figure.figsize']     = (13, 4)
plt.rcParams['font.size']          = 11

# ── 안전한 파일 다운로드 ─────────────────────────────────────
def safe_download(url: str, save_path: pathlib.Path, min_size_kb: int = 5) -> bool:
    """
    urllib로 파일 다운로드. 이미 있거나 작은 파일이면 재시도.

    Returns True if file is ready, False if download failed.
    """
    fpath = pathlib.Path(save_path)
    if fpath.exists() and fpath.stat().st_size > min_size_kb * 1024:
        print(f"  ✅ {fpath.name} — 이미 있음 ({fpath.stat().st_size // 1024} KB)")
        return True

    print(f"  ⬇️  {fpath.name} 다운로드 중...")
    try:
        urllib.request.urlretrieve(url, fpath)
        size = fpath.stat().st_size
        if size < min_size_kb * 1024:
            fpath.unlink(missing_ok=True)
            raise RuntimeError(f"파일이 너무 작음 ({size} bytes). 서버 오류 가능.")
        print(f"     완료 — {size // 1024} KB")
        return True
    except Exception as e:
        print(f"  ❌ 실패: {e}")
        print(f"     수동 다운로드: {url}")
        print(f"     저장 위치:    {fpath.resolve()}")
        return False


# ── CWRU .mat 파일 로드 ──────────────────────────────────────
def load_cwru_mat(fpath) -> tuple:
    """
    CWRU .mat → (신호 배열, 샘플링주파수)

    fpath: str 또는 pathlib.Path
    Returns: (np.ndarray 1D, int fs)
    """
    fpath = pathlib.Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(
            f"파일 없음: {fpath.resolve()}\n"
            "위 다운로드 셀을 다시 실행하거나 수동으로 파일을 복사해주세요."
        )
    mat = scipy.io.loadmat(str(fpath))
    de_keys = [k for k in mat.keys() if 'DE_time' in k]
    if not de_keys:
        available = [k for k in mat.keys() if not k.startswith('__')]
        raise KeyError(f"'DE_time' 키 없음. 파일 내 키: {available}")
    return mat[de_keys[0]].flatten(), 12000


# ── 시간영역 특징 추출 ───────────────────────────────────────
def extract_features(x: np.ndarray) -> dict:
    """
    1D 진동 신호 → 시간영역 특징 7종

    Kurtosis: fisher=False → 정상분포 = 3 (직관적)
    """
    rms      = np.sqrt(np.mean(x ** 2))
    peak     = np.max(np.abs(x))
    mean_abs = np.mean(np.abs(x))
    return {
        'RMS':          rms,
        'Peak':         peak,
        'Peak-to-Peak': np.ptp(x),
        'Crest_Factor': peak / (rms + 1e-10),
        'Kurtosis':     float(kurtosis(x, fisher=False)),
        'Skewness':     float(skew(x)),
        'Shape_Factor': rms / (mean_abs + 1e-10),
    }


print(f"notebook_setup 로드 완료 | OS: {_OS} | 폰트: {plt.rcParams['font.family']}")
