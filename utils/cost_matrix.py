"""
座標生成・距離行列計算ユーティリティ
"""

import numpy as np


def generate_positions(n: int, seed: int = None, area: float = 100.0) -> np.ndarray:
    """
    ランダムな座標を生成

    Parameters
    ----------
    n : int
        座標数
    seed : int
        乱数シード
    area : float
        座標範囲 [0, area]

    Returns
    -------
    positions : np.ndarray (n, 2)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, area, size=(n, 2))


def compute_distance_matrix(pos_a: np.ndarray, pos_b: np.ndarray) -> np.ndarray:
    """
    2つの座標群間のユークリッド距離行列を計算

    Parameters
    ----------
    pos_a : np.ndarray (M, 2)
    pos_b : np.ndarray (N, 2)

    Returns
    -------
    dist : np.ndarray (M, N)
    """
    return np.linalg.norm(
        pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :],
        axis=2
    )