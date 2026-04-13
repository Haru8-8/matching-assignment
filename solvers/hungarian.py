"""
ハンガリアン法スクラッチ実装
LP双対性・相補性条件に基づく割当問題の厳密解法

問題設定：
  N人のドライバーをN件の配送先に1対1で割り当て、総移動コストを最小化

双対問題との対応：
  主問題: min Σ c_ij * x_ij
  双対問題: max Σ u_i + Σ v_j  s.t. u_i + v_j <= c_ij
  相補性条件: x_ij=1 => u_i + v_j = c_ij（割当ペアはタイト）
"""

import numpy as np


class HungarianSolver:
    """
    ハンガリアン法による割当問題ソルバー

    Attributes
    ----------
    cost : np.ndarray
        コスト行列 (N x N)
    n : int
        問題サイズ
    u : np.ndarray
        行ポテンシャル（双対変数）
    v : np.ndarray
        列ポテンシャル（双対変数）
    match_row : np.ndarray
        match_row[i] = j : ドライバーiが配送先jに割り当て済み（-1は未割当）
    match_col : np.ndarray
        match_col[j] = i : 配送先jにドライバーiが割り当て済み（-1は未割当）
    """

    def __init__(self, cost: np.ndarray, verbose: bool = False):
        assert cost.ndim == 2 and cost.shape[0] == cost.shape[1], \
            "コスト行列は正方行列である必要があります"
        self.cost = cost.astype(float)
        self.n = cost.shape[0]
        self.verbose = verbose

        # 双対変数
        self.u = np.zeros(self.n)
        self.v = np.zeros(self.n)

        # マッチング状態
        self.match_row = np.full(self.n, -1)  # match_row[i] = j
        self.match_col = np.full(self.n, -1)  # match_col[j] = i

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _slack(self, i: int, j: int) -> float:
        """緩み: c_ij - u_i - v_j（0なら相補性条件を満たせる候補）"""
        return self.cost[i, j] - self.u[i] - self.v[j]

    def _initialize(self):
        """
        初期ポテンシャル設定
        u_i = min_j(c_ij)  （各行の最小値）
        v_j = 0
        """
        self.u = self.cost.min(axis=1)
        self.v = np.zeros(self.n)
        self._log("=== 初期化 ===")
        self._log(f"u = {self.u}")
        self._log(f"v = {self.v}")

    def _find_augmenting_path(self, start_row: int) -> list[int] | None:
        """
        start_rowから始まる拡張路をBFSで探索

        ラベリング操作：
          - ラベル付き行から零要素で辿れる列にラベルを付ける
          - ラベル付き列がマッチング済みなら、その相手行にもラベルを付ける
          - ラベル付き列が未マッチングなら拡張路発見

        Returns
        -------
        path : list[int] or None
            [start_row, col_0, row_1, col_1, ..., col_end]
            拡張路が見つかった場合。見つからない場合はNone（δ計算が必要）
        """
        # 前回ラベルのリセット
        self._labeled_rows = set()
        self._labeled_cols = set()

        # ラベル管理
        labeled_rows = {start_row}
        labeled_cols = set()

        # 逆追跡用：prev_row[j] = jに来た行i、prev_col[i] = iに来た列j
        prev_row: dict[int, int] = {}  # col -> row（この列はどの行から来たか）
        prev_col: dict[int, int] = {}  # row -> col（この行はどの列から来たか）

        # BFSキュー（探索待ちの行）
        queue = [start_row]

        while queue:
            row = queue.pop(0)

            # この行から零要素で辿れる、まだラベルのない列を探す
            for col in range(self.n):
                if col in labeled_cols:
                    continue
                if abs(self._slack(row, col)) > 1e-9:
                    continue

                # 零要素を発見 → 列にラベル付与
                labeled_cols.add(col)
                prev_row[col] = row

                if self.match_col[col] == -1:
                    # 未マッチング列 → 拡張路発見！逆追跡してパスを構築
                    path = []
                    c = col
                    while c is not None:
                        r = prev_row[c]
                        path.append((r, c))
                        c = prev_col.get(r)
                    path.reverse()
                    self._log(f"  拡張路発見: {path}")
                    return path
                else:
                    # マッチング済み列 → その相手行にラベル付与してキューへ
                    next_row = self.match_col[col]
                    if next_row not in labeled_rows:
                        labeled_rows.add(next_row)
                        prev_col[next_row] = col
                        queue.append(next_row)

        # 拡張路なし → δ計算のためにラベル集合を返す
        self._labeled_rows = labeled_rows.copy()
        self._labeled_cols = labeled_cols.copy()
        return None

    def _update_potentials(self):
        """
        δ計算とポテンシャル更新

        δ = min{ c_ij - u_i - v_j | i∈ラベル付き行, j∉ラベル付き列 }

        更新：
          u_i += δ  （i∈ラベル付き行）
          v_j -= δ  （j∈ラベル付き列）

        効果：
          ラベル付き行 × ラベルなし列 → スラックがδ減少 → 最小のものが0に
          ラベル付き行 × ラベル付き列 → スラック変化なし（既存零要素を保持）
          ラベルなし行 × ラベルなし列 → スラック変化なし
          ラベルなし行 × ラベル付き列 → スラックがδ増加
        """
        delta = float("inf")
        for i in self._labeled_rows:
            for j in range(self.n):
                if j not in self._labeled_cols:
                    delta = min(delta, self._slack(i, j))

        self._log(f"  δ = {delta:.4f}")
        self._log(f"  ラベル付き行: {self._labeled_rows}, ラベル付き列: {self._labeled_cols}")

        for i in self._labeled_rows:
            self.u[i] += delta
        for j in self._labeled_cols:
            self.v[j] -= delta

        self._log(f"  更新後 u = {self.u}")
        self._log(f"  更新後 v = {self.v}")

    def solve(self) -> tuple[list[tuple[int, int]], float]:
        """
        ハンガリアン法を実行して最適割当を返す

        Returns
        -------
        assignment : list of (row, col)
            最適割当のペアリスト
        total_cost : float
            総コスト
        """
        self._initialize()

        # 各ドライバーを順番にマッチングしていく
        for start_row in range(self.n):
            self._log(f"\n=== ドライバー{start_row}のマッチング開始 ===")

            # 拡張路が見つかるまでδ更新を繰り返す
            while True:
                path = self._find_augmenting_path(start_row)

                if path is not None:
                    # 拡張路に沿ってマッチングを更新
                    for r, c in path:
                        self.match_row[r] = c
                        self.match_col[c] = r
                    self._log(f"  マッチング更新: {path}")
                    break
                else:
                    # δ計算・ポテンシャル更新
                    self._update_potentials()

        assignment = [(i, self.match_row[i]) for i in range(self.n)]
        total_cost = sum(self.cost[i, j] for i, j in assignment)

        self._log(f"\n=== 最適割当 ===")
        self._log(f"割当: {assignment}")
        self._log(f"総コスト: {total_cost:.4f}")

        return assignment, total_cost


# ============================================================
# 動作確認
# ============================================================

def test_small():
    """小さいケースで手計算と照合"""
    print("=== テスト1：3x3 手計算ケース ===")
    cost = np.array([
        [4, 2, 8],
        [4, 3, 7],
        [3, 1, 6],
    ], dtype=float)

    solver = HungarianSolver(cost, verbose=True)
    assignment, total_cost = solver.solve()
    print(f"割当: {assignment}")
    print(f"総コスト: {total_cost}")
    # 期待値: (0,1),(1,0),(2,2) or 同コストの別解 → 合計コスト: 2+4+6=12


def test_medium():
    """中規模ケース（scipyと比較用）"""
    print("\n=== テスト2：5x5 ランダムケース ===")
    rng = np.random.default_rng(42)
    cost = rng.integers(1, 20, size=(5, 5)).astype(float)
    print(f"コスト行列:\n{cost}")

    solver = HungarianSolver(cost, verbose=False)
    assignment, total_cost = solver.solve()
    print(f"割当: {assignment}")
    print(f"総コスト: {total_cost}")

    # scipy で検証
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    scipy_cost = cost[row_ind, col_ind].sum()
    print(f"scipy総コスト: {scipy_cost}")
    print(f"一致: {abs(total_cost - scipy_cost) < 1e-6}")


def test_distance_matrix():
    """地図上の距離行列ケース（Streamlit想定）"""
    print("\n=== テスト3：距離行列ケース ===")
    rng = np.random.default_rng(0)
    n = 6
    # ドライバーと配送先の座標をランダム生成
    driver_pos = rng.uniform(0, 100, size=(n, 2))
    delivery_pos = rng.uniform(0, 100, size=(n, 2))

    # ユークリッド距離行列
    cost = np.linalg.norm(
        driver_pos[:, np.newaxis, :] - delivery_pos[np.newaxis, :, :],
        axis=2
    )
    print(f"コスト行列（距離）:\n{cost.round(2)}")

    solver = HungarianSolver(cost, verbose=False)
    assignment, total_cost = solver.solve()
    print(f"割当: {assignment}")
    print(f"総移動コスト: {total_cost:.2f}")

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    scipy_cost = cost[row_ind, col_ind].sum()
    print(f"scipy総コスト: {scipy_cost:.2f}")
    print(f"一致: {abs(total_cost - scipy_cost) < 1e-6}")


if __name__ == "__main__":
    test_small()
    test_medium()
    test_distance_matrix()