"""
MIPによる拡張割当問題
1人のドライバーが複数の配送先を担当可能なケース

基本割当（ハンガリアン法）との違い：
  - ドライバー数Mと配送先数Nが異なってもよい
  - 1人が最大capacity件まで担当可能
  - 担当件数の偏りを抑えるバランス制約をオプションで追加可能
"""

import numpy as np
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpBinary,
    lpSum, value, LpStatus, PULP_CBC_CMD
)


class MIPAssignmentSolver:
    """
    PuLP/HiGHSによる拡張割当ソルバー

    Parameters
    ----------
    cost : np.ndarray
        コスト行列 (M x N)  M=ドライバー数, N=配送先数
    capacity : int
        1ドライバーが担当できる最大配送先数
    balance : bool
        担当件数の偏りを抑えるバランス制約を追加するか
    """

    def __init__(self, cost: np.ndarray, capacity: int = None, balance: bool = False):
        assert cost.ndim == 2, "コスト行列は2次元配列である必要があります"
        self.cost = cost.astype(float)
        self.m = cost.shape[0]  # ドライバー数
        self.n = cost.shape[1]  # 配送先数
        self.capacity = capacity if capacity is not None else self.n
        self.balance = balance

        assert self.m <= self.n, "ドライバー数は配送先数以下である必要があります"
        assert self.capacity >= -(-self.n // self.m), \
            f"capacityが小さすぎます（最低{-(-self.n // self.m)}必要）"

    def solve(self) -> tuple[list[tuple[int, int]], float, str]:
        """
        MIPを解いて最適割当を返す

        Returns
        -------
        assignment : list of (driver, delivery)
        total_cost : float
        status : str
        """
        prob = LpProblem("extended_assignment", LpMinimize)

        # 決定変数：x[i][j] = 1 ならドライバーiが配送先jを担当
        x = [[LpVariable(f"x_{i}_{j}", cat=LpBinary)
              for j in range(self.n)]
             for i in range(self.m)]

        # 目的関数
        prob += lpSum(self.cost[i, j] * x[i][j]
                      for i in range(self.m)
                      for j in range(self.n))

        # 制約1：各配送先は必ず1人が担当
        for j in range(self.n):
            prob += lpSum(x[i][j] for i in range(self.m)) == 1

        # 制約2：各ドライバーの担当件数はcapacity以下
        for i in range(self.m):
            prob += lpSum(x[i][j] for j in range(self.n)) <= self.capacity

        # 制約3（オプション）：担当件数のバランス
        # 最大担当数 - 最小担当数 <= 1
        if self.balance:
            min_load = self.n // self.m
            max_load = -(-self.n // self.m)  # ceil
            for i in range(self.m):
                prob += lpSum(x[i][j] for j in range(self.n)) >= min_load
                prob += lpSum(x[i][j] for j in range(self.n)) <= max_load

        prob.solve(PULP_CBC_CMD(msg=0))

        status = LpStatus[prob.status]
        if status != "Optimal":
            return [], float("inf"), status

        assignment = [
            (i, j)
            for i in range(self.m)
            for j in range(self.n)
            if value(x[i][j]) > 0.5
        ]
        total_cost = value(prob.objective)

        return assignment, total_cost, status


# ============================================================
# 動作確認
# ============================================================

def test_square():
    """正方ケース（基本割当との比較）"""
    print("=== テスト1：4x4（基本割当と同じ条件）===")
    rng = np.random.default_rng(42)
    cost = rng.integers(1, 20, size=(4, 4)).astype(float)
    print(f"コスト行列:\n{cost}")

    solver = MIPAssignmentSolver(cost, capacity=1)
    assignment, total_cost, status = solver.solve()
    print(f"status: {status}")
    print(f"割当: {assignment}")
    print(f"総コスト: {total_cost}")

    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    print(f"scipy総コスト: {cost[row_ind, col_ind].sum()}")
    print(f"一致: {abs(total_cost - cost[row_ind, col_ind].sum()) < 1e-4}")


def test_extended():
    """拡張ケース：ドライバー3人、配送先6件"""
    print("\n=== テスト2：3ドライバー × 6配送先 ===")
    rng = np.random.default_rng(0)
    cost = rng.uniform(1, 100, size=(3, 6))
    print(f"コスト行列:\n{cost.round(2)}")

    print("\n-- capacity=3（上限なし相当）--")
    solver = MIPAssignmentSolver(cost, capacity=3)
    assignment, total_cost, status = solver.solve()
    print(f"割当: {assignment}")
    load = [sum(1 for a in assignment if a[0] == i) for i in range(3)]
    print(f"担当件数: {load}")
    print(f"総コスト: {total_cost:.2f}")

    print("\n-- capacity=3 + balance制約 --")
    solver_b = MIPAssignmentSolver(cost, capacity=3, balance=True)
    assignment_b, total_cost_b, status_b = solver_b.solve()
    print(f"割当: {assignment_b}")
    load_b = [sum(1 for a in assignment_b if a[0] == i) for i in range(3)]
    print(f"担当件数: {load_b}")
    print(f"総コスト: {total_cost_b:.2f}")


def test_distance_matrix():
    """距離行列ケース（Streamlit想定）"""
    print("\n=== テスト3：距離行列 4ドライバー × 8配送先 ===")
    rng = np.random.default_rng(1)
    driver_pos = rng.uniform(0, 100, size=(4, 2))
    delivery_pos = rng.uniform(0, 100, size=(8, 2))

    cost = np.linalg.norm(
        driver_pos[:, np.newaxis, :] - delivery_pos[np.newaxis, :, :],
        axis=2
    )
    print(f"コスト行列（距離）:\n{cost.round(2)}")

    solver = MIPAssignmentSolver(cost, capacity=3, balance=True)
    assignment, total_cost, status = solver.solve()
    print(f"status: {status}")
    print(f"割当: {assignment}")
    load = [sum(1 for a in assignment if a[0] == i) for i in range(4)]
    print(f"担当件数: {load}")
    print(f"総コスト: {total_cost:.2f}")


if __name__ == "__main__":
    test_square()
    test_extended()
    test_distance_matrix()