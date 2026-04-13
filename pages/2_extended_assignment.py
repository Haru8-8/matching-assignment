# pages/2_extended_assignment.py
"""
ページ2：拡張割当
MIP（PuLP/HiGHS）による1対多割当
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from solvers.mip_assignment import MIPAssignmentSolver
from utils.cost_matrix import generate_positions, compute_distance_matrix

st.title("拡張割当：MIP（1ドライバーが複数配送先を担当）")
st.markdown("M人のドライバーがN件の配送先を分担します。capacity・バランス制約を調整できます。")

# ─── サイドバー ───────────────────────────────────────────
with st.sidebar:
    st.header("設定")
    n_drivers = st.slider("ドライバー数 M", min_value=2, max_value=8, value=3)
    n_deliveries = st.slider("配送先数 N", min_value=n_drivers, max_value=20, value=8)
    capacity = st.slider(
        "1ドライバーの最大担当数",
        min_value=-(-n_deliveries // n_drivers),
        max_value=n_deliveries,
        value=min(-(-n_deliveries // n_drivers) + 1, n_deliveries)
    )
    balance = st.checkbox("担当数バランス制約", value=True)
    seed = st.number_input("乱数シード", min_value=0, max_value=999, value=0)
    run = st.button("割当を実行", type="primary")

# ─── 計算 ─────────────────────────────────────────────────
if run:
    driver_pos = generate_positions(n_drivers, seed=seed)
    delivery_pos = generate_positions(n_deliveries, seed=seed + 100)
    cost = compute_distance_matrix(driver_pos, delivery_pos)

    solver = MIPAssignmentSolver(cost, capacity=capacity, balance=balance)
    assignment, total_cost, status = solver.solve()
    load = [sum(1 for a in assignment if a[0] == i) for i in range(n_drivers)]

    st.session_state["extended"] = dict(
        n_drivers=n_drivers, n_deliveries=n_deliveries,
        cost=cost, driver_pos=driver_pos, delivery_pos=delivery_pos,
        assignment=assignment, total_cost=total_cost, status=status, load=load,
    )

# ─── 表示 ─────────────────────────────────────────────────
if "extended" not in st.session_state:
    st.info("サイドバーで設定を行い「割当を実行」ボタンを押してください。")
    st.stop()

r = st.session_state["extended"]

if r["status"] != "Optimal":
    st.error(f"最適解が見つかりませんでした（status: {r['status']}）")
    st.stop()

n_drivers = r["n_drivers"]
n_deliveries = r["n_deliveries"]
cost = r["cost"]
driver_pos = r["driver_pos"]
delivery_pos = r["delivery_pos"]
assignment = r["assignment"]
total_cost = r["total_cost"]
load = r["load"]

st.subheader("結果")
cols = st.columns(3)
cols[0].metric("総コスト", f"{total_cost:.2f}")
cols[1].metric("最大担当数", max(load))
cols[2].metric("最小担当数", min(load))

load_df = pd.DataFrame({
    "ドライバー": [f"R{i}" for i in range(n_drivers)],
    "担当件数": load,
    "担当配送先": [
        ", ".join(f"D{j}" for _, j in assignment if _ == i)
        for i in range(n_drivers)
    ]
})
st.dataframe(load_df, use_container_width=True)

st.subheader("割当の可視化")
colors = [
    f"hsl({int(360 * i / n_drivers)}, 70%, 50%)"
    for i in range(n_drivers)
]

fig = go.Figure()
for i, j in assignment:
    fig.add_trace(go.Scatter(
        x=[driver_pos[i, 0], delivery_pos[j, 0]],
        y=[driver_pos[i, 1], delivery_pos[j, 1]],
        mode="lines",
        line=dict(color=colors[i], width=2),
        showlegend=False
    ))
fig.add_trace(go.Scatter(
    x=delivery_pos[:, 0], y=delivery_pos[:, 1],
    mode="markers+text",
    marker=dict(symbol="square", size=12, color="lightgray", line=dict(width=1, color="gray")),
    text=[f"D{j}" for j in range(n_deliveries)],
    textposition="top center",
    name="配送先",
    showlegend=True
))
for i in range(n_drivers):
    fig.add_trace(go.Scatter(
        x=[driver_pos[i, 0]], y=[driver_pos[i, 1]],
        mode="markers+text",
        marker=dict(symbol="circle", size=14, color=colors[i]),
        text=[f"R{i}"],
        textposition="bottom center",
        name=f"R{i}（{load[i]}件）",
        showlegend=True
    ))
fig.update_layout(
    title=f"拡張割当結果（総コスト: {total_cost:.2f}）",
    xaxis=dict(range=[-5, 105], showgrid=True),
    yaxis=dict(range=[-5, 105], showgrid=True),
    height=550,
    legend=dict(orientation="v", x=1.02)
)
st.plotly_chart(fig, use_container_width=True)

with st.expander("コスト行列を表示"):
    df = pd.DataFrame(
        cost.round(2),
        index=[f"R{i}" for i in range(n_drivers)],
        columns=[f"D{j}" for j in range(n_deliveries)]
    )
    st.dataframe(df)