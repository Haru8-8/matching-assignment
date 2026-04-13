# pages/1_basic_assignment.py
"""
ページ1：基本割当
ハンガリアン法スクラッチ vs scipy.optimize.linear_sum_assignment
"""

import time
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import linear_sum_assignment

sys.path.append(str(Path(__file__).parent.parent))
from solvers.hungarian import HungarianSolver
from utils.cost_matrix import generate_positions, compute_distance_matrix

st.title("基本割当：ハンガリアン法 vs scipy")
st.markdown("N人のドライバーをN件の配送先に1対1で割り当て、総移動コストを最小化します。")

# ─── サイドバー ───────────────────────────────────────────
with st.sidebar:
    st.header("設定")
    n = st.slider("ドライバー・配送先数 N", min_value=3, max_value=20, value=6)
    seed = st.number_input("乱数シード", min_value=0, max_value=999, value=42)
    run = st.button("割当を実行", type="primary")

# ─── 計算 ─────────────────────────────────────────────────
if run:
    driver_pos = generate_positions(n, seed=seed)
    delivery_pos = generate_positions(n, seed=seed + 100)
    cost = compute_distance_matrix(driver_pos, delivery_pos)

    t0 = time.perf_counter()
    hungarian = HungarianSolver(cost)
    h_assignment, h_cost = hungarian.solve()
    h_time = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    row_ind, col_ind = linear_sum_assignment(cost)
    s_assignment = list(zip(row_ind, col_ind))
    s_cost = cost[row_ind, col_ind].sum()
    s_time = (time.perf_counter() - t0) * 1000

    st.session_state["basic"] = dict(
        n=n, cost=cost,
        driver_pos=driver_pos, delivery_pos=delivery_pos,
        h_assignment=h_assignment, h_cost=h_cost, h_time=h_time,
        s_assignment=s_assignment, s_cost=s_cost, s_time=s_time,
    )

# ─── 表示 ─────────────────────────────────────────────────
if "basic" not in st.session_state:
    st.info("サイドバーで設定を行い「割当を実行」ボタンを押してください。")
    st.stop()

r = st.session_state["basic"]
cost = r["cost"]
driver_pos = r["driver_pos"]
delivery_pos = r["delivery_pos"]
n = r["n"]

st.subheader("結果比較")
col1, col2 = st.columns(2)
with col1:
    st.metric("ハンガリアン法（スクラッチ）総コスト", f"{r['h_cost']:.2f}")
    st.metric("実行時間", f"{r['h_time']:.3f} ms")
with col2:
    st.metric("scipy 総コスト", f"{r['s_cost']:.2f}")
    st.metric("実行時間", f"{r['s_time']:.3f} ms")

match = abs(r["h_cost"] - r["s_cost"]) < 1e-4
if match:
    st.success("✅ 両手法の結果が一致しています")
else:
    st.error("❌ 結果が一致しません")

st.subheader("割当の可視化")
tab1, tab2 = st.tabs(["ハンガリアン法", "scipy"])

def make_figure(assignment, driver_pos, delivery_pos, title):
    fig = go.Figure()
    colors = [
        f"hsl({int(360 * i / len(assignment))}, 70%, 50%)"
        for i in range(len(assignment))
    ]
    for idx, (i, j) in enumerate(assignment):
        fig.add_trace(go.Scatter(
            x=[driver_pos[i, 0], delivery_pos[j, 0]],
            y=[driver_pos[i, 1], delivery_pos[j, 1]],
            mode="lines",
            line=dict(color=colors[idx], width=2),
            showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=delivery_pos[:, 0], y=delivery_pos[:, 1],
        mode="markers+text",
        marker=dict(symbol="square", size=12, color="lightgray", line=dict(width=1, color="gray")),
        text=[f"D{j}" for j in range(len(delivery_pos))],
        textposition="top center",
        name="配送先"
    ))
    for idx, (i, j) in enumerate(assignment):
        fig.add_trace(go.Scatter(
            x=[driver_pos[i, 0]], y=[driver_pos[i, 1]],
            mode="markers+text",
            marker=dict(symbol="circle", size=14, color=colors[idx]),
            text=[f"R{i}"],
            textposition="bottom center",
            name=f"R{i}→D{j}",
            showlegend=True
        ))
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-5, 105], showgrid=True),
        yaxis=dict(range=[-5, 105], showgrid=True),
        height=500,
        legend=dict(orientation="v", x=1.02)
    )
    return fig

with tab1:
    st.plotly_chart(
        make_figure(r["h_assignment"], driver_pos, delivery_pos, f"ハンガリアン法（総コスト: {r['h_cost']:.2f}）"),
        use_container_width=True
    )
with tab2:
    st.plotly_chart(
        make_figure(r["s_assignment"], driver_pos, delivery_pos, f"scipy（総コスト: {r['s_cost']:.2f}）"),
        use_container_width=True
    )

with st.expander("コスト行列を表示"):
    df = pd.DataFrame(
        cost.round(2),
        index=[f"R{i}" for i in range(n)],
        columns=[f"D{j}" for j in range(n)]
    )
    st.dataframe(df)