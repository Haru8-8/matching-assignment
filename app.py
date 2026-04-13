# app.py
import streamlit as st

st.set_page_config(
    page_title="マッチング・割当問題",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 マッチング・割当問題")
st.markdown("""
### 配送ドライバーの最適割当

このアプリでは2種類の割当問題を可視化します。

| ページ | 問題 | 手法 |
|---|---|---|
| 基本割当 | N人→N件（1対1） | ハンガリアン法スクラッチ vs scipy |
| 拡張割当 | M人→N件（1対多可） | MIP（PuLP/HiGHS） |
""")