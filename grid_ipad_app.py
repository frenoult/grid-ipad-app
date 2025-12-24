import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

# =========================
# Data structure
# =========================
@dataclass
class Line:
    level: int
    buy_price: float
    sell_price: float
    qty: float = 0.0
    is_open: bool = False


# =========================
# Grid builder
# =========================
def build_grid(grid_type, pa1, n_lines, ra, rv, ca, cv):
    lines = []
    pa = pa1
    for i in range(1, n_lines + 1):
        if grid_type == "Arithmétique":
            buy = pa
            sell = pa + rv
            pa -= ra
        else:
            buy = pa
            sell = pa * cv
            pa *= ca
        lines.append(Line(i, buy, sell))
    return lines


# =========================
# Backtest engine
# =========================
def backtest(
    df,
    grid_type,
    sell_mode,
    pa1,
    n_lines,
    cash_init,
    line_value,
    ra,
    rv,
    ca,
    cv,
    reset
):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["mid"] = (df["high"] + df["low"]) / 2
    df["prev_mid"] = df["mid"].shift(1)

    lines = build_grid(grid_type, pa1, n_lines, ra, rv, ca, cv)

    cash = cash_init
    qty_total = 0.0
    realized = 0.0
    ops = []
    nav = []

    for _, row in df.iterrows():
        date = row["date"]
        high = row["high"]
        low = row["low"]
        prev_mid = row["prev_mid"] if not pd.isna(row["prev_mid"]) else row["mid"]

        sold_today = False

        # ===== SELL =====
        if sell_mode == "Ligne à ligne":
            for ln in sorted([l for l in lines if l.is_open and high >= l.sell_price],
                             key=lambda x: x.sell_price, reverse=True):
                proceeds = ln.qty * ln.sell_price
                cost = ln.qty * ln.buy_price
                gain = proceeds - cost

                cash += proceeds
                qty_total -= ln.qty
                realized += gain

                ops.append([date, "SELL", ln.level, ln.sell_price, ln.qty, gain])
                ln.is_open = False
                ln.qty = 0
                sold_today = True

        else:
            open_lines = [l for l in lines if l.is_open]
            if open_lines:
                qty = sum(l.qty for l in open_lines)
                avg_entry = sum(l.qty * l.buy_price for l in open_lines) / qty
                trigger = avg_entry + rv if grid_type == "Arithmétique" else avg_entry * cv

                if high >= trigger:
                    proceeds = qty * trigger
                    cost = sum(l.qty * l.buy_price for l in open_lines)
                    gain = proceeds - cost

                    cash += proceeds
                    qty_total = 0
                    realized += gain

                    ops.append([date, "SELL_BLOCK", 0, trigger, qty, gain])
                    for l in open_lines:
                        l.is_open = False
                        l.qty = 0
                    sold_today = True

        # ===== BUY =====
        if not sold_today:
            for ln in sorted([l for l in lines if not l.is_open and prev_mid > l.buy_price and low <= l.buy_price],
                             key=lambda x: x.buy_price, reverse=True):
                if cash < line_value:
                    break
                qty = line_value / ln.buy_price
                cash -= line_value
                qty_total += qty

                ln.qty = qty
                ln.is_open = True
                ops.append([date, "BUY", ln.level, ln.buy_price, qty, 0])

        nav.append([date, cash + qty_total * row["mid"], realized])

        if reset and all(not l.is_open for l in lines):
            lines = build_grid(grid_type, row["mid"], n_lines, ra, rv, ca, cv)

    ops_df = pd.DataFrame(ops, columns=["date", "type", "level", "price", "qty", "gain"])
    nav_df = pd.DataFrame(nav, columns=["date", "nav", "realized"])

    nav_df["drawdown"] = nav_df["nav"] / nav_df["nav"].cummax() - 1

    return ops_df, nav_df


# =========================
# Streamlit UI
# =========================
st.set_page_config(layout="wide")
st.title("Grid Investment – Backtest iPad")

uploaded = st.file_uploader("Importer CSV (date, high, low)", type="csv")

grid_type = st.selectbox("Type de grille", ["Géométrique", "Arithmétique"])
sell_mode = st.selectbox("Mode de vente", ["Ligne à ligne", "Vente en bloc"])

cash = st.number_input("Capital total", value=40000.0)
line_value = st.number_input("Montant par ligne", value=10000.0)
n_lines = st.number_input("Nombre de lignes", value=20)

pa1 = st.number_input("PA₁", value=16.75)

reset = st.checkbox("Réinitialisation dynamique", value=True)

if grid_type == "Arithmétique":
    ra = st.number_input("RA", value=0.9)
    rv = st.number_input("RV", value=1.2)
    ca = cv = 0.0
else:
    ca = st.number_input("Coeff achat", value=0.96)
    cv = st.number_input("Coeff vente", value=1.12)
    ra = rv = 0.0

if st.button("Lancer le backtest") and uploaded:
    df = pd.read_csv(uploaded)
    ops, nav = backtest(df, grid_type, sell_mode, pa1, n_lines,
                         cash, line_value, ra, rv, ca, cv, reset)

    st.subheader("Résultats")
    st.metric("Gains réalisés", f"{nav.realized.iloc[-1]:,.2f}")
    st.metric("NAV finale", f"{nav.nav.iloc[-1]:,.2f}")
    st.metric("Max drawdown", f"{nav.drawdown.min()*100:.2f}%")

    fig, ax = plt.subplots()
    ax.plot(nav.date, nav.nav)
    ax.set_title("Valeur liquidative")
    st.pyplot(fig)

    st.subheader("Journal des opérations")
    st.dataframe(ops)

    st.download_button("Télécharger operations.csv",
                       ops.to_csv(index=False).encode(),
                       "operations.csv")

    st.download_button("Télécharger nav.csv",
                       nav.to_csv(index=False).encode(),
                       "nav.csv")
