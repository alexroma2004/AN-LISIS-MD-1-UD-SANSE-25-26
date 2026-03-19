
import io
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="MD-1 Staff Elite Definitiva", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path("app_data")
APP_DIR.mkdir(exist_ok=True)
DB_PATH = APP_DIR / "md1_staff_elite_definitiva_v2.db"

OBJECTIVE_METRICS = ["CMJ", "RSI_mod", "VMP"]
ALL_METRICS = ["CMJ", "RSI_mod", "VMP", "sRPE"]

LABELS = {"CMJ": "CMJ", "RSI_mod": "RSI mod", "VMP": "VMP sentadilla", "sRPE": "sRPE"}
RISK_ORDER = ["Estado óptimo","Buen estado","Fatiga leve","Fatiga leve-moderada","Fatiga moderada","Fatiga moderada-alta","Fatiga crítica"]
SEVERITY_COLORS = {
    "Buen estado": "#16A34A",
    "Fatiga leve": "#EAB308",
    "Fatiga moderada": "#F97316",
    "Fatiga crítica": "#C62828",
    "Sin referencia": "#94A3B8",
}
RISK_COLORS = {
    "Estado óptimo": "#15803D",
    "Buen estado": "#2E8B57",
    "Fatiga leve": "#E3A008",
    "Fatiga leve-moderada": "#F59E0B",
    "Fatiga moderada": "#F97316",
    "Fatiga moderada-alta": "#EA580C",
    "Fatiga crítica": "#B91C1C",
}

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem;}
.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1F4E79 100%);
    color: white; border-radius: 20px; padding: 22px 24px; margin-bottom: 14px;
    box-shadow: 0 10px 28px rgba(2,6,23,0.22);
}
.card {
    background: white; border-radius: 18px; padding: 16px 18px;
    border: 1px solid rgba(15,23,42,0.08);
    box-shadow: 0 6px 18px rgba(15,23,42,0.06); margin-bottom: 10px;
}
.kpi {
    background: white; border-radius: 16px; padding: 14px 16px;
    border: 1px solid rgba(15,23,42,0.08);
    box-shadow: 0 6px 18px rgba(15,23,42,0.06);
    min-height: 94px;
}
.kpi-label {font-size: 0.82rem; color: #475467;}
.kpi-value {font-size: 1.65rem; font-weight: 800; color: #101828; line-height: 1.1;}
.kpi-sub {font-size: 0.80rem; color: #667085;}
.section-title {font-size: 1.15rem; font-weight: 800; color: #101828; margin: 0.4rem 0 0.7rem 0;}
.pill {
    display:inline-block; padding:4px 10px; border-radius:999px; color:white;
    font-weight:700; font-size:0.78rem; margin-right:6px; margin-top:4px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DB
# =========================================================
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS monitoring (
            Fecha TEXT NOT NULL,
            Jugador TEXT NOT NULL,
            Posicion TEXT,
            Minutos REAL,
            CMJ REAL,
            RSI_mod REAL,
            VMP REAL,
            sRPE REAL,
            Observaciones TEXT,
            updated_at TEXT,
            PRIMARY KEY (Fecha, Jugador)
        )
    """)
    conn.commit()
    conn.close()

def load_monitoring():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM monitoring", conn)
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=["Fecha","Jugador","Posicion","Minutos","CMJ","RSI_mod","VMP","sRPE","Observaciones"])
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    for c in ["Minutos", *ALL_METRICS]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Jugador"] = df["Jugador"].astype(str).str.strip()
    return df.sort_values(["Jugador","Fecha"]).reset_index(drop=True)

def upsert_monitoring(df):
    if df.empty:
        return
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(pd.to_datetime(r["Fecha"]).date()),
            str(r["Jugador"]),
            None if pd.isna(r.get("Posicion")) else str(r.get("Posicion")),
            None if pd.isna(r.get("Minutos")) else float(r.get("Minutos")),
            None if pd.isna(r.get("CMJ")) else float(r.get("CMJ")),
            None if pd.isna(r.get("RSI_mod")) else float(r.get("RSI_mod")),
            None if pd.isna(r.get("VMP")) else float(r.get("VMP")),
            None if pd.isna(r.get("sRPE")) else float(r.get("sRPE")),
            None if pd.isna(r.get("Observaciones")) else str(r.get("Observaciones")),
            now,
        ))
    conn = get_conn()
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO monitoring
        (Fecha, Jugador, Posicion, Minutos, CMJ, RSI_mod, VMP, sRPE, Observaciones, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(Fecha, Jugador) DO UPDATE SET
            Posicion=excluded.Posicion,
            Minutos=excluded.Minutos,
            CMJ=excluded.CMJ,
            RSI_mod=excluded.RSI_mod,
            VMP=excluded.VMP,
            sRPE=excluded.sRPE,
            Observaciones=excluded.Observaciones,
            updated_at=excluded.updated_at
    """, rows)
    conn.commit()
    conn.close()

# =========================================================
# PARSER
# =========================================================
def safe_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "").replace(",", ".")
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def std_name(x):
    if pd.isna(x):
        return np.nan
    return " ".join(str(x).strip().split()).title()

def try_parse_date(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    try:
        d = pd.to_datetime(s, dayfirst=True, errors="raise")
        if 2000 <= d.year <= 2100:
            return d
    except Exception:
        pass
    m = re.search(r"(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?", s)
    if m:
        day = int(m.group(1))
        month = int(m.group(2))
        year = m.group(3)
        if year is None:
            year = pd.Timestamp.now().year
        else:
            year = int(year)
            if year < 100:
                year += 2000
        try:
            return pd.Timestamp(year=year, month=month, day=day)
        except Exception:
            return None
    return None

def first_valid_numeric(values):
    vals = [safe_num(v) for v in values]
    vals = [v for v in vals if not pd.isna(v)]
    return vals[0] if vals else np.nan

def detect_format(df_raw):
    cols = [str(c).lower().strip() for c in df_raw.columns]
    joined_cols = " ".join(cols)
    if any(k in joined_cols for k in ["jugador","player"]) and any(k in joined_cols for k in ["cmj","rsi","vmp"]):
        return "tidy"
    flat = " ".join(df_raw.astype(str).fillna("").values.flatten().tolist()).lower()
    if "cmj" in flat or "rsi" in flat or "vmp" in flat:
        return "block"
    if df_raw.shape[1] <= 6:
        return "block"
    return "unknown"

def read_uploaded(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, header=None)
    raise ValueError("Formato no soportado. Usa .csv, .xlsx o .xls")

def parse_tidy(df_raw):
    df = df_raw.copy()
    tmp_cols = [str(c).strip().lower() for c in df.columns]
    first_row = [str(v).strip().lower() if pd.notna(v) else "" for v in df.iloc[0].tolist()] if len(df) else []
    if not any(k in " ".join(tmp_cols) for k in ["jugador","player","cmj","rsi","vmp"]) and any(k in " ".join(first_row) for k in ["jugador","player","cmj","rsi","vmp","fecha","date"]):
        df.columns = [str(v).strip() for v in df.iloc[0].tolist()]
        df = df.iloc[1:].reset_index(drop=True)
    else:
        df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        low = c.lower().strip()
        if low in ["fecha","date"]: rename[c] = "Fecha"
        elif low in ["jugador","player","nombre"]: rename[c] = "Jugador"
        elif "pos" in low: rename[c] = "Posicion"
        elif "min" in low: rename[c] = "Minutos"
        elif "cmj" in low: rename[c] = "CMJ"
        elif "rsi" in low: rename[c] = "RSI_mod"
        elif "vmp" in low: rename[c] = "VMP"
        elif "srpe" in low or "s-rpe" in low or low == "rpe": rename[c] = "sRPE"
        elif "obs" in low: rename[c] = "Observaciones"
    df = df.rename(columns=rename)

    needed = ["Fecha","Jugador","CMJ","RSI_mod","VMP"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    for optional in ["Posicion","Minutos","Observaciones","sRPE"]:
        if optional not in df.columns:
            df[optional] = np.nan

    df = df[["Fecha","Jugador","Posicion","Minutos","CMJ","RSI_mod","VMP","sRPE","Observaciones"]].copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")
    df["Jugador"] = df["Jugador"].apply(std_name)
    for c in ["Minutos", *ALL_METRICS]:
        df[c] = df[c].apply(safe_num)
    return df.dropna(subset=["Fecha","Jugador"]).drop_duplicates(subset=["Fecha","Jugador"], keep="last")

def parse_block(df_raw):
    df = df_raw.copy()
    df.columns = range(df.shape[1])
    df = df.replace(r"^\s*$", np.nan, regex=True)

    first_row = [str(v).strip().lower() if pd.notna(v) else "" for v in df.iloc[0].tolist()] if len(df) else []
    if first_row and "nombre" in first_row[0] and any("variable" in x for x in first_row):
        date_cols = []
        for c in range(2, df.shape[1]):
            d = try_parse_date(df.iat[0, c])
            if d is not None:
                date_cols.append((c, d))
        records = []
        i = 1
        while i + 3 < len(df):
            player = df.iat[i, 0]
            var1 = str(df.iat[i, 1]).strip().lower() if pd.notna(df.iat[i, 1]) else ""
            var2 = str(df.iat[i+1, 1]).strip().lower() if pd.notna(df.iat[i+1, 1]) else ""
            var3 = str(df.iat[i+2, 1]).strip().lower() if pd.notna(df.iat[i+2, 1]) else ""
            var4 = str(df.iat[i+3, 1]).strip().lower() if pd.notna(df.iat[i+3, 1]) else ""
            looks_like_group = pd.notna(player) and "cmj" in var1 and "rsi" in var2 and "vmp" in var3 and ("rpe" in var4 or "srpe" in var4 or "s-rpe" in var4)
            if looks_like_group:
                player_name = std_name(player)
                for c, d in date_cols:
                    cmj = safe_num(df.iat[i, c]) if c < df.shape[1] else np.nan
                    rsi = safe_num(df.iat[i+1, c]) if c < df.shape[1] else np.nan
                    vmp = safe_num(df.iat[i+2, c]) if c < df.shape[1] else np.nan
                    srpe = safe_num(df.iat[i+3, c]) if c < df.shape[1] else np.nan
                    if sum(pd.notna(x) for x in [cmj, rsi, vmp, srpe]) >= 2:
                        records.append({"Fecha": d, "Jugador": player_name, "Posicion": np.nan, "Minutos": np.nan, "CMJ": cmj, "RSI_mod": rsi, "VMP": vmp, "sRPE": srpe, "Observaciones": np.nan})
                i += 4
                continue
            i += 1
        out = pd.DataFrame(records)
        if not out.empty:
            return out.drop_duplicates(subset=["Fecha","Jugador"], keep="last")

    records = []
    current_date = None
    current_player = None
    i = 0
    while i < len(df):
        row = df.iloc[i].tolist()
        parsed_dates = [try_parse_date(v) for v in row if pd.notna(v)]
        parsed_dates = [d for d in parsed_dates if d is not None]
        if parsed_dates:
            current_date = parsed_dates[0]
        tokens = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() != ""]
        if tokens:
            first = tokens[0]
            if first.lower() not in ["cmj","rsi mod","rsi_mod","vmp","srpe","s-rpe","rpe"]:
                if try_parse_date(first) is None and pd.isna(safe_num(first)):
                    current_player = std_name(first)

        if current_player is not None and current_date is not None and i + 3 < len(df):
            labels = []; nums = []
            for j in range(0,4):
                row_j = df.iloc[i+j].tolist()
                non_empty = [str(v).strip().lower() for v in row_j if pd.notna(v) and str(v).strip() != ""]
                label_j = ""
                for tok in non_empty:
                    if any(k in tok for k in ["cmj","rsi","vmp","rpe","srpe","s-rpe"]):
                        label_j = tok
                        break
                labels.append(label_j)
                nums.append(first_valid_numeric(row_j))
            if "cmj" in labels[0] and "rsi" in labels[1] and "vmp" in labels[2] and any(k in labels[3] for k in ["rpe","srpe","s-rpe"]) and sum(pd.notna(v) for v in nums) >= 3:
                records.append({"Fecha": current_date, "Jugador": current_player, "Posicion": np.nan, "Minutos": np.nan, "CMJ": nums[0], "RSI_mod": nums[1], "VMP": nums[2], "sRPE": nums[3], "Observaciones": np.nan})
                i += 4
                continue

        if current_player is not None and current_date is not None and i + 4 < len(df):
            labels = []; nums = []
            for j in range(1,5):
                row_j = df.iloc[i+j].tolist()
                non_empty = [str(v).strip().lower() for v in row_j if pd.notna(v) and str(v).strip() != ""]
                label_j = ""
                for tok in non_empty:
                    if any(k in tok for k in ["cmj","rsi","vmp","rpe","srpe","s-rpe"]):
                        label_j = tok
                        break
                labels.append(label_j)
                nums.append(first_valid_numeric(row_j))
            if "cmj" in labels[0] and "rsi" in labels[1] and "vmp" in labels[2] and any(k in labels[3] for k in ["rpe","srpe","s-rpe"]) and sum(pd.notna(v) for v in nums) >= 3:
                records.append({"Fecha": current_date, "Jugador": current_player, "Posicion": np.nan, "Minutos": np.nan, "CMJ": nums[0], "RSI_mod": nums[1], "VMP": nums[2], "sRPE": nums[3], "Observaciones": np.nan})
                i += 5
                continue
        i += 1

    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("No se pudieron interpretar los bloques del archivo.")
    return out.drop_duplicates(subset=["Fecha","Jugador"], keep="last")

def parse_uploaded(uploaded_file):
    df_raw = read_uploaded(uploaded_file)
    fmt = detect_format(df_raw)
    if fmt == "tidy":
        return parse_tidy(df_raw)
    if fmt == "block":
        return parse_block(df_raw)
    raise ValueError("No se pudo detectar el formato del archivo.")

# =========================================================
# METRICS / DIAGNOSTIC
# =========================================================
def severity_from_pct(pct_change):
    if pd.isna(pct_change):
        return "Sin referencia", np.nan
    if pct_change >= -2.5:
        return "Buen estado", 0.0
    if pct_change > -5.0:
        return "Fatiga leve", 1.0
    if pct_change > -10.0:
        return "Fatiga moderada", 2.0
    return "Fatiga crítica", 3.0

def classify_risk_from_counts(n_leve, n_mod, n_crit):
    if n_crit >= 2:
        return "Fatiga crítica"
    if n_mod >= 2 and n_crit >= 1:
        return "Fatiga moderada-alta"
    if n_mod >= 2 or n_crit >= 1:
        return "Fatiga moderada"
    if n_leve >= 2 and n_mod >= 1:
        return "Fatiga leve-moderada"
    if n_leve >= 2 or n_mod >= 1:
        return "Fatiga leve"
    if n_leve == 1:
        return "Buen estado"
    return "Estado óptimo"

def slope_last_n(series, n=3):
    s = pd.Series(series).dropna()
    if len(s) < 2:
        return np.nan
    s = s.tail(n).reset_index(drop=True)
    x = np.arange(len(s))
    try:
        return np.polyfit(x, s, 1)[0]
    except Exception:
        return np.nan

def trend_label_from_slope(slope):
    if pd.isna(slope):
        return "Sin tendencia"
    if slope > 0.15:
        return "Empeorando"
    if slope < -0.15:
        return "Mejorando"
    return "Estable"

def zscore_prior_or_full(group_series):
    full_mean = group_series.mean()
    full_std = group_series.std(ddof=0)
    prior_mean = group_series.shift(1).expanding().mean()
    prior_std = group_series.shift(1).expanding().std(ddof=0)
    mean = prior_mean.fillna(full_mean)
    std = prior_std.fillna(full_std).replace(0, np.nan)
    return mean, std

def historical_percentile(player_df, value, metric):
    series = player_df[metric].dropna()
    if len(series) == 0 or pd.isna(value):
        return np.nan
    return round((series <= value).mean() * 100, 1)

def infer_fatigue_profile(row):
    vals = {m: row.get(f"{m}_pct_vs_baseline", np.nan) for m in OBJECTIVE_METRICS}
    clean = {k:v for k,v in vals.items() if pd.notna(v)}
    worst_metric, worst_value = (None, None)
    if clean:
        worst_metric = min(clean, key=clean.get)
        worst_value = clean[worst_metric]
    if worst_metric == "CMJ":
        main = "perfil de fatiga predominantemente explosivo"
    elif worst_metric == "RSI_mod":
        main = "perfil de fatiga predominantemente reactivo"
    elif worst_metric == "VMP":
        main = "perfil de fatiga predominantemente de fuerza/velocidad"
    else:
        main = "perfil sin una afectación dominante clara"
    moderate_or_worse = sum(int(row.get(f"{m}_severity_points", np.nan) >= 2) if pd.notna(row.get(f"{m}_severity_points", np.nan)) else 0 for m in OBJECTIVE_METRICS)
    if moderate_or_worse >= 2:
        pattern = "afectación global"
    elif moderate_or_worse == 1:
        pattern = "afectación específica"
    else:
        pattern = "alteración ligera"
    return main, pattern, worst_metric, worst_value

def flags_for_player(player_df, row):
    flags = []
    if row["objective_loss_score"] == player_df["objective_loss_score"].max():
        flags.append("Peor objective loss score del periodo")
    if row.get("trend_label") == "Empeorando":
        flags.append("Tendencia reciente negativa")
    if pd.notna(row.get("objective_z_score")) and row.get("objective_z_score") <= -1.5:
        flags.append("Valor anómalo para su patrón habitual")
    if row.get("risk_label") in ["Fatiga moderada","Fatiga moderada-alta","Fatiga crítica"]:
        flags.append("Requiere decisión individual")
    return flags

def compute_metrics(df):
    if df.empty:
        return df.copy()
    df = df.copy().sort_values(["Jugador","Fecha"]).reset_index(drop=True)

    for metric in ALL_METRICS:
        full_mean = df.groupby("Jugador")[metric].transform("mean")
        expanding_sum = df.groupby("Jugador")[metric].transform(lambda s: s.shift(1).expanding().sum())
        expanding_count = df.groupby("Jugador")[metric].transform(lambda s: s.shift(1).expanding().count())
        prior_mean = expanding_sum / expanding_count
        baseline = prior_mean.where(expanding_count > 0, full_mean)

        df[f"{metric}_baseline"] = baseline
        df[f"{metric}_pct_vs_baseline"] = np.where(
            df[f"{metric}_baseline"].notna() & (df[f"{metric}_baseline"] != 0),
            (df[metric] - df[f"{metric}_baseline"]) / df[f"{metric}_baseline"] * 100,
            np.nan,
        )

        means, stds = [], []
        for _, g in df.groupby("Jugador")[metric]:
            m, s = zscore_prior_or_full(g)
            means.extend(m.tolist()); stds.extend(s.tolist())
        df[f"{metric}_z_mean_ref"] = means
        df[f"{metric}_z_std_ref"] = stds
        df[f"{metric}_z"] = np.where(
            pd.notna(df[f"{metric}_z_std_ref"]) & (df[f"{metric}_z_std_ref"] != 0),
            (df[metric] - df[f"{metric}_z_mean_ref"]) / df[f"{metric}_z_std_ref"],
            np.nan,
        )

        df[f"{metric}_ma3"] = df.groupby("Jugador")[metric].transform(lambda s: s.rolling(window=3, min_periods=1).mean())

    for metric in OBJECTIVE_METRICS:
        sev = df[f"{metric}_pct_vs_baseline"].apply(severity_from_pct)
        df[f"{metric}_severity"] = sev.apply(lambda x: x[0])
        df[f"{metric}_severity_points"] = sev.apply(lambda x: x[1])

    df["n_leve"] = sum((df[f"{m}_severity"] == "Fatiga leve").astype(int) for m in OBJECTIVE_METRICS)
    df["n_mod"] = sum((df[f"{m}_severity"] == "Fatiga moderada").astype(int) for m in OBJECTIVE_METRICS)
    df["n_crit"] = sum((df[f"{m}_severity"] == "Fatiga crítica").astype(int) for m in OBJECTIVE_METRICS)
    df["objective_loss_score"] = df[[f"{m}_severity_points" for m in OBJECTIVE_METRICS]].mean(axis=1, skipna=True)
    df["objective_loss_mean_pct"] = df[[f"{m}_pct_vs_baseline" for m in OBJECTIVE_METRICS]].mean(axis=1, skipna=True)
    df["risk_label"] = df.apply(lambda r: classify_risk_from_counts(int(r["n_leve"]), int(r["n_mod"]), int(r["n_crit"])), axis=1)
    df["objective_z_score"] = df[[f"{m}_z" for m in OBJECTIVE_METRICS]].mean(axis=1, skipna=True)
    df["readiness_score"] = np.clip(100 - (df["objective_loss_score"] / 3.0) * 100, 0, 100)
    df["objective_loss_score_ma3"] = df.groupby("Jugador")["objective_loss_score"].transform(lambda s: s.rolling(window=3, min_periods=1).mean())

    trend_slopes = []
    for _, g in df.groupby("Jugador"):
        vals = g["objective_loss_score"].tolist()
        local = []
        for i in range(len(vals)):
            local.append(slope_last_n(vals[: i + 1], n=3))
        trend_slopes.extend(local)
    df["objective_loss_slope_3"] = trend_slopes
    df["trend_label"] = df["objective_loss_slope_3"].apply(trend_label_from_slope)

    team = df.groupby("Fecha")[OBJECTIVE_METRICS].mean().reset_index().rename(columns={m: f"{m}_team_mean" for m in OBJECTIVE_METRICS})
    df = df.merge(team, on="Fecha", how="left")
    for m in OBJECTIVE_METRICS:
        df[f"{m}_vs_team_pct"] = np.where(
            df[f"{m}_team_mean"].notna() & (df[f"{m}_team_mean"] != 0),
            (df[m] - df[f"{m}_team_mean"]) / df[f"{m}_team_mean"] * 100,
            np.nan,
        )

    for metric in OBJECTIVE_METRICS + ["objective_loss_score","readiness_score"]:
        asc = metric != "readiness_score"
        df[f"{metric}_team_rank"] = df.groupby("Fecha")[metric].rank(method="min", ascending=asc)

    perc = {m: [] for m in OBJECTIVE_METRICS}
    for _, g in df.groupby("Jugador"):
        g = g.sort_values("Fecha")
        for _, r in g.iterrows():
            hist = g[g["Fecha"] <= r["Fecha"]]
            for m in OBJECTIVE_METRICS:
                perc[m].append(historical_percentile(hist, r[m], m))
    for m in OBJECTIVE_METRICS:
        df[f"{m}_historical_percentile"] = perc[m]
    return df

# =========================================================
# UI HELPERS
# =========================================================
def kpi(label, value, sub=""):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)

def render_pills(row):
    html = ""
    for m in OBJECTIVE_METRICS:
        status = row.get(f"{m}_severity", "Sin referencia")
        color = SEVERITY_COLORS.get(status, "#94A3B8")
        html += f'<span class="pill" style="background:{color};">{LABELS[m]} · {status}</span>'
    return html

def recommendation_from_row(row):
    risk = row["risk_label"]; trend = row.get("trend_label", "Sin tendencia")
    if risk == "Fatiga crítica":
        return f"Reducir claramente la exigencia neuromuscular en MD-1. Tendencia: {trend.lower()}."
    if risk in ["Fatiga moderada","Fatiga moderada-alta"]:
        return f"Conviene controlar volumen e intensidad y vigilar la respuesta en el calentamiento. Tendencia: {trend.lower()}."
    if risk in ["Fatiga leve","Fatiga leve-moderada"]:
        return f"Hay una pérdida objetiva leve/moderada; prioriza una MD-1 conservadora y sin estímulos residuales. Tendencia: {trend.lower()}."
    return f"Estado compatible con normalidad funcional para MD-1. Tendencia: {trend.lower()}."

def player_comment(row):
    issues = []
    for metric in OBJECTIVE_METRICS:
        sev = row.get(f"{metric}_severity"); pct = row.get(f"{metric}_pct_vs_baseline")
        if sev not in [None, "Buen estado", "Sin referencia"] and pd.notna(pct):
            issues.append(f"{LABELS[metric]}: {sev.lower()} ({pct:.1f}%)")
    base = "Pérdida objetiva detectada en " + "; ".join(issues) + "." if issues else "No se observan pérdidas objetivas relevantes respecto a la línea base."
    return base + " " + recommendation_from_row(row)

def team_interpretation(df_last):
    if df_last.empty:
        return "No hay registros para interpretar."
    critical = (df_last["risk_label"] == "Fatiga crítica").sum()
    mod_or_worse = df_last["risk_label"].isin(["Fatiga moderada","Fatiga moderada-alta","Fatiga crítica"]).sum()
    mean_loss = df_last["objective_loss_score"].mean()
    if critical >= 2 or mean_loss >= 2.0:
        return "El grupo presenta una señal colectiva alta de pérdida de rendimiento en MD-1. Conviene minimizar la carga neuromuscular y priorizar frescura."
    if mod_or_worse >= max(2, round(len(df_last) * 0.25)) or mean_loss >= 1.2:
        return "Existe una afectación grupal moderada. La sesión de MD-1 debería ser breve, controlada y con estímulos de activación muy medidos."
    if (df_last["risk_label"].isin(["Fatiga leve","Fatiga leve-moderada"])).sum() >= max(2, round(len(df_last) * 0.3)):
        return "El grupo está globalmente estable, pero aparecen varias señales leves que aconsejan una MD-1 prudente y bien dosificada."
    return "El estado global del grupo es compatible con una MD-1 estable, con pérdidas individuales puntuales y controlables."

# =========================================================
# PLOTS
# =========================================================
def improved_radar(row):
    labels = [LABELS[m] for m in OBJECTIVE_METRICS]
    scores = []
    for m in OBJECTIVE_METRICS:
        pct = row.get(f"{m}_pct_vs_baseline", np.nan)
        if pd.isna(pct): score = 50
        elif pct >= 0: score = 100
        elif pct >= -2.5: score = 85
        elif pct > -5: score = 65
        elif pct > -10: score = 40
        else: score = 15
        scores.append(score)
    labels_closed = labels + [labels[0]]
    scores_closed = scores + [scores[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=scores_closed, theta=labels_closed, fill="toself", line=dict(color="#1F4E79", width=3), fillcolor="rgba(31,78,121,0.28)", name="Actual"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100], tickvals=[15,40,65,85,100], ticktext=["Crítica","Moderada","Leve","Óptimo","Máx"])), showlegend=False, title="Radar de estado neuromuscular", height=360, margin=dict(l=10,r=10,t=35,b=10))
    return fig

def radar_current_vs_baseline(row):
    labels = [LABELS[m] for m in OBJECTIVE_METRICS]
    current_scores, baseline_scores = [], []
    for m in OBJECTIVE_METRICS:
        pct = row.get(f"{m}_pct_vs_baseline", np.nan)
        if pd.isna(pct): score = 50
        elif pct >= 0: score = 100
        elif pct >= -2.5: score = 85
        elif pct > -5: score = 65
        elif pct > -10: score = 40
        else: score = 15
        current_scores.append(score)
        baseline_scores.append(85)
    labels_closed = labels + [labels[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=baseline_scores + [baseline_scores[0]], theta=labels_closed, fill="toself", line=dict(color="#98A2B3", width=2, dash="dash"), fillcolor="rgba(152,162,179,0.10)", name="Baseline visual"))
    fig.add_trace(go.Scatterpolar(r=current_scores + [current_scores[0]], theta=labels_closed, fill="toself", line=dict(color="#1F4E79", width=3), fillcolor="rgba(31,78,121,0.28)", name="Actual"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100], tickvals=[15,40,65,85,100], ticktext=["Crítica","Moderada","Leve","Óptimo","Máx"])), title="Actual vs baseline visual", height=360, margin=dict(l=10,r=10,t=35,b=10))
    return fig

def plot_metric_main(player_df, metric, selected_date):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=player_df["Fecha"], y=player_df[metric], mode="lines+markers", name="Valor real", line=dict(color="#1F4E79", width=3)))
    fig.add_trace(go.Scatter(x=player_df["Fecha"], y=player_df[f"{metric}_ma3"], mode="lines", name="MA3", line=dict(color="#64748B", width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=player_df["Fecha"], y=player_df[f"{metric}_baseline"], mode="lines", name="Baseline", line=dict(color="#0F766E", width=2, dash="dot")))
    sel = player_df[player_df["Fecha"].dt.normalize() == pd.to_datetime(selected_date).normalize()]
    if not sel.empty:
        fig.add_trace(go.Scatter(x=sel["Fecha"], y=sel[metric], mode="markers", name="Fecha", marker=dict(size=12, color="#C62828", symbol="diamond")))
    fig.update_layout(title=f"{LABELS[metric]} · valor real, MA3 y baseline", height=300, margin=dict(l=10,r=10,t=35,b=10))
    return fig

def plot_metric_pct(player_df, metric, selected_date):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=player_df["Fecha"], y=player_df[f"{metric}_pct_vs_baseline"], mode="lines+markers", name="% vs baseline", line=dict(color="#1F4E79", width=3)))
    sel = player_df[player_df["Fecha"].dt.normalize() == pd.to_datetime(selected_date).normalize()]
    if not sel.empty:
        fig.add_trace(go.Scatter(x=sel["Fecha"], y=sel[f"{metric}_pct_vs_baseline"], mode="markers", name="Fecha", marker=dict(size=12, color="#C62828", symbol="diamond")))
    fig.add_hline(y=0, line_dash="dot")
    fig.add_hrect(y0=-2.5, y1=15, fillcolor="rgba(46,139,87,0.10)", line_width=0)
    fig.add_hrect(y0=-5, y1=-2.5, fillcolor="rgba(227,160,8,0.12)", line_width=0)
    fig.add_hrect(y0=-10, y1=-5, fillcolor="rgba(249,115,22,0.12)", line_width=0)
    fig.add_hrect(y0=-30, y1=-10, fillcolor="rgba(198,40,40,0.10)", line_width=0)
    fig.update_layout(title=f"{LABELS[metric]} · % vs baseline", height=300, margin=dict(l=10,r=10,t=35,b=10))
    return fig

def plot_objective_timeline(player_df, selected_date):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=player_df["Fecha"], y=player_df["objective_loss_score"], mode="lines+markers", name="Loss score", line=dict(color="#C62828", width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=player_df["Fecha"], y=player_df["objective_loss_score_ma3"], mode="lines", name="Loss MA3", line=dict(color="#111827", width=2, dash="dash")), secondary_y=False)
    fig.add_trace(go.Scatter(x=player_df["Fecha"], y=player_df["objective_z_score"], mode="lines+markers", name="Z-score objetivo", line=dict(color="#F97316", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=player_df["Fecha"], y=player_df["readiness_score"], mode="lines+markers", name="Readiness", line=dict(color="#0F766E", width=3)), secondary_y=True)
    sel = player_df[player_df["Fecha"].dt.normalize() == pd.to_datetime(selected_date).normalize()]
    if not sel.empty:
        fig.add_trace(go.Scatter(x=sel["Fecha"], y=sel["objective_loss_score"], mode="markers", marker=dict(size=12, color="#111827", symbol="diamond"), name="Fecha"), secondary_y=False)
    fig.update_layout(title="Timeline combinado: loss + readiness + z-score", height=340, margin=dict(l=10,r=10,t=35,b=10))
    fig.update_yaxes(title_text="Loss / Z", secondary_y=False)
    fig.update_yaxes(title_text="Readiness", secondary_y=True)
    return fig

def plot_player_snapshot_compare(row):
    labels = [LABELS[m] for m in OBJECTIVE_METRICS]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[row.get(f"{m}_pct_vs_baseline", np.nan) for m in OBJECTIVE_METRICS], name="% vs baseline", marker_color="#1F4E79"))
    fig.add_trace(go.Bar(x=labels, y=[row.get(f"{m}_vs_team_pct", np.nan) for m in OBJECTIVE_METRICS], name="% vs equipo", marker_color="#0F766E"))
    fig.add_hline(y=-2.5, line_dash="dot"); fig.add_hline(y=-5, line_dash="dot"); fig.add_hline(y=-10, line_dash="dot")
    fig.update_layout(barmode="group", title="Snapshot de la sesión", height=320, margin=dict(l=10,r=10,t=35,b=10))
    return fig

def plot_team_heatmap(team_df):
    data = team_df.set_index("Jugador")[[f"{m}_pct_vs_baseline" for m in OBJECTIVE_METRICS]].copy()
    data.columns = [LABELS[m] for m in OBJECTIVE_METRICS]
    fig = px.imshow(data, color_continuous_scale=["#B91C1C","#F97316","#E3A008","#16A34A"], zmin=-15, zmax=5, text_auto=".1f", aspect="auto")
    fig.update_layout(title="Heatmap del equipo · % vs línea base", height=max(340, len(data) * 35 + 100), margin=dict(l=10,r=10,t=35,b=10))
    return fig

def plot_team_risk_distribution(team_df):
    temp = team_df["risk_label"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    temp.columns = ["Estado","N"]
    fig = px.bar(temp, x="Estado", y="N", color="Estado", color_discrete_map=RISK_COLORS, title="Distribución del riesgo")
    fig.update_layout(height=300, margin=dict(l=10,r=10,t=35,b=10), showlegend=False)
    return fig

def plot_team_objective_bar(team_df):
    temp = team_df[["Jugador","objective_loss_score","objective_loss_mean_pct","risk_label"]].copy().sort_values(["objective_loss_score","objective_loss_mean_pct"], ascending=[False, True])
    fig = px.bar(temp, x="Jugador", y="objective_loss_score", color="risk_label", color_discrete_map=RISK_COLORS, text="objective_loss_score", title="Objective loss score por jugador")
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=35,b=10), showlegend=False)
    return fig

def plot_team_score_trend(df):
    temp = df.groupby("Fecha").agg(objective_loss_score=("objective_loss_score","mean"), readiness_score=("readiness_score","mean"), objective_z_score=("objective_z_score","mean")).reset_index().sort_values("Fecha")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=temp["Fecha"], y=temp["objective_loss_score"], mode="lines+markers", name="Loss equipo"), secondary_y=False)
    fig.add_trace(go.Scatter(x=temp["Fecha"], y=temp["objective_z_score"], mode="lines+markers", name="Z-score equipo"), secondary_y=False)
    fig.add_trace(go.Scatter(x=temp["Fecha"], y=temp["readiness_score"], mode="lines+markers", name="Readiness equipo"), secondary_y=True)
    fig.update_layout(title="Evolución global del grupo", height=320, margin=dict(l=10,r=10,t=35,b=10))
    fig.update_yaxes(title_text="Loss / Z", secondary_y=False)
    fig.update_yaxes(title_text="Readiness", secondary_y=True)
    return fig

def plot_percentile_bars(row):
    temp = pd.DataFrame({"Métrica":[LABELS[m] for m in OBJECTIVE_METRICS], "Percentil histórico":[row.get(f"{m}_historical_percentile", np.nan) for m in OBJECTIVE_METRICS]})
    fig = px.bar(temp, x="Métrica", y="Percentil histórico", text="Percentil histórico", title="Percentil histórico individual")
    fig.update_traces(texttemplate="%{text:.0f}")
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=35,b=10), showlegend=False)
    return fig

def plot_rank_vs_team(row):
    temp = pd.DataFrame({
        "Métrica":[LABELS[m] for m in OBJECTIVE_METRICS] + ["Loss score","Readiness"],
        "Ranking sesión":[
            row.get("CMJ_team_rank", np.nan),
            row.get("RSI_mod_team_rank", np.nan),
            row.get("VMP_team_rank", np.nan),
            row.get("objective_loss_score_team_rank", np.nan),
            row.get("readiness_score_team_rank", np.nan)
        ]
    })
    fig = px.bar(temp, x="Métrica", y="Ranking sesión", text="Ranking sesión", title="Posición relativa en la sesión")
    fig.update_traces(texttemplate="%{text:.0f}")
    fig.update_layout(height=280, margin=dict(l=10,r=10,t=35,b=10), showlegend=False)
    return fig

def plotly_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False})

# =========================================================
# REPORT HTML / PDF
# =========================================================
def report_css():
    return """
    <style>
    body {font-family: Arial, sans-serif; margin: 24px; color: #111827; background: #F8FAFC;}
    .hero {background: linear-gradient(135deg, #0F172A 0%, #1F4E79 100%); color: white; border-radius: 20px; padding: 24px; margin-bottom: 16px;}
    .cards {display:grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 16px;}
    .card {background:#fff; border:1px solid #E5E7EB; border-radius:16px; padding:14px 16px;}
    .label {font-size:12px; color:#667085;}
    .value {font-size:26px; font-weight:800; color:#101828; margin-top:4px;}
    .section {background:#fff; border:1px solid #E5E7EB; border-radius:16px; padding:16px 18px; margin-bottom:14px;}
    .title {font-size:18px; font-weight:800; margin-bottom:10px; color:#101828;}
    .grid2 {display:grid; grid-template-columns:1fr 1fr; gap:14px;}
    .grid1 {display:grid; grid-template-columns:1fr; gap:14px;}
    .badge {display:inline-block; padding:6px 10px; border-radius:999px; color:#fff; font-weight:700; font-size:12px;}
    ul {margin:8px 0 0 18px;}
    table.report-table {width:100%; border-collapse:collapse; font-size:13px;}
    table.report-table th, table.report-table td {border-bottom:1px solid #E5E7EB; padding:8px 10px; text-align:left;}
    table.report-table th {background:#F8FAFC;}
    .diag {font-size:14px; line-height:1.5;}
    </style>
    """

def html_risk_badge(label):
    color = RISK_COLORS.get(label, "#475467")
    return f'<span class="badge" style="background:{color};">{label}</span>'

def coach_session_html(team_day, selected_date):
    rows = ""
    temp = team_day.copy().sort_values(["objective_loss_score","objective_loss_mean_pct"], ascending=[False, True])
    for _, r in temp.iterrows():
        rows += f"<tr><td>{r['Jugador']}</td><td>{html_risk_badge(r['risk_label'])}</td><td>{r['CMJ_pct_vs_baseline']:.1f}%</td><td>{r['RSI_mod_pct_vs_baseline']:.1f}%</td><td>{r['VMP_pct_vs_baseline']:.1f}%</td><td>{r['objective_loss_mean_pct']:.1f}%</td><td>{r['objective_loss_score']:.2f}</td></tr>"
    heat_html = plotly_html(plot_team_heatmap(team_day))
    bar_html = plotly_html(plot_team_objective_bar(team_day))
    risk_html = plotly_html(plot_team_risk_distribution(team_day))
    return f"""
    <html><head><meta charset="utf-8">{report_css()}</head><body>
    <div class="hero"><div style="font-size:12px;opacity:0.9;">Informe de sesión · MD-1</div><div style="font-size:30px;font-weight:900;">Estado neuromuscular del equipo</div><div style="font-size:15px;margin-top:6px;">Fecha analizada: {selected_date}</div></div>
    <div class="cards">
      <div class="card"><div class="label">Jugadores evaluados</div><div class="value">{team_day['Jugador'].nunique()}</div></div>
      <div class="card"><div class="label">Objective loss medio</div><div class="value">{team_day['objective_loss_score'].mean():.2f}</div></div>
      <div class="card"><div class="label">Pérdida media %</div><div class="value">{team_day['objective_loss_mean_pct'].mean():.1f}%</div></div>
      <div class="card"><div class="label">Readiness media</div><div class="value">{team_day['readiness_score'].mean():.0f}</div></div>
    </div>
    <div class="section"><div class="title">Lectura rápida para el entrenador</div><div class="diag">{team_interpretation(team_day)}</div></div>
    <div class="section"><div class="title">Panel visual</div><div class="grid2">{risk_html}{bar_html}</div><div class="grid1">{heat_html}</div></div>
    <div class="section"><div class="title">Resumen por jugador</div><table class="report-table"><thead><tr><th>Jugador</th><th>Riesgo</th><th>CMJ %</th><th>RSI mod %</th><th>VMP %</th><th>Pérdida media %</th><th>Loss score</th></tr></thead><tbody>{rows}</tbody></table></div>
    </body></html>
    """

def player_session_html(row, player_df, session_df):
    main, pattern, worst_metric, worst_value = infer_fatigue_profile(row)
    flags = flags_for_player(player_df, row)
    flags_html = "".join([f"<li>{f}</li>" for f in flags]) if flags else "<li>Sin flags adicionales</li>"

    rows = ""
    for m in OBJECTIVE_METRICS:
        z_val = row.get(f"{m}_z", np.nan)
        perc_val = row.get(f"{m}_historical_percentile", np.nan)
        z_txt = "NA" if pd.isna(z_val) else f"{z_val:.2f}"
        perc_txt = "NA" if pd.isna(perc_val) else f"{perc_val:.0f}"
        rows += f"<tr><td>{LABELS[m]}</td><td>{row.get(m, np.nan):.2f}</td><td>{row.get(f'{m}_baseline', np.nan):.2f}</td><td>{row.get(f'{m}_pct_vs_baseline', np.nan):.1f}%</td><td>{z_txt}</td><td>{row.get(f'{m}_severity', 'Sin referencia')}</td><td>{perc_txt}</td></tr>"

    rank_rows = ""
    for m in OBJECTIVE_METRICS:
        rk = row.get(f"{m}_team_rank", np.nan)
        rk_txt = "NA" if pd.isna(rk) else f"{rk:.0f}"
        rank_rows += f"<tr><td>{LABELS[m]}</td><td>{rk_txt}</td><td>{row.get(f'{m}_vs_team_pct', np.nan):.1f}%</td></tr>"

    radar_html = plotly_html(radar_current_vs_baseline(row))
    snapshot_html = plotly_html(plot_player_snapshot_compare(row))
    timeline_html = plotly_html(plot_objective_timeline(player_df, row['Fecha']))
    ranks_html = plotly_html(plot_rank_vs_team(row))
    percentiles_html = plotly_html(plot_percentile_bars(row))

    return f"""
    <html><head><meta charset="utf-8">{report_css()}</head><body>
    <div class="hero"><div style="font-size:12px;opacity:0.9;">Informe individual · Sesión específica</div><div style="font-size:30px;font-weight:900;">{row['Jugador']}</div><div style="font-size:15px;margin-top:6px;">Fecha: {pd.to_datetime(row['Fecha']).date()}</div></div>
    <div class="cards">
      <div class="card"><div class="label">Riesgo</div><div class="value" style="font-size:22px;">{row['risk_label']}</div></div>
      <div class="card"><div class="label">Loss score</div><div class="value">{row['objective_loss_score']:.2f}</div></div>
      <div class="card"><div class="label">Pérdida media %</div><div class="value">{row['objective_loss_mean_pct']:.1f}%</div></div>
      <div class="card"><div class="label">Readiness</div><div class="value">{row['readiness_score']:.0f}</div></div>
    </div>
    <div class="section"><div class="title">Diagnóstico staff</div><div class="diag"><p><b>Perfil principal:</b> {main}.</p><p><b>Patrón:</b> {pattern}.</p><p><b>Variable dominante:</b> {LABELS.get(worst_metric, 'NA')} ({'NA' if worst_value is None else f'{worst_value:.1f}%'}).</p><p>{player_comment(row)}</p></div></div>
    <div class="section"><div class="title">Flags automáticos</div><ul>{flags_html}</ul></div>
    <div class="section"><div class="title">Panel visual</div><div class="grid2">{radar_html}{snapshot_html}</div><div class="grid2">{timeline_html}{ranks_html}</div><div class="grid1">{percentiles_html}</div></div>
    <div class="section"><div class="title">Detalle por variable</div><table class="report-table"><thead><tr><th>Métrica</th><th>Valor</th><th>Línea base</th><th>% vs baseline</th><th>Z-score</th><th>Estado</th><th>Percentil histórico</th></tr></thead><tbody>{rows}</tbody></table></div>
    <div class="section"><div class="title">Comparación vs equipo en la sesión</div><table class="report-table"><thead><tr><th>Métrica</th><th>Ranking sesión</th><th>% vs equipo</th></tr></thead><tbody>{rank_rows}</tbody></table></div>
    </body></html>
    """

def player_season_html(player_df, player):
    latest = player_df.iloc[-1]
    main, pattern, worst_metric, worst_value = infer_fatigue_profile(latest)
    flags = flags_for_player(player_df, latest)
    flags_html = "".join([f"<li>{f}</li>" for f in flags]) if flags else "<li>Sin flags adicionales</li>"
    summary = player_df[["Fecha","objective_loss_score","objective_loss_mean_pct","readiness_score","risk_label","trend_label"]].copy()
    summary["Fecha"] = summary["Fecha"].dt.strftime("%Y-%m-%d")
    rows = ""
    for _, r in summary.iterrows():
        rows += f"<tr><td>{r['Fecha']}</td><td>{r['objective_loss_score']:.2f}</td><td>{r['objective_loss_mean_pct']:.1f}%</td><td>{r['readiness_score']:.0f}</td><td>{r['risk_label']}</td><td>{r['trend_label']}</td></tr>"

    radar_html = plotly_html(radar_current_vs_baseline(latest))
    timeline_html = plotly_html(plot_objective_timeline(player_df, latest["Fecha"]))
    cmj_html = plotly_html(plot_metric_pct(player_df, "CMJ", latest["Fecha"]))
    rsi_html = plotly_html(plot_metric_pct(player_df, "RSI_mod", latest["Fecha"]))
    vmp_html = plotly_html(plot_metric_pct(player_df, "VMP", latest["Fecha"]))

    return f"""
    <html><head><meta charset="utf-8">{report_css()}</head><body>
    <div class="hero"><div style="font-size:12px;opacity:0.9;">Informe individual · Evolución anual</div><div style="font-size:30px;font-weight:900;">{player}</div><div style="font-size:15px;margin-top:6px;">Registros: {len(player_df)} · Última fecha: {pd.to_datetime(latest['Fecha']).date()}</div></div>
    <div class="cards">
      <div class="card"><div class="label">Riesgo actual</div><div class="value" style="font-size:22px;">{latest['risk_label']}</div></div>
      <div class="card"><div class="label">Loss score actual</div><div class="value">{latest['objective_loss_score']:.2f}</div></div>
      <div class="card"><div class="label">Readiness actual</div><div class="value">{latest['readiness_score']:.0f}</div></div>
      <div class="card"><div class="label">Tendencia actual</div><div class="value" style="font-size:22px;">{latest['trend_label']}</div></div>
    </div>
    <div class="section"><div class="title">Diagnóstico longitudinal</div><div class="diag"><p><b>Perfil principal actual:</b> {main}.</p><p><b>Patrón:</b> {pattern}.</p><p><b>Variable dominante:</b> {LABELS.get(worst_metric, 'NA')} ({'NA' if worst_value is None else f'{worst_value:.1f}%'}).</p><p>{player_comment(latest)}</p></div></div>
    <div class="section"><div class="title">Flags actuales</div><ul>{flags_html}</ul></div>
    <div class="section"><div class="title">Panel visual</div><div class="grid2">{radar_html}{timeline_html}</div><div class="grid2">{cmj_html}{rsi_html}</div><div class="grid1">{vmp_html}</div></div>
    <div class="section"><div class="title">Resumen longitudinal</div><table class="report-table"><thead><tr><th>Fecha</th><th>Loss</th><th>Pérdida %</th><th>Readiness</th><th>Riesgo</th><th>Tendencia</th></tr></thead><tbody>{rows}</tbody></table></div>
    </body></html>
    """

def report_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallGrey", fontSize=9, textColor=colors.HexColor("#667085"), leading=11))
    styles.add(ParagraphStyle(name="TitleDark", fontSize=18, textColor=colors.HexColor("#0F172A"), leading=22, spaceAfter=8))
    styles.add(ParagraphStyle(name="SectionDark", fontSize=13, textColor=colors.HexColor("#0F172A"), leading=16, spaceAfter=6))
    return styles

def build_pdf_bytes_player_session(row, player_df):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.2*cm, rightMargin=1.2*cm, topMargin=1.0*cm, bottomMargin=1.0*cm)
    styles = report_styles()
    elems = []
    main, pattern, worst_metric, worst_value = infer_fatigue_profile(row)
    flags = flags_for_player(player_df, row)
    elems.append(Paragraph(f"Informe individual · {row['Jugador']}", styles["TitleDark"]))
    elems.append(Paragraph(f"Sesión: {pd.to_datetime(row['Fecha']).date()}", styles["SmallGrey"]))
    elems.append(Spacer(1, 0.2*cm))
    elems.append(Paragraph(player_comment(row), styles["BodyText"]))
    elems.append(Spacer(1, 0.25*cm))
    diag = [
        ["Riesgo", str(row["risk_label"])],
        ["Loss score", f"{row['objective_loss_score']:.2f}"],
        ["Pérdida media %", f"{row['objective_loss_mean_pct']:.1f}%"],
        ["Readiness", f"{row['readiness_score']:.0f}"],
        ["Z-score objetivo", "NA" if pd.isna(row["objective_z_score"]) else f"{row['objective_z_score']:.2f}"],
        ["Perfil", main],
        ["Patrón", pattern],
        ["Variable dominante", f"{LABELS.get(worst_metric, 'NA')} ({'NA' if worst_value is None else f'{worst_value:.1f}%'})"],
    ]
    t = Table(diag, colWidths=[4.8*cm, 10*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.whitesmoke),("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold")]))
    elems.append(t)
    elems.append(Spacer(1, 0.25*cm))
    elems.append(Paragraph("Flags automáticos", styles["SectionDark"]))
    for f in (flags or ["Sin flags adicionales"]):
        elems.append(Paragraph(f"• {f}", styles["BodyText"]))
    elems.append(Spacer(1, 0.2*cm))
    data = [["Métrica","Valor","Baseline","% vs baseline","Z-score","Estado"]]
    for m in OBJECTIVE_METRICS:
        z_val = row.get(f"{m}_z", np.nan)
        z_txt = "NA" if pd.isna(z_val) else f"{z_val:.2f}"
        data.append([LABELS[m], f"{row.get(m, np.nan):.2f}", f"{row.get(f'{m}_baseline', np.nan):.2f}", f"{row.get(f'{m}_pct_vs_baseline', np.nan):.1f}%", z_txt, str(row.get(f"{m}_severity", "Sin referencia"))])
    t2 = Table(data, repeatRows=1)
    t2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0F172A")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8.4)]))
    elems.append(t2)
    doc.build(elems)
    buf.seek(0)
    return buf.getvalue()

def build_pdf_bytes_player_season(player_df, player):
    latest = player_df.iloc[-1]
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1.0*cm, rightMargin=1.0*cm, topMargin=1.0*cm, bottomMargin=1.0*cm)
    styles = report_styles()
    elems = []
    main, pattern, worst_metric, worst_value = infer_fatigue_profile(latest)
    flags = flags_for_player(player_df, latest)
    elems.append(Paragraph(f"Informe anual · {player}", styles["TitleDark"]))
    elems.append(Paragraph(f"Registros: {len(player_df)} · Última fecha: {pd.to_datetime(latest['Fecha']).date()}", styles["SmallGrey"]))
    elems.append(Spacer(1, 0.2*cm))
    diag = [
        ["Riesgo actual", str(latest["risk_label"]), "Loss score", f"{latest['objective_loss_score']:.2f}"],
        ["Readiness", f"{latest['readiness_score']:.0f}", "Tendencia", str(latest["trend_label"])],
        ["Perfil", main, "Patrón", pattern],
        ["Variable dominante", f"{LABELS.get(worst_metric, 'NA')} ({'NA' if worst_value is None else f'{worst_value:.1f}%'})", "", ""],
    ]
    t = Table(diag, colWidths=[3.8*cm, 5.2*cm, 3.4*cm, 5.4*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.whitesmoke),("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold")]))
    elems.append(t)
    elems.append(Spacer(1, 0.2*cm))
    for f in (flags or ["Sin flags adicionales"]):
        elems.append(Paragraph(f"• {f}", styles["BodyText"]))
    elems.append(Spacer(1, 0.2*cm))
    data = [["Fecha","Loss","Pérdida %","Readiness","Riesgo","Tendencia"]]
    temp = player_df[["Fecha","objective_loss_score","objective_loss_mean_pct","readiness_score","risk_label","trend_label"]].copy()
    temp["Fecha"] = temp["Fecha"].dt.strftime("%Y-%m-%d")
    for _, r in temp.iterrows():
        data.append([str(r["Fecha"]), f"{r['objective_loss_score']:.2f}", f"{r['objective_loss_mean_pct']:.1f}%", f"{r['readiness_score']:.0f}", str(r["risk_label"]), str(r["trend_label"])])
    t2 = Table(data, repeatRows=1)
    t2.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0F172A")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8.2)]))
    elems.append(t2)
    doc.build(elems)
    buf.seek(0)
    return buf.getvalue()

def build_pdf_bytes_team_session(team_day, selected_date):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1.0*cm, rightMargin=1.0*cm, topMargin=1.0*cm, bottomMargin=1.0*cm)
    styles = report_styles()
    elems = []
    elems.append(Paragraph("Informe de sesión · MD-1", styles["TitleDark"]))
    elems.append(Paragraph(f"Fecha analizada: {selected_date}", styles["SmallGrey"]))
    elems.append(Spacer(1, 0.2*cm))
    elems.append(Paragraph(team_interpretation(team_day), styles["BodyText"]))
    elems.append(Spacer(1, 0.25*cm))
    data = [["Jugador","Riesgo","CMJ %","RSI %","VMP %","Pérdida %","Loss score"]]
    temp = team_day.copy().sort_values(["objective_loss_score","objective_loss_mean_pct"], ascending=[False, True])
    for _, r in temp.iterrows():
        data.append([str(r["Jugador"]), str(r["risk_label"]), f"{r['CMJ_pct_vs_baseline']:.1f}%", f"{r['RSI_mod_pct_vs_baseline']:.1f}%", f"{r['VMP_pct_vs_baseline']:.1f}%", f"{r['objective_loss_mean_pct']:.1f}%", f"{r['objective_loss_score']:.2f}"])
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#0F172A")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),("FONTSIZE",(0,0),(-1,-1),8.2)]))
    elems.append(t)
    doc.build(elems)
    buf.seek(0)
    return buf.getvalue()

# =========================================================
# PAGES
# =========================================================
def page_cargar():
    st.markdown("### Cargar archivo semanal")
    uploaded = st.file_uploader("Sube tu Excel/CSV semanal", type=["xlsx","xls","csv"])
    if uploaded is not None:
        try:
            parsed = parse_uploaded(uploaded)
            st.success(f"Archivo interpretado correctamente: {parsed['Jugador'].nunique()} jugadores · {parsed['Fecha'].nunique()} fecha(s)")
            st.dataframe(parsed, use_container_width=True, hide_index=True)
            if st.button("Guardar en base de datos", type="primary"):
                upsert_monitoring(parsed)
                st.success("Datos guardados correctamente.")
                st.rerun()
        except Exception as e:
            st.error(f"No se pudo interpretar el archivo: {e}")

def page_equipo(metrics_df):
    if metrics_df.empty:
        st.info("No hay datos disponibles.")
        return

    st.markdown('<div class="hero"><div style="font-size:0.92rem; opacity:0.9;">Monitorización neuromuscular MD-1</div><div style="font-size:2.05rem; font-weight:900; margin-top:0.15rem;">Equipo</div><div style="font-size:1rem; opacity:0.92; margin-top:0.4rem;">Lectura global del estado del grupo en MD-1.</div></div>', unsafe_allow_html=True)

    dates = sorted(metrics_df["Fecha"].dropna().unique())
    opts = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates]
    selected_date = pd.to_datetime(st.selectbox("Fecha de análisis", opts, index=len(opts)-1))
    team_day = metrics_df[metrics_df["Fecha"].dt.normalize() == selected_date.normalize()].copy()

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    with c1: kpi("Fecha", selected_date.strftime("%Y-%m-%d"), f"{team_day['Jugador'].nunique()} jugadores")
    with c2: kpi("Readiness media", f"{team_day['readiness_score'].mean():.1f}", "0-100")
    with c3: kpi("Objective loss medio", f"{team_day['objective_loss_score'].mean():.2f}", "0-3")
    with c4: kpi("Pérdida media %", f"{team_day['objective_loss_mean_pct'].mean():.1f}%", "grupo")
    with c5:
        z_mean = team_day["objective_z_score"].mean() if "objective_z_score" in team_day.columns else np.nan
        kpi("Z-score objetivo", "NA" if pd.isna(z_mean) else f"{z_mean:.2f}", "media grupo")
    with c6: kpi("Moderada o peor", int(team_day["risk_label"].isin(["Fatiga moderada","Fatiga moderada-alta","Fatiga crítica"]).sum()), "requieren atención")
    with c7: kpi("Casos críticos", int((team_day["risk_label"] == "Fatiga crítica").sum()), "prioridad")

    st.success(team_interpretation(team_day))

    a,b = st.columns(2)
    with a: st.plotly_chart(plot_team_risk_distribution(team_day), use_container_width=True)
    with b: st.plotly_chart(plot_team_score_trend(metrics_df), use_container_width=True)
    c,d = st.columns(2)
    with c: st.plotly_chart(plot_team_heatmap(team_day), use_container_width=True)
    with d: st.plotly_chart(plot_team_objective_bar(team_day), use_container_width=True)

    table = team_day[["Jugador","CMJ_pct_vs_baseline","RSI_mod_pct_vs_baseline","VMP_pct_vs_baseline","objective_loss_mean_pct","objective_loss_score","objective_z_score","readiness_score","risk_label","trend_label"]].copy()
    table.columns = ["Jugador","CMJ %","RSI mod %","VMP %","Pérdida media %","Loss score","Z-score objetivo","Readiness","Riesgo","Tendencia"]
    for col in ["CMJ %","RSI mod %","VMP %","Pérdida media %","Loss score","Z-score objetivo","Readiness"]:
        table[col] = table[col].round(2)
    st.dataframe(table.sort_values(["Loss score","Pérdida media %"], ascending=[False, True]), use_container_width=True, hide_index=True)

def page_jugador(metrics_df):
    if metrics_df.empty:
        st.info("No hay datos disponibles.")
        return
    st.markdown('<div class="section-title">Jugador</div>', unsafe_allow_html=True)
    players = sorted(metrics_df["Jugador"].dropna().unique().tolist())
    player = st.selectbox("Selecciona jugador", players)
    player_df = metrics_df[metrics_df["Jugador"] == player].copy().sort_values("Fecha")
    opts = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in player_df["Fecha"].dropna().unique()]
    selected_date = pd.to_datetime(st.selectbox("Fecha del jugador", opts, index=len(opts)-1))
    current = player_df[player_df["Fecha"].dt.normalize() == selected_date.normalize()]
    row = current.iloc[-1] if not current.empty else player_df.iloc[-1]
    risk_color = RISK_COLORS.get(row["risk_label"], "#475467")
    st.markdown(f'<div class="card"><div style="font-size:1.7rem; font-weight:900; color:#101828;">{player}</div><div style="margin-top:0.35rem;">{render_pills(row)}</div><div style="margin-top:0.55rem;"><span class="pill" style="background:{risk_color};">{row["risk_label"]}</span></div><div style="margin-top:0.7rem; color:#475467;">{player_comment(row)}</div></div>', unsafe_allow_html=True)

    main, pattern, worst_metric, worst_value = infer_fatigue_profile(row)
    flags = flags_for_player(player_df, row)
    if flags:
        st.warning(" | ".join(flags))
    st.markdown(f"**Diagnóstico:** {main}, con {pattern}. Variable dominante: {LABELS.get(worst_metric, 'NA')} ({'NA' if worst_value is None else f'{worst_value:.1f}%'}).")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi("Fecha", selected_date.strftime("%Y-%m-%d"), "control")
    with c2: kpi("Loss score", f"{row['objective_loss_score']:.2f}", "0-3")
    with c3: kpi("Pérdida media %", f"{row['objective_loss_mean_pct']:.1f}%", "CMJ + RSI + VMP")
    with c4: kpi("Readiness", f"{row['readiness_score']:.0f}", "ese mismo día")
    with c5: kpi("Riesgo", row["risk_label"], "clasificación")
    with c6: kpi("Z-score objetivo", "NA" if pd.isna(row["objective_z_score"]) else f"{row['objective_z_score']:.2f}", "apoyo")

    a,b = st.columns(2)
    with a: st.plotly_chart(improved_radar(row), use_container_width=True)
    with b: st.plotly_chart(radar_current_vs_baseline(row), use_container_width=True)
    c,d = st.columns(2)
    with c: st.plotly_chart(plot_player_snapshot_compare(row), use_container_width=True)
    with d: st.plotly_chart(plot_objective_timeline(player_df, selected_date), use_container_width=True)
    e,f = st.columns(2)
    with e: st.plotly_chart(plot_percentile_bars(row), use_container_width=True)
    with f: st.plotly_chart(plot_rank_vs_team(row), use_container_width=True)

    st.markdown("### Gráficas principales por variable")
    for m in OBJECTIVE_METRICS:
        l, r = st.columns(2)
        with l: st.plotly_chart(plot_metric_main(player_df, m, selected_date), use_container_width=True)
        with r: st.plotly_chart(plot_metric_pct(player_df, m, selected_date), use_container_width=True)

def page_comparador(metrics_df):
    if metrics_df.empty:
        st.info("No hay datos disponibles.")
        return
    players = sorted(metrics_df["Jugador"].dropna().unique().tolist())
    c1,c2 = st.columns(2)
    with c1: p1 = st.selectbox("Jugador A", players, index=0)
    with c2: p2 = st.selectbox("Jugador B", players, index=min(1, len(players)-1))
    if p1 == p2:
        st.warning("Selecciona dos jugadores distintos.")
        return
    metric = st.selectbox("Métrica", OBJECTIVE_METRICS, format_func=lambda x: LABELS[x])
    a = metrics_df[metrics_df["Jugador"] == p1].sort_values("Fecha")
    b = metrics_df[metrics_df["Jugador"] == p2].sort_values("Fecha")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a["Fecha"], y=a[f"{metric}_pct_vs_baseline"], mode="lines+markers", name=p1))
    fig.add_trace(go.Scatter(x=b["Fecha"], y=b[f"{metric}_pct_vs_baseline"], mode="lines+markers", name=p2))
    fig.add_hline(y=-2.5, line_dash="dot"); fig.add_hline(y=-5, line_dash="dot"); fig.add_hline(y=-10, line_dash="dot")
    fig.update_layout(title=f"Comparación · {LABELS[metric]} (% vs línea base)", height=360)
    st.plotly_chart(fig, use_container_width=True)

def page_informes(metrics_df):
    if metrics_df.empty:
        st.info("No hay datos disponibles.")
        return
    st.markdown('<div class="section-title">Informes descargables</div>', unsafe_allow_html=True)
    tab1,tab2,tab3 = st.tabs(["Informe individual anual","Informe individual por sesión","Informe entrenador sesión"])

    with tab1:
        players = sorted(metrics_df["Jugador"].dropna().unique().tolist())
        player = st.selectbox("Jugador · informe anual", players, key="yr")
        player_df = metrics_df[metrics_df["Jugador"] == player].copy().sort_values("Fecha")
        latest = player_df.iloc[-1]
        st.markdown(f"**Resumen actual:** {latest['risk_label']} · loss {latest['objective_loss_score']:.2f} · readiness {latest['readiness_score']:.0f}")
        html = player_season_html(player_df, player)
        st.download_button("Descargar HTML anual", data=html.encode("utf-8"), file_name=f"informe_anual_{player.replace(' ','_')}.html", mime="text/html")
        st.download_button("Descargar PDF anual", data=build_pdf_bytes_player_season(player_df, player), file_name=f"informe_anual_{player.replace(' ','_')}.pdf", mime="application/pdf")

    with tab2:
        players = sorted(metrics_df["Jugador"].dropna().unique().tolist())
        player = st.selectbox("Jugador · informe sesión", players, key="ses")
        player_df = metrics_df[metrics_df["Jugador"] == player].copy().sort_values("Fecha")
        dates = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in player_df["Fecha"].dropna().unique()]
        sel_date = st.selectbox("Fecha", dates, key="sesd")
        row = player_df[player_df["Fecha"].dt.strftime("%Y-%m-%d") == sel_date].iloc[-1]
        session_df = metrics_df[metrics_df["Fecha"].dt.strftime("%Y-%m-%d") == sel_date].copy()
        st.markdown(f"**Resumen:** {row['risk_label']} · pérdida media {row['objective_loss_mean_pct']:.1f}% · readiness {row['readiness_score']:.0f} · tendencia {row['trend_label']}")
        html = player_session_html(row, player_df, session_df)
        st.download_button("Descargar HTML sesión", data=html.encode("utf-8"), file_name=f"informe_sesion_{player.replace(' ','_')}_{sel_date}.html", mime="text/html")
        st.download_button("Descargar PDF sesión", data=build_pdf_bytes_player_session(row, player_df), file_name=f"informe_sesion_{player.replace(' ','_')}_{sel_date}.pdf", mime="application/pdf")

    with tab3:
        dates = sorted(metrics_df["Fecha"].dropna().unique())
        opts = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates]
        sel_date = st.selectbox("Fecha de sesión", opts, key="teamr")
        team_day = metrics_df[metrics_df["Fecha"].dt.strftime("%Y-%m-%d") == sel_date].copy()
        st.markdown(f"**Lectura rápida:** {team_interpretation(team_day)}")
        html = coach_session_html(team_day, sel_date)
        st.download_button("Descargar HTML entrenador", data=html.encode("utf-8"), file_name=f"informe_equipo_md1_{sel_date}.html", mime="text/html")
        st.download_button("Descargar PDF entrenador", data=build_pdf_bytes_team_session(team_day, sel_date), file_name=f"informe_equipo_md1_{sel_date}.pdf", mime="application/pdf")

def page_admin(base_df):
    if base_df.empty:
        st.info("La base está vacía.")
        return
    c1,c2,c3 = st.columns(3)
    with c1: kpi("Registros", len(base_df), "total")
    with c2: kpi("Jugadores", base_df["Jugador"].nunique(), "únicos")
    with c3: kpi("Fechas", base_df["Fecha"].nunique(), "controles")
    st.download_button("Descargar base CSV", data=base_df.to_csv(index=False).encode("utf-8"), file_name="md1_staff_elite_definitiva_v2.csv", mime="text/csv")

# =========================================================
# MAIN
# =========================================================
def main():
    init_db()
    st.sidebar.markdown("## Filtros")
    base_df = load_monitoring()
    if not base_df.empty and "Posicion" in base_df.columns and base_df["Posicion"].notna().any():
        options = sorted([x for x in base_df["Posicion"].dropna().astype(str).unique().tolist() if x.strip() != ""])
        positions = st.sidebar.multiselect("Posición", options=options)
        if positions:
            base_df = base_df[base_df["Posicion"].isin(positions)].copy()

    metrics_df = compute_metrics(base_df) if not base_df.empty else base_df.copy()
    menu = st.sidebar.radio("Sección", ["Cargar MD-1","Equipo","Jugador","Comparador","Informes","Administración"])

    if menu == "Cargar MD-1":
        page_cargar()
    elif menu == "Equipo":
        page_equipo(metrics_df)
    elif menu == "Jugador":
        page_jugador(metrics_df)
    elif menu == "Comparador":
        page_comparador(metrics_df)
    elif menu == "Informes":
        page_informes(metrics_df)
    else:
        page_admin(base_df)

if __name__ == "__main__":
    main()
