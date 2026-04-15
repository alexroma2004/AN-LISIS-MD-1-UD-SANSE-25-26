
import io
import re
from pathlib import Path
from supabase import create_client, Client

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="MD-1 Staff Elite Definitiva", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path("app_data")
APP_DIR.mkdir(exist_ok=True)

OBJECTIVE_METRICS = ["CMJ", "RSI_mod", "VMP"]
ALL_METRICS = ["CMJ", "RSI_mod", "VMP", "sRPE"]

LABELS = {"CMJ": "CMJ", "RSI_mod": "RSI mod", "VMP": "VMP sentadilla", "sRPE": "sRPE"}
MICROCYCLE_OPTIONS = ["MD+1", "MD+2", "MD-4", "MD-3", "MD-2", "MD-1"]
VALID_BASELINE_DAYS = ["MD-4", "MD-3", "MD-2", "MD-1"]

FORCE_PROFILE_COLORS = {
    "Avión": "#2563EB",
    "Tanque": "#7C3AED",
    "Elástico": "#059669",
    "Base por desarrollar": "#DC2626",
}
PROFILE_TABLE_NAME = "player_profiles"
PLAYER_PROFILES_SQL = """
create table if not exists public.player_profiles (
  "Jugador" text primary key,
  "Peso_corporal" double precision,
  "Carga_sentadilla" double precision,
  "updated_at" timestamp default now()
);

alter table public.player_profiles enable row level security;

create policy if not exists "player_profiles_select"
on public.player_profiles
for select
using (true);

create policy if not exists "player_profiles_insert"
on public.player_profiles
for insert
with check (true);

create policy if not exists "player_profiles_update"
on public.player_profiles
for update
using (true)
with check (true);

create policy if not exists "player_profiles_delete"
on public.player_profiles
for delete
using (true);

NOTIFY pgrst, 'reload schema';
"""
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
.block-container {padding-top: 2rem; padding-bottom: 1.25rem;}
.hero {
    background: linear-gradient(135deg, #0F172A 0%, #1F4E79 100%);
    color: white; border-radius: 20px; padding: 24px 26px; margin-bottom: 14px;
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
.player-card {
    background: linear-gradient(135deg, #0F172A 0%, #1D4ED8 100%);
    color: white; border-radius: 24px; padding: 20px 22px; margin-bottom: 12px;
    box-shadow: 0 16px 36px rgba(15,23,42,0.22);
}
.player-card h3 {margin:0; font-size:1.7rem; line-height:1.1;}
.player-card .sub {opacity:0.9; margin-top:0.25rem; font-size:0.95rem;}
.player-badge {
    display:inline-block; padding:6px 12px; border-radius:999px; font-weight:800; font-size:0.82rem;
    color:white; margin-top:10px; margin-bottom:12px;
}
.player-msg {
    background:#F8FAFC; border:1px solid rgba(15,23,42,0.08); border-radius:16px; padding:14px 16px;
    color:#0F172A; font-size:0.96rem; line-height:1.45; margin-top:10px;
}
.mini-grid {
    display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:10px; margin-top:12px;
}
.mini-stat {
    background: rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.12); border-radius:16px; padding:12px 14px;
}
.mini-stat .lab {font-size:0.78rem; opacity:0.85;}
.mini-stat .val {font-size:1.2rem; font-weight:800; margin-top:2px;}
.tag-soft {
    display:inline-block; padding:5px 10px; border-radius:999px; font-weight:700; font-size:0.78rem;
    background:#EEF2FF; color:#1D4ED8; margin-right:6px; margin-top:4px;
}
.summary-strip {
    display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:14px; margin:8px 0 16px 0;
}
.summary-tile {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 55%, #2563EB 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 16px 18px 16px 18px;
    box-shadow: 0 16px 34px rgba(15,23,42,0.18);
}
.summary-tile::after {
    content: "";
    position: absolute;
    top: -30px;
    right: -20px;
    width: 90px;
    height: 90px;
    border-radius: 999px;
    background: rgba(255,255,255,0.10);
}
.summary-tile .icon {
    width: 34px;
    height: 34px;
    border-radius: 12px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: rgba(255,255,255,0.12);
    font-size: 1rem;
    margin-bottom: 10px;
}
.summary-tile .top {font-size:0.78rem; color:rgba(255,255,255,0.78); font-weight:700;}
.summary-tile .big {font-size:1.72rem; font-weight:900; color:#FFFFFF; line-height:1.06; margin-top:4px;}
.summary-tile .sub {font-size:0.84rem; color:rgba(255,255,255,0.84); margin-top:6px; line-height:1.3;}
.premium-panel {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border:1px solid rgba(15,23,42,0.08);
    border-radius:22px;
    padding:16px 18px;
    box-shadow: 0 10px 26px rgba(15,23,42,0.08);
    margin-bottom:12px;
}
.premium-heading {
    display:flex;
    align-items:center;
    gap:10px;
    font-size:1rem;
    font-weight:900;
    color:#0F172A;
    margin-bottom:12px;
}
.premium-heading .dot {
    width:10px;
    height:10px;
    border-radius:999px;
    background:#2563EB;
    box-shadow: 0 0 0 6px rgba(37,99,235,0.12);
}
.quick-chip {
    display:inline-block; padding:7px 11px; border-radius:999px; font-weight:800; font-size:0.79rem;
    margin-right:8px; margin-bottom:8px; color:white; box-shadow: 0 6px 16px rgba(15,23,42,0.10);
}

.results-grid {
    display:grid; grid-template-columns: repeat(5, minmax(0,1fr)); gap:12px; margin:10px 0 14px 0;
}
.result-card {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border:1px solid rgba(15,23,42,0.08);
    border-radius:20px;
    padding:14px 16px;
    box-shadow: 0 8px 20px rgba(15,23,42,0.06);
}
.result-card .lab {font-size:0.78rem; color:#667085; font-weight:700;}
.result-card .big {font-size:1.5rem; color:#101828; font-weight:900; margin-top:4px; line-height:1.05;}
.result-card .mini {font-size:0.80rem; color:#475467; margin-top:6px; line-height:1.35;}
.result-up {color:#15803D; font-weight:800;}
.result-down {color:#C2410C; font-weight:800;}
.result-neutral {color:#334155; font-weight:800;}

.prepost-grid {
    display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:12px; margin:10px 0 14px 0;
}
.prepost-card {
    background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
    border:1px solid rgba(15,23,42,0.08);
    border-radius:20px;
    padding:14px 16px;
    box-shadow: 0 8px 20px rgba(15,23,42,0.06);
}
.prepost-card .lab {font-size:0.82rem; color:#667085; font-weight:700;}
.prepost-card .big {font-size:1.35rem; color:#101828; font-weight:900; margin-top:4px;}
.prepost-card .mini {font-size:0.80rem; color:#475467; margin-top:6px; line-height:1.35;}
.soft-note {
    background:#EFF6FF;
    border:1px solid #DBEAFE;
    color:#1D4ED8;
    border-radius:16px;
    padding:12px 14px;
    font-size:0.9rem;
    font-weight:700;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# DB
# =========================================================
@st.cache_resource
def get_supabase() -> Client:
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )

def init_db():
    # Supabase gestiona la persistencia; no se crea DB local.
    pass

def load_monitoring():
    try:
        supabase = get_supabase()
        res = supabase.table("monitoring").select("*").execute()
        data = res.data if getattr(res, "data", None) else []
    except Exception as e:
        st.error(f"Error al leer desde Supabase: {e}")
        return pd.DataFrame(columns=["Fecha","Jugador","Microciclo","Posicion","Minutos","CMJ","RSI_mod","CMJ_post","RSI_mod_post","VMP","sRPE","Observaciones"])

    if not data:
        return pd.DataFrame(columns=["Fecha","Jugador","Microciclo","Posicion","Minutos","CMJ","RSI_mod","CMJ_post","RSI_mod_post","VMP","sRPE","Observaciones"])

    df = pd.DataFrame(data)

    expected_cols = ["Fecha","Jugador","Microciclo","Posicion","Minutos","CMJ","RSI_mod","CMJ_post","RSI_mod_post","VMP","sRPE","Observaciones","updated_at"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    df = df[expected_cols].copy()

    if "Fecha" in df.columns:
        df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    for c in ["Minutos", *ALL_METRICS, "CMJ_post", "RSI_mod_post"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Jugador" in df.columns:
        df["Jugador"] = df["Jugador"].astype(str).str.strip()

    return df.sort_values(["Jugador","Fecha"]).reset_index(drop=True)

def upsert_monitoring(df):
    if df.empty:
        return
    supabase = get_supabase()
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "Fecha": str(pd.to_datetime(r["Fecha"]).date()),
            "Jugador": str(r["Jugador"]),
            "Microciclo": None if pd.isna(r.get("Microciclo")) else str(r.get("Microciclo")),
            "Posicion": None if pd.isna(r.get("Posicion")) else str(r.get("Posicion")),
            "Minutos": None if pd.isna(r.get("Minutos")) else float(r.get("Minutos")),
            "CMJ": None if pd.isna(r.get("CMJ")) else float(r.get("CMJ")),
            "RSI_mod": None if pd.isna(r.get("RSI_mod")) else float(r.get("RSI_mod")),
            "CMJ_post": None if pd.isna(r.get("CMJ_post")) else float(r.get("CMJ_post")),
            "RSI_mod_post": None if pd.isna(r.get("RSI_mod_post")) else float(r.get("RSI_mod_post")),
            "VMP": None if pd.isna(r.get("VMP")) else float(r.get("VMP")),
            "sRPE": None if pd.isna(r.get("sRPE")) else float(r.get("sRPE")),
            "Observaciones": None if pd.isna(r.get("Observaciones")) else str(r.get("Observaciones")),
            "updated_at": now,
        })
    supabase.table("monitoring").upsert(rows, on_conflict="Fecha,Jugador").execute()

def delete_session_by_date(date_str):
    supabase = get_supabase()
    supabase.table("monitoring").delete().eq("Fecha", date_str).execute()


def load_player_profiles():
    try:
        supabase = get_supabase()
        res = supabase.table(PROFILE_TABLE_NAME).select("*").execute()
        data = res.data if getattr(res, "data", None) else []
    except Exception as e:
        return pd.DataFrame(columns=["Jugador","Peso_corporal","Carga_sentadilla"]), str(e)

    if not data:
        return pd.DataFrame(columns=["Jugador","Peso_corporal","Carga_sentadilla"]), None

    df = pd.DataFrame(data)
    keep_cols = [c for c in ["Jugador","Peso_corporal","Carga_sentadilla","updated_at"] if c in df.columns]
    df = df[keep_cols].copy()
    if "Jugador" in df.columns:
        df["Jugador"] = df["Jugador"].astype(str).str.strip()
    for c in ["Peso_corporal","Carga_sentadilla"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, None

def upsert_player_profiles(df):
    if df.empty:
        return
    supabase = get_supabase()
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    rows = []
    for _, r in df.iterrows():
        jugador = str(r.get("Jugador", "")).strip()
        if not jugador:
            continue
        peso = r.get("Peso_corporal")
        carga = r.get("Carga_sentadilla")
        rows.append({
            "Jugador": jugador,
            "Peso_corporal": None if pd.isna(peso) else float(peso),
            "Carga_sentadilla": None if pd.isna(carga) else float(carga),
            "updated_at": now,
        })
    if rows:
        supabase.table(PROFILE_TABLE_NAME).upsert(rows, on_conflict="Jugador").execute()

def estimate_pct_1rm_from_vmp(vmp):
    if pd.isna(vmp):
        return np.nan
    # Aproximación lineal basada en anclajes prácticos del proyecto:
    # 1.07 m/s ≈ 55% 1RM y 0.92 m/s ≈ 65% 1RM
    pct = 126.3333333333 - 66.6666666667 * float(vmp)
    return float(np.clip(pct, 30, 100))

def estimate_1rm_from_load_vmp(load_kg, vmp):
    if pd.isna(load_kg) or pd.isna(vmp) or float(vmp) <= 0:
        return np.nan
    pct = estimate_pct_1rm_from_vmp(vmp)
    if pd.isna(pct) or pct <= 0:
        return np.nan
    return float(load_kg) / (pct / 100.0)

def force_reactivity_profile_label(rsi_mod, est_1rm_rel, rsi_ref, rel_ref):
    if pd.isna(rsi_mod) or pd.isna(est_1rm_rel) or pd.isna(rsi_ref) or pd.isna(rel_ref):
        return np.nan
    if rsi_mod >= rsi_ref and est_1rm_rel >= rel_ref:
        return "Avión"
    if rsi_mod < rsi_ref and est_1rm_rel >= rel_ref:
        return "Tanque"
    if rsi_mod >= rsi_ref and est_1rm_rel < rel_ref:
        return "Elástico"
    return "Base por desarrollar"

def build_force_reactivity_df(metrics_df, selected_date):
    profiles_df, err = load_player_profiles()
    day_df = metrics_df[metrics_df["Fecha"].dt.normalize() == pd.to_datetime(selected_date).normalize()].copy()
    if day_df.empty:
        return day_df, profiles_df, err, np.nan, np.nan

    # último registro del día por jugador
    day_df = day_df.sort_values(["Jugador","Fecha"]).groupby("Jugador", as_index=False).tail(1)
    if profiles_df.empty:
        merged = day_df.copy()
        merged["Peso_corporal"] = np.nan
        merged["Carga_sentadilla"] = np.nan
    else:
        merged = day_df.merge(
            profiles_df[["Jugador","Peso_corporal","Carga_sentadilla"]],
            on="Jugador", how="left"
        )

    merged["est_1rm"] = merged.apply(lambda r: estimate_1rm_from_load_vmp(r.get("Carga_sentadilla"), r.get("VMP")), axis=1)
    merged["est_1rm_rel"] = np.where(
        merged["Peso_corporal"].notna() & (merged["Peso_corporal"] > 0),
        merged["est_1rm"] / merged["Peso_corporal"],
        np.nan
    )

    valid = merged.dropna(subset=["RSI_mod", "est_1rm_rel"]).copy()
    rsi_ref = valid["RSI_mod"].median() if not valid.empty else np.nan
    rel_ref = valid["est_1rm_rel"].median() if not valid.empty else np.nan
    merged["perfil_fr"] = merged.apply(lambda r: force_reactivity_profile_label(r.get("RSI_mod"), r.get("est_1rm_rel"), rsi_ref, rel_ref), axis=1)

    return merged.sort_values("Jugador"), profiles_df, err, rsi_ref, rel_ref

def plot_force_reactivity_scatter(df, rsi_ref, rel_ref):
    fig = go.Figure()
    plot_df = df.dropna(subset=["RSI_mod", "est_1rm_rel"]).copy()

    if plot_df.empty:
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), title="Perfil fuerza-reactividad")
        return fig

    for profile_name in ["Avión", "Tanque", "Elástico", "Base por desarrollar"]:
        sub = plot_df[plot_df["perfil_fr"] == profile_name].copy()
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["RSI_mod"],
            y=sub["est_1rm_rel"],
            mode="markers+text",
            name=profile_name,
            text=sub["Jugador"],
            textposition="top center",
            marker=dict(size=13, color=FORCE_PROFILE_COLORS[profile_name], line=dict(color="white", width=1.5)),
            customdata=np.stack([
                sub["VMP"].astype(float),
                sub["Carga_sentadilla"].fillna(np.nan).astype(float),
                sub["Peso_corporal"].fillna(np.nan).astype(float),
                sub["est_1rm"].fillna(np.nan).astype(float),
            ], axis=1),
            hovertemplate="<b>%{text}</b><br>RSI mod: %{x:.3f}<br>1RM relativa: %{y:.2f} kg/kg<br>VMP: %{customdata[0]:.3f} m/s<br>Carga usada: %{customdata[1]:.1f} kg<br>Peso corporal: %{customdata[2]:.1f} kg<br>1RM estimada: %{customdata[3]:.1f} kg<extra></extra>",
        ))

    if pd.notna(rsi_ref):
        fig.add_vline(x=float(rsi_ref), line_width=2, line_color="#475467")
    if pd.notna(rel_ref):
        fig.add_hline(y=float(rel_ref), line_width=2, line_color="#475467")

    x_min = max(0.10, float(plot_df["RSI_mod"].min()) - 0.05)
    x_max = float(plot_df["RSI_mod"].max()) + 0.05
    y_min = max(0.4, float(plot_df["est_1rm_rel"].min()) - 0.15)
    y_max = float(plot_df["est_1rm_rel"].max()) + 0.15
    fig.update_xaxes(range=[x_min, x_max], title="RSI modificado")
    fig.update_yaxes(range=[y_min, y_max], title="1RM relativa estimada (kg/kg)")

    if pd.notna(rsi_ref) and pd.notna(rel_ref):
        fig.add_annotation(x=x_max, y=y_max, text="AVIÓN", showarrow=False, xanchor="right", yanchor="top", font=dict(size=13, color=FORCE_PROFILE_COLORS["Avión"]))
        fig.add_annotation(x=x_min, y=y_max, text="TANQUE", showarrow=False, xanchor="left", yanchor="top", font=dict(size=13, color=FORCE_PROFILE_COLORS["Tanque"]))
        fig.add_annotation(x=x_max, y=y_min, text="ELÁSTICO", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=13, color=FORCE_PROFILE_COLORS["Elástico"]))
        fig.add_annotation(x=x_min, y=y_min, text="BASE POR DESARROLLAR", showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=12, color=FORCE_PROFILE_COLORS["Base por desarrollar"]))

    fig.update_layout(
        title="RSI modificado vs 1RM relativa estimada",
        height=560,
        margin=dict(l=10, r=10, t=45, b=10),
        legend_title="Perfil",
    )
    return fig


def fr_delta_text(value, ref, decimals=2, suffix=""):
    if pd.isna(value) or pd.isna(ref):
        return "Sin referencia"
    delta = value - ref
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{decimals}f}{suffix}"

def force_profile_focus(profile_name):
    mapping = {
        "Avión": "Perfil muy completo: combina buena reactividad y buena fuerza relativa. La prioridad es mantener.",
        "Tanque": "Predomina la fuerza relativa. La pista de mejora está en ganar reactividad y velocidad de expresión.",
        "Elástico": "Predomina la reactividad. La pista de mejora está en ganar fuerza relativa sin perder frescura.",
        "Base por desarrollar": "Todavía hay margen claro tanto en reactividad como en fuerza relativa. Conviene un trabajo global."
    }
    return mapping.get(profile_name, "Perfil no disponible.")

def build_force_profile_message(row, rsi_ref, rel_ref):
    profile = row.get("perfil_fr")
    name = row.get("Jugador", "Jugador")
    rsi = row.get("RSI_mod")
    rel = row.get("est_1rm_rel")
    parts = [f"{name} se sitúa ahora mismo en el perfil **{profile}**."]
    parts.append(force_profile_focus(profile))

    if pd.notna(rsi) and pd.notna(rsi_ref):
        if rsi >= rsi_ref:
            parts.append(f"Su RSI mod ({rsi:.3f}) está por encima de la referencia del grupo ({rsi_ref:.3f}).")
        else:
            parts.append(f"Su RSI mod ({rsi:.3f}) está por debajo de la referencia del grupo ({rsi_ref:.3f}).")

    if pd.notna(rel) and pd.notna(rel_ref):
        if rel >= rel_ref:
            parts.append(f"Su 1RM relativa estimada ({rel:.2f} kg/kg) también está por encima de la referencia del grupo ({rel_ref:.2f} kg/kg).")
        else:
            parts.append(f"Su 1RM relativa estimada ({rel:.2f} kg/kg) está por debajo de la referencia del grupo ({rel_ref:.2f} kg/kg).")

    if profile == "Avión":
        parts.append("Mensaje práctico: mantener la base actual y afinar detalles sin meter fatiga innecesaria.")
    elif profile == "Tanque":
        parts.append("Mensaje práctico: priorizar tareas de salto, RSI, stiffness y acciones explosivas de baja pérdida técnica.")
    elif profile == "Elástico":
        parts.append("Mensaje práctico: introducir trabajo de fuerza útil para elevar la base sin apagar su componente reactivo.")
    elif profile == "Base por desarrollar":
        parts.append("Mensaje práctico: progresión equilibrada de fuerza general, técnica de sentadilla y tareas reactivas básicas.")
    return " ".join(parts)


def latest_previous_player_row(metrics_df, player, selected_date):
    if metrics_df.empty:
        return None
    dfp = metrics_df[metrics_df["Jugador"] == player].copy()
    dfp = dfp[dfp["Fecha"].dt.normalize() < pd.to_datetime(selected_date).normalize()]
    if dfp.empty:
        return None
    return dfp.sort_values("Fecha").iloc[-1]

def player_ma3_context(metrics_df, player, selected_date):
    if metrics_df.empty:
        return {}
    dfp = metrics_df[metrics_df["Jugador"] == player].copy()
    dfp = dfp[dfp["Fecha"].dt.normalize() <= pd.to_datetime(selected_date).normalize()]
    if dfp.empty:
        return {}
    dfp = dfp.sort_values("Fecha").tail(3)
    out = {}
    for col in ["RSI_mod", "VMP", "CMJ"]:
        vals = pd.to_numeric(dfp[col], errors="coerce").dropna()
        out[f"{col}_ma3"] = vals.mean() if not vals.empty else np.nan
    return out

def force_profile_priority(profile_name):
    mapping = {
        "Avión": "Mantener y afinar",
        "Tanque": "Potenciar reactividad",
        "Elástico": "Potenciar fuerza relativa",
        "Base por desarrollar": "Desarrollo global",
    }
    return mapping.get(profile_name, "Sin prioridad definida")

def force_profile_strengths(profile_name):
    mapping = {
        "Avión": ("Buena reactividad", "Buena fuerza relativa"),
        "Tanque": ("Base de fuerza sólida", "Potencial de transferencia alto"),
        "Elástico": ("Buen componente reactivo", "Expresión neuromuscular interesante"),
        "Base por desarrollar": ("Margen de mejora amplio", "Perfil muy moldeable"),
    }
    return mapping.get(profile_name, ("—", "—"))

def force_profile_score(rsi, rel, rsi_ref, rel_ref):
    if pd.isna(rsi) or pd.isna(rel) or pd.isna(rsi_ref) or pd.isna(rel_ref) or rsi_ref <= 0 or rel_ref <= 0:
        return np.nan
    score = 50 * (rsi / rsi_ref) + 50 * (rel / rel_ref)
    return float(np.clip(score, 0, 100))

def score_label(score):
    if pd.isna(score):
        return "Sin score"
    if score >= 85:
        return "Perfil muy completo"
    if score >= 70:
        return "Buen perfil"
    if score >= 55:
        return "Perfil con margen"
    return "Perfil por desarrollar"

def compact_delta(curr, prev, decimals=3, suffix=""):
    if pd.isna(curr) or pd.isna(prev):
        return "Sin referencia"
    delta = curr - prev
    pct = (delta / prev * 100) if prev != 0 else np.nan
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    if pd.isna(pct):
        return f"{arrow} {delta:.{decimals}f}{suffix}"
    return f"{arrow} {delta:.{decimals}f}{suffix} ({pct:+.1f}%)"


def render_force_profile_card(row, rsi_ref, rel_ref, metrics_df=None, selected_date=None):
    profile = row.get("perfil_fr", "Perfil no disponible")
    color = FORCE_PROFILE_COLORS.get(profile, "#334155")
    rsi = row.get("RSI_mod")
    rel = row.get("est_1rm_rel")
    est = row.get("est_1rm")
    vmp = row.get("VMP")
    peso = row.get("Peso_corporal")
    carga = row.get("Carga_sentadilla")
    player = row.get("Jugador", "Jugador")

    prev_row = latest_previous_player_row(metrics_df, player, selected_date) if metrics_df is not None and selected_date is not None else None
    ctx_ma3 = player_ma3_context(metrics_df, player, selected_date) if metrics_df is not None and selected_date is not None else {}
    prev_rsi = prev_row["RSI_mod"] if prev_row is not None and "RSI_mod" in prev_row else np.nan
    prev_vmp = prev_row["VMP"] if prev_row is not None and "VMP" in prev_row else np.nan
    score = force_profile_score(rsi, rel, rsi_ref, rel_ref)
    strength_1, strength_2 = force_profile_strengths(profile)
    priority = force_profile_priority(profile)

    def fnum(x, d=2):
        return "—" if pd.isna(x) else f"{x:.{d}f}"

    card = f"""
    <div class="player-card">
        <div style="font-size:0.9rem; opacity:0.88;">Tarjeta individual</div>
        <h3>{player}</h3>
        <div class="sub">Lectura rápida del perfil fuerza-reactividad en la fecha seleccionada.</div>
        <div class="player-badge" style="background:{color};">{profile}</div>
        <div class="mini-grid">
            <div class="mini-stat"><div class="lab">RSI mod</div><div class="val">{fnum(rsi,3)}</div><div class="lab">vs última: {compact_delta(rsi, prev_rsi, 3)}</div></div>
            <div class="mini-stat"><div class="lab">1RM relativa</div><div class="val">{fnum(rel,2)} kg/kg</div><div class="lab">vs equipo: {fr_delta_text(rel, rel_ref, 2)}</div></div>
            <div class="mini-stat"><div class="lab">1RM estimada</div><div class="val">{fnum(est,1)} kg</div><div class="lab">estimación</div></div>
            <div class="mini-stat"><div class="lab">VMP</div><div class="val">{fnum(vmp,3)} m/s</div><div class="lab">vs última: {compact_delta(vmp, prev_vmp, 3, " m/s")}</div></div>
        </div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(f'<span class="tag-soft">Fortaleza 1: {strength_1}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="tag-soft">Fortaleza 2: {strength_2}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="tag-soft">Prioridad: {priority}</span>', unsafe_allow_html=True)
    with t2:
        tags = []
        if pd.notna(peso): tags.append(f'<span class="tag-soft">Peso corporal: {peso:.1f} kg</span>')
        if pd.notna(carga): tags.append(f'<span class="tag-soft">Carga sentadilla: {carga:.1f} kg</span>')
        if pd.notna(ctx_ma3.get("RSI_mod_ma3", np.nan)): tags.append(f'<span class="tag-soft">RSI MA3: {ctx_ma3["RSI_mod_ma3"]:.3f}</span>')
        if pd.notna(score): tags.append(f'<span class="tag-soft">Score F-R: {score:.0f}/100 · {score_label(score)}</span>')
        for tag in tags:
            st.markdown(tag, unsafe_allow_html=True)

    st.markdown(f'<div class="player-msg">{build_force_profile_message(row, rsi_ref, rel_ref)}</div>', unsafe_allow_html=True)


def classify_balance_level(rsi, rel, rsi_ref, rel_ref):
    if pd.isna(rsi) or pd.isna(rel) or pd.isna(rsi_ref) or pd.isna(rel_ref) or rsi_ref <= 0 or rel_ref <= 0:
        return "Sin clasificar"
    nr = rsi / rsi_ref
    nf = rel / rel_ref
    diff = abs(nr - nf)
    mean_level = (nr + nf) / 2

    if diff <= 0.10:
        if mean_level >= 1.05:
            return "Equilibrado alto"
        elif mean_level >= 0.90:
            return "Equilibrado medio"
        else:
            return "Equilibrado bajo"
    else:
        if nr > nf:
            return "Desequilibrado reactivo"
        else:
            return "Desequilibrado fuerza"

def action_priority_label(profile_name, balance_label):
    if balance_label == "Equilibrado alto":
        return "Mantener"
    if balance_label == "Equilibrado medio":
        return "Ajustar"
    if balance_label == "Equilibrado bajo":
        return "Desarrollo global"
    if profile_name == "Tanque":
        return "Potenciar reactividad"
    if profile_name == "Elástico":
        return "Potenciar fuerza"
    if profile_name == "Base por desarrollar":
        return "Desarrollo global"
    return "Mantener"

def trend_arrow(curr, prev):
    if pd.isna(curr) or pd.isna(prev):
        return "—"
    if curr > prev:
        return "↑"
    if curr < prev:
        return "↓"
    return "→"

def build_team_force_summary(valid_df, metrics_df, selected_date, rsi_ref, rel_ref):
    rows = []
    for _, row in valid_df.iterrows():
        prev = latest_previous_player_row(metrics_df, row["Jugador"], selected_date)
        prev_rsi = prev["RSI_mod"] if prev is not None and "RSI_mod" in prev else np.nan
        prev_vmp = prev["VMP"] if prev is not None and "VMP" in prev else np.nan
        prev_est_1rm = np.nan
        if prev is not None:
            prev_est_1rm = estimate_1rm_from_load_vmp(row.get("Carga_sentadilla"), prev.get("VMP"))
        prev_rel = prev_est_1rm / row["Peso_corporal"] if pd.notna(prev_est_1rm) and pd.notna(row.get("Peso_corporal")) and row.get("Peso_corporal") > 0 else np.nan
        score = force_profile_score(row.get("RSI_mod"), row.get("est_1rm_rel"), rsi_ref, rel_ref)
        balance_label = classify_balance_level(row.get("RSI_mod"), row.get("est_1rm_rel"), rsi_ref, rel_ref)
        rows.append({
            "Jugador": row["Jugador"],
            "Perfil": row["perfil_fr"],
            "Equilibrio": balance_label,
            "Score F-R": round(score, 1) if pd.notna(score) else np.nan,
            "RSI mod": row["RSI_mod"],
            "1RM relativa (kg/kg)": row["est_1rm_rel"],
            "Cambio RSI": trend_arrow(row["RSI_mod"], prev_rsi),
            "Cambio 1RM rel": trend_arrow(row["est_1rm_rel"], prev_rel),
            "Prioridad": action_priority_label(row["perfil_fr"], balance_label),
        })
    return pd.DataFrame(rows)

def plot_force_reactivity_filtered(df, rsi_ref, rel_ref, selected_profiles=None, show_names=True):
    if selected_profiles:
        df = df[df["perfil_fr"].isin(selected_profiles)].copy()
    fig = go.Figure()
    plot_df = df.dropna(subset=["RSI_mod", "est_1rm_rel"]).copy()
    if plot_df.empty:
        fig.update_layout(height=560, title="RSI modificado vs 1RM relativa estimada")
        return fig

    for profile_name in ["Avión", "Tanque", "Elástico", "Base por desarrollar"]:
        sub = plot_df[plot_df["perfil_fr"] == profile_name].copy()
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["RSI_mod"],
            y=sub["est_1rm_rel"],
            mode="markers+text" if show_names else "markers",
            name=profile_name,
            text=sub["Jugador"] if show_names else None,
            textposition="top center",
            marker=dict(size=13, color=FORCE_PROFILE_COLORS[profile_name], line=dict(color="white", width=1.5)),
            customdata=np.stack([
                sub["VMP"].astype(float),
                sub["Carga_sentadilla"].fillna(np.nan).astype(float),
                sub["Peso_corporal"].fillna(np.nan).astype(float),
                sub["est_1rm"].fillna(np.nan).astype(float),
            ], axis=1),
            hovertemplate="<b>%{text}</b><br>RSI mod: %{x:.3f}<br>1RM relativa: %{y:.2f} kg/kg<br>VMP: %{customdata[0]:.3f} m/s<br>Carga usada: %{customdata[1]:.1f} kg<br>Peso corporal: %{customdata[2]:.1f} kg<br>1RM estimada: %{customdata[3]:.1f} kg<extra></extra>",
        ))

    if pd.notna(rsi_ref):
        fig.add_vline(x=float(rsi_ref), line_width=2, line_color="#475467")
    if pd.notna(rel_ref):
        fig.add_hline(y=float(rel_ref), line_width=2, line_color="#475467")

    x_min = max(0.10, float(plot_df["RSI_mod"].min()) - 0.05)
    x_max = float(plot_df["RSI_mod"].max()) + 0.05
    y_min = max(0.4, float(plot_df["est_1rm_rel"].min()) - 0.15)
    y_max = float(plot_df["est_1rm_rel"].max()) + 0.15
    fig.update_xaxes(range=[x_min, x_max], title="RSI modificado")
    fig.update_yaxes(range=[y_min, y_max], title="1RM relativa estimada (kg/kg)")

    if pd.notna(rsi_ref) and pd.notna(rel_ref):
        fig.add_annotation(x=x_max, y=y_max, text="AVIÓN", showarrow=False, xanchor="right", yanchor="top", font=dict(size=13, color=FORCE_PROFILE_COLORS["Avión"]))
        fig.add_annotation(x=x_min, y=y_max, text="TANQUE", showarrow=False, xanchor="left", yanchor="top", font=dict(size=13, color=FORCE_PROFILE_COLORS["Tanque"]))
        fig.add_annotation(x=x_max, y=y_min, text="ELÁSTICO", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=13, color=FORCE_PROFILE_COLORS["Elástico"]))
        fig.add_annotation(x=x_min, y=y_min, text="BASE POR DESARROLLAR", showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=12, color=FORCE_PROFILE_COLORS["Base por desarrollar"]))
    fig.update_layout(title="RSI modificado vs 1RM relativa estimada", height=560, margin=dict(l=10, r=10, t=45, b=10), legend_title="Perfil")
    return fig



def render_team_summary_tiles(team_summary):
    if team_summary.empty:
        return
    top_score = team_summary.sort_values("Score F-R", ascending=False).iloc[0]
    top_balance = team_summary[team_summary["Equilibrio"].isin(["Equilibrado alto","Equilibrado medio"])]
    top_balance_name = top_balance.iloc[0]["Jugador"] if not top_balance.empty else "—"
    top_priority = team_summary["Prioridad"].value_counts().idxmax() if not team_summary["Prioridad"].empty else "—"
    mean_score = team_summary["Score F-R"].dropna().mean() if "Score F-R" in team_summary.columns else np.nan
    html = f"""
    <div class="summary-strip">
        <div class="summary-tile">
            <div class="icon">🏆</div>
            <div class="top">Mejor score global</div>
            <div class="big">{top_score['Jugador']}</div>
            <div class="sub">{top_score['Score F-R']:.0f}/100 · {top_score['Perfil']}</div>
        </div>
        <div class="summary-tile">
            <div class="icon">📈</div>
            <div class="top">Media del equipo</div>
            <div class="big">{mean_score:.0f}/100</div>
            <div class="sub">score fuerza-reactividad del grupo</div>
        </div>
        <div class="summary-tile">
            <div class="icon">⚖️</div>
            <div class="top">Equilibrio destacado</div>
            <div class="big">{top_balance_name}</div>
            <div class="sub">perfil más consistente del día</div>
        </div>
        <div class="summary-tile">
            <div class="icon">🎯</div>
            <div class="top">Prioridad dominante</div>
            <div class="big">{top_priority}</div>
            <div class="sub">tendencia principal del grupo</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def plot_team_priority_bar(team_summary):
    counts = team_summary["Prioridad"].value_counts().reset_index()
    counts.columns = ["Prioridad","Jugadores"]
    fig = px.bar(
        counts,
        x="Prioridad",
        y="Jugadores",
        text="Jugadores",
        title="Prioridades de trabajo del grupo"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=45,b=10), xaxis_title="", yaxis_title="Jugadores")
    return fig

def plot_balance_donut(team_summary):
    counts = team_summary["Equilibrio"].value_counts().reset_index()
    counts.columns = ["Equilibrio","Jugadores"]
    fig = px.pie(
        counts,
        names="Equilibrio",
        values="Jugadores",
        hole=0.55,
        title="Distribución del equilibrio del equipo"
    )
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=45,b=10), legend_title="")
    return fig

def style_team_summary(df):
    def color_priority(val):
        colors = {
            "Mantener":"background-color: rgba(22,163,74,0.16); color:#166534; font-weight:700;",
            "Ajustar":"background-color: rgba(59,130,246,0.14); color:#1D4ED8; font-weight:700;",
            "Potenciar reactividad":"background-color: rgba(124,58,237,0.14); color:#6D28D9; font-weight:700;",
            "Potenciar fuerza":"background-color: rgba(234,88,12,0.14); color:#C2410C; font-weight:700;",
            "Desarrollo global":"background-color: rgba(220,38,38,0.14); color:#B91C1C; font-weight:700;",
        }
        return colors.get(val, "")
    def color_profile(val):
        colors = {
            "Avión":"background-color: rgba(37,99,235,0.14); color:#1D4ED8; font-weight:700;",
            "Tanque":"background-color: rgba(124,58,237,0.14); color:#6D28D9; font-weight:700;",
            "Elástico":"background-color: rgba(5,150,105,0.14); color:#047857; font-weight:700;",
            "Base por desarrollar":"background-color: rgba(220,38,38,0.14); color:#B91C1C; font-weight:700;",
        }
        return colors.get(val, "")
    def color_balance(val):
        colors = {
            "Equilibrado alto":"background-color: rgba(22,163,74,0.14); color:#166534; font-weight:700;",
            "Equilibrado medio":"background-color: rgba(59,130,246,0.14); color:#1D4ED8; font-weight:700;",
            "Equilibrado bajo":"background-color: rgba(245,158,11,0.16); color:#B45309; font-weight:700;",
            "Desequilibrado reactivo":"background-color: rgba(124,58,237,0.14); color:#6D28D9; font-weight:700;",
            "Desequilibrado fuerza":"background-color: rgba(234,88,12,0.14); color:#C2410C; font-weight:700;",
        }
        return colors.get(val, "")
    styler = df.style.format({
        "Score F-R":"{:.1f}",
        "RSI mod":"{:.3f}",
        "1RM relativa (kg/kg)":"{:.2f}",
    })
    styler = styler.map(color_profile, subset=["Perfil"])
    styler = styler.map(color_balance, subset=["Equilibrio"])
    styler = styler.map(color_priority, subset=["Prioridad"])
    return styler

def page_force_reactivity(metrics_df):
    if metrics_df.empty:
        st.info("No hay datos disponibles.")
        return

    st.markdown('<div class="hero"><div style="font-size:0.92rem; opacity:0.9;">Perfil fuerza-reactividad</div><div style="font-size:2.05rem; font-weight:900; margin-top:0.15rem;">RSI mod vs 1RM relativa estimada</div><div style="font-size:1rem; opacity:0.92; margin-top:0.4rem;">Comparación de la reactividad y la fuerza relativa del equipo con perfiles por cuadrantes.</div></div>', unsafe_allow_html=True)

    dates = sorted(metrics_df["Fecha"].dropna().unique())
    if not dates:
        st.info("No hay fechas disponibles.")
        return
    opts = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates]
    selected_date = pd.to_datetime(st.selectbox("Fecha de análisis del perfil", opts, index=len(opts)-1, key="fr_date"))

    fr_df, profiles_df, profiles_err, rsi_ref, rel_ref = build_force_reactivity_df(metrics_df, selected_date)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Jugadores del día", fr_df["Jugador"].nunique(), selected_date.strftime("%Y-%m-%d"))
    valid = fr_df.dropna(subset=["RSI_mod", "est_1rm_rel"]).copy()
    with c2:
        kpi("Con perfil calculado", len(valid), "peso + carga completos")
    with c3:
        kpi("RSI referencia", f"{rsi_ref:.3f}" if pd.notna(rsi_ref) else "—", "mediana equipo")
    with c4:
        kpi("1RM relativa ref.", f"{rel_ref:.2f}" if pd.notna(rel_ref) else "—", "kg/kg mediana")

    if profiles_err:
        st.warning("La tabla de perfiles de jugador no está disponible todavía. Crea la tabla en Supabase con el SQL del desplegable inferior para poder guardar pesos y cargas.")
    with st.expander("SQL para crear la tabla de pesos y cargas en Supabase"):
        st.code(PLAYER_PROFILES_SQL, language="sql")

    missing_body = fr_df[fr_df["Peso_corporal"].isna()]["Jugador"].dropna().tolist() if "Peso_corporal" in fr_df.columns else []
    missing_load = fr_df[fr_df["Carga_sentadilla"].isna()]["Jugador"].dropna().tolist() if "Carga_sentadilla" in fr_df.columns else []
    if missing_body or missing_load:
        msg = []
        if missing_body:
            msg.append("sin peso corporal: " + ", ".join(missing_body))
        if missing_load:
            msg.append("sin carga de sentadilla: " + ", ".join(missing_load))
        st.info("Para que el perfil se calcule completo, faltan datos en: " + " · ".join(msg))

    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.markdown('<div class="premium-heading"><span class="dot"></span><span>Scatter colectivo</span></div>', unsafe_allow_html=True)
        filter_col1, filter_col2 = st.columns([1.5, 1])
        with filter_col1:
            selected_profiles = st.multiselect(
                "Filtrar por perfil",
                ["Avión", "Tanque", "Elástico", "Base por desarrollar"],
                default=["Avión", "Tanque", "Elástico", "Base por desarrollar"],
                key="fr_profiles_filter"
            )
        with filter_col2:
            show_names = st.toggle("Mostrar nombres", value=True, key="fr_show_names")

        st.markdown('<div class="premium-panel">', unsafe_allow_html=True)
        st.plotly_chart(
            plot_force_reactivity_filtered(fr_df, rsi_ref, rel_ref, selected_profiles=selected_profiles, show_names=show_names),
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if not valid.empty:
            valid["score_fr"] = valid.apply(lambda r: force_profile_score(r.get("RSI_mod"), r.get("est_1rm_rel"), rsi_ref, rel_ref), axis=1)
            team_summary = build_team_force_summary(valid, metrics_df, selected_date, rsi_ref, rel_ref)

            st.markdown('<div class="premium-heading"><span class="dot"></span><span>Panel colectivo del equipo</span></div>', unsafe_allow_html=True)
            sort_by = st.selectbox(
                "Ordenar tabla por",
                ["Score F-R", "Jugador", "Perfil", "Equilibrio", "Prioridad"],
                index=0,
                key="fr_sort_team"
            )
            ascending = sort_by in ["Jugador", "Perfil", "Equilibrio", "Prioridad"]
            render_team_summary_tiles(team_summary)

            cvis1, cvis2 = st.columns([1,1], gap="large")
            with cvis1:
                st.markdown('<div class="premium-panel">', unsafe_allow_html=True)
                st.plotly_chart(plot_balance_donut(team_summary), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with cvis2:
                st.markdown('<div class="premium-panel">', unsafe_allow_html=True)
                st.plotly_chart(plot_team_priority_bar(team_summary), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="premium-heading"><span class="dot"></span><span>Tabla-resumen visual</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="premium-panel">', unsafe_allow_html=True)
            st.dataframe(
                style_team_summary(team_summary.sort_values(sort_by, ascending=ascending)),
                use_container_width=True,
                hide_index=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="premium-heading"><span class="dot"></span><span>Lectura rápida del grupo</span></div>', unsafe_allow_html=True)
            balance_counts = team_summary["Equilibrio"].value_counts()
            priority_counts = team_summary["Prioridad"].value_counts()
            chips = []
            for label in ["Equilibrado alto", "Equilibrado medio", "Equilibrado bajo", "Desequilibrado reactivo", "Desequilibrado fuerza"]:
                n = int(balance_counts.get(label, 0))
                if n > 0:
                    color = "#166534" if "alto" in label else "#1D4ED8" if "medio" in label else "#B45309" if "bajo" in label else "#6D28D9" if "reactivo" in label else "#C2410C"
                    chips.append(f'<span class="quick-chip" style="background:{color};">{label}: {n}</span>')
            for label in ["Mantener", "Ajustar", "Potenciar reactividad", "Potenciar fuerza", "Desarrollo global"]:
                n = int(priority_counts.get(label, 0))
                if n > 0:
                    color = "#166534" if label=="Mantener" else "#1D4ED8" if label=="Ajustar" else "#6D28D9" if label=="Potenciar reactividad" else "#C2410C" if label=="Potenciar fuerza" else "#B91C1C"
                    chips.append(f'<span class="quick-chip" style="background:{color};">{label}: {n}</span>')
            st.markdown(f'<div class="premium-panel">{"".join(chips)}</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="premium-heading"><span class="dot"></span><span>Vista jugador-friendly</span></div>', unsafe_allow_html=True)
        if valid.empty:
            st.info("No hay suficientes datos para generar tarjetas individuales.")
        else:
            valid["score_fr"] = valid.apply(lambda r: force_profile_score(r.get("RSI_mod"), r.get("est_1rm_rel"), rsi_ref, rel_ref), axis=1)
            players = valid["Jugador"].dropna().astype(str).sort_values().unique().tolist()
            selected_player = st.selectbox("Selecciona un jugador", players, index=0, key="fr_player")
            row = valid[valid["Jugador"] == selected_player].iloc[0]
            render_force_profile_card(row, rsi_ref, rel_ref, metrics_df=metrics_df, selected_date=selected_date)

            st.markdown('<div class="premium-heading"><span class="dot"></span><span>Recomendación automática</span></div>', unsafe_allow_html=True)
            balance_label = classify_balance_level(row.get("RSI_mod"), row.get("est_1rm_rel"), rsi_ref, rel_ref)
            st.markdown(f'<div class="soft-note">Prioridad principal para <b>{selected_player}</b>: <b>{action_priority_label(row["perfil_fr"], balance_label)}</b>.</div>', unsafe_allow_html=True)

            st.markdown('<div class="premium-heading"><span class="dot"></span><span>Rankings rápidos</span></div>', unsafe_allow_html=True)
            rr1, rr2 = st.columns(2)
            with rr1:
                top_global = valid.sort_values("score_fr", ascending=False)[["Jugador","score_fr"]].head(3).copy()
                st.markdown("**Top 3 score global**")
                for i, (_, rr) in enumerate(top_global.iterrows(), start=1):
                    st.markdown(f"{i}. **{rr['Jugador']}** · {rr['score_fr']:.0f}/100")
            with rr2:
                team_summary = build_team_force_summary(valid, metrics_df, selected_date, rsi_ref, rel_ref)
                top_bal = team_summary[team_summary["Equilibrio"].isin(["Equilibrado alto","Equilibrado medio"])].copy()
                top_bal = top_bal.sort_values(["Equilibrio","Score F-R"], ascending=[True, False]).head(3)
                st.markdown("**Perfiles equilibrados**")
                if top_bal.empty:
                    st.markdown("Sin jugadores equilibrados con datos suficientes.")
                else:
                    for i, (_, rr) in enumerate(top_bal.iterrows(), start=1):
                        st.markdown(f"{i}. **{rr['Jugador']}** · {rr['Equilibrio']}")

            st.markdown('<div class="premium-heading"><span class="dot"></span><span>Cómo leer el perfil</span></div>', unsafe_allow_html=True)
            st.markdown(
                "- **Avión**: alto RSI mod y alta fuerza relativa.\n"
                "- **Tanque**: buena fuerza relativa, pero menor componente reactivo.\n"
                "- **Elástico**: buena reactividad, pero menor base de fuerza relativa.\n"
                "- **Base por desarrollar**: margen claro en ambas dimensiones."
            )

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

def parse_tidy(df_raw, forced_date=None):
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
        low = str(c).lower().strip()
        if low in ["fecha","date"]:
            rename[c] = "Fecha"
        elif low in ["jugador","player","nombre"]:
            rename[c] = "Jugador"
        elif "pos" in low:
            rename[c] = "Posicion"
        elif "min" in low:
            rename[c] = "Minutos"
        elif "cmj" in low and "post" in low:
            rename[c] = "CMJ_post"
        elif ("rsi" in low or "rsi mod" in low or "rsi_mod" in low) and "post" in low:
            rename[c] = "RSI_mod_post"
        elif "cmj" in low:
            rename[c] = "CMJ"
        elif "rsi" in low:
            rename[c] = "RSI_mod"
        elif "vmp" in low:
            rename[c] = "VMP"
        elif "srpe" in low or "s-rpe" in low or low == "rpe":
            rename[c] = "sRPE"
        elif "obs" in low:
            rename[c] = "Observaciones"
    df = df.rename(columns=rename)

    needed = ["Jugador","CMJ","RSI_mod"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    if "Fecha" not in df.columns:
        if forced_date is None:
            raise ValueError("Falta la columna 'Fecha'. Selecciona una fecha antes de subir el archivo.")
        df["Fecha"] = pd.to_datetime(forced_date)

    for optional in ["Posicion","Minutos","Observaciones","sRPE","VMP","CMJ_post","RSI_mod_post"]:
        if optional not in df.columns:
            df[optional] = np.nan

    df = df[["Fecha","Jugador","Posicion","Minutos","CMJ","RSI_mod","CMJ_post","RSI_mod_post","VMP","sRPE","Observaciones"]].copy()

    if forced_date is not None:
        df["Fecha"] = pd.to_datetime(forced_date)
    else:
        df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True, errors="coerce")

    df["Jugador"] = df["Jugador"].apply(std_name)
    for c in ["Minutos", *ALL_METRICS, "CMJ_post", "RSI_mod_post"]:
        df[c] = df[c].apply(safe_num)
    return df.dropna(subset=["Fecha","Jugador"]).drop_duplicates(subset=["Fecha","Jugador"], keep="last")

def parse_block(df_raw):
    df = df_raw.copy()
    df.columns = range(df.shape[1])
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # =====================================================
    # FORMATO BLOQUE PRE/POST
    # NOMBRE | VARIABLES | PRE | POST
    # =====================================================
    header = [str(v).strip().lower() if pd.notna(v) else "" for v in df.iloc[0].tolist()] if len(df) else []
    if header and any("nombre" in x for x in header) and any("variable" in x for x in header) and any(x == "pre" for x in header) and any(x == "post" for x in header):
        col_name = next((i for i, x in enumerate(header) if "nombre" in x), 0)
        col_var = next((i for i, x in enumerate(header) if "variable" in x), 1)
        col_pre = next((i for i, x in enumerate(header) if x == "pre"), 2)
        col_post = next((i for i, x in enumerate(header) if x == "post"), 3)

        records = []
        current_player = None
        bucket = {
            "CMJ": np.nan,
            "RSI_mod": np.nan,
            "CMJ_post": np.nan,
            "RSI_mod_post": np.nan,
            "VMP": np.nan,
            "sRPE": np.nan,
        }

        def flush_player(player_name, data_bucket):
            if player_name is None:
                return
            if all(pd.isna(data_bucket[k]) for k in ["CMJ", "RSI_mod", "CMJ_post", "RSI_mod_post", "VMP", "sRPE"]):
                return
            records.append({
                "Fecha": pd.NaT,
                "Jugador": std_name(player_name),
                "Posicion": np.nan,
                "Minutos": np.nan,
                "CMJ": data_bucket["CMJ"],
                "RSI_mod": data_bucket["RSI_mod"],
                "CMJ_post": data_bucket["CMJ_post"],
                "RSI_mod_post": data_bucket["RSI_mod_post"],
                "VMP": data_bucket["VMP"],
                "sRPE": data_bucket["sRPE"],
                "Observaciones": np.nan,
            })

        for i in range(1, len(df)):
            player_cell = df.iat[i, col_name] if col_name < df.shape[1] else np.nan
            var_cell = df.iat[i, col_var] if col_var < df.shape[1] else np.nan
            pre_cell = df.iat[i, col_pre] if col_pre < df.shape[1] else np.nan
            post_cell = df.iat[i, col_post] if col_post < df.shape[1] else np.nan

            if pd.notna(player_cell) and str(player_cell).strip() != "":
                flush_player(current_player, bucket)
                current_player = str(player_cell).strip()
                bucket = {
                    "CMJ": np.nan,
                    "RSI_mod": np.nan,
                    "CMJ_post": np.nan,
                    "RSI_mod_post": np.nan,
                    "VMP": np.nan,
                    "sRPE": np.nan,
                }

            var_txt = str(var_cell).strip().lower() if pd.notna(var_cell) else ""
            pre_val = safe_num(pre_cell)
            post_val = safe_num(post_cell)

            if "cmj" in var_txt:
                bucket["CMJ"] = pre_val
                bucket["CMJ_post"] = post_val
            elif "rsi" in var_txt:
                bucket["RSI_mod"] = pre_val
                bucket["RSI_mod_post"] = post_val
            elif "vmp" in var_txt:
                bucket["VMP"] = pre_val
            elif "rpe" in var_txt:
                bucket["sRPE"] = pre_val

        flush_player(current_player, bucket)
        out = pd.DataFrame(records)
        if not out.empty:
            return out.drop_duplicates(subset=["Jugador"], keep="last")

    # =====================================================
    # FORMATOS BLOQUE HISTÓRICOS ANTIGUOS
    # =====================================================
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
                        records.append({"Fecha": d, "Jugador": player_name, "Posicion": np.nan, "Minutos": np.nan, "CMJ": cmj, "RSI_mod": rsi, "CMJ_post": np.nan, "RSI_mod_post": np.nan, "VMP": vmp, "sRPE": srpe, "Observaciones": np.nan})
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
                records.append({"Fecha": current_date, "Jugador": current_player, "Posicion": np.nan, "Minutos": np.nan, "CMJ": nums[0], "RSI_mod": nums[1], "CMJ_post": np.nan, "RSI_mod_post": np.nan, "VMP": nums[2], "sRPE": nums[3], "Observaciones": np.nan})
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
                records.append({"Fecha": current_date, "Jugador": current_player, "Posicion": np.nan, "Minutos": np.nan, "CMJ": nums[0], "RSI_mod": nums[1], "CMJ_post": np.nan, "RSI_mod_post": np.nan, "VMP": nums[2], "sRPE": nums[3], "Observaciones": np.nan})
                i += 5
                continue
        i += 1

    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("No se pudo interpretar el formato bloque del archivo.")
    return out.drop_duplicates(subset=["Fecha","Jugador"], keep="last")

def parse_uploaded(uploaded_file, forced_date=None):
    df_raw = read_uploaded(uploaded_file)
    fmt = detect_format(df_raw)
    if fmt == "tidy":
        parsed = parse_tidy(df_raw, forced_date=forced_date)
    elif fmt == "block":
        parsed = parse_block(df_raw)
        if forced_date is not None:
            parsed["Fecha"] = pd.to_datetime(forced_date)
    else:
        raise ValueError("No se pudo detectar el formato del archivo.")
    if "CMJ_post" not in parsed.columns:
        parsed["CMJ_post"] = np.nan
    if "RSI_mod_post" not in parsed.columns:
        parsed["RSI_mod_post"] = np.nan
    if forced_date is not None:
        parsed["Fecha"] = pd.to_datetime(forced_date)
    return parsed

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



def compute_pre_post_fields(df):
    df = df.copy()
    for metric in ["CMJ", "RSI_mod"]:
        post_col = f"{metric}_post"
        if metric not in df.columns:
            df[metric] = np.nan
        if post_col not in df.columns:
            df[post_col] = np.nan

        df[f"{metric}_delta_abs"] = np.where(
            df[metric].notna() & df[post_col].notna(),
            df[post_col] - df[metric],
            np.nan,
        )
        df[f"{metric}_delta_pct"] = np.where(
            df[metric].notna() & (df[metric] != 0) & df[post_col].notna(),
            (df[post_col] - df[metric]) / df[metric] * 100,
            np.nan,
        )
    return df

def render_pre_post_cards(row):
    cards = []
    for metric, label, suffix, dec in [("CMJ","CMJ"," cm",1),("RSI_mod","RSI mod","",3)]:
        pre = row.get(metric, np.nan)
        post = row.get(f"{metric}_post", np.nan)
        delta_abs = row.get(f"{metric}_delta_abs", np.nan)
        delta_pct = row.get(f"{metric}_delta_pct", np.nan)
        if pd.isna(delta_pct):
            trend_class = "result-neutral"; trend_text = "Sin dato post"
        elif delta_pct > 0:
            trend_class = "result-up"; trend_text = f"{delta_pct:+.1f}%"
        elif delta_pct < 0:
            trend_class = "result-down"; trend_text = f"{delta_pct:+.1f}%"
        else:
            trend_class = "result-neutral"; trend_text = f"{delta_pct:+.1f}%"
        pre_txt = "—" if pd.isna(pre) else f"{pre:.{dec}f}{suffix}"
        post_txt = "—" if pd.isna(post) else f"{post:.{dec}f}{suffix}"
        delta_abs_txt = "—" if pd.isna(delta_abs) else f"{delta_abs:+.{dec}f}{suffix}"
        cards.append(
            f'<div class="prepost-card">'
            f'<div class="lab">{label} · PRE vs POST</div>'
            f'<div class="big">{pre_txt} → {post_txt}</div>'
            f'<div class="mini"><span class="{trend_class}">{trend_text} intra-sesión</span></div>'
            f'<div class="mini">cambio absoluto: <b>{delta_abs_txt}</b></div>'
            f'</div>'
        )
    st.markdown(f'<div class="prepost-grid">{"".join(cards)}</div>', unsafe_allow_html=True)

def plot_pre_post_current(row):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("CMJ", "RSI mod"))

    cmj_pre = row.get("CMJ", np.nan)
    cmj_post = row.get("CMJ_post", np.nan)
    rsi_pre = row.get("RSI_mod", np.nan)
    rsi_post = row.get("RSI_mod_post", np.nan)

    fig.add_trace(go.Bar(x=["PRE","POST"], y=[cmj_pre, cmj_post]), row=1, col=1)
    fig.add_trace(go.Bar(x=["PRE","POST"], y=[rsi_pre, rsi_post]), row=1, col=2)

    fig.update_layout(height=340)
    return fig

def plot_delta_timeline(player_df, metric, selected_date):
    fig = go.Figure()
    y = player_df.get(f"{metric}_delta_pct", pd.Series(index=player_df.index, dtype=float))
    fig.add_trace(go.Scatter(x=player_df["Fecha"], y=y, mode="lines+markers", name="Δ % post-pre", line=dict(width=3)))
    fig.add_hline(y=0, line_dash="dot")
    sel = player_df[player_df["Fecha"].dt.normalize() == pd.to_datetime(selected_date).normalize()]
    if not sel.empty:
        val = sel.iloc[-1].get(f"{metric}_delta_pct", np.nan)
        fig.add_trace(go.Scatter(x=[sel.iloc[-1]["Fecha"]], y=[val], mode="markers", name="Fecha", marker=dict(size=12, color="#C62828", symbol="diamond")))
    fig.update_layout(title=f"{LABELS.get(metric, metric)} · Δ % POST vs PRE", height=300, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_team_pre_post_delta(team_df):
    rows = []
    for metric in ["CMJ","RSI_mod"]:
        col = f"{metric}_delta_pct"
        vals = pd.to_numeric(team_df[col], errors="coerce").dropna() if col in team_df.columns else pd.Series(dtype=float)
        rows.append({"Métrica": LABELS.get(metric, metric), "Delta_pct": vals.mean() if not vals.empty else np.nan})
    temp = pd.DataFrame(rows)
    fig = px.bar(temp, x="Métrica", y="Delta_pct", text="Delta_pct", title="Respuesta media PRE → POST del equipo")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=10), yaxis_title="%")
    return fig
def progressive_filtered_baseline(group, metric):
    out = []
    full_mean = pd.to_numeric(group[metric], errors="coerce").mean()
    has_micro = "Microciclo" in group.columns
    for i in range(len(group)):
        prev = group.iloc[:i].copy()
        if has_micro:
            prev = prev[prev["Microciclo"].isin(VALID_BASELINE_DAYS)]
        vals = pd.to_numeric(prev[metric], errors="coerce").dropna()
        if len(vals) > 0:
            out.append(vals.mean())
        else:
            out.append(full_mean)
    return pd.Series(out, index=group.index)

def progressive_ma3_by_cycle(group, metric):
    if "Microciclo" not in group.columns:
        return pd.to_numeric(group[metric], errors="coerce").rolling(window=3, min_periods=1).mean()
    out = pd.Series(index=group.index, dtype=float)
    for cycle, sub in group.groupby("Microciclo", sort=False):
        vals = pd.to_numeric(sub[metric], errors="coerce").rolling(window=3, min_periods=1).mean()
        out.loc[sub.index] = vals.values
    return out

def compute_metrics(df):
    if df.empty:
        return df.copy()
    sort_cols = ["Jugador","Fecha"] + (["Microciclo"] if "Microciclo" in df.columns else [])
    df = df.copy().sort_values(sort_cols).reset_index(drop=True)

    for metric in ALL_METRICS:
        df[f"{metric}_baseline"] = (
            df.groupby("Jugador", group_keys=False)
              .apply(lambda g: progressive_filtered_baseline(g, metric))
              .reset_index(level=0, drop=True)
        )

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

        df[f"{metric}_ma3"] = (
            df.groupby("Jugador", group_keys=False)
              .apply(lambda g: progressive_ma3_by_cycle(g, metric))
              .reset_index(level=0, drop=True)
        )

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
    df = compute_pre_post_fields(df)
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
    df = compute_pre_post_fields(df)
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
        return f"Reducir claramente la exigencia neuromuscular en esta sesión. Tendencia: {trend.lower()}."
    if risk in ["Fatiga moderada","Fatiga moderada-alta"]:
        return f"Conviene controlar volumen e intensidad y vigilar la respuesta en el calentamiento. Tendencia: {trend.lower()}."
    if risk in ["Fatiga leve","Fatiga leve-moderada"]:
        return f"Hay una pérdida objetiva leve/moderada; prioriza una sesión conservadora y sin estímulos residuales. Tendencia: {trend.lower()}."
    return f"Estado compatible con normalidad funcional para esta sesión. Tendencia: {trend.lower()}."

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
        return "El grupo presenta una señal colectiva alta de pérdida de rendimiento en la sesión analizada. Conviene minimizar la carga neuromuscular y priorizar frescura."
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

def radar_relative_loss(row):
    labels = [LABELS[m] for m in OBJECTIVE_METRICS]
    losses = []
    hover_txt = []
    for m in OBJECTIVE_METRICS:
        pct = row.get(f"{m}_pct_vs_baseline", np.nan)
        if pd.isna(pct):
            loss = 0
            hover_txt.append(f"{LABELS[m]}: sin referencia")
        else:
            loss = max(0, -float(pct))
            hover_txt.append(f"{LABELS[m]}: {pct:.1f}% vs baseline")
        losses.append(min(loss, 15))
    labels_closed = labels + [labels[0]]
    losses_closed = losses + [losses[0]]
    hover_closed = hover_txt + [hover_txt[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[2.5, 2.5, 2.5, 2.5],
        theta=labels_closed,
        mode="lines",
        line=dict(color="rgba(22,163,74,0.55)", dash="dot", width=2),
        name="Umbral leve"
    ))
    fig.add_trace(go.Scatterpolar(
        r=[5, 5, 5, 5],
        theta=labels_closed,
        mode="lines",
        line=dict(color="rgba(227,160,8,0.60)", dash="dot", width=2),
        name="Umbral moderado"
    ))
    fig.add_trace(go.Scatterpolar(
        r=[10, 10, 10, 10],
        theta=labels_closed,
        mode="lines",
        line=dict(color="rgba(249,115,22,0.65)", dash="dot", width=2),
        name="Umbral crítico"
    ))
    fig.add_trace(go.Scatterpolar(
        r=losses_closed,
        theta=labels_closed,
        fill="toself",
        line=dict(color="#C62828", width=3),
        fillcolor="rgba(198,40,40,0.28)",
        name="Pérdida relativa",
        text=hover_closed,
        hovertemplate="%{text}<extra></extra>"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 15],
                tickvals=[0, 2.5, 5, 10, 15],
                ticktext=["0", "2.5", "5", "10", "15+"]
            )
        ),
        title="Radar de pérdida relativa (%)",
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    )
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
    temp = temp[temp["N"] > 0]
    fig = px.bar(
        temp, y="Estado", x="N", orientation="h", color="Estado",
        color_discrete_map=RISK_COLORS, title="Distribución del riesgo"
    )
    fig.update_layout(
        height=330, margin=dict(l=40,r=20,t=70,b=30), showlegend=False, title_x=0.5
    )
    fig.update_yaxes(automargin=True, title="")
    fig.update_xaxes(automargin=True, title="N")
    return fig

def plot_team_objective_bar(team_df):
    temp = team_df[["Jugador","objective_loss_score","objective_loss_mean_pct","risk_label"]].copy().sort_values(
        ["objective_loss_score","objective_loss_mean_pct"], ascending=[True, True]
    )
    fig = px.bar(
        temp, y="Jugador", x="objective_loss_score", orientation="h",
        color="risk_label", color_discrete_map=RISK_COLORS,
        text="objective_loss_score", title="Objective loss score por jugador",
        hover_data=["objective_loss_mean_pct"]
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig.update_layout(
        height=max(420, 24 * len(temp) + 160),
        margin=dict(l=90,r=30,t=70,b=30),
        showlegend=False, title_x=0.5
    )
    fig.update_yaxes(automargin=True, title="")
    fig.update_xaxes(automargin=True, title="objective_loss_score")
    return fig

def plot_team_score_trend(df):
    temp = df.groupby("Fecha").agg(
        objective_loss_score=("objective_loss_score", "mean"),
        readiness_score=("readiness_score", "mean"),
        objective_z_score=("objective_z_score", "mean"),
    ).reset_index().sort_values("Fecha")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=temp["Fecha"],
            y=temp["objective_loss_score"],
            mode="lines+markers",
            name="Loss equipo",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=temp["Fecha"],
            y=temp["readiness_score"],
            mode="lines+markers",
            name="Readiness equipo",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Evolución global del grupo",
        height=340,
        margin=dict(l=30, r=30, t=70, b=40),
        title_x=0.5,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(title_text="Loss", secondary_y=False, automargin=True)
    fig.update_yaxes(title_text="Readiness", secondary_y=True, automargin=True)
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

def fig_to_rl_image(fig, width_cm=16, height_cm=8.5):
    try:
        img_bytes = fig.to_image(format="png", scale=2)
        bio = io.BytesIO(img_bytes)
        return Image(bio, width=width_cm * cm, height=height_cm * cm)
    except Exception:
        return None

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
    .loss-wrap {background:#E5E7EB;border-radius:999px;height:14px;overflow:hidden;min-width:140px;}
    .loss-bar {height:14px;border-radius:999px;}
    .muted {color:#667085;font-size:12px;}
    </style>
    """

def html_risk_badge(label):
    color = RISK_COLORS.get(label, "#475467")
    return f'<span class="badge" style="background:{color};">{label}</span>'

def html_loss_bar(score):
    if pd.isna(score):
        return '<span class="muted">NA</span>'
    color = "#16A34A"
    if score >= 2.5:
        color = "#B91C1C"
    elif score >= 1.5:
        color = "#F97316"
    elif score >= 0.5:
        color = "#E3A008"
    width = max(2, min(100, float(score) / 3 * 100))
    return f'<div class="loss-wrap"><div class="loss-bar" style="width:{width}%;background:{color};"></div></div><div class="muted" style="margin-top:4px;">{score:.2f}</div>'

def coach_session_html(team_day, selected_date):
    rows = ""
    temp = team_day.copy().sort_values(["objective_loss_score","objective_loss_mean_pct"], ascending=[False, True])
    for _, r in temp.iterrows():
        rows += f"<tr><td>{r['Jugador']}</td><td>{html_risk_badge(r['risk_label'])}</td><td>{r['CMJ_pct_vs_baseline']:.1f}%</td><td>{r['RSI_mod_pct_vs_baseline']:.1f}%</td><td>{r['VMP_pct_vs_baseline']:.1f}%</td><td>{r['objective_loss_mean_pct']:.1f}%</td><td>{html_loss_bar(r['objective_loss_score'])}</td></tr>"
    bar_html = plotly_html(plot_team_objective_bar(team_day))
    risk_html = plotly_html(plot_team_risk_distribution(team_day))
    return f"""
    <html><head><meta charset="utf-8">{report_css()}</head><body>
    <div class="hero"><div style="font-size:12px;opacity:0.9;">Informe de sesión · MD-1</div><div style="font-size:32px;font-weight:900;line-height:1.15;">Estado neuromuscular del equipo</div><div style="font-size:15px;margin-top:6px;">Fecha analizada: {selected_date}</div></div>
    <div class="cards">
      <div class="card"><div class="label">Jugadores evaluados</div><div class="value">{team_day['Jugador'].nunique()}</div></div>
      <div class="card"><div class="label">Objective loss medio</div><div class="value">{team_day['objective_loss_score'].mean():.2f}</div></div>
      <div class="card"><div class="label">Pérdida media %</div><div class="value">{team_day['objective_loss_mean_pct'].mean():.1f}%</div></div>
      <div class="card"><div class="label">Readiness media</div><div class="value">{team_day['readiness_score'].mean():.0f}</div></div>
    </div>
    <div class="section"><div class="title">Lectura rápida para el entrenador</div><div class="diag">{team_interpretation(team_day)}</div></div>
    <div class="section"><div class="title">Panel visual</div><div class="grid2">{risk_html}{bar_html}</div></div>
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
    snapshot_html = plotly_html(radar_relative_loss(row))
    timeline_html = plotly_html(plot_objective_timeline(player_df, row['Fecha']))
    return f"""
    <html><head><meta charset="utf-8">{report_css()}</head><body>
    <div class="hero"><div style="font-size:12px;opacity:0.9;">Informe individual · Sesión específica</div><div style="font-size:32px;font-weight:900;line-height:1.15;">{row['Jugador']}</div><div style="font-size:15px;margin-top:6px;">Fecha: {pd.to_datetime(row['Fecha']).date()}</div></div>
    <div class="cards">
      <div class="card"><div class="label">Riesgo</div><div class="value" style="font-size:22px;">{row['risk_label']}</div></div>
      <div class="card"><div class="label">Loss score</div><div class="value">{row['objective_loss_score']:.2f}</div></div>
      <div class="card"><div class="label">Pérdida media %</div><div class="value">{row['objective_loss_mean_pct']:.1f}%</div></div>
      <div class="card"><div class="label">Readiness</div><div class="value">{row['readiness_score']:.0f}</div></div>
    </div>
    <div class="section"><div class="title">Diagnóstico staff</div><div class="diag"><p><b>Perfil principal:</b> {main}.</p><p><b>Patrón:</b> {pattern}.</p><p><b>Variable dominante:</b> {LABELS.get(worst_metric, 'NA')} ({'NA' if worst_value is None else f'{worst_value:.1f}%'}).</p><p>{player_comment(row)}</p></div></div>
    <div class="section"><div class="title">Flags automáticos</div><ul>{flags_html}</ul></div>
    <div class="section"><div class="title">Panel visual</div><div class="grid2">{radar_html}{snapshot_html}</div><div class="grid1">{timeline_html}</div></div>
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
    timeline_html = plotly_html(radar_relative_loss(latest))
    cmj_html = plotly_html(plot_metric_pct(player_df, "CMJ", latest["Fecha"]))
    rsi_html = plotly_html(plot_metric_pct(player_df, "RSI_mod", latest["Fecha"]))
    vmp_html = plotly_html(plot_metric_pct(player_df, "VMP", latest["Fecha"]))

    return f"""
    <html><head><meta charset="utf-8">{report_css()}</head><body>
    <div class="hero"><div style="font-size:12px;opacity:0.9;">Informe individual · Evolución anual</div><div style="font-size:32px;font-weight:900;line-height:1.15;">{player}</div><div style="font-size:15px;margin-top:6px;">Registros: {len(player_df)} · Última fecha: {pd.to_datetime(latest['Fecha']).date()}</div></div>
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
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=1.0*cm, rightMargin=1.0*cm, topMargin=1.0*cm, bottomMargin=1.0*cm)
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
        ["Riesgo", str(row["risk_label"]), "Readiness", f"{row['readiness_score']:.0f}"],
        ["Loss score", f"{row['objective_loss_score']:.2f}", "Pérdida media %", f"{row['objective_loss_mean_pct']:.1f}%"],
        ["Z-score objetivo", "NA" if pd.isna(row["objective_z_score"]) else f"{row['objective_z_score']:.2f}", "Tendencia", str(row["trend_label"])],
        ["Perfil", main, "Patrón", pattern],
        ["Variable dominante", f"{LABELS.get(worst_metric, 'NA')} ({'NA' if worst_value is None else f'{worst_value:.1f}%'})", "", ""],
    ]
    t = Table(diag, colWidths=[3.4*cm, 5.0*cm, 3.4*cm, 5.0*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.whitesmoke),("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold")]))
    elems.append(t)
    elems.append(Spacer(1, 0.25*cm))
    elems.append(Paragraph("Flags automáticos", styles["SectionDark"]))
    for f in (flags or ["Sin flags adicionales"]):
        elems.append(Paragraph(f"• {f}", styles["BodyText"]))
    elems.append(Spacer(1, 0.2*cm))
    for fig in [radar_current_vs_baseline(row), radar_relative_loss(row)]:
        rl_img = fig_to_rl_image(fig, width_cm=11.8, height_cm=6.8)
        if rl_img is not None:
            elems.append(rl_img)
    rl_img = fig_to_rl_image(plot_objective_timeline(player_df, row["Fecha"]), width_cm=24.0, height_cm=7.2)
    if rl_img is not None:
        elems.append(Spacer(1, 0.2*cm))
        elems.append(rl_img)
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
        ["Riesgo actual", str(latest["risk_label"]), "Readiness", f"{latest['readiness_score']:.0f}"],
        ["Loss score", f"{latest['objective_loss_score']:.2f}", "Pérdida media %", f"{latest['objective_loss_mean_pct']:.1f}%"],
        ["Z-score objetivo", "NA" if pd.isna(latest["objective_z_score"]) else f"{latest['objective_z_score']:.2f}", "Tendencia", str(latest["trend_label"])],
        ["Perfil", main, "Patrón", pattern],
        ["Variable dominante", f"{LABELS.get(worst_metric, 'NA')} ({'NA' if worst_value is None else f'{worst_value:.1f}%'})", "", ""],
    ]
    t = Table(diag, colWidths=[3.4*cm, 5.0*cm, 3.4*cm, 5.0*cm])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,-1),colors.whitesmoke),("GRID",(0,0),(-1,-1),0.3,colors.lightgrey),("FONTNAME",(0,0),(-1,-1),"Helvetica-Bold")]))
    elems.append(t)
    elems.append(Spacer(1, 0.2*cm))
    for f in (flags or ["Sin flags adicionales"]):
        elems.append(Paragraph(f"• {f}", styles["BodyText"]))
    elems.append(Spacer(1, 0.2*cm))
    for fig in [radar_current_vs_baseline(latest), radar_relative_loss(latest), plot_metric_pct(player_df, "CMJ", latest["Fecha"]), plot_metric_pct(player_df, "RSI_mod", latest["Fecha"]), plot_metric_pct(player_df, "VMP", latest["Fecha"])]:
        rl_img = fig_to_rl_image(fig, width_cm=11.8, height_cm=6.5)
        if rl_img is not None:
            elems.append(rl_img)
            elems.append(Spacer(1, 0.12*cm))
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
    elems.append(Spacer(1, 0.2*cm))
    for fig in [plot_team_risk_distribution(team_day), plot_team_objective_bar(team_day)]:
        rl_img = fig_to_rl_image(fig, width_cm=12.0, height_cm=7.0)
        if rl_img is not None:
            elems.append(rl_img)
            elems.append(Spacer(1, 0.15*cm))
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
    st.markdown("### CARGAR SESIÓN")
    c1, c2 = st.columns(2)
    with c1:
        fecha_sesion = st.date_input(
            "Selecciona la fecha a la que corresponde el archivo",
            value=pd.Timestamp.today().date(),
            format="DD/MM/YYYY"
        )
    with c2:
        md_label = st.selectbox("Selecciona el día del microciclo", MICROCYCLE_OPTIONS, index=MICROCYCLE_OPTIONS.index("MD-1"))

    st.caption("MD+1 y MD+2 se guardan y analizan, pero no cuentan para el baseline funcional. El baseline se construye con MD-4 a MD-1. Si tu Excel incluye columnas PRE y POST de CMJ y RSI mod, ambas se guardarán dentro de la misma sesión.")
    uploaded = st.file_uploader("Sube tu Excel/CSV de la sesión", type=["xlsx","xls","csv"])

    if uploaded is not None:
        try:
            parsed = parse_uploaded(uploaded, forced_date=fecha_sesion)
            parsed["Microciclo"] = md_label
            if md_label != "MD-1":
                parsed["VMP"] = np.nan
            st.success(
                f"Archivo interpretado correctamente: {parsed['Jugador'].nunique()} jugadores · "
                f"fecha asignada: {pd.to_datetime(fecha_sesion).strftime('%Y-%m-%d')} · "
                f"microciclo: {md_label}"
            )
            st.dataframe(parsed, use_container_width=True, hide_index=True)
            if st.button("Guardar en base de datos", type="primary"):
                try:
                    upsert_monitoring(parsed)
                    st.success("Datos guardados correctamente.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar en Supabase: {e}")
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
    with c1: kpi("Fecha", selected_date.strftime("%Y-%m-%d"), f"{team_day['Jugador'].nunique()} jugadores · {team_day['Microciclo'].dropna().iloc[0] if 'Microciclo' in team_day.columns and team_day['Microciclo'].notna().any() else 'NA'}")
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

    st.markdown("### Respuesta intra-sesión · PRE vs POST")
    p1,p2,p3,p4 = st.columns(4)
    with p1: kpi("CMJ Δ medio", "NA" if team_day["CMJ_delta_pct"].dropna().empty else f"{team_day['CMJ_delta_pct'].mean():+.1f}%", "post vs pre")
    with p2: kpi("RSI Δ medio", "NA" if team_day["RSI_mod_delta_pct"].dropna().empty else f"{team_day['RSI_mod_delta_pct'].mean():+.1f}%", "post vs pre")
    with p3: kpi("CMJ post disponibles", int(team_day["CMJ_post"].notna().sum()) if "CMJ_post" in team_day.columns else 0, "jugadores")
    with p4: kpi("RSI post disponibles", int(team_day["RSI_mod_post"].notna().sum()) if "RSI_mod_post" in team_day.columns else 0, "jugadores")
    st.plotly_chart(plot_team_pre_post_delta(team_day), use_container_width=True)

    table = team_day[["Jugador","CMJ_pct_vs_baseline","RSI_mod_pct_vs_baseline","VMP_pct_vs_baseline","CMJ_delta_pct","RSI_mod_delta_pct","objective_loss_mean_pct","objective_loss_score","objective_z_score","readiness_score","risk_label","trend_label"]].copy()
    table.columns = ["Jugador","CMJ %","RSI mod %","VMP %","CMJ Δ post-pre %","RSI Δ post-pre %","Pérdida media %","Loss score","Z-score objetivo","Readiness","Riesgo","Tendencia"]
    for col in ["CMJ %","RSI mod %","VMP %","CMJ Δ post-pre %","RSI Δ post-pre %","Pérdida media %","Loss score","Z-score objetivo","Readiness"]:
        table[col] = table[col].round(2)
    st.dataframe(table.sort_values(["Loss score","Pérdida media %"], ascending=[False, True]), use_container_width=True, hide_index=True)


def player_results_reference(player_df, metric):
    vals = pd.to_numeric(player_df[metric], errors="coerce").dropna()
    if vals.empty:
        return {"current": np.nan, "baseline": np.nan, "initial": np.nan, "best": np.nan}
    current = vals.iloc[-1]
    baseline_col = f"{metric}_baseline"
    baseline = pd.to_numeric(player_df[baseline_col], errors="coerce").iloc[-1] if baseline_col in player_df.columns else np.nan
    initial = vals.iloc[0]
    best = vals.max()
    return {"current": current, "baseline": baseline, "initial": initial, "best": best}

def pct_change_safe(current, ref):
    if pd.isna(current) or pd.isna(ref) or ref == 0:
        return np.nan
    return (current - ref) / ref * 100.0

def trend_class(pct):
    if pd.isna(pct):
        return "result-neutral", "Sin referencia"
    if pct > 1.0:
        return "result-up", f"+{pct:.1f}%"
    if pct < -1.0:
        return "result-down", f"{pct:.1f}%"
    return "result-neutral", f"{pct:.1f}%"

def player_estimated_1rm_series(player_df, load_kg):
    ser = player_df["VMP"].apply(lambda x: estimate_1rm_from_load_vmp(load_kg, x))
    return pd.to_numeric(ser, errors="coerce")

def render_results_cards(player_df, row, load_kg, body_mass=np.nan):
    refs = {}
    for metric in OBJECTIVE_METRICS:
        vals_pre = pd.to_numeric(player_df[metric], errors="coerce").dropna() if metric in player_df.columns else pd.Series(dtype=float)
        best_series = vals_pre.copy()

        post_col = f"{metric}_post"
        if post_col in player_df.columns:
            vals_post = pd.to_numeric(player_df[post_col], errors="coerce").dropna()
            if not vals_post.empty:
                best_series = pd.concat([best_series, vals_post], ignore_index=True)

        refs[metric] = {
            "current": pd.to_numeric(pd.Series([row.get(metric, np.nan)]), errors="coerce").iloc[0],
            "baseline": pd.to_numeric(pd.Series([row.get(f"{metric}_baseline", np.nan)]), errors="coerce").iloc[0],
            "initial": vals_pre.iloc[0] if not vals_pre.empty else np.nan,
            "best": best_series.max() if not best_series.empty else np.nan,
        }

    if pd.notna(load_kg):
        est_series = player_estimated_1rm_series(player_df, load_kg)
        current_est = estimate_1rm_from_load_vmp(load_kg, row.get("VMP", np.nan))
        est_hist = est_series.dropna()
        refs["1RM_est"] = {
            "current": current_est if pd.notna(current_est) else (est_hist.iloc[-1] if not est_hist.empty else np.nan),
            "baseline": est_hist.expanding().mean().shift(1).iloc[-1] if len(est_hist) > 1 else (est_hist.mean() if not est_hist.empty else np.nan),
            "initial": est_hist.iloc[0] if not est_hist.empty else np.nan,
            "best": est_hist.max() if not est_hist.empty else np.nan,
        }

        if pd.notna(body_mass) and body_mass > 0:
            rel_series = est_series / body_mass
            rel_hist = rel_series.dropna()
            current_rel = current_est / body_mass if pd.notna(current_est) else np.nan
            refs["1RM_rel"] = {
                "current": current_rel if pd.notna(current_rel) else (rel_hist.iloc[-1] if not rel_hist.empty else np.nan),
                "baseline": rel_hist.expanding().mean().shift(1).iloc[-1] if len(rel_hist) > 1 else (rel_hist.mean() if not rel_hist.empty else np.nan),
                "initial": rel_hist.iloc[0] if not rel_hist.empty else np.nan,
                "best": rel_hist.max() if not rel_hist.empty else np.nan,
            }
        else:
            refs["1RM_rel"] = {"current": np.nan, "baseline": np.nan, "initial": np.nan, "best": np.nan}
    else:
        refs["1RM_est"] = {"current": np.nan, "baseline": np.nan, "initial": np.nan, "best": np.nan}
        refs["1RM_rel"] = {"current": np.nan, "baseline": np.nan, "initial": np.nan, "best": np.nan}

    order = [("CMJ","cm"), ("RSI_mod",""), ("VMP"," m/s"), ("1RM_est"," kg"), ("1RM_rel"," kg/kg")]
    labels = {"CMJ":"CMJ", "RSI_mod":"RSI mod", "VMP":"VMP", "1RM_est":"1RM estimada", "1RM_rel":"1RM relativa"}
    decimals = {"CMJ":1, "RSI_mod":3, "VMP":3, "1RM_est":1, "1RM_rel":2}

    cards = []
    for key, suffix in order:
        ref = refs[key]
        current = ref["current"]
        vs_base = pct_change_safe(current, ref["baseline"])
        vs_initial = pct_change_safe(current, ref["initial"])
        css_class, txt = trend_class(vs_base)
        dist_best = pct_change_safe(current, ref["best"])

        best_text = "—" if pd.isna(ref["best"]) else f"{ref['best']:.{decimals[key]}f}{suffix}"
        current_text = "—" if pd.isna(current) else f"{current:.{decimals[key]}f}{suffix}"
        init_text = "—" if pd.isna(vs_initial) else f"{vs_initial:+.1f}%"
        best_gap = "—" if pd.isna(dist_best) else f"{dist_best:+.1f}%"

        card_html = (
            f'<div class="result-card">'
            f'<div class="lab">{labels[key]}</div>'
            f'<div class="big">{current_text}</div>'
            f'<div class="mini"><span class="{css_class}">{txt} vs baseline</span></div>'
            f'<div class="mini">vs inicio: <b>{init_text}</b></div>'
            f'<div class="mini">mejor marca: <b>{best_text}</b> · distancia: <b>{best_gap}</b></div>'
            f'</div>'
        )
        cards.append(card_html)

    html = '<div class="results-grid">' + ''.join(cards) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

def results_summary_text(player_df, row, load_kg=None, body_mass=np.nan):
    msgs = []
    for metric in OBJECTIVE_METRICS:
        current = row.get(metric, np.nan)
        baseline = row.get(f"{metric}_baseline", np.nan)
        pct = pct_change_safe(current, baseline)
        if pd.notna(pct):
            msgs.append((metric, pct))
    if pd.notna(load_kg):
        current_est = estimate_1rm_from_load_vmp(load_kg, row.get("VMP", np.nan))
        est_series = player_estimated_1rm_series(player_df, load_kg).dropna()
        est_base = est_series.expanding().mean().shift(1).iloc[-1] if len(est_series) > 1 else (est_series.mean() if not est_series.empty else np.nan)
        pct = pct_change_safe(current_est, est_base)
        if pd.notna(pct):
            msgs.append(("1RM_est", pct))
        if pd.notna(body_mass) and body_mass > 0:
            current_rel = current_est / body_mass if pd.notna(current_est) else np.nan
            rel_series = (player_estimated_1rm_series(player_df, load_kg) / body_mass).dropna()
            rel_base = rel_series.expanding().mean().shift(1).iloc[-1] if len(rel_series) > 1 else (rel_series.mean() if not rel_series.empty else np.nan)
            pct_rel = pct_change_safe(current_rel, rel_base)
            if pd.notna(pct_rel):
                msgs.append(("1RM_rel", pct_rel))
    if not msgs:
        return "Sin datos suficientes para resumir la evolución del jugador."
    msgs_sorted = sorted(msgs, key=lambda x: x[1], reverse=True)
    best_metric, best_pct = msgs_sorted[0]
    worst_metric, worst_pct = msgs_sorted[-1]
    metric_name = lambda m: {"1RM_est":"1RM estimada","1RM_rel":"1RM relativa"}.get(m, LABELS.get(m,m))
    return f"Evolución global del jugador: la señal más positiva aparece en {metric_name(best_metric)} ({best_pct:+.1f}% vs baseline). La variable más rezagada es {metric_name(worst_metric)} ({worst_pct:+.1f}% vs baseline)."


def plot_session_candlestick(player_df, metric, selected_date):
    post_col = f"{metric}_post"
    fig = go.Figure()

    if metric not in player_df.columns or post_col not in player_df.columns:
        fig.update_layout(
            title=f"{LABELS.get(metric, metric)} · histórico sesión a sesión (PRE→POST)",
            height=340,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False,
        )
        return fig

    cols = ["Fecha", metric, post_col]
    baseline_col = f"{metric}_baseline"
    if baseline_col in player_df.columns:
        cols.append(baseline_col)

    temp = player_df[cols].copy()
    temp = temp.dropna(subset=[metric, post_col], how="any")

    if temp.empty:
        fig.update_layout(
            title=f"{LABELS.get(metric, metric)} · histórico sesión a sesión (PRE→POST)",
            height=340,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False,
        )
        return fig

    temp["open"] = pd.to_numeric(temp[metric], errors="coerce")
    temp["close"] = pd.to_numeric(temp[post_col], errors="coerce")
    temp["high"] = temp[["open", "close"]].max(axis=1)
    temp["low"] = temp[["open", "close"]].min(axis=1)
    temp["x_label"] = pd.to_datetime(temp["Fecha"]).dt.strftime("%d-%m-%Y")

    fig.add_trace(go.Candlestick(
        x=temp["x_label"],
        open=temp["open"],
        high=temp["high"],
        low=temp["low"],
        close=temp["close"],
        name="Sesión",
        increasing_line_color="#16A34A",
        decreasing_line_color="#DC2626",
        increasing_fillcolor="rgba(22,163,74,0.35)",
        decreasing_fillcolor="rgba(220,38,38,0.35)",
    ))

    if baseline_col in temp.columns:
        baseline_vals = pd.to_numeric(temp[baseline_col], errors="coerce")
        if baseline_vals.notna().any():
            fig.add_trace(go.Scatter(
                x=temp["x_label"],
                y=baseline_vals,
                mode="lines+markers",
                name="Baseline PRE",
                line=dict(color="#111827", width=4, dash="dash"),
            ))

    y_values = pd.concat([temp["open"], temp["close"]]).dropna()
    if not y_values.empty:
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        margin = (y_max - y_min) * 0.35 if y_max != y_min else 0.05
        fig.update_yaxes(range=[y_min - margin, y_max + margin])

    fig.update_layout(height=340)
    return fig

    cols = ["Fecha", metric, post_col]
    baseline_col = f"{metric}_baseline"
    if baseline_col in player_df.columns:
        cols.append(baseline_col)

    temp = player_df[cols].copy()
    temp = temp.dropna(subset=[metric, post_col], how="any")

    if temp.empty:
        fig.update_layout(
            title=f"{LABELS.get(metric, metric)} · histórico sesión a sesión (PRE→POST)",
            height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False,
        )
        return fig

    temp["open"] = pd.to_numeric(temp[metric], errors="coerce")
    temp["close"] = pd.to_numeric(temp[post_col], errors="coerce")
    temp["high"] = temp[["open", "close"]].max(axis=1)
    temp["low"] = temp[["open", "close"]].min(axis=1)

    fig.add_trace(go.Candlestick(
        x=temp["Fecha"],
        open=temp["open"],
        high=temp["high"],
        low=temp["low"],
        close=temp["close"],
        name="Sesión",
        increasing_line_color="#16A34A",
        decreasing_line_color="#DC2626",
        increasing_fillcolor="rgba(22,163,74,0.35)",
        decreasing_fillcolor="rgba(220,38,38,0.35)",
        whiskerwidth=0.4,
    ))

    if baseline_col in temp.columns:
        baseline_vals = pd.to_numeric(temp[baseline_col], errors="coerce")
        if baseline_vals.notna().any():
            fig.add_trace(go.Scatter(
                x=temp["Fecha"],
                y=baseline_vals,
                mode="lines",
                name="Baseline PRE",
                line=dict(color="#0F766E", width=2, dash="dot")
            ))

    sel = temp[temp["Fecha"].dt.normalize() == pd.to_datetime(selected_date).normalize()]
    if not sel.empty:
        fig.add_trace(go.Scatter(
            x=[sel["Fecha"].iloc[-1]],
            y=[sel["high"].iloc[-1]],
            mode="markers",
            name="Fecha",
            marker=dict(size=10, color="#1D4ED8", symbol="diamond")
        ))

    fig.update_layout(
        title=f"{LABELS.get(metric, metric)} · histórico sesión a sesión (PRE→POST)",
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
    )
    return fig


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

    profiles_df, _profiles_err = load_player_profiles()
    load_kg = np.nan
    body_mass = np.nan
    if not profiles_df.empty and player in profiles_df["Jugador"].values:
        prof_row = profiles_df[profiles_df["Jugador"] == player].iloc[0]
        load_kg = pd.to_numeric(prof_row.get("Carga_sentadilla"), errors="coerce")
        body_mass = pd.to_numeric(prof_row.get("Peso_corporal"), errors="coerce")

    risk_color = RISK_COLORS.get(row["risk_label"], "#475467")
    st.markdown(f'<div class="card"><div style="font-size:1.7rem; font-weight:900; color:#101828;">{player}</div><div style="margin-top:0.35rem;">{render_pills(row)}</div><div style="margin-top:0.55rem;"><span class="pill" style="background:{risk_color};">{row["risk_label"]}</span></div><div style="margin-top:0.7rem; color:#475467;">{player_comment(row)}</div></div>', unsafe_allow_html=True)

    main, pattern_txt, worst_metric, worst_value = infer_fatigue_profile(row)
    flags = flags_for_player(player_df, row)
    if flags:
        st.warning(" | ".join(flags))
    st.markdown(f"**Diagnóstico:** {main}, con {pattern_txt}. Variable dominante: {LABELS.get(worst_metric, 'NA')} ({'NA' if worst_value is None else f'{worst_value:.1f}%'}).")

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi("Fecha", selected_date.strftime("%Y-%m-%d"), f"{row.get('Microciclo', 'NA')}")
    with c2: kpi("Loss score", f"{row['objective_loss_score']:.2f}", "0-3")
    with c3: kpi("Pérdida media %", f"{row['objective_loss_mean_pct']:.1f}%", "CMJ + RSI + VMP")
    with c4: kpi("Readiness", f"{row['readiness_score']:.0f}", "ese mismo día")
    with c5: kpi("Riesgo", row["risk_label"], "clasificación")
    with c6: kpi("Z-score objetivo", "NA" if pd.isna(row["objective_z_score"]) else f"{row['objective_z_score']:.2f}", "apoyo")

    a,b = st.columns(2)
    with a: st.plotly_chart(radar_relative_loss(row), use_container_width=True)
    with b: st.plotly_chart(radar_current_vs_baseline(row), use_container_width=True)
    c,d = st.columns(2)
    with c: st.plotly_chart(plot_player_snapshot_compare(row), use_container_width=True)
    with d: st.plotly_chart(plot_objective_timeline(player_df, selected_date), use_container_width=True)

    st.markdown("### Resultados")
    render_results_cards(player_df, row, load_kg, body_mass=body_mass)
    st.markdown(f'<div class="soft-note">{results_summary_text(player_df, row, load_kg, body_mass=body_mass)}</div>', unsafe_allow_html=True)

    st.markdown("### Respuesta intra-sesión · PRE vs POST")
    render_pre_post_cards(row)
    cmj_delta_txt = "NA" if pd.isna(row.get("CMJ_delta_pct")) else f"{row.get('CMJ_delta_pct'):+.1f}%"
    rsi_delta_txt = "NA" if pd.isna(row.get("RSI_mod_delta_pct")) else f"{row.get('RSI_mod_delta_pct'):+.1f}%"
    x1, x2 = st.columns(2)
    with x1:
        st.plotly_chart(plot_pre_post_current(row), use_container_width=True)
    with x2:
        st.markdown(f'<div class="soft-note">CMJ post-pre: {cmj_delta_txt} · RSI post-pre: {rsi_delta_txt}.</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_delta_timeline(player_df, "CMJ", selected_date), use_container_width=True)
        st.plotly_chart(plot_delta_timeline(player_df, "RSI_mod", selected_date), use_container_width=True)

    st.markdown("### Histórico sesión a sesión · gráfico de velas")
    y1, y2 = st.columns(2)
    with y1:
        st.plotly_chart(plot_session_candlestick(player_df, "CMJ", selected_date), use_container_width=True)
    with y2:
        st.plotly_chart(plot_session_candlestick(player_df, "RSI_mod", selected_date), use_container_width=True)

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

def delete_session_by_date(date_str):
    supabase = get_supabase()
    supabase.table("monitoring").delete().eq("Fecha", date_str).execute()


def page_admin(base_df):
    st.markdown('<div class="section-title">Administración</div>', unsafe_allow_html=True)

    st.markdown("### Pesos corporales y cargas de sentadilla")
    players = sorted(base_df["Jugador"].dropna().astype(str).unique().tolist()) if not base_df.empty else []
    profiles_df, profiles_err = load_player_profiles()

    roster = pd.DataFrame({"Jugador": players})
    if not profiles_df.empty:
        roster = roster.merge(profiles_df[["Jugador","Peso_corporal","Carga_sentadilla"]], on="Jugador", how="left")
    else:
        roster["Peso_corporal"] = np.nan
        roster["Carga_sentadilla"] = np.nan

    st.caption("Introduce o modifica manualmente el peso corporal y la carga fija usada en sentadilla para estimar la 1RM relativa de cada jugador.")
    edited_profiles = st.data_editor(
        roster,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        key="profiles_editor",
        column_config={
            "Jugador": st.column_config.TextColumn("Jugador", disabled=True),
            "Peso_corporal": st.column_config.NumberColumn("Peso corporal (kg)", min_value=40.0, max_value=130.0, step=0.1, format="%.1f"),
            "Carga_sentadilla": st.column_config.NumberColumn("Carga sentadilla (kg)", min_value=20.0, max_value=200.0, step=2.5, format="%.1f"),
        },
    )
    c_save1, c_save2 = st.columns([1,1])
    with c_save1:
        if st.button("Guardar pesos y cargas", type="primary"):
            try:
                upsert_player_profiles(edited_profiles)
                st.success("Pesos y cargas guardados correctamente.")
                st.rerun()
            except Exception as e:
                st.error(f"No se pudieron guardar los pesos/cargas: {e}")
    with c_save2:
        st.download_button(
            "Descargar plantilla de pesos/cargas",
            data=roster.to_csv(index=False).encode("utf-8"),
            file_name="plantilla_pesos_cargas.csv",
            mime="text/csv",
        )

    if profiles_err:
        st.warning("La tabla de perfiles aún no existe en Supabase. Créala con este SQL y recarga la app.")
        st.code(PLAYER_PROFILES_SQL, language="sql")

    st.divider()

    if base_df.empty:
        st.info("La base está vacía.")
        return

    c1,c2,c3 = st.columns(3)
    with c1: kpi("Registros", len(base_df), "total")
    with c2: kpi("Jugadores", base_df["Jugador"].nunique(), "únicos")
    with c3: kpi("Fechas", base_df["Fecha"].nunique(), "controles")
    st.download_button("Descargar base CSV", data=base_df.to_csv(index=False).encode("utf-8"), file_name="md1_staff_elite_definitiva_v2.csv", mime="text/csv")
    st.markdown("### Eliminar una sesión")
    date_opts = sorted(base_df["Fecha"].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    if date_opts:
        selected_delete = st.selectbox("Selecciona la sesión/fecha a eliminar", date_opts)
        if st.button("Eliminar sesión seleccionada", type="secondary"):
            delete_session_by_date(selected_delete)
            st.success(f"Sesión {selected_delete} eliminada.")
            st.rerun()
    else:
        st.info("No hay sesiones para eliminar.")


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
    menu = st.sidebar.radio("Sección", ["CARGAR SESIÓN","Equipo","Perfil F-R","Jugador","Comparador","Informes","Administración"])

    if menu == "CARGAR SESIÓN":
        page_cargar()
    elif menu == "Equipo":
        page_equipo(metrics_df)
    elif menu == "Perfil F-R":
        page_force_reactivity(metrics_df)
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