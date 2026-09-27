"""Constantes, textes et fonctions partages par les pages de l'interface."""
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import altair as alt
import psutil
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
UPLOADS = ROOT / "uploads"
PAGES = {"upload": "views/upload.py", "dashboard": "views/dashboard.py", "guide": "views/guide.py"}

INK, INK_2, MUTED, GRID = "#0F2233", "#52514e", "#898781", "#e1e0d9"
TEAM_COLOR = {"A": "#eb6834", "B": "#2a78d6"}
DEFAULT_TEAM_NAMES = {"back": {"A": "Côté opposé", "B": "Côté caméra"},
                      "side": {"A": "Côté gauche", "B": "Côté droit"}}
VIEW_LABEL = {"back": "Caméra derrière le terrain", "side": "Caméra sur le côté"}

ACTIONS = ["service", "reception", "defense", "passe", "attaque", "contre"]
ACTION_LABEL = {"service": "Service", "reception": "Réception", "defense": "Défense",
                "passe": "Passe", "attaque": "Attaque", "contre": "Contre"}
ACTION_HELP = {
    "service": "Première touche d'un rallye, jouée seule ou depuis derrière la ligne de fond.",
    "reception": "Première touche d'une équipe juste après le service adverse.",
    "defense": "Première touche d'une équipe après une attaque adverse.",
    "passe": "Touche au milieu d'une possession, le plus souvent la passe du passeur.",
    "attaque": "Dernière touche d'une possession, celle qui renvoie la balle chez l'adversaire.",
    "contre": "Touche près du filet moins de 0,8 s après une touche adverse. "
              "Elle ne compte pas dans les 3 touches autorisées.",
}
STEPS = ["prepare", "inference", "analysis", "render", "encode"]
STEP_LABEL = {"prepare": "Préparation de la vidéo", "inference": "Détection des joueurs et de la balle",
              "analysis": "Calcul des statistiques", "render": "Création de la vidéo annotée",
              "encode": "Encodage de la vidéo"}

CSS = """
<style>
:root { --ink: #0F2233; --ink-2: #52514e; --muted: #6b6a66; --line: #e3e7ec; --navy: #13314F;
        --team-a: #eb6834; --team-b: #2a78d6; }
.block-container { padding-top: 2.2rem; max-width: 1280px; }
h1, h2, h3 { color: var(--navy); letter-spacing: .2px; }
.lead { color: var(--ink-2); font-size: 1.05rem; margin: -.4rem 0 1rem; }
.step { display: flex; align-items: center; gap: .6rem; margin-bottom: .2rem; }
.step .num { background: var(--navy); color: #fff; border-radius: 50%; width: 1.9rem; height: 1.9rem;
  display: inline-flex; align-items: center; justify-content: center; font-weight: 700; flex: none; }
.step .num.done { background: #0ca30c; }
.step .txt { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.45rem; color: var(--navy); }
.hint { color: var(--ink-2); font-size: .95rem; margin: 0 0 .6rem; }

.faceoff { border: 1px solid var(--line); border-radius: 14px; overflow: hidden; background: #fff; }
.faceoff .head { display: grid; grid-template-columns: 1fr auto 1fr; align-items: end; gap: 12px;
  background: var(--navy); color: #fff; padding: 16px 22px; }
.faceoff .head .team { font-family: 'Barlow Condensed', sans-serif; font-size: 1.5rem; font-weight: 700; line-height: 1.1; }
.faceoff .head .team.b { text-align: right; }
.faceoff .head .team i { display: block; width: 44px; height: 5px; border-radius: 3px; margin-top: 6px; }
.faceoff .head .team.b i { margin-left: auto; }
.faceoff .head .vs { font-size: .9rem; opacity: .75; padding-bottom: 4px; text-align: center; }
.faceoff .row { display: grid; grid-template-columns: 1fr 170px 1fr; align-items: center; gap: 12px;
  padding: 7px 22px; border-top: 1px solid var(--line); }
.faceoff .row .lab { text-align: center; color: var(--ink-2); font-size: .95rem; line-height: 1.15; }
.faceoff .row .lab small { display: block; color: var(--muted); font-size: .78rem; }
.faceoff .side { display: flex; align-items: center; gap: 10px; }
.faceoff .side.a { flex-direction: row-reverse; }
.faceoff .val { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.35rem;
  color: var(--ink); min-width: 3.2rem; }
.faceoff .side.a .val { text-align: right; }
.faceoff .track { flex: 1; height: 8px; display: flex; }
.faceoff .side.a .track { justify-content: flex-end; }
.faceoff .bar { height: 8px; border-radius: 4px; }
@media (max-width: 700px) { .faceoff .row { grid-template-columns: 1fr 110px 1fr; padding: 7px 12px; } }

.insights { margin: 0; padding-left: 1.1rem; }
.insights li { margin: .25rem 0; color: var(--ink); }
.muted { color: var(--muted); }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def step_header(n, text, done=False):
    st.markdown(f'<div class="step"><span class="num{" done" if done else ""}">{"✓" if done else n}</span>'
                f'<span class="txt">{text}</span></div>', unsafe_allow_html=True)


def team_names(meta):
    names = dict(DEFAULT_TEAM_NAMES[meta.get("view", "back")])
    names.update({k: v for k, v in (meta.get("team_names") or {}).items() if v})
    return names


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def job_status(run_dir):
    """Etat d'une analyse (status.json) ; signale un processus disparu sans avoir fini."""
    s = read_json(Path(run_dir) / "status.json")
    if s and s.get("state") == "running" and not psutil.pid_exists(s.get("pid", -1)):
        s.update(state="error", message="Le processus d'analyse s'est arrêté avant la fin (fermé ou interrompu).")
    return s


def list_runs():
    """Dossiers d'analyse, du plus récent au plus ancien."""
    runs = []
    for d in RESULTS.glob("*/"):
        if not ((d / "results.json").exists() or (d / "status.json").exists()):
            continue
        meta = read_json(d / "meta.json") or {}
        stamp = max((f.stat().st_mtime for f in (d / "results.json", d / "status.json") if f.exists()))
        runs.append({"name": d.name, "dir": d, "title": meta.get("title") or d.name, "meta": meta,
                     "has_results": (d / "results.json").exists(), "mtime": stamp})
    return sorted(runs, key=lambda r: -r["mtime"])


@st.cache_data(show_spinner=False)
def _load(path, mtime):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_results(run_dir):
    p = Path(run_dir) / "results.json"
    return _load(str(p), p.stat().st_mtime)


def slug(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "video"


def unique_run_name(base):
    name, k = slug(base), 2
    while (RESULTS / name).exists():
        name, k = f"{slug(base)}_{k}", k + 1
    return name


def fmt_clock(s):
    s = max(0, float(s))
    return f"{int(s // 60)}:{int(s % 60):02d}"


def fmt_duration(s):
    s = int(round(s))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min {s % 60:02d}"
    return f"{s // 3600} h {s % 3600 // 60:02d}"


def fmt_num(x, dec=1):
    return f"{x:.{dec}f}".replace(".", ",")


def fmt_date(iso):
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%Y à %H:%M")
    except (TypeError, ValueError):
        return ""


def style_chart(chart, height=None):
    """Axes et grille discrets, police de l'app, fond transparent."""
    if height:
        chart = chart.properties(height=height)
    return (chart.configure(font="Barlow", background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=INK_2, titleColor=INK_2, gridColor=GRID, domainColor="#c3c2b7",
                            tickColor="#c3c2b7", labelFontSize=12, titleFontSize=12, titleFontWeight=500)
            .configure_legend(labelColor=INK_2, titleColor=INK_2, labelFontSize=12, titleFontSize=12,
                              orient="top", titleOrient="left")
            .configure_title(color=INK, fontSize=14, anchor="start"))


def team_scale(names):
    return alt.Scale(domain=[names["A"], names["B"]], range=[TEAM_COLOR["A"], TEAM_COLOR["B"]])


def open_run(name):
    st.session_state["run"] = name
    st.switch_page(PAGES["dashboard"])


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


def pid_alive(pid):
    return bool(pid) and psutil.pid_exists(pid)


def kill(pid):
    try:
        psutil.Process(pid).kill()
    except psutil.Error:
        pass


def env_utf8():
    return {**os.environ, "PYTHONIOENCODING": "utf-8"}
