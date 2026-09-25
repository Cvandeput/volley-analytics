"""Tableau de bord des analyses volley : streamlit run dashboard.py"""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RESULTS = Path("results")
TEAM_NAME = {"proche": "Côté caméra", "loin": "Côté opposé"}
TEAM_COLOR = {"Côté caméra": "#2F6FDE", "Côté opposé": "#E0A800"}
KIND_NAME = {"rebond": "Réception / passe", "frappe": "Frappe"}

st.set_page_config(page_title="Analyse volley", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;700&display=swap');
.stApp { background: #EEF2F6; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp td, .stApp th {
  font-family: 'Barlow', system-ui, sans-serif; }
.stApp h1, .stApp h2, .stApp h3 {
  font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif; font-weight: 700; color: #13314F; }
.board { background: #13314F; color: #fff; border-radius: 16px; padding: 20px 28px;
  display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 24px; }
.team .name { font-size: 1.05rem; opacity: .82; }
.team .score { font-family: 'Barlow Condensed', sans-serif; font-weight: 700;
  font-size: 4.4rem; line-height: 1; }
.team .bar { height: 6px; width: 72px; border-radius: 3px; margin-top: 8px; }
.team.right { text-align: right; }
.team.right .bar { margin-left: auto; }
.mid { text-align: center; font-family: 'Barlow Condensed', sans-serif; }
.mid .big { font-size: 1.7rem; font-weight: 700; }
.mid .small { font-size: 1.05rem; opacity: .75; }
.stats { display: flex; gap: 36px; flex-wrap: wrap; margin: 18px 4px 4px; color: #13314F; }
.stat b { display: block; font-family: 'Barlow Condensed', sans-serif; font-size: 2rem; line-height: 1.05; }
.stat span { font-size: .95rem; color: #4A5A6C; }
@media (max-width: 700px) { .team .score { font-size: 3rem; } .board { padding: 16px; gap: 12px; } }
</style>
""", unsafe_allow_html=True)

runs = sorted((p.parent for p in RESULTS.glob("*/results.json")),
              key=lambda p: (p / "results.json").stat().st_mtime, reverse=True)
if not runs:
    st.title("Analyse volley")
    st.info("Aucune analyse dans le dossier results. Lance d'abord : "
            "python analyze.py <video> --ball <poids_balle.pt>")
    st.stop()

names = [p.name for p in runs]
run = RESULTS / st.sidebar.selectbox("Analyse", names)
data = json.loads((run / "results.json").read_text(encoding="utf-8"))
S = data["summary"]
min_vis = st.sidebar.slider("Présence minimale d'un joueur (s)", 0.0, 20.0, 2.0, 0.5)
st.sidebar.caption("Les numéros de joueur sont les identifiants du tracker. "
                   "Un même joueur peut en avoir plusieurs si le suivi a décroché.")

touches = pd.DataFrame(data["touches"])
jumps = pd.DataFrame(data["jumps"])
players = pd.DataFrame(data["players"])
poss = pd.DataFrame(data["possessions"])

# ---------------------------------------------------------------- en-tête
st.title(run.name)
tb = S["touches_by_team"]
st.markdown(f"""
<div class="board">
  <div class="team"><div class="name">{TEAM_NAME['proche']}</div>
    <div class="score">{tb['proche']}</div>
    <div class="bar" style="background:{TEAM_COLOR[TEAM_NAME['proche']]}"></div></div>
  <div class="mid"><div class="big">{S['touches_total']} touches</div>
    <div class="small">{S['rallies']} rallyes, {S['duration_s']:.0f} s de vidéo</div></div>
  <div class="team right"><div class="name">{TEAM_NAME['loin']}</div>
    <div class="score">{tb['loin']}</div>
    <div class="bar" style="background:{TEAM_COLOR[TEAM_NAME['loin']]}"></div></div>
</div>
<div class="stats">
  <div class="stat"><b>{S['jumps_total']}</b><span>sauts</span></div>
  <div class="stat"><b>{S['possessions']}</b><span>possessions</span></div>
  <div class="stat"><b>{S['touches_per_possession']:.1f}</b><span>touches par possession</span></div>
  <div class="stat"><b>{S['ball_detection_rate']:.0%}</b><span>balle détectée</span></div>
  <div class="stat"><b>{S['contacts_sans_joueur']}</b><span>contacts sans joueur</span></div>
</div>
""", unsafe_allow_html=True)

if S["ball_detection_rate"] < 0.5:
    st.warning("La balle n'est détectée que sur "
               f"{S['ball_detection_rate']:.0%} des images : touches et possessions sont sous-estimées. "
               "Essaie --ball-imgsz 1280 et --ball-conf 0.15, ou un modèle balle mieux entraîné.")
if S["possessions_suspectes"]:
    st.caption(f"{S['possessions_suspectes']} possession(s) comptent plus de 3 touches : "
               "probable fausse détection ou changement de camp manqué.")

# ---------------------------------------------------------------- vidéo + joueurs
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Vidéo annotée")
    start = 0
    if not touches.empty:
        labels = ["Début"] + [f"Touche {i + 1} à {r.t:.1f} s, joueur {r.player}".replace(".", ",")
                              for i, r in enumerate(touches.itertuples())]
        pick = st.selectbox("Aller à", range(len(labels)), format_func=lambda i: labels[i])
        start = 0 if pick == 0 else max(0, int(touches.iloc[pick - 1]["t"]) - 2)
    video = run / "annotated.mp4"
    if video.exists():
        st.video(str(video), start_time=start)
    else:
        st.info("Pas de vidéo annotée pour cette analyse. Relance analyze.py sans --no-video.")

with right:
    st.subheader("Joueurs")
    if players.empty:
        st.info("Aucun joueur suivi assez longtemps sur le terrain.")
    else:
        view = players[players["visible_s"] >= min_vis].copy()
        view["team"] = view["team"].map(TEAM_NAME)
        st.dataframe(
            view[["id", "team", "touches", "jumps", "max_jump_cm", "avg_jump_cm", "visible_s"]],
            hide_index=True, width="stretch", height=420,
            column_config={
                "id": st.column_config.NumberColumn("Joueur", format="%d"),
                "team": "Côté",
                "touches": st.column_config.ProgressColumn(
                    "Touches", format="%d", min_value=0,
                    max_value=int(max(1, players["touches"].max()))),
                "jumps": st.column_config.NumberColumn("Sauts", format="%d"),
                "max_jump_cm": st.column_config.NumberColumn("Saut max (cm)", format="%d"),
                "avg_jump_cm": st.column_config.NumberColumn("Saut moyen (cm)", format="%d"),
                "visible_s": st.column_config.NumberColumn("Présence (s)", format="%.1f"),
            })
        st.caption("Hauteur estimée à partir du temps de vol : ordre de grandeur, pas une mesure.")

team_scale = alt.Scale(domain=list(TEAM_COLOR), range=list(TEAM_COLOR.values()))

# ---------------------------------------------------------------- chronologie
st.subheader("Chronologie des touches")
if touches.empty:
    st.info("Aucune touche détectée. Vérifie le taux de détection de la balle, ou baisse "
            "touch_min_speed dans analyze.py puis relance avec --reuse.")
else:
    tdf = touches.assign(Côté=touches["team"].map(TEAM_NAME), Type=touches["kind"].map(KIND_NAME),
                         Joueur=touches["player"].astype(str))
    chart = alt.Chart(tdf).mark_point(filled=True, size=140).encode(
        x=alt.X("t:Q", title="Temps (s)"),
        y=alt.Y("Joueur:N", sort="ascending"),
        color=alt.Color("Côté:N", scale=team_scale),
        shape=alt.Shape("Type:N"),
        tooltip=["t", "Joueur", "Côté", "Type", "rally", "possession"],
    ).properties(height=280)
    st.altair_chart(chart, width="stretch")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("Touches par joueur")
    if not players.empty and players["touches"].sum():
        pdf = players[players["touches"] > 0].assign(Côté=lambda d: d["team"].map(TEAM_NAME),
                                                     Joueur=lambda d: d["id"].astype(str))
        st.altair_chart(alt.Chart(pdf).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("touches:Q", title="Touches"),
            y=alt.Y("Joueur:N", sort="-x"),
            color=alt.Color("Côté:N", scale=team_scale, legend=None),
        ).properties(height=260), width="stretch")
    else:
        st.info("Pas encore de touche attribuée.")
with c2:
    st.subheader("Sauts")
    if jumps.empty:
        st.info("Aucun saut détecté.")
    else:
        jdf = jumps.assign(Côté=jumps["team"].map(TEAM_NAME), Joueur=jumps["player"].astype(str))
        st.altair_chart(alt.Chart(jdf).mark_circle(size=120).encode(
            x=alt.X("t:Q", title="Temps (s)"),
            y=alt.Y("height_cm:Q", title="Hauteur estimée (cm)"),
            color=alt.Color("Côté:N", scale=team_scale, legend=None),
            tooltip=["Joueur", "t", "height_cm", "air_s"],
        ).properties(height=260), width="stretch")

# ---------------------------------------------------------------- détails
with st.expander("Possessions"):
    if poss.empty:
        st.write("Aucune possession.")
    else:
        pv = poss.assign(team=poss["team"].map(TEAM_NAME),
                         players=poss["players"].map(lambda l: ", ".join(map(str, l))))
        st.dataframe(pv, hide_index=True, width="stretch", column_config={
            "rally": "Rallye", "team": "Côté", "start_s": "Début (s)", "end_s": "Fin (s)",
            "touches": "Touches", "players": "Joueurs"})
with st.expander("Trajectoire de la balle"):
    if data["ball_track"]:
        bt = pd.DataFrame(data["ball_track"], columns=["t", "x", "y"])
        bt["Hauteur à l'image"] = data["height"] - bt["y"]
        st.line_chart(bt, x="t", y="Hauteur à l'image", height=220)
    else:
        st.write("Balle jamais détectée.")
with st.expander("Seuils utilisés"):
    st.json(data["config"])
