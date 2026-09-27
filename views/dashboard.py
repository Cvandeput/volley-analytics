"""Page : tableau de bord d'une analyse."""
import html
import json
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import analyze as A
from views.common import (ACTION_HELP, ACTION_LABEL, ACTIONS, INK, INK_2, MUTED, PAGES, RESULTS, ROOT,
                          TEAM_COLOR, VIEW_LABEL, fmt_clock, fmt_date, fmt_duration, fmt_num, job_status, list_runs,
                          load_results, style_chart, team_names, team_scale)

ss = st.session_state
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
COURT_FILL, LINE = "#F6EFE6", "#0F2233"
CFG_HELP = {
    "court_side_margin_m": "Marge gardée hors des lignes de côté (m)",
    "court_end_margin_m": "Marge gardée derrière les lignes de fond, pour les serveurs (m)",
    "track_max_side_m": "Piste écartée si elle reste hors des lignes de côté : arbitres, banc (m)",
    "track_min_spread_m": "Piste immobile hors du terrain écartée : juges de ligne (m)",
    "net_switch_s": "Piste coupée si elle change de camp plus longtemps que ça (s)",
    "stitch_gap_s": "Trou maximal pour recoller deux morceaux de piste (s)",
    "stitch_speed_mps": "Vitesse maximale supposée pendant ce trou (m/s)",
    "stitch_radius_m": "Tolérance de position au recollage (m)",
    "ball_max_jump": "Saut maximal de la balle entre deux images (part de la largeur)",
    "ball_max_gap": "Trous de détection de la balle comblés (images)",
    "touch_k": "Fenêtre de calcul des vitesses de balle (images)",
    "touch_min_speed": "Vitesse minimale de la balle (part de la largeur par image)",
    "touch_strike_ratio": "Accélération qui signale une frappe (rapport des vitesses)",
    "touch_min_gap_s": "Écart minimal entre deux touches (s)",
    "touch_reach": "Distance balle-mains maximale (en hauteurs de joueur)",
    "assign_w_dist": "Attribution : poids de la distance balle-mains",
    "assign_w_four": "Attribution : pénalité d'une 4e touche du même camp",
    "assign_w_double": "Attribution : pénalité de deux touches de suite du même joueur",
    "block_net_m": "Contre : distance maximale au filet (m)",
    "block_delay_s": "Contre : délai maximal après la touche adverse (s)",
    "jump_rise": "Saut : montée minimale des chevilles (part de la taille)",
    "jump_air": "Saut : seuil « en l'air » pour le temps de vol (part de la taille)",
    "jump_min_air_s": "Saut : temps de vol minimal (s)",
    "jump_max_air_s": "Saut : temps de vol maximal (s)",
    "jump_min_gap_s": "Écart minimal entre deux sauts (s)",
    "jump_land_tol": "Saut : écart maximal entre appel et réception (part de la taille)",
    "rally_gap_s": "Nouveau rallye après ce temps sans touche (s)",
    "min_track_s": "Présence minimale pour figurer dans le tableau des joueurs (s)",
    "move_max_speed_mps": "Pas ignorés au-delà de cette vitesse dans la distance (m/s)",
}


def pct(x):
    return f"{round(100 * x)} %"


def reliability(S):
    """Niveau de confiance (2 bonne, 1 moyenne, 0 faible) et raisons."""
    level, notes = 2, []
    rate = S["ball_detection_rate"]
    if rate < 0.5:
        level = 0
    elif rate < 0.75:
        level = 1
    notes.append(f"balle vue sur {pct(rate)} des images")
    contacts = S["touches_total"] + S["contacts_sans_joueur"]
    if contacts and S["contacts_sans_joueur"] / contacts > 0.35:
        level = min(level, 1)
        notes.append(f"{S['contacts_sans_joueur']} contacts de balle sans joueur identifié")
    if S["possessions_suspectes"]:
        level = min(level, 1 if S["possessions_suspectes"] / max(1, S["possessions"]) <= 0.3 else 0)
        notes.append(f"{S['possessions_suspectes']} possession(s) de plus de 3 touches")
    if S["touches_total"] < 3:
        level = 0
        notes.append("très peu de touches détectées")
    return level, notes


def to_uv(df, view, x="x", y="y"):
    """Coordonnées terrain -> axes du schéma (vertical en vue de fond, horizontal en vue de côté)."""
    df = df.copy()
    df["u"], df["v"] = (df[y], df[x]) if view == "side" else (df[x], df[y])
    return df


def court_dims(view):
    """Longueurs du schéma (U horizontal, V vertical) et marges affichées autour du terrain."""
    if view == "side":
        return A.COURT_L, A.COURT_W, 4.0, 1.5
    return A.COURT_W, A.COURT_L, 1.5, 4.0


def cx(view, field="u"):
    U, _, mu, _ = court_dims(view)
    return alt.X(f"{field}:Q", scale=alt.Scale(domain=[-mu, U + mu], nice=False, zero=False), axis=None)


def cy(view, field="v"):
    _, V, _, mv = court_dims(view)
    return alt.Y(f"{field}:Q", scale=alt.Scale(domain=[-mv, V + mv], nice=False, zero=False, reverse=True),
                 axis=None)


def court_chart(layers, view, names, width=300):
    """Schéma du terrain vu de dessus, sous les couches données (encodées avec cx/cy)."""
    side = view == "side"
    U, V, mu, mv = court_dims(view)
    height = int(width * (V + 2 * mv) / (U + 2 * mu))

    def seg(a, b, w):
        return pd.DataFrame({"u": [a[0]], "v": [a[1]], "u2": [b[0]], "v2": [b[1]], "w": [w]})

    lines = [seg((0, 0), (U, 0), 1.5), seg((0, V), (U, V), 1.5), seg((0, 0), (0, V), 1.5), seg((U, 0), (U, V), 1.5)]
    for c, w in ((A.NET_Y, 3.5), (A.NET_Y - A.ATTACK_M, 1), (A.NET_Y + A.ATTACK_M, 1)):
        lines.append(seg((c, 0), (c, V), w) if side else seg((0, c), (U, c), w))
    base = alt.Chart(pd.DataFrame({"u": [0], "v": [0], "u2": [U], "v2": [V]})).mark_rect(
        color=COURT_FILL).encode(cx(view), cy(view), x2="u2:Q", y2="v2:Q")
    rules = alt.Chart(pd.concat(lines)).mark_rule(color=LINE, opacity=0.8).encode(
        cx(view), cy(view), x2="u2:Q", y2="v2:Q", strokeWidth=alt.StrokeWidth("w:Q", scale=None))
    if side:
        lab = [{"u": U * 0.25, "v": V + mv / 2, "t": names["A"]}, {"u": U * 0.75, "v": V + mv / 2, "t": names["B"]}]
    else:
        lab = [{"u": U / 2, "v": -mv / 2, "t": names["A"]}, {"u": U / 2, "v": V + mv / 2, "t": names["B"]}]
    labels = alt.Chart(pd.DataFrame(lab)).mark_text(color=INK_2, fontSize=12, fontWeight=600).encode(
        cx(view), cy(view), text="t:N")
    return alt.layer(base, *layers, rules, labels).properties(width=width, height=height)


@st.cache_data(show_spinner=False, max_entries=4)
def court_still(video, corners, view, t):
    try:
        img = A.read_frame(video, t)
    except SystemExit:
        return None
    A.Court(corners, view).draw(img, (0, 230, 255), 3)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def faceoff(T, names):
    rows = [("Touches de balle", T["A"]["touches"], T["B"]["touches"], "toutes actions", None)]
    for a in ACTIONS:
        rows.append((ACTION_LABEL[a] + "s", T["A"][a], T["B"][a], None, ACTION_HELP[a]))
    rows.append(("Attaques en suspension", T["A"]["attacks_air"], T["B"]["attacks_air"],
                 "joueur en l'air à la frappe", "Attaque touchée pendant un saut détecté du joueur."))
    rows.append(("Constructions en 3 touches", T["A"]["built"], T["B"]["built"],
                 None, "Possessions jouées réception (ou défense) → passe → attaque : le schéma idéal."))
    rows.append(("Sauts", T["A"]["jumps"], T["B"]["jumps"], None, "Sauts détectés par la montée des chevilles."))
    dist = [T[k]["distance_m"] / T[k]["players"] if T[k]["players"] else 0 for k in "AB"]
    rows.append(("Distance par joueur", dist[0], dist[1], "moyenne, en mètres",
                 "Distance parcourue au sol (hors sauts), moyenne des joueurs suivis au moins 1 s."))
    hidden = [r[0].lower() for r in rows if r[1] == 0 and r[2] == 0]
    rows = [r for r in rows if r[1] or r[2]]

    def side(v, top, team):
        val = f"{v:.0f}" if isinstance(v, float) else str(v)
        bar = (f'<div class="bar" style="width:{100 * v / top:.0f}%;background:{TEAM_COLOR[team]}"></div>'
               if v and top else "")
        return f'<div class="side {team.lower()}"><span class="val">{val}</span><div class="track">{bar}</div></div>'

    out = [f'<div class="faceoff"><div class="head">'
           f'<div class="team">{html.escape(names["A"])}<i style="background:{TEAM_COLOR["A"]}"></i></div>'
           f'<div class="vs">face à face</div>'
           f'<div class="team b">{html.escape(names["B"])}<i style="background:{TEAM_COLOR["B"]}"></i></div></div>']
    for label, a, b, sub, tip in rows:
        top = max(a, b)
        title = f' title="{html.escape(tip)}"' if tip else ""
        small = f"<small>{sub}</small>" if sub else ""
        out.append(f'<div class="row">{side(a, top, "A")}<div class="lab"{title}>{label}{small}</div>'
                   f'{side(b, top, "B")}</div>')
    out.append("</div>")
    return "".join(out), hidden


def insights(S, T, names, players, jumps, rallies, poss):
    out, star = [], None
    if not players.empty and players["touches"].max() > 0:
        p = players.sort_values(["touches", "jumps"], ascending=False).iloc[0]
        star = p["id"]
        acts = [f"{int(p[a])} {ACTION_LABEL[a].lower()}{'s' if p[a] > 1 else ''}" for a in ACTIONS if p[a]]
        out.append(f"**#{p['id']}** ({names[p['team']]}) est le plus impliqué : {int(p['touches'])} touches "
                   f"({', '.join(acts)}).")
    if not players.empty and players["attaque"].max() > 0:
        p = players.sort_values("attaque", ascending=False).iloc[0]
        n = int(p["attaque"])
        if p["id"] != star:
            out.append(f"**#{p['id']}** ({names[p['team']]}) est le principal attaquant : "
                       f"{n} attaque{'s' if n > 1 else ''}.")
    if not jumps.empty:
        j = jumps.sort_values("height_cm", ascending=False).iloc[0]
        out.append(f"Plus haut saut : **#{j['player']}**, environ {j['height_cm']} cm "
                   f"({fmt_num(j['air_s'], 2)} s en l'air).")
    if not players.empty and players["distance_m"].max() > 0:
        p = players.sort_values("distance_m", ascending=False).iloc[0]
        out.append(f"**#{p['id']}** a le plus couru : environ {int(p['distance_m'])} m "
                   f"en {fmt_duration(p['visible_s'])} de présence.")
    if not rallies.empty and len(rallies) > 1:
        r = rallies.sort_values("duration_s", ascending=False).iloc[0]
        out.append(f"Plus long rallye : n° {r['rally']}, {fmt_num(r['duration_s'])} s et "
                   f"{r['crossings']} passage(s) de filet.")
    built = [k for k in "AB" if T[k]["built"]]
    for k in built:
        out.append(f"{names[k]} a construit {T[k]['built']} de ses {T[k]['possessions']} possessions "
                   "en 3 touches (réception/défense → passe → attaque).")
    if not built and S["possessions"] >= 3:
        out.append("Aucune possession construite en 3 touches n'a été détectée.")
    return out[:6]


# ---------------------------------------------------------------- sélection de l'analyse

runs = [r for r in list_runs() if r["has_results"]]
if not runs:
    st.title("Tableau de bord")
    st.info("Aucune analyse pour l'instant. Commence par importer une vidéo.", icon=":material/info:")
    st.page_link(PAGES["upload"], label="Lancer une première analyse", icon=":material/upload:")
    st.stop()

titles = {r["name"]: r["title"] for r in runs}
if ss.get("run") not in titles:
    ss["run"] = runs[0]["name"]
with st.sidebar:
    st.selectbox("Analyse affichée", list(titles), key="run", format_func=titles.get)
    job = ss.get("job")
    if job and (job_status(RESULTS / job) or {}).get("state") == "running":
        st.info("Une analyse est en cours.", icon=":material/progress_activity:")
        st.page_link(PAGES["upload"], label="Voir l'avancement", icon=":material/arrow_forward:")

run_dir = RESULTS / ss["run"]
data = load_results(run_dir)
if data.get("version", 1) < 2:
    st.title(titles[ss["run"]])
    st.warning("Cette analyse a été produite par une ancienne version. Relance le calcul (quelques secondes, "
               f"sans GPU) :\n\n`python analyze.py <video> --reuse --name {ss['run']}`", icon=":material/update:")
    st.stop()

meta, S, T = data["meta"], data["summary"], data["teams"]
view = meta.get("view", "back")
names = team_names(meta)
tscale = team_scale(names)
touches = pd.DataFrame(data["touches"])
players = pd.DataFrame(data["players"])
jumps = pd.DataFrame(data["jumps"])
poss = pd.DataFrame(data["possessions"])
rallies = pd.DataFrame(data["rallies"])
positions = pd.DataFrame(data["positions"], columns=["t", "id", "team", "x", "y"])
if not touches.empty:
    touches["Équipe"] = touches["team"].map(names)
    touches["Action"] = touches["action"].map(ACTION_LABEL)
    touches[["px", "py"]] = pd.DataFrame(touches["pos"].tolist(), index=touches.index)

# ---------------------------------------------------------------- en-tête

st.title(meta.get("title") or ss["run"])
sub = [f"{fmt_duration(S['duration_s'])} de vidéo", VIEW_LABEL.get(view, view)]
if meta.get("analyzed"):
    sub.append(f"analysée le {fmt_date(meta['analyzed'])}")
st.markdown(f'<p class="lead">{" · ".join(sub)}</p>', unsafe_allow_html=True)

level, notes = reliability(S)
badge = [":red-badge[:material/error: Fiabilité faible]", ":orange-badge[:material/warning: Fiabilité moyenne]",
         ":green-badge[:material/check_circle: Fiabilité bonne]"][level]
st.markdown(f"{badge} &nbsp; {', '.join(notes).capitalize()}. Détails dans l'onglet *Fiabilité et méthode*.")

left, right = st.columns([3, 2], gap="large")
with left:
    card, hidden = faceoff(T, names)
    st.markdown(card, unsafe_allow_html=True)
    if hidden:
        st.caption("Non détecté dans cette vidéo : " + ", ".join(hidden) + ".")
with right:
    with st.container(border=True):
        st.markdown("#### À retenir")
        tips = insights(S, T, names, players, jumps, rallies, poss)
        st.markdown("\n".join(f"- {t}" for t in tips) if tips else "Pas assez de touches détectées pour en tirer "
                    "des conclusions.")
    st.caption("Les joueurs sont numérotés par le suivi vidéo (#1, #2…), pas par leur numéro de maillot. "
               "Survole un intitulé du face-à-face pour sa définition.")

m = st.columns(5)
avg_len = rallies["duration_s"].mean() if not rallies.empty else 0
avg_cross = rallies["crossings"].mean() if not rallies.empty else 0
m[0].metric("Rallyes", S["rallies"], border=True, help="Un rallye s'arrête après 4 s sans touche de balle.")
m[1].metric("Durée d'un rallye", f"{fmt_num(avg_len)} s", border=True,
            help="Moyenne, du premier au dernier contact détecté.")
m[2].metric("Passages de filet", fmt_num(avg_cross), border=True,
            help="Moyenne par rallye du nombre de fois où la balle change de camp : "
                 "plus il est haut, plus les échanges sont longs.")
m[3].metric("Touches par camp", fmt_num(S["touches_per_possession"]), border=True,
            help="Moyenne des touches d'une équipe avant de renvoyer la balle (3 au maximum, contre non compté).")
m[4].metric("Joueurs suivis", S["players_tracked"], border=True,
            help=f"Silhouettes suivies au moins {data['config']['min_track_s']:g} s sur le terrain.")

# ---------------------------------------------------------------- onglets

tab_video, tab_players, tab_flow, tab_court, tab_quality = st.tabs([
    ":material/movie: Vidéo", ":material/groups: Joueurs", ":material/timeline: Déroulé",
    ":material/sports_volleyball: Terrain", ":material/fact_check: Fiabilité et méthode"])

with tab_video:
    c1, c2 = st.columns([3, 2], gap="large")
    events = []
    for tc in data["touches"]:
        events.append({"t": tc["t"], "Temps": fmt_clock(tc["t"]), "Action": ACTION_LABEL[tc["action"]],
                       "Joueur": f"#{tc['player']}", "Équipe": names[tc["team"]]})
    for j in data["jumps"]:
        events.append({"t": j["t"], "Temps": fmt_clock(j["t"]), "Action": f"Saut ≈ {j['height_cm']} cm",
                       "Joueur": f"#{j['player']}", "Équipe": names[j["team"]]})
    ev = pd.DataFrame(events, columns=["t", "Temps", "Action", "Joueur", "Équipe"]).sort_values("t")
    with c2:
        st.markdown("**Moments clés**")
        kinds = st.multiselect("Filtrer", [ACTION_LABEL[a] for a in ACTIONS] + ["Saut"],
                               placeholder="Toutes les actions", label_visibility="collapsed")
        if kinds:
            ev = ev[ev["Action"].map(lambda s: s.split(" ")[0]).isin(kinds)]
        ev = ev.reset_index(drop=True)
        st.caption("Clique une ligne pour lancer la vidéo 2 s avant ce moment.")
        sel = st.dataframe(ev.drop(columns="t"), hide_index=True, height=min(420, 38 + 35 * max(1, len(ev))),
                           width="stretch",
                           on_select="rerun", selection_mode="single-row", key=f"ev_{ss['run']}_{kinds}")
    start = 0
    if sel.selection.rows:
        start = max(0, int(ev.iloc[sel.selection.rows[0]]["t"] - 2))
    with c1:
        video = run_dir / "annotated.mp4"
        if video.exists():
            st.video(str(video), start_time=start, autoplay=bool(sel.selection.rows), muted=True)
            st.caption(f"Cadres orange : {names['A']} · cadres bleus : {names['B']} · trait jaune : balle · "
                       "cercle blanc : touche détectée · lignes blanches : terrain, filet (épaisse) et lignes des 3 m.")
        else:
            st.info("Pas de vidéo annotée pour cette analyse (option désactivée au lancement).")

with tab_players:
    if players.empty:
        st.info("Aucun joueur suivi assez longtemps sur le terrain.")
    else:
        c1, c2, c3 = st.columns([3, 2, 2], vertical_alignment="bottom")
        team_f = c1.segmented_control("Équipe", ["all", "A", "B"], default="all",
                                      format_func=lambda k: "Les deux" if k == "all" else names[k]) or "all"
        top_vis = float(players["visible_s"].max())
        min_vis = c2.slider("Présence minimale (s)", 0.0, max(1.0, top_vis), min(2.0, top_vis), 0.5,
                            help="Masque les silhouettes suivies peu de temps (joueur mal suivi, passage bref).")
        active = c3.toggle("Seulement les joueurs avec une touche ou un saut")
        pv = players[players["visible_s"] >= min_vis].copy()
        if team_f != "all":
            pv = pv[pv["team"] == team_f]
        if active:
            pv = pv[(pv["touches"] > 0) | (pv["jumps"] > 0)]
        pv["Joueur"] = "#" + pv["id"].astype(str)
        pv["Équipe"] = pv["team"].map(names)
        pv["rythme"] = (pv["distance_m"] / (pv["visible_s"] / 60)).round().astype(int)
        pv["max_jump_cm"] = pv["max_jump_cm"].map(lambda v: f"{int(v)} cm" if pd.notna(v) else "—")
        cols = ["Joueur", "Équipe", "touches", *ACTIONS, "jumps", "max_jump_cm", "distance_m", "rythme", "visible_s"]
        cfg = {
            "touches": st.column_config.ProgressColumn("Touches", format="%d", min_value=0,
                                                       max_value=int(max(1, players["touches"].max())),
                                                       help="Toutes les touches de balle attribuées au joueur."),
            **{a: st.column_config.NumberColumn(ACTION_LABEL[a], format="%d", help=ACTION_HELP[a]) for a in ACTIONS},
            "jumps": st.column_config.NumberColumn("Sauts", format="%d", help="Sauts détectés."),
            "max_jump_cm": st.column_config.TextColumn("Saut max",
                                                       help="Estimé par le temps de vol : un ordre de grandeur."),
            "distance_m": st.column_config.NumberColumn("Distance (m)", format="%d",
                                                        help="Distance parcourue au sol pendant sa présence à l'image."),
            "rythme": st.column_config.NumberColumn("Rythme (m/min)", format="%d",
                                                    help="Distance ramenée à une minute de présence : "
                                                         "compare des joueurs vus plus ou moins longtemps."),
            "visible_s": st.column_config.NumberColumn("Présence (s)", format="%.1f",
                                                       help="Temps pendant lequel le joueur est suivi sur le terrain."),
        }
        st.dataframe(pv[cols], hide_index=True, width="stretch", column_config=cfg,
                     height=min(560, 38 + 35 * len(pv)))
        st.download_button("Télécharger le tableau (CSV)", pv[cols].to_csv(index=False, sep=";").encode("utf-8-sig"),
                           f"{ss['run']}_joueurs.csv", "text/csv", icon=":material/download:")

        st.markdown("#### Fiche joueur")
        ids = pv["id"].tolist() or players["id"].tolist()
        pid = st.selectbox("Joueur", ids, format_func=lambda i: f"#{i} · {names[players.set_index('id').at[i, 'team']]}")
        p = players.set_index("id").loc[pid]
        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            mm = st.columns(3)
            mm[0].metric("Touches", int(p["touches"]))
            mm[1].metric("Sauts", int(p["jumps"]))
            mm[2].metric("Distance", f"{int(p['distance_m'])} m")
            acts = [f"{ACTION_LABEL[a]} : {int(p[a])}" for a in ACTIONS if p[a]]
            st.markdown("**Actions** · " + (" · ".join(acts) if acts else "aucune"))
            if pd.notna(p["max_jump_cm"]):
                st.markdown(f"**Saut** · max ≈ {int(p['max_jump_cm'])} cm, moyen ≈ {int(p['avg_jump_cm'])} cm")
            st.markdown(f"**Présence** · {fmt_num(p['visible_s'])} s · suivi recollé depuis "
                        f"{int(p['fragments'])} morceau(x) de piste")
            mine = touches[touches["player"] == pid] if not touches.empty else touches
            if not mine.empty:
                st.dataframe(pd.DataFrame({"Temps": mine["t"].map(fmt_clock), "Action": mine["Action"]}),
                             hide_index=True, width="stretch", height=min(250, 38 + 35 * len(mine)))
        with c2:
            pos = to_uv(positions[positions["id"] == pid], view)
            layers = [alt.Chart(pos).mark_circle(size=40, opacity=0.45, color=TEAM_COLOR[p["team"]], clip=True
                                                 ).encode(cx(view), cy(view))]
            if not mine.empty:
                mt = to_uv(mine, view, "px", "py")
                layers.append(alt.Chart(mt).mark_point(size=120, filled=True, color=INK, stroke="white",
                                                       strokeWidth=2, clip=True).encode(
                    cx(view), cy(view), tooltip=[alt.Tooltip("Action:N"), alt.Tooltip("t:Q", title="Temps (s)")]))
            st.altair_chart(style_chart(court_chart(layers, view, names, 280)), theme=None, width="content")
            st.caption("Points de couleur : positions du joueur (5 par seconde). Points foncés : ses touches.")

with tab_flow:
    if touches.empty:
        st.info("Aucune touche détectée : vérifie la fiabilité de la détection de balle.")
    else:
        st.markdown("#### Chronologie des actions")
        rows = [ACTION_LABEL[a] for a in ACTIONS] + ["Saut"]
        pts = touches.assign(Joueur="#" + touches["player"].astype(str))[["t", "Action", "Équipe", "Joueur", "rally"]]
        if not jumps.empty:
            pts = pd.concat([pts, pd.DataFrame({"t": jumps["t"], "Action": "Saut", "Équipe": jumps["team"].map(names),
                                                "Joueur": "#" + jumps["player"].astype(str), "rally": None})])
        band = rallies.assign(s0=rallies["start_s"] - 0.3, s1=rallies["end_s"] + 0.3,
                              label="Rallye " + rallies["rally"].astype(str))
        band["lx"] = band["s0"].clip(lower=0)
        xs = alt.Scale(domain=[0, S["duration_s"]], nice=False)
        chart = alt.layer(
            alt.Chart(band).mark_rect(color="#F3F5F8", clip=True).encode(alt.X("s0:Q", scale=xs, title="Temps (s)"), x2="s1:Q"),
            alt.Chart(band).mark_text(align="left", baseline="top", dx=4, dy=4, color=MUTED, fontSize=11,
                                      clip=True).encode(
                alt.X("lx:Q", scale=xs, title="Temps (s)"), y=alt.value(0), text="label:N"),
            alt.Chart(pts).mark_point(filled=True, size=120, opacity=1, stroke="white", strokeWidth=1.5).encode(
                alt.X("t:Q", scale=xs, title="Temps (s)"), y=alt.Y("Action:N", sort=rows, title=None, scale=alt.Scale(domain=rows)),
                color=alt.Color("Équipe:N", scale=tscale, title=None),
                tooltip=[alt.Tooltip("t:Q", title="Temps (s)", format=".1f"), "Action:N", "Joueur:N", "Équipe:N"]),
        )
        st.altair_chart(style_chart(chart, 300), theme=None, width="stretch")
        st.caption("Chaque point est une touche (ou un saut), placée selon le moment et le rôle déduit. "
                   "Les bandes grises délimitent les rallyes.")

        c1, c2 = st.columns([2, 3], gap="large")
        with c1:
            st.markdown("#### Rallyes")
            rv = pd.DataFrame({
                "Rallye": rallies["rally"], "Début": rallies["start_s"].map(fmt_clock),
                "Durée (s)": rallies["duration_s"], "Touches": rallies["touches"],
                "Passages de filet": rallies["crossings"],
                "Au service": rallies["serve_team"].map(lambda k: names.get(k, "?") if k else "non vu"),
            })
            st.dataframe(rv, hide_index=True, width="stretch",
                         column_config={"Durée (s)": st.column_config.NumberColumn(format="%.1f")})
        with c2:
            st.markdown("#### Possessions")
            pv = pd.DataFrame({
                "Rallye": poss["rally"], "Équipe": poss["team"].map(names), "Début": poss["start_s"].map(fmt_clock),
                "Touches": poss["touches"],
                "Enchaînement": poss["actions"].map(lambda l: " → ".join(ACTION_LABEL[a] for a in l)),
                "Construite": poss["built"],
            })
            st.dataframe(pv, hide_index=True, width="stretch", column_config={
                "Construite": st.column_config.CheckboxColumn(help="Réception/défense → passe → attaque."),
                "Touches": st.column_config.NumberColumn(help="Plus de 3 : une touche est sans doute mal attribuée.")})

with tab_court:
    st.caption("Terrain vu de dessus. Positions calculées à partir des pieds des joueurs grâce au repère du terrain.")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### Occupation du terrain")
        team_h = st.segmented_control("Équipe ", ["all", "A", "B"], default="all", key="heat_team",
                                      format_func=lambda k: "Les deux" if k == "all" else names[k]) or "all"
        hp = positions if team_h == "all" else positions[positions["team"] == team_h]
        if hp.empty:
            st.info("Pas de positions disponibles.")
        else:
            period = max(1, round(data["fps"] / 5)) / data["fps"]
            g = hp.assign(x0=np.floor(hp["x"]), y0=np.floor(hp["y"])).groupby(["x0", "y0"]).size().reset_index(name="n")
            g["sec"] = (g["n"] * period).round(1)
            g = g.assign(x1=g["x0"] + 1, y1=g["y0"] + 1)
            if view == "side":
                g = g.assign(u=g["y0"], u2=g["y1"], v=g["x0"], v2=g["x1"])
            else:
                g = g.assign(u=g["x0"], u2=g["x1"], v=g["y0"], v2=g["y1"])
            heat = alt.Chart(g).mark_rect(opacity=0.9, clip=True).encode(
                cx(view), cy(view), x2="u2:Q", y2="v2:Q",
                color=alt.Color("sec:Q", title="Temps (s)", scale=alt.Scale(range=SEQ_BLUE)),
                tooltip=[alt.Tooltip("sec:Q", title="Temps cumulé (s)")])
            st.altair_chart(style_chart(court_chart([heat], view, names, 300)), theme=None, width="content")
            st.caption("Plus la case est foncée, plus les joueurs y ont passé de temps (cases de 1 m).")
    with c2:
        st.markdown("#### Où ont lieu les touches")
        if touches.empty:
            st.info("Aucune touche détectée.")
        else:
            acts = st.pills("Actions", [ACTION_LABEL[a] for a in ACTIONS if (touches["action"] == a).any()],
                            selection_mode="multi", key="court_acts")
            tp = touches if not acts else touches[touches["Action"].isin(acts)]
            tp = to_uv(tp.assign(Joueur="#" + tp["player"].astype(str)), view, "px", "py")
            dots = alt.Chart(tp).mark_point(filled=True, size=130, opacity=1, stroke="white", strokeWidth=2,
                                            clip=True).encode(
                cx(view), cy(view), color=alt.Color("Équipe:N", scale=tscale, title=None),
                tooltip=["Action:N", "Joueur:N", "Équipe:N", alt.Tooltip("t:Q", title="Temps (s)", format=".1f")])
            st.altair_chart(style_chart(court_chart([dots], view, names, 300)), theme=None, width="content")
            st.caption("Position au sol du joueur au moment de la touche (avant son saut s'il était en l'air).")

with tab_quality:
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown("#### Indicateurs de qualité")
        q = st.columns(2)
        q[0].metric("Balle détectée", pct(S["ball_detection_rate"]), border=True,
                    help="Part des images où la balle est trouvée. En dessous de 75 %, des touches manquent.")
        q[1].metric("Contacts sans joueur", S["contacts_sans_joueur"], border=True,
                    help="Changements de trajectoire sans joueur à portée : sol, filet, antenne, "
                         "ou joueur non détecté. Ils ne comptent pas comme touches.")
        q = st.columns(2)
        q[0].metric("Possessions suspectes", S["possessions_suspectes"], border=True,
                    help="Plus de 3 touches du même camp : une touche adverse (souvent au filet) a été attribuée "
                         "au mauvais camp, ou une touche a été vue deux fois.")
        q[1].metric("Pistes → joueurs", f"{S['track_ids_raw']} → {S['players_tracked']}", border=True,
                    help="Le suivi crée une piste par silhouette et en perd parfois (croisements, filet). "
                         "Les morceaux d'un même joueur sont recollés ; arbitres et juges de ligne sont écartés.")
        st.markdown("#### Comment les chiffres sont obtenus")
        st.markdown(
            "- **Joueurs** : détectés image par image (squelette), suivis dans le temps, puis placés en mètres "
            "sur le terrain grâce aux 4 coins repérés. Le côté du filet donne l'équipe.\n"
            "- **Touches** : la balle change brusquement de trajectoire près des mains d'un joueur. "
            "Quand deux joueurs de camps opposés sont à portée (au filet), on choisit celui qui respecte "
            "les règles : 3 touches maximum par camp, pas deux touches de suite pour le même joueur.\n"
            "- **Rôles** : déduits de l'ordre des touches dans chaque possession (voir le guide).\n"
            "- **Sauts** : montée nette des chevilles puis retour au même endroit ; la hauteur vient du temps "
            "de vol (h = g·t²/8), c'est un ordre de grandeur.\n"
            "- **Distances** : positions au sol lissées, 5 fois par seconde, sauts exclus.")
        st.page_link(PAGES["guide"], label="Guide complet et lexique", icon=":material/menu_book:")
    with c2:
        st.markdown("#### Repère du terrain")
        vid = Path(meta.get("video", ""))
        vid = vid if vid.is_absolute() else ROOT / vid
        still = court_still(str(vid), meta["corners"], view, min(1.0, S["duration_s"] / 2)) if vid.exists() else None
        if still is not None:
            st.image(still, width="stretch")
        st.caption("La ligne épaisse doit tomber sous le filet et les lignes fines sur les lignes des 3 m. "
                   "Sinon, les camps et les distances sont faux : recalibre (quelques secondes, sans GPU).")
        if st.button("Recalibrer le terrain", icon=":material/crop_free:", disabled=not vid.exists()):
            ss["recal"] = ss["run"]
            st.switch_page(PAGES["upload"])
        with st.expander("Seuils utilisés"):
            cfg_df = pd.DataFrame([{"Paramètre": k, "Valeur": v, "Rôle": CFG_HELP.get(k, "")}
                                   for k, v in data["config"].items()])
            st.dataframe(cfg_df, hide_index=True, width="stretch")
            st.caption("Modifiables dans CFG en haut de analyze.py, puis `python analyze.py <video> --reuse`.")
        st.download_button("Télécharger toutes les données (JSON)",
                           json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"),
                           f"{ss['run']}.json", "application/json", icon=":material/download:")
