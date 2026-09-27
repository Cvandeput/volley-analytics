"""Page : importer une vidéo, repérer le terrain, lancer l'analyse en arrière-plan."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates

import analyze as A
from views.common import (DEFAULT_TEAM_NAMES, RESULTS, ROOT, STEP_LABEL, STEPS, UPLOADS, VIEW_LABEL,
                          env_utf8, fmt_date, fmt_duration, fmt_num, gpu_name, job_status, kill, list_runs,
                          open_run, read_json, slug, step_header, unique_run_name)

VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "m4v", "webm"]
PREVIEW_MAX_MB = 600
ss = st.session_state


# ---------------------------------------------------------------- helpers

@st.cache_data(show_spinner=False)
def cached_info(path, mtime):
    return A.video_info(path)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_frame(path, t):
    return A.read_frame(path, t)


@st.cache_resource(show_spinner="Chargement du modèle de détection du terrain…")
def court_model():
    from ultralytics import YOLO
    return YOLO(str(A.COURT_MODEL)) if A.COURT_MODEL.exists() else None


def set_source(path):
    ss["src"] = str(path)
    ss["corners"] = []
    ss.pop("recal", None)


def local_videos():
    vids = [p for ext in VIDEO_TYPES for p in UPLOADS.glob(f"*.{ext}")]
    vids += sorted((ROOT / "VolleyVision" / "data").glob("**/*.mp4"))
    return vids


def last_speed():
    """Images analysées par seconde lors des analyses précédentes (pour estimer la durée)."""
    for r in list_runs():
        if r["meta"].get("inference_fps"):
            return r["meta"]["inference_fps"]
    return None


def download(url):
    import yt_dlp

    UPLOADS.mkdir(exist_ok=True)
    bar = st.progress(0.0, text="Téléchargement…")

    def hook(d):
        total = d.get("total_bytes") or d.get("total_bytes_estimate")
        if d["status"] == "downloading" and total:
            frac = min(1.0, d["downloaded_bytes"] / total)
            bar.progress(frac, text=f"Téléchargement… {frac:.0%}")

    opts = {"outtmpl": str(UPLOADS / "%(title).60B.%(ext)s"), "restrictfilenames": True,
            "format": "bv*[height<=1080][ext=mp4]/bv*[height<=1080]/b[height<=1080]/b",
            "quiet": True, "no_warnings": True, "noprogress": True, "progress_hooks": [hook]}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            path = Path(ydl.prepare_filename(ydl.extract_info(url, download=True)))
    except Exception as e:
        bar.empty()
        st.error(f"Téléchargement impossible : {e}")
        return
    bar.empty()
    set_source(path)
    st.rerun()


def auto_detect(frame):
    model = court_model()
    pts = A.detect_court(frame, model) if model is not None else None
    if pts:
        ss["corners"] = pts
        st.toast("Terrain détecté automatiquement. Vérifie le tracé.", icon=":material/auto_fix_high:")
    else:
        st.toast("Détection automatique impossible : clique les 4 coins.", icon=":material/touch_app:")


def overlay(frame, corners, view):
    img = frame.copy()
    thick = max(2, img.shape[1] // 400)
    if len(corners) == 4:
        A.Court(corners, view).draw(img, (0, 230, 255), thick)
    elif len(corners) > 1:
        cv2.polylines(img, [np.int32(corners)], False, (0, 230, 255), thick, cv2.LINE_AA)
    for x, y in corners:
        cv2.circle(img, (int(x), int(y)), 5 * thick, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 4 * thick, (0, 230, 255), -1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def launch(src, name, corners, view, title, teams, opts, portion=None, reuse=False):
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    ordered = [[int(x), int(y)] for x, y in A.order_corners(corners)]
    (out / "corners.json").write_text(json.dumps(ordered))
    cmd = [sys.executable, "-u", str(ROOT / "analyze.py"), str(src), "--out", str(RESULTS), "--name", name,
           "--title", title, "--view", view, "--team-a", teams["A"], "--team-b", teams["B"]]
    if reuse:
        cmd.append("--reuse")
    else:
        cmd += ["--ball", opts["ball"], "--ball-imgsz", str(opts["ball_imgsz"]),
                "--ball-conf", str(opts["ball_conf"])]
        if opts["device"]:
            cmd += ["--device", opts["device"]]
        if portion:
            cmd += ["--start", f"{portion[0]:.2f}", "--end", f"{portion[1]:.2f}"]
    if not opts["video"]:
        cmd.append("--no-video")
    with open(out / "log.txt", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env_utf8(),
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    now = time.time()
    (out / "status.json").write_text(json.dumps({"state": "running", "step": "prepare", "pid": proc.pid,
                                                 "started": now, "step_started": now, "updated": now}))
    ss["job"] = name


# ---------------------------------------------------------------- suivi de l'analyse

@st.fragment(run_every=1.0)
def job_progress(name):
    s = job_status(RESULTS / name) or {"state": "running", "step": "prepare"}
    if s["state"] != "running":
        st.rerun(scope="app")
    meta = read_json(RESULTS / name / "meta.json") or {}
    step = s.get("step", "prepare")
    k = min(STEPS.index(step), 3) if step in STEPS else 0
    done, total = s.get("done", 0), s.get("total", 0)
    now = time.time()
    with st.container(border=True):
        st.markdown(f"**Analyse en cours : {meta.get('title', name)}**")
        text = f"Étape {k + 1}/4 · {STEP_LABEL.get(step, step)}"
        if total:
            text += f" — {done:,}/{total:,} images".replace(",", " ")
        st.progress(done / total if total else 0.0, text=text)
        info = f"Temps écoulé : {fmt_duration(now - s.get('started', now))}"
        elapsed = s.get("updated", now) - s.get("step_started", now)
        if total and done > 10 and elapsed > 2:
            info += f" · reste ≈ {fmt_duration((total - done) * elapsed / done)} pour cette étape"
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.caption(info + ". Tu peux changer de page, l'analyse continue.")
        if c2.button("Annuler", icon=":material/stop_circle:", width="stretch"):
            kill(s.get("pid"))
            (RESULTS / name / "status.json").write_text(json.dumps({"state": "error", "message": "Analyse annulée."}))
            st.rerun(scope="app")


def job_result(name, s):
    meta = read_json(RESULTS / name / "meta.json") or {}
    with st.container(border=True):
        if s["state"] == "done":
            st.success(f"Analyse « {meta.get('title', name)} » terminée.", icon=":material/check_circle:")
            c1, c2, _ = st.columns([1.3, 1, 3])
            if c1.button("Voir le tableau de bord", type="primary", icon=":material/monitoring:", width="stretch"):
                ss.pop("job", None)
                open_run(name)
            if c2.button("Fermer", width="stretch"):
                ss.pop("job", None)
                st.rerun()
        else:
            st.error(f"L'analyse « {meta.get('title', name)} » n'a pas abouti : {s.get('message', 'erreur inconnue')}",
                     icon=":material/error:")
            log = RESULTS / name / "log.txt"
            if log.exists():
                with st.expander("Journal technique"):
                    st.code("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]),
                            language=None)
            if st.button("Fermer"):
                ss.pop("job", None)
                st.rerun()


# ---------------------------------------------------------------- page

st.title("Nouvelle analyse")
st.markdown('<p class="lead">Importe une vidéo filmée en plan fixe, repère le terrain, puis lance l\'analyse. '
            "Elle tourne en arrière-plan : le tableau de bord s'ouvre quand elle est finie.</p>",
            unsafe_allow_html=True)

if ss.get("job"):
    s = job_status(RESULTS / ss["job"])
    if s is None or s["state"] == "running":
        job_progress(ss["job"])
    else:
        job_result(ss["job"], s)

recal = ss.get("recal")
recal_meta = read_json(RESULTS / recal / "meta.json") if recal else None
if recal and recal_meta:
    video = Path(recal_meta["video"])
    ss["src"] = str(video if video.is_absolute() else ROOT / video)
    if ss.get("recal_loaded") != recal:
        ss["recal_loaded"] = recal
        ss["corners"] = read_json(RESULTS / recal / "corners.json") or []
        ss["auto_tried"] = ss["src"]
    with st.container(border=True):
        st.markdown(f"**Recalibrage de « {recal_meta.get('title', recal)} »**")
        st.caption("Replace les 4 coins du terrain puis relance le calcul. La détection des joueurs et de la balle "
                   "est réutilisée : c'est l'affaire de quelques secondes (plus la vidéo annotée).")
        if st.button("Annuler le recalibrage"):
            ss.pop("recal", None)
            ss.pop("recal_loaded", None)
            ss.pop("src", None)
            st.rerun()
else:
    recal = None
    with st.container(border=True):
        step_header(1, "Choisir la vidéo", done=bool(ss.get("src")))
        t_file, t_url, t_disk = st.tabs([":material/upload_file: Importer un fichier",
                                         ":material/link: Lien YouTube ou web",
                                         ":material/folder_open: Vidéo déjà sur l'ordinateur"])
        with t_file:
            up = st.file_uploader("Vidéo", type=VIDEO_TYPES, label_visibility="collapsed",
                                  help="Plan fixe, terrain entier visible. MP4 conseillé.")
            if up is not None and ss.get("uploaded") != (up.name, up.size):
                UPLOADS.mkdir(exist_ok=True)
                dst = UPLOADS / f"{slug(Path(up.name).stem)}{Path(up.name).suffix.lower()}"
                with st.spinner("Enregistrement de la vidéo…"), open(dst, "wb") as f:
                    shutil.copyfileobj(up, f, 16 << 20)
                ss["uploaded"] = (up.name, up.size)
                set_source(dst)
        with t_url:
            url = st.text_input("Adresse de la vidéo", placeholder="https://www.youtube.com/watch?v=…")
            st.caption("Téléchargée en 1080p maximum, sans le son, dans le dossier uploads.")
            if st.button("Télécharger", icon=":material/download:", disabled=not url):
                download(url)
        with t_disk:
            vids = local_videos()
            pick = st.selectbox("Vidéo", vids, index=None, placeholder="Choisis une vidéo",
                                format_func=lambda p: str(p.relative_to(ROOT)), label_visibility="collapsed")
            if st.button("Utiliser cette vidéo", disabled=pick is None):
                set_source(pick)
        if ss.get("src"):
            st.caption(f":material/movie: Vidéo sélectionnée : **{Path(ss['src']).name}**")

src = ss.get("src")
if src and not Path(src).exists():
    st.error(f"Vidéo introuvable : {src}")
    src = None

if src:
    info = cached_info(src, Path(src).stat().st_mtime)
    dur, fps = info["duration"], info["fps"]
    if not info["frames"]:
        st.error("Impossible de lire cette vidéo (format ou codec non pris en charge).")
        st.stop()

    # -------- étape 2 : portion
    portion = None
    t_lo, t_hi = 0.0, dur
    if not recal:
        with st.container(border=True):
            step_header(2, "Choisir la portion à analyser", done=True)
            c1, c2 = st.columns([3, 2], gap="large")
            with c1:
                if Path(src).stat().st_size < PREVIEW_MAX_MB * 2 ** 20:
                    st.video(src)
                else:
                    st.caption("Aperçu désactivé : fichier volumineux.")
            with c2:
                st.markdown(f"**{fmt_duration(dur)}** · {info['W']}×{info['H']} · {fmt_num(fps, 0)} images/s")
                step = 0.5 if dur < 600 else 1.0
                t_lo, t_hi = st.slider("Portion (secondes)", 0.0, float(round(dur, 1)), (0.0, float(round(dur, 1))),
                                       step=step, key=f"range_{src}",
                                       help="Seule cette portion sera analysée. Pratique pour un set ou un rallye précis.")
                n = int((t_hi - t_lo) * fps)
                speed = last_speed()
                gpu = gpu_name()
                est = f" · durée estimée ≈ **{fmt_duration(n / speed * 1.25)}**" if speed else ""
                n_txt = f"{n:,}".replace(",", " ")
                st.markdown(f"{fmt_duration(t_hi - t_lo)} à analyser, soit {n_txt} images{est}.")
                st.caption((f"Calcul sur la carte graphique {gpu}." if gpu else
                            "Pas de carte graphique détectée : l'analyse sera lente (processeur).")
                           + " Pour un premier essai, 1 à 2 minutes suffisent.")
                if t_hi - t_lo < 1:
                    st.warning("Choisis une portion d'au moins une seconde.")
                    st.stop()
                if t_lo > 0 or t_hi < round(dur, 1):
                    portion = (t_lo, t_hi)

    # -------- étape 3 : terrain
    ss.setdefault("corners", [])
    with st.container(border=True):
        step_header(3 if not recal else 1, "Repérer le terrain", done=len(ss["corners"]) == 4)
        st.markdown('<p class="hint">Ce repère sert à placer chaque joueur en mètres, à savoir de quel côté du filet '
                    "il joue et à mesurer ses déplacements. Clique les 4 coins extérieurs du terrain, "
                    "dans n'importe quel ordre.</p>", unsafe_allow_html=True)
        c1, c2 = st.columns([3, 1.25], gap="large")
        with c2:
            default_view = (recal_meta or {}).get("view", ss.get("view", "back"))
            view = st.radio("Position de la caméra", ["back", "side"], format_func=VIEW_LABEL.get,
                            index=["back", "side"].index(default_view), key=f"view_{src}")
            ss["view"] = view
            t_img = st.slider("Image utilisée", float(t_lo), float(max(t_lo, t_hi - 0.1)), float(t_lo), step=0.5,
                              key=f"frame_{src}", help="Choisis une image où les 4 coins sont visibles.")
            frame = cached_frame(src, t_img)
            if ss.get("auto_tried") != src and not ss["corners"] and A.COURT_MODEL.exists():
                ss["auto_tried"] = src
                auto_detect(frame)
            b1, b2 = st.columns(2)
            if b1.button("Auto", icon=":material/auto_fix_high:", width="stretch",
                         help="Détection automatique du terrain", disabled=not A.COURT_MODEL.exists()):
                auto_detect(frame)
            if b2.button("Effacer", icon=":material/restart_alt:", width="stretch"):
                ss["corners"] = []
            n_pts = len(ss["corners"])
            if n_pts < 4:
                st.info(f"Coins placés : {n_pts}/4. Clique sur l'image pour en ajouter.", icon=":material/touch_app:")
            else:
                st.success("Terrain repéré.", icon=":material/check:")
                st.caption("Vérifie le tracé : la ligne épaisse doit tomber sur la ligne centrale, sous le filet, "
                           "et les deux lignes fines sur les lignes des 3 m. Sinon, efface et clique les coins à la main.")
            if view == "side":
                st.caption("Vue de côté : les équipes sont « côté gauche » et « côté droit ».")
        with c1:
            click = streamlit_image_coordinates(overlay(frame, ss["corners"], view), key=f"court_{src}",
                                                width="stretch", image_format="JPEG", jpeg_quality=85,
                                                cursor="crosshair")
            if click and click.get("unix_time") != ss.get("last_click"):
                ss["last_click"] = click["unix_time"]
                if len(ss["corners"]) < 4:
                    scale = info["W"] / click["width"]
                    ss["corners"].append([int(click["x"] * scale), int(click["y"] * scale)])
                    st.rerun()

    # -------- étape 4 : lancement
    names = DEFAULT_TEAM_NAMES[view]
    saved = (recal_meta or {}).get("team_names", {})
    with st.container(border=True):
        step_header(4 if not recal else 2, "Lancer l'analyse" if not recal else "Relancer le calcul")
        default_title = (recal_meta or {}).get("title") or Path(src).stem.replace("_", " ")
        title = st.text_input("Nom de l'analyse", value=default_title, key=f"title_{src}")
        c1, c2 = st.columns(2)
        teams = {
            "A": c1.text_input(f"Équipe {names['A'].lower()}", value=saved.get("A", ""), key=f"team_a_{src}",
                               placeholder="ex. France", help=f"Laisser vide pour afficher « {names['A']} »."),
            "B": c2.text_input(f"Équipe {names['B'].lower()}", value=saved.get("B", ""), key=f"team_b_{src}",
                               placeholder="ex. Italie", help=f"Laisser vide pour afficher « {names['B']} »."),
        }
        opts = {"ball": str(A.DEFAULT_BALL), "ball_imgsz": 1280, "ball_conf": 0.25, "device": None, "video": True}
        with st.expander("Options avancées"):
            if not recal:
                models = [A.DEFAULT_BALL] + [p for p in ROOT.glob("*.pt") if "ball" in p.name.lower()]
                opts["ball"] = str(st.selectbox("Modèle de détection de la balle", [m for m in models if m.exists()]
                                                or [A.DEFAULT_BALL], format_func=lambda p: str(p.relative_to(ROOT))))
                c1, c2, c3 = st.columns(3)
                opts["device"] = {"Automatique": None, "Carte graphique": "0", "Processeur": "cpu"}[
                    c1.selectbox("Calcul", ["Automatique", "Carte graphique", "Processeur"])]
                opts["ball_imgsz"] = c2.selectbox("Résolution balle", [960, 1280, 1600], index=1,
                                                  help="Plus haut : balle mieux vue de loin, analyse plus lente.")
                opts["ball_conf"] = c3.slider("Seuil balle", 0.05, 0.6, 0.25, 0.05,
                                              help="Plus bas : plus de balles détectées, mais plus de fausses alertes.")
            opts["video"] = st.toggle("Créer la vidéo annotée", value=True,
                                      help="Vidéo avec joueurs, balle et actions dessinés. Ajoute quelques minutes.")
        ready = len(ss["corners"]) == 4
        busy = bool(ss.get("job")) and (job_status(RESULTS / ss["job"]) or {}).get("state") == "running"
        label = "Lancer l'analyse" if not recal else "Relancer le calcul"
        if st.button(label, type="primary", icon=":material/play_arrow:", disabled=not ready or busy):
            name = recal or unique_run_name(title)
            launch(src, name, ss["corners"], view, title.strip() or name, teams, opts, portion, reuse=bool(recal))
            ss.pop("recal", None)
            ss.pop("recal_loaded", None)
            st.rerun()
        if not ready:
            st.caption("Place d'abord les 4 coins du terrain.")
        elif busy:
            st.caption("Une analyse est déjà en cours.")

# ---------------------------------------------------------------- historique

runs = list_runs()
if runs:
    st.subheader("Analyses récentes")
    for r in runs[:8]:
        s = job_status(r["dir"]) or {"state": "done" if r["has_results"] else "error"}
        state = {"running": ":blue-badge[:material/progress_activity: En cours]",
                 "done": ":green-badge[:material/check: Terminée]",
                 "error": ":red-badge[:material/error: Échec]"}.get(s["state"], "")
        c1, c2, c3 = st.columns([4, 2, 1.2], vertical_alignment="center")
        c1.markdown(f"**{r['title']}**  \n<span class='muted'>{fmt_date(r['meta'].get('created'))}</span>",
                    unsafe_allow_html=True)
        c2.markdown(state)
        if c3.button("Ouvrir", key=f"open_{r['name']}", disabled=not r["has_results"], width="stretch"):
            open_run(r["name"])
