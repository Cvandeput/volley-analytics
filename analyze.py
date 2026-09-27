"""
Analyse d'une video de volley : joueurs (pose + suivi), balle, touches, actions, sauts, deplacements.

Premiere analyse (GPU) :
    python analyze.py <video> [--ball <poids_balle.pt>] [--start 30 --end 90]
Reanalyse sans GPU apres modification des seuils CFG (quelques secondes) :
    python analyze.py <video> --reuse
Recliquer les coins du terrain :
    python analyze.py <video> --reuse --reclick

L'interface (streamlit run app.py) lance ce script en arriere-plan et suit status.json.
Sorties dans results/<nom>/ : results.json, annotated.mp4, corners.json, meta.json, raw.json, status.json
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import time
import warnings
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_BALL = ROOT / "VolleyVision/models/yolov8_tracking/best.pt"
COURT_MODEL = ROOT / "VolleyVision/models/yolov8_court/best.pt"
TRACKER = ROOT / "tracker.yaml"
VERSION = 2

# Seuils : modifie-les puis relance avec --reuse
CFG = {
    "court_side_margin_m": 1.5,   # joueurs gardes jusqu'a X m hors des lignes de cote
    "court_end_margin_m": 6.0,    # et jusqu'a X m derriere les lignes de fond (serveurs)
    "track_max_side_m": 0.3,      # piste ecartee si sa position mediane est hors des lignes de cote (arbitres, banc)
    "track_min_spread_m": 1.0,    # piste immobile hors du terrain ecartee (juges de ligne)
    "net_switch_s": 1.0,          # piste coupee si elle passe de l'autre cote du filet plus de X s (echange d'identite)
    "stitch_gap_s": 2.0,          # recollage des pistes coupees : trou max
    "stitch_speed_mps": 6.0,      # vitesse max supposee pendant le trou
    "stitch_radius_m": 1.5,       # tolerance de position au recollage
    "ball_max_jump": 0.08,        # deplacement max entre 2 images tant que la vitesse est inconnue (part de la largeur)
    "ball_gate": 0.04,            # ecart max a la position predite par la vitesse (part de la largeur)
    "ball_track_gap": 8,          # images sans detection avant de clore une piste de balle
    "ball_min_track": 3,          # pistes plus courtes ignorees (fausses detections isolees)
    "ball_static": 0.012,         # piste qui bouge moins que ca : balle immobile ecartee (part de la largeur)
    "ball_spike": 0.02,           # point retire s'il s'ecarte autant des courbes d'avant et d'apres (part de la largeur)
    "ball_max_gap": 6,            # trous de detection combles (images)
    "touch_k": 3,                 # fenetre (frames) des vitesses avant/apres
    "touch_min_speed": 0.004,     # vitesse mini balle (fraction largeur / frame)
    "touch_strike_ratio": 2.0,    # vitesse apres > ratio * vitesse avant = frappe
    "touch_min_gap_s": 0.30,      # ecart mini entre 2 touches
    "touch_reach": 0.6,           # distance balle-mains max, en hauteurs de joueur
    "assign_w_dist": 2.0,         # attribution des touches : cout de la distance balle-mains
    "assign_w_four": 1.5,         # penalite d'une 4e touche d'affilee du meme camp
    "assign_w_double": 1.0,       # penalite de deux touches de suite du meme joueur
    "block_net_m": 1.5,           # contre : touche a moins de X m du filet...
    "block_delay_s": 0.8,         # ... moins de X s apres une touche adverse
    "jump_rise": 0.12,            # montee mini des chevilles (fraction hauteur joueur)
    "jump_air": 0.03,             # seuil "en l'air" pour le temps de vol
    "jump_min_air_s": 0.3,        # en dessous : bruit de detection
    "jump_max_air_s": 1.0,        # au-dessus : deplacement pris pour un saut (record humain ~0.9 s)
    "jump_min_gap_s": 0.5,
    "jump_land_tol": 0.35,        # ecart max sol avant/apres (fraction hauteur)
    "rally_gap_s": 4.0,           # 4 s sans touche = nouveau rallye
    "min_track_s": 1.0,           # pistes plus courtes ignorees dans le tableau joueurs
    "move_max_speed_mps": 8.0,    # pas plus rapides ignores dans la distance (saut d'identite)
}

COURT_W, COURT_L, NET_Y, ATTACK_M = 9.0, 18.0, 9.0, 3.0
TEAMS = ("A", "B")  # A : y < 9 (loin de la camera, ou a gauche en vue de cote), B : y > 9
ACTIONS = ("service", "reception", "defense", "passe", "attaque", "contre")
ACTION_TXT = {"service": "Service", "reception": "Reception", "defense": "Defense",
              "passe": "Passe", "attaque": "Attaque", "contre": "Contre"}
KP_CONF = 0.3
L_WRIST, R_WRIST, L_ANKLE, R_ANKLE = 9, 10, 15, 16
TEAM_BGR = {"A": (52, 104, 235), "B": (214, 120, 42)}  # orange / bleu, comme le tableau de bord
BALL_BGR = (0, 230, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- avancement

class Status:
    """Ecrit l'avancement dans status.json, lu par l'interface."""

    def __init__(self, path):
        self.path, self.step, self.last = path, None, 0.0
        self.started = self.step_started = time.time()

    def __call__(self, step, done=0, total=0):
        now = time.time()
        if step != self.step:
            self.step, self.step_started, self.last = step, now, 0.0
        if now - self.last < 0.5 and done != total:
            return
        self.last = now
        self._write({"state": "running", "step": step, "done": done, "total": total})

    def finish(self, state="done", message=""):
        self._write({"state": state, "step": self.step, "message": message})

    def _write(self, d):
        d.update(pid=os.getpid(), started=self.started, step_started=self.step_started, updated=time.time())
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        for _ in range(5):
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:  # fichier lu au meme moment par l'interface (Windows)
                time.sleep(0.05)


# ---------------------------------------------------------------- video

def video_info(video):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frames": n, "W": W, "H": H, "duration": n / fps if fps else 0}


def read_frame(video, t=0.0):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Impossible de lire {video}")
    return img


def trim(src, dst, start, end):
    """Copie la portion [start, end] (s) de la video, sans le son."""
    ff = shutil.which("ffmpeg")
    if ff:
        cmd = [ff, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src)]
        if end:
            cmd += ["-t", f"{end - start:.3f}"]
        cmd += ["-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]
        subprocess.run(cmd, check=True)
        return
    info = video_info(src)
    cap = cv2.VideoCapture(str(src))
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["W"], info["H"]))
    for _ in range(int(((end or info["duration"]) - start) * info["fps"])):
        ok, img = cap.read()
        if not ok:
            break
        vw.write(img)
    cap.release()
    vw.release()


# ---------------------------------------------------------------- terrain

def order_corners(pts):
    """4 coins dans n'importe quel ordre -> haut gauche, haut droit, bas droit, bas gauche (image)."""
    pts = sorted((tuple(float(v) for v in p) for p in pts), key=lambda p: p[1])
    (tl, tr), (bl, br) = sorted(pts[:2]), sorted(pts[2:])
    return [tl, tr, br, bl]


class Court:
    """Terrain de 9 x 18 m. Repere : x en travers (0-9), y dans la longueur (0-18), filet en y = 9.

    view = "back" : camera derriere une ligne de fond (le filet traverse l'image),
    view = "side" : camera sur le cote (le filet est vertical dans l'image).
    """

    def __init__(self, corners, view="back"):
        self.corners = order_corners(corners)
        self.view = view
        if view == "side":
            dst = [[0, 0], [0, COURT_L], [COURT_W, COURT_L], [COURT_W, 0]]
        else:
            dst = [[0, 0], [COURT_W, 0], [COURT_W, COURT_L], [0, COURT_L]]
        self.H = cv2.getPerspectiveTransform(np.float32(self.corners), np.float32(dst))
        self.Hinv = np.linalg.inv(self.H)
        center = np.mean(self.corners, axis=0)
        self._w_sign = np.sign(self.H[2] @ [center[0], center[1], 1.0])

    @staticmethod
    def _apply(M, pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        h = np.c_[p, np.ones(len(p))] @ M.T
        with np.errstate(divide="ignore", invalid="ignore"):
            return h[:, :2] / h[:, 2:], h[:, 2]

    def to_court(self, pts):
        """Points image (N, 2) -> metres (N, 2) ; NaN au-dela de l'horizon."""
        out, w = self._apply(self.H, pts)
        out[w * self._w_sign <= 1e-9] = np.nan
        return out

    def to_image(self, pts):
        return self._apply(self.Hinv, pts)[0]

    def contains(self, x, y, cfg):
        sm, em = cfg["court_side_margin_m"], cfg["court_end_margin_m"]
        return -sm <= x <= COURT_W + sm and -em <= y <= COURT_L + em

    def draw(self, img, color=(255, 255, 255), thick=1):
        """Contour, ligne centrale (filet) et lignes des 3 m projetees sur l'image."""
        def seg(y, c, t):
            a, b = self.to_image([(0, y), (COURT_W, y)]).astype(int)
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), c, t, cv2.LINE_AA)
        cv2.polylines(img, [np.int32(self.corners)], True, color, thick, cv2.LINE_AA)
        seg(NET_Y, color, thick + 1)
        for y in (NET_Y - ATTACK_M, NET_Y + ATTACK_M):
            seg(y, color, max(1, thick - 1))
        return img


def side_of(y):
    return "A" if y < NET_Y else "B"


def detect_court(img, model):
    """Propose les 4 coins avec le modele de segmentation du terrain (None si echec)."""
    r = model.predict(img, verbose=False, retina_masks=True)[0]
    if r.masks is None or not len(r.masks):
        return None
    i = int(r.boxes.conf.argmax())
    mask = (r.masks.data[i].cpu().numpy() > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 30):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) <= 4:
            break
    if len(approx) != 4:
        return None
    return [[int(x), int(y)] for x, y in order_corners(approx.reshape(-1, 2).tolist())]


def pick_corners(video, path, reclick=False):
    if path.exists() and not reclick:
        return json.loads(path.read_text())
    base = read_frame(video)
    pts = []
    if COURT_MODEL.exists():
        from ultralytics import YOLO
        pts = [tuple(p) for p in detect_court(base, YOLO(str(COURT_MODEL))) or []]

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    win = "Coins du terrain"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)
    while True:
        view = base.copy()
        for p in pts:
            cv2.circle(view, tuple(map(int, p)), 7, (0, 0, 255), -1)
        if len(pts) == 4:
            Court(pts).draw(view, (0, 255, 255), 2)
        msg = (f"Clique le coin {len(pts) + 1}/4 (ordre libre)" if len(pts) < 4
               else "Entree : valider   R : recommencer")
        cv2.putText(view, msg, (20, 45), FONT, 1.0, (0, 0, 0), 5)
        cv2.putText(view, msg, (20, 45), FONT, 1.0, (255, 255, 255), 2)
        cv2.imshow(win, view)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(pts) == 4:
            break
        if key in (ord("r"), ord("R")):
            pts.clear()
        if key == 27:
            raise SystemExit("Annule")
    cv2.destroyAllWindows()
    pts = [[int(x), int(y)] for x, y in order_corners(pts)]
    path.write_text(json.dumps(pts))
    return pts


# ---------------------------------------------------------------- inference (GPU)

def default_device():
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def run_inference(video, ball_weights, args, status):
    from ultralytics import YOLO

    pose = YOLO(args.pose)
    ball = YOLO(ball_weights)
    ball_cls = [i for i, n in ball.names.items() if "ball" in n.lower()] or None
    print(f"Classes balle : {[ball.names[i] for i in ball_cls] if ball_cls else 'toutes'}")

    info = video_info(video)
    total = info["frames"]
    tracker = str(TRACKER) if TRACKER.exists() else "botsort.yaml"
    frames = []
    stream = pose.track(str(video), tracker=tracker, imgsz=args.imgsz, conf=0.2,
                        device=args.device, stream=True, verbose=False)
    for i, r in enumerate(stream):
        players = []
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id.int().cpu().tolist()
            kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            for j, (b, tid) in enumerate(zip(boxes, ids)):
                players.append({
                    "id": tid,
                    "box": [round(float(v), 1) for v in b],
                    "feet": [round(float((b[0] + b[2]) / 2), 1), round(float(b[3]), 1)],
                    "kpts": np.round(kps[j], 2).tolist() if kps is not None else None,
                })
        br = ball.predict(r.orig_img, imgsz=args.ball_imgsz, conf=args.ball_conf,
                          classes=ball_cls, device=args.device, verbose=False)[0]
        cands = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                 for (x, y, _, _), c in zip(br.boxes.xywh.cpu().numpy(), br.boxes.conf.cpu().numpy())]
        frames.append({"players": players, "ball": cands})
        status("inference", i + 1, max(total, i + 1))
        if i % 100 == 0:
            print(f"  frame {i}/{total}", flush=True)
    return {"fps": info["fps"], "W": info["W"], "H": info["H"], "frames": frames}


# ---------------------------------------------------------------- joueurs

def place_players(raw_frames, court, cfg):
    """Garde les joueurs dont les pieds sont sur le terrain (avec marge), position en metres."""
    frames = []
    for fr in raw_frames:
        players = []
        if fr["players"]:
            pos = court.to_court([p["feet"] for p in fr["players"]])
            for p, (x, y) in zip(fr["players"], pos):
                if not np.isnan(x) and court.contains(x, y, cfg):
                    players.append({**p, "pos": (round(float(x), 2), round(float(y), 2))})
        frames.append({"players": players, "ball": fr["ball"]})
    return frames


def side_runs(f, pos, min_len):
    """Morceaux d'une piste restant du meme cote du filet ; un passage bref de l'autre cote
    (joueur en l'air au filet, pieds mal projetes) ne coupe pas la piste."""
    runs = []
    for k, (_, y) in enumerate(pos):
        s = side_of(y)
        if runs and runs[-1][0] == s:
            runs[-1][2] = k
        else:
            runs.append([s, k, k])
    merged = []
    for r in runs:
        if merged and (merged[-1][0] == r[0] or f[r[2]] - f[r[1]] + 1 < min_len):
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    if len(merged) > 1 and f[merged[0][2]] - f[merged[0][1]] + 1 < min_len:
        merged[1][1] = merged[0][1]
        merged.pop(0)
    return [(a, b) for _, a, b in merged]


def is_player(pos, cfg):
    mx, my = np.median(pos, axis=0)
    m = cfg["track_max_side_m"]
    if not -m <= mx <= COURT_W + m:
        return False  # le long des lignes de cote : arbitres, banc, photographes
    if 0 <= mx <= COURT_W and 0 <= my <= COURT_L:
        return True
    spread = float(np.ptp(np.percentile(pos, [5, 95], axis=0), axis=0).max())
    return spread >= cfg["track_min_spread_m"]  # immobile hors du terrain : juge de ligne


def clean_tracks(frames, fps, cfg):
    """Coupe les pistes qui changent de camp, recolle les pistes coupees, ecarte les non-joueurs,
    renumerote 1, 2, 3... et fixe l'equipe de chaque joueur."""
    raw = {}
    for i, fr in enumerate(frames):
        for p in fr["players"]:
            t = raw.setdefault(p["id"], {"f": [], "pos": []})
            t["f"].append(i)
            t["pos"].append(p["pos"])

    # le suivi echange parfois deux joueurs qui se croisent au filet : on coupe a chaque changement de camp
    segs, seg_of = {}, {}
    for tid, t in raw.items():
        for k, (a, b) in enumerate(side_runs(t["f"], t["pos"], int(cfg["net_switch_s"] * fps))):
            segs[(tid, k)] = {"f": t["f"][a:b + 1], "pos": np.array(t["pos"][a:b + 1])}
            for fi in t["f"][a:b + 1]:
                seg_of[(tid, fi)] = (tid, k)

    # un morceau qui demarre peu apres la fin d'un autre, au meme endroit et du meme cote, est le meme joueur
    gap_max = cfg["stitch_gap_s"] * fps
    chains = []
    for key in sorted(segs, key=lambda k: segs[k]["f"][0]):
        s = segs[key]
        start, p0 = s["f"][0], np.median(s["pos"][:5], axis=0)
        best, bd = None, float("inf")
        for c in chains:
            gap = start - c["end"]
            if gap <= 0 or gap > gap_max or side_of(c["pos"][1]) != side_of(p0[1]):
                continue
            d = float(np.hypot(*(p0 - c["pos"])))
            if d <= cfg["stitch_radius_m"] + cfg["stitch_speed_mps"] * gap / fps and d < bd:
                best, bd = c, d
        if best is None:
            best = {"keys": []}
            chains.append(best)
        best["keys"].append(key)
        best["end"], best["pos"] = s["f"][-1], np.median(s["pos"][-5:], axis=0)

    for c in chains:
        c["pos_all"] = np.concatenate([segs[k]["pos"] for k in c["keys"]])
    chains = [c for c in chains if is_player(c["pos_all"], cfg)]
    chains.sort(key=lambda c: -len(c["pos_all"]))
    new_id = {key: k + 1 for k, c in enumerate(chains) for key in c["keys"]}
    fragments = {k + 1: len({key[0] for key in c["keys"]}) for k, c in enumerate(chains)}
    team = {k + 1: Counter(side_of(y) for _, y in c["pos_all"]).most_common(1)[0][0]
            for k, c in enumerate(chains)}

    out = []
    for i, fr in enumerate(frames):
        players = []
        for p in fr["players"]:
            pid = new_id.get(seg_of[(p["id"], i)])
            if pid is not None:
                players.append({**p, "id": pid, "team": team[pid]})
        out.append({"players": players, "ball": fr["ball"]})
    return out, team, fragments, len(raw)


# ---------------------------------------------------------------- balle

def ball_tracklets(frames, W, cfg):
    """Relie les detections de balle d'une image a l'autre en pistes. Chaque piste predit sa
    position suivante d'apres sa vitesse ; plusieurs balles peuvent etre suivies en meme temps."""
    active, closed = [], []
    for i, fr in enumerate(frames):
        cands = fr["ball"]
        still = []
        for t in active:
            (closed if i - t["f"][-1] > cfg["ball_track_gap"] else still).append(t)
        active = still

        pairs = []
        for ti, t in enumerate(active):
            dt = i - t["f"][-1]
            if len(t["f"]) >= 2:
                k = -min(3, len(t["f"]))
                v = np.subtract(t["p"][-1], t["p"][k]) / (t["f"][-1] - t["f"][k])
                pred = np.add(t["p"][-1], v * dt)
                gate = cfg["ball_gate"] * W * (1 + 0.5 * (dt - 1)) + 0.3 * float(np.hypot(*v)) * dt
            else:  # vitesse encore inconnue
                pred, gate = t["p"][-1], cfg["ball_max_jump"] * W * dt
            for ci, c in enumerate(cands):
                d = math.hypot(c[0] - pred[0], c[1] - pred[1])
                if d <= gate:
                    pairs.append((d, ti, ci))
        used_t, used_c = set(), set()
        for _, ti, ci in sorted(pairs):
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            active[ti]["f"].append(i)
            active[ti]["p"].append(cands[ci][:2])
            active[ti]["c"].append(cands[ci][2])
        active += [{"f": [i], "p": [c[:2]], "c": [c[2]]} for ci, c in enumerate(cands) if ci not in used_c]
    return closed + active


def select_ball(frames, W, cfg):
    """Une position de balle par image : la piste en mouvement la plus sure. Les balles immobiles
    (reserve, ramasseurs, serveur avant son lancer) et les detections isolees sont ecartees."""
    n = len(frames)
    track = np.full((n, 2), np.nan)
    moving, n_static = [], 0
    for t in ball_tracklets(frames, W, cfg):
        if len(t["f"]) < cfg["ball_min_track"]:
            continue
        P = np.array(t["p"])
        spread = float(np.percentile(np.hypot(*(P - np.median(P, axis=0)).T), 90))
        if spread < cfg["ball_static"] * W:
            n_static += 1
            continue
        moving.append(t)
    taken = np.zeros(n, bool)
    for t in sorted(moving, key=lambda t: -sum(t["c"])):
        f, P = np.array(t["f"]), np.array(t["p"])
        free = ~taken[f]  # les images deja couvertes par une piste plus sure lui restent
        # on ne comble qu'avec des morceaux continus : pas d'alternance image par image entre deux objets
        edges = np.flatnonzero(np.diff(np.r_[0, free.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a >= cfg["ball_min_track"]:
                track[f[a:b]] = P[a:b]
                taken[f[a:b]] = True
    return drop_spikes(track, cfg["ball_spike"] * W), n_static


def drop_spikes(track, tol):
    """Retire les points isoles qui ne suivent ni la courbe d'arrivee ni celle de depart
    (fausse detection au milieu d'une trajectoire). Un point de touche suit la courbe d'arrivee."""
    out = track.copy()
    n = len(out)

    def error(i):
        valid = ~np.isnan(out[:, 0])
        before = [j for j in range(i - 6, i) if j >= 0 and valid[j]][-5:]
        after = [j for j in range(i + 1, i + 7) if j < n and valid[j]][:5]
        if np.isnan(out[i, 0]) or len(before) < 3 or len(after) < 3:
            return 0.0
        errs = []
        for idx in (before, after):
            deg = 2 if len(idx) >= 4 else 1
            pred = [np.polyval(np.polyfit(idx, out[idx, d], deg), i) for d in (0, 1)]
            errs.append(float(np.hypot(pred[0] - out[i, 0], pred[1] - out[i, 1])))
        return min(errs)

    # passes successives : on retire le pire point de chaque voisinage, puis on recalcule autour,
    # pour qu'un point parasite ne fasse pas tomber le vrai point d'a cote
    err = np.array([error(i) for i in range(n)])
    for _ in range(20):
        bad = [i for i in np.flatnonzero(err > tol) if err[i] >= err[max(0, i - 6):i + 7].max()]
        if not bad:
            break
        out[bad] = np.nan
        for i in {j for b in bad for j in range(max(0, b - 7), min(n, b + 8))}:
            err[i] = error(i)
    return out


def fill_gaps(track, max_gap, W):
    """Comble les petits trous en suivant la courbe de la balle (parabole ajustee sur les points
    voisins), ou en ligne droite si la balle a ete frappee pendant le trou."""
    out = track.copy()
    valid = np.where(~np.isnan(track[:, 0]))[0]
    for k in range(len(valid) - 1):
        a, b = valid[k], valid[k + 1]
        if not 1 < b - a <= max_gap + 1:
            continue
        near = [j for j in valid[max(0, k - 3):k + 5] if a - 10 <= j <= b + 10]
        t_new = np.arange(a + 1, b)
        if len(near) >= 5:
            fits = [np.polyfit(near, track[near, d], 2) for d in (0, 1)]
            resid = max(np.abs(np.polyval(fits[d], near) - track[near, d]).max() for d in (0, 1))
            if resid <= 0.006 * W:
                out[a + 1:b] = np.stack([np.polyval(fits[d], t_new) for d in (0, 1)], axis=1)
                continue
        w = ((t_new - a) / (b - a))[:, None]
        out[a + 1:b] = track[a] * (1 - w) + track[b] * w
    return out


# ---------------------------------------------------------------- touches et actions

def hand_distance(p, bp):
    """Distance balle-mains, en hauteurs de joueur (les poignets, sinon le haut du corps)."""
    x1, y1, x2, y2 = p["box"]
    h = max(1.0, y2 - y1)
    pts = []
    if p["kpts"]:
        pts = [(p["kpts"][k][0], p["kpts"][k][1]) for k in (L_WRIST, R_WRIST) if p["kpts"][k][2] >= KP_CONF]
    if not pts:
        pts = [((x1 + x2) / 2, y1 + 0.3 * h)]
    return min(math.hypot(px - bp[0], py - bp[1]) for px, py in pts) / h


def touch_options(frames, ball, t, reach):
    """Joueur le plus proche de la balle dans chaque equipe, sur les frames t-1 a t+1."""
    best = {}
    for dt in (-1, 0, 1):
        if 0 <= t + dt < len(frames) and not np.isnan(ball[t + dt]).any():
            for p in frames[t + dt]["players"]:
                d = hand_distance(p, ball[t + dt])
                if d <= reach and d < best.get(p["team"], (None, reach))[1] + 1e-9:
                    best[p["team"]] = (p, d)
    return list(best.values())


def assign_sequence(opts, cfg):
    """Choix du joueur de chaque touche d'un rallye (Viterbi) : le plus proche de la balle, sauf si
    cela donne 4 touches d'affilee au meme camp ou deux touches de suite au meme joueur."""
    V = []
    for i, options in enumerate(opts):
        cur = {}
        for oi, (p, d) in enumerate(options):
            e = cfg["assign_w_dist"] * d / cfg["touch_reach"]
            if i == 0:
                cur[(oi, 1)] = (e, None)
                continue
            for (poi, pc), (pcost, _) in V[-1].items():
                pp = opts[i - 1][poi][0]
                if pp["team"] == p["team"]:
                    c = min(pc + 1, 4)
                    tr = ((cfg["assign_w_four"] if pc == 3 else 0.0)
                          + (cfg["assign_w_double"] if pp["id"] == p["id"] else 0.0))
                else:
                    c, tr = 1, 0.0
                if (oi, c) not in cur or pcost + tr + e < cur[(oi, c)][0]:
                    cur[(oi, c)] = (pcost + tr + e, (poi, pc))
        V.append(cur)
    state = min(V[-1], key=lambda s: V[-1][s][0])
    picks = []
    for i in range(len(opts) - 1, -1, -1):
        picks.append(opts[i][state[0]])
        state = V[i][state][1]
    return picks[::-1]


def ground_pos(track, pid, t, airborne, fps):
    """Position au sol du joueur au moment t (derniere position hors saut : en l'air, les pieds
    sont projetes trop loin, parfois de l'autre cote du filet)."""
    for fr in range(t, max(-1, t - int(fps)), -1):
        if fr in track[pid] and pid not in airborne.get(fr, ()):
            return list(track[pid][fr])
    return list(track[pid][min(track[pid], key=lambda fr: abs(fr - t))])


def detect_touches(ball, frames, airborne, fps, W, cfg):
    k = cfg["touch_k"]
    vmin = cfg["touch_min_speed"] * W
    cands = []
    for t in range(k, len(ball) - k):
        p0, p1, p2 = ball[t - k], ball[t], ball[t + k]
        if np.isnan(p0).any() or np.isnan(p1).any() or np.isnan(p2).any():
            continue
        vb, va = (p1 - p0) / k, (p2 - p1) / k
        sb, sa = float(np.hypot(*vb)), float(np.hypot(*va))
        rebound = vb[1] > 0.5 * vmin and va[1] < -0.5 * vmin  # descendait puis remonte
        strike = sa > max(cfg["touch_strike_ratio"] * sb, 3 * vmin)  # acceleration brutale
        if rebound or strike:
            cands.append((t, float(np.hypot(*(va - vb))), "rebond" if rebound else "frappe"))

    gap = max(1, int(cfg["touch_min_gap_s"] * fps))
    kept = []
    for c in sorted(cands, key=lambda c: -c[1]):
        if all(abs(c[0] - o[0]) >= gap for o in kept):
            kept.append(c)
    kept.sort()

    cands = [(t, kind, touch_options(frames, ball, t, cfg["touch_reach"])) for t, _, kind in kept]
    orphans = sum(1 for c in cands if not c[2])  # sol, filet, ou joueur non detecte
    cands = [c for c in cands if c[2]]

    track = {}
    for i, fr in enumerate(frames):
        for p in fr["players"]:
            track.setdefault(p["id"], {})[i] = p["pos"]

    touches, rally = [], []
    for n, c in enumerate(cands):
        rally.append(c)
        if n + 1 < len(cands) and (cands[n + 1][0] - c[0]) / fps <= cfg["rally_gap_s"]:
            continue
        for (t, kind, _), (p, _) in zip(rally, assign_sequence([o for _, _, o in rally], cfg)):
            in_air = any(p["id"] in airborne.get(t + dt, ()) for dt in range(-2, 3))
            touches.append({"frame": int(t), "t": round(t / fps, 2), "player": p["id"],
                            "team": p["team"], "kind": kind, "airborne": in_air,
                            "pos": ground_pos(track, p["id"], t, airborne, fps),
                            "x": round(float(ball[t][0]), 1), "y": round(float(ball[t][1]), 1)})
        rally = []
    return touches, orphans


def build_possessions(touches, fps, cfg):
    """Rallyes et possessions ; un contre (touche au filet juste apres l'adversaire) ne compte pas."""
    poss, rally, last = [], 0, None
    for tc in touches:
        dt = None if last is None else (tc["frame"] - last["frame"]) / fps
        new_rally = dt is None or dt > cfg["rally_gap_s"]
        if new_rally:
            rally += 1
        tc["action"] = None
        if (not new_rally and tc["team"] != last["team"] and dt <= cfg["block_delay_s"]
                and abs(tc["pos"][1] - NET_Y) <= cfg["block_net_m"]):
            tc["action"] = "contre"
        if new_rally or tc["team"] != poss[-1]["team"]:
            poss.append({"rally": rally, "team": tc["team"], "start_s": tc["t"], "end_s": tc["t"],
                         "touches": 0, "players": [], "items": []})
        cur = poss[-1]
        cur["touches"] += tc["action"] != "contre"
        cur["end_s"] = tc["t"]
        cur["players"].append(tc["player"])
        cur["items"].append(tc)
        tc["rally"], tc["possession"] = rally, len(poss)
        last = tc
    return poss, rally


def label_actions(poss):
    """Role de chaque touche d'apres sa place dans la possession : reception, passe, attaque..."""
    by_rally = {}
    for p in poss:
        by_rally.setdefault(p["rally"], []).append(p)
    for rp in by_rally.values():
        after_serve = False
        for k, p in enumerate(rp):
            seq = [tc for tc in p["items"] if tc["action"] != "contre"]
            if k == 0 and seq and (len(seq) == 1 or abs(seq[0]["pos"][1] - NET_Y) > NET_Y - 0.5):
                seq[0]["action"] = "service"  # 1re touche du rallye, seule ou derriere la ligne de fond
                seq, served = seq[1:], True
            else:
                served = False
            last_poss = k == len(rp) - 1
            n = len(seq)
            for j, tc in enumerate(seq):
                strong = tc["kind"] == "frappe" or tc["airborne"]
                if j == 0 and not (n == 1 and strong and not after_serve):
                    tc["action"] = "reception" if after_serve else "defense"
                elif j == n - 1:
                    tc["action"] = "passe" if last_poss and not strong and n > 1 else "attaque"
                else:
                    tc["action"] = "passe"
            p["actions"] = [tc["action"] for tc in p["items"]]
            p["built"] = [tc["action"] for tc in seq] in (["reception", "passe", "attaque"],
                                                          ["defense", "passe", "attaque"])
            after_serve = served and not seq


def summarize_rallies(poss, touches):
    rallies = []
    for r in sorted({p["rally"] for p in poss}):
        rp = [p for p in poss if p["rally"] == r]
        tt = [t for t in touches if t["rally"] == r]
        serve = next((t["team"] for t in tt if t["action"] == "service"), None)
        rallies.append({"rally": r, "start_s": rp[0]["start_s"], "end_s": rp[-1]["end_s"],
                        "duration_s": round(rp[-1]["end_s"] - rp[0]["start_s"], 2),
                        "touches": len(tt), "crossings": len(rp) - 1, "serve_team": serve,
                        "last_team": tt[-1]["team"]})
    return rallies


# ---------------------------------------------------------------- sauts

def rolling_nanmedian(a, w):
    half = w // 2
    padded = np.pad(a, (half, half), constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=1)


def ankle_y(p):
    if p["kpts"]:
        ys = [p["kpts"][k][1] for k in (L_ANKLE, R_ANKLE) if p["kpts"][k][2] >= KP_CONF]
        if ys:
            return float(np.mean(ys))
    return float(p["box"][3])


def player_series(frames):
    series = {}
    for i, fr in enumerate(frames):
        for p in fr["players"]:
            s = series.setdefault(p["id"], {"f": [], "y": [], "h": [], "pos": [], "team": p["team"]})
            s["f"].append(i)
            s["y"].append(ankle_y(p))
            s["h"].append(p["box"][3] - p["box"][1])
            s["pos"].append(p["pos"])
    return series


def detect_jumps(f, y, h, fps, cfg):
    f = np.asarray(f)
    if len(f) < fps * 0.5:
        return []
    n = int(f[-1] - f[0] + 1)
    Y = np.full(n, np.nan)
    Y[f - f[0]] = y
    hmed = float(np.median(h))
    if hmed <= 1:
        return []
    base = rolling_nanmedian(Y, max(3, int(fps * 1.5)))
    with np.errstate(invalid="ignore"):
        above = (base - Y) / hmed > cfg["jump_rise"]
    pre, shift = max(2, int(0.3 * fps)), max(1, int(0.1 * fps))
    max_air, min_gap = int(cfg["jump_max_air_s"] * fps), int(cfg["jump_min_gap_s"] * fps)

    jumps, i = [], 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        s, e = i, j
        i = j + 1
        pre_seg = Y[max(0, s - shift - pre):max(0, s - shift)]
        post_seg = Y[e + 1 + shift:e + 1 + shift + pre]
        if np.count_nonzero(~np.isnan(pre_seg)) < 2 or np.count_nonzero(~np.isnan(post_seg)) < 2:
            continue
        g0, g1 = float(np.nanmedian(pre_seg)), float(np.nanmedian(post_seg))
        if abs(g1 - g0) / hmed > cfg["jump_land_tol"]:
            continue  # deplacement avant/arriere, pas un saut

        def ground(t):
            return g0 + (g1 - g0) * (t - s) / max(1, e - s)

        peak = s + int(np.nanargmin(Y[s:e + 1]))
        if (ground(peak) - Y[peak]) / hmed <= cfg["jump_rise"]:
            continue
        a = b = peak
        while a - 1 >= 0 and not np.isnan(Y[a - 1]) and (ground(a - 1) - Y[a - 1]) / hmed > cfg["jump_air"]:
            a -= 1
        while b + 1 < n and not np.isnan(Y[b + 1]) and (ground(b + 1) - Y[b + 1]) / hmed > cfg["jump_air"]:
            b += 1
        t_air = (b - a + 1) / fps
        if b - a + 1 > max_air or t_air < cfg["jump_min_air_s"]:
            continue
        start = int(f[0] + a)
        if jumps and start - jumps[-1]["end"] < min_gap:
            continue
        jumps.append({"start": start, "end": int(f[0] + b), "peak": int(f[0] + peak),
                      "t": round(float(f[0] + peak) / fps, 2), "air_s": round(t_air, 2),
                      "height_cm": int(round(min(1.2, 9.81 * t_air ** 2 / 8) * 100))})
    return jumps


# ---------------------------------------------------------------- deplacements

def movement(f, pos, air_frames, fps, cfg):
    """Positions lissees a 5 Hz (hors sauts) et distance parcourue en metres."""
    f = np.asarray(f)
    X = np.full((int(f[-1] - f[0] + 1), 2), np.nan)
    X[f - f[0]] = pos
    for fr in air_frames:
        if f[0] <= fr <= f[-1]:
            X[fr - f[0]] = np.nan
    w = max(3, int(fps * 0.4))
    X = np.stack([rolling_nanmedian(X[:, 0], w), rolling_nanmedian(X[:, 1], w)], axis=1)
    step = max(1, int(round(fps / 5)))
    S = X[::step]
    d = np.hypot(*np.diff(S, axis=0).T)
    with np.errstate(invalid="ignore"):
        ok = ~np.isnan(d) & (d / (step / fps) <= cfg["move_max_speed_mps"])
    samples = [(round(float(f[0] + k * step) / fps, 2), round(float(x), 1), round(float(y), 1))
               for k, (x, y) in enumerate(S) if not np.isnan(x)]
    return float(d[ok].sum()), samples


# ---------------------------------------------------------------- rendu video

def render(video, frames, ball, touches, jumps_by_id, court, fps, out_dir, status):
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw = out_dir / "annotated_raw.mp4"
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    t_frames = np.array([t["frame"] for t in touches], dtype=int)
    j_starts = np.array(sorted(j["start"] for js in jumps_by_id.values() for j in js), dtype=int)
    airborne = {}
    for pid, js in jumps_by_id.items():
        for j in js:
            for fr in range(j["start"], j["end"] + 1):
                airborne.setdefault(fr, set()).add(pid)
    flash = int(0.8 * fps)
    trail = deque(maxlen=int(fps * 0.5))

    for i in range(len(frames)):
        ok, img = cap.read()
        if not ok:
            break
        court.draw(img)
        for p in frames[i]["players"]:
            x1, y1, x2, y2 = map(int, p["box"])
            col = TEAM_BGR[p["team"]]
            air = p["id"] in airborne.get(i, ())
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if air else 1)
            label = f"#{p['id']} saut" if air else f"#{p['id']}"
            cv2.putText(img, label, (x1, y1 - 6), FONT, 0.55, (0, 0, 0), 4)
            cv2.putText(img, label, (x1, y1 - 6), FONT, 0.55, col, 2)

        bp = ball[i]
        if np.isnan(bp).any():
            trail.clear()
        else:
            trail.append((int(bp[0]), int(bp[1])))
            cv2.circle(img, trail[-1], 6, BALL_BGR, -1, cv2.LINE_AA)
        if len(trail) > 1:
            cv2.polylines(img, [np.array(trail, np.int32)], False, BALL_BGR, 2, cv2.LINE_AA)

        idx = int(np.searchsorted(t_frames, i, side="right"))
        if idx and i - touches[idx - 1]["frame"] < flash:
            tc = touches[idx - 1]
            c = (int(tc["x"]), int(tc["y"]))
            cv2.circle(img, c, 22, (255, 255, 255), 3, cv2.LINE_AA)
            txt = f"{ACTION_TXT[tc['action']]}  #{tc['player']}"
            cv2.putText(img, txt, (c[0] + 26, c[1]), FONT, 0.8, (0, 0, 0), 5)
            cv2.putText(img, txt, (c[0] + 26, c[1]), FONT, 0.8, (255, 255, 255), 2)

        n_j = int(np.searchsorted(j_starts, i, side="right"))
        rally = touches[idx - 1]["rally"] if idx else 0
        overlay = img.copy()
        cv2.rectangle(overlay, (12, 12), (470, 62), (79, 49, 19), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.putText(img, f"Rallye {rally}   Touches {idx}   Sauts {n_j}", (26, 47), FONT, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
        status("render", i + 1, len(frames))
    cap.release()
    vw.release()

    out = out_dir / "annotated.mp4"
    ff = shutil.which("ffmpeg")
    if ff:
        status("encode")
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
        raw.unlink()
    else:
        raw.replace(out)
        print("ffmpeg introuvable : la video annotee risque de ne pas se lire dans le navigateur")
    return out


# ---------------------------------------------------------------- analyse

def analyze(raw, court, cfg):
    fps, W = raw["fps"], raw["W"]
    frames = place_players(raw["frames"], court, cfg)
    frames, team_of, fragments, n_raw_ids = clean_tracks(frames, fps, cfg)

    series = player_series(frames)
    jumps_by_id = {pid: detect_jumps(s["f"], s["y"], s["h"], fps, cfg) for pid, s in series.items()}
    jumps_by_id = {pid: js for pid, js in jumps_by_id.items() if js}
    airborne = {}
    for pid, js in jumps_by_id.items():
        for j in js:
            for fr in range(j["start"], j["end"] + 1):
                airborne.setdefault(fr, set()).add(pid)

    ball_raw, n_static = select_ball(frames, W, cfg)
    ball = fill_gaps(ball_raw, cfg["ball_max_gap"], W)
    touches, orphans = detect_touches(ball, frames, airborne, fps, W, cfg)
    poss, n_rallies = build_possessions(touches, fps, cfg)
    label_actions(poss)
    rallies = summarize_rallies(poss, touches)

    jumps = []
    for pid, js in jumps_by_id.items():
        for j in js:
            hit = next((t["action"] for t in touches
                        if t["player"] == pid and j["start"] - 2 <= t["frame"] <= j["end"] + 2), None)
            jumps.append({"player": pid, "team": team_of[pid], **j, "action": hit})
    jumps.sort(key=lambda j: j["start"])

    rows, positions = [], []
    for pid, s in series.items():
        vis = len(s["f"]) / fps
        air = [fr for j in jumps_by_id.get(pid, []) for fr in range(j["start"], j["end"] + 1)]
        dist, samples = movement(s["f"], s["pos"], air, fps, cfg)
        positions += [[t, pid, s["team"], x, y] for t, x, y in samples]
        if vis < cfg["min_track_s"]:
            continue
        js = jumps_by_id.get(pid, [])
        hs = [j["height_cm"] for j in js]
        mine = [t for t in touches if t["player"] == pid]
        rows.append({
            "id": pid,
            "team": s["team"],
            "visible_s": round(vis, 1),
            "touches": len(mine),
            **{a: sum(1 for t in mine if t["action"] == a) for a in ACTIONS},
            "jumps": len(js),
            "max_jump_cm": max(hs) if hs else None,
            "avg_jump_cm": int(round(float(np.mean(hs)))) if hs else None,
            "distance_m": int(round(dist)),
            "fragments": fragments.get(pid, 1),
        })
    rows.sort(key=lambda r: (-r["touches"], -r["jumps"], r["id"]))

    teams = {}
    for tm in TEAMS:
        tt = [t for t in touches if t["team"] == tm]
        ps = [p for p in poss if p["team"] == tm]
        pr = [r for r in rows if r["team"] == tm]
        teams[tm] = {
            "touches": len(tt),
            **{a: sum(1 for t in tt if t["action"] == a) for a in ACTIONS},
            "attacks_air": sum(1 for t in tt if t["action"] == "attaque" and t["airborne"]),
            "possessions": len(ps),
            "built": sum(1 for p in ps if p["built"]),
            "jumps": sum(1 for j in jumps if j["team"] == tm),
            "players": len(pr),
            "distance_m": sum(r["distance_m"] for r in pr),
        }

    summary = {
        "duration_s": round(len(frames) / fps, 1),
        "touches_total": len(touches),
        "touches_by_team": {tm: teams[tm]["touches"] for tm in TEAMS},
        "contacts_sans_joueur": orphans,
        "jumps_total": len(jumps),
        "rallies": n_rallies,
        "possessions": len(poss),
        "touches_per_possession": round(float(np.mean([p["touches"] for p in poss])), 2) if poss else 0,
        "possessions_suspectes": sum(1 for p in poss if p["touches"] > 3),
        "ball_detection_rate": round(float(np.mean(~np.isnan(ball_raw[:, 0]))), 3) if len(frames) else 0,
        "ball_static_tracks": n_static,
        "players_tracked": len(rows),
        "track_ids_raw": n_raw_ids,
    }
    for p in poss:
        del p["items"]
    step = max(1, int(fps // 10))
    ball_track = [[round(i / fps, 2), round(float(b[0]), 1), round(float(b[1]), 1)]
                  for i, b in enumerate(ball) if i % step == 0 and not np.isnan(b).any()]
    results = {"version": VERSION, "fps": fps, "width": W, "height": raw["H"], "summary": summary,
               "teams": teams, "players": rows, "touches": touches, "jumps": jumps, "possessions": poss,
               "rallies": rallies, "positions": positions, "ball_track": ball_track, "config": cfg}
    return results, frames, ball, touches, jumps_by_id


# ---------------------------------------------------------------- main

def run(args, out_dir, status):
    status("prepare")
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    video = Path(args.video)
    if args.start is not None or args.end is not None:
        clip = out_dir / "clip.mp4"
        print("Decoupage de la video...", flush=True)
        trim(video, clip, args.start or 0.0, args.end)
        meta.update(source=str(video), start=args.start or 0.0, end=args.end)
        video = clip
    elif args.reuse and meta.get("video") and Path(meta["video"]).exists():
        video = Path(meta["video"])
    meta.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    meta.setdefault("source", str(Path(args.video)))
    meta["video"] = str(video)
    meta["view"] = args.view or meta.get("view", "back")
    meta["title"] = args.title or meta.get("title") or out_dir.name
    names = meta.setdefault("team_names", {})
    for tm, name in (("A", args.team_a), ("B", args.team_b)):
        if name is not None:
            names[tm] = name
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    court = Court(pick_corners(video, out_dir / "corners.json", args.reclick), meta["view"])

    raw_path = out_dir / "raw.json"
    if args.reuse and raw_path.exists():
        raw = json.loads(raw_path.read_text())
        print("raw.json reutilise")
    else:
        if not args.ball:
            raise SystemExit("--ball <poids.pt> requis pour la premiere analyse")
        args.device = args.device or default_device()
        print(f"Inference joueurs + balle sur {args.device}...", flush=True)
        t0 = time.time()
        raw = run_inference(video, args.ball, args, status)
        raw_path.write_text(json.dumps(raw))
        meta["inference_fps"] = round(len(raw["frames"]) / max(1e-6, time.time() - t0), 2)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    status("analysis")
    results, frames, ball, touches, jumps_by_id = analyze(raw, court, CFG)
    results["meta"] = {**meta, "corners": court.corners, "analyzed": datetime.now().isoformat(timespec="seconds")}
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.no_video:
        print("Rendu video...", flush=True)
        render(video, frames, ball, touches, jumps_by_id, court, raw["fps"], out_dir, status)

    s = results["summary"]
    print(f"\nTouches {s['touches_total']} (A {s['touches_by_team']['A']}, B {s['touches_by_team']['B']}), "
          f"sauts {s['jumps_total']}, rallyes {s['rallies']}, balle detectee {s['ball_detection_rate']:.0%}")
    print(f"Resultats : {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Analyse d'une video de volley en camera fixe")
    ap.add_argument("video")
    ap.add_argument("--ball", default=str(DEFAULT_BALL) if DEFAULT_BALL.exists() else None,
                    help="poids du modele balle (.pt)")
    ap.add_argument("--pose", default="yolo26m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ball-imgsz", type=int, default=1280)
    ap.add_argument("--ball-conf", type=float, default=0.25)
    ap.add_argument("--device", default=None, help="0 (GPU), cpu... ; auto par defaut")
    ap.add_argument("--out", default="results")
    ap.add_argument("--name", help="dossier de sortie (defaut : nom de la video)")
    ap.add_argument("--title", help="titre affiche dans le tableau de bord")
    ap.add_argument("--view", choices=["back", "side"], help="camera derriere le terrain ou sur le cote")
    ap.add_argument("--team-a", help="nom de l'equipe cote oppose (ou gauche)")
    ap.add_argument("--team-b", help="nom de l'equipe cote camera (ou droit)")
    ap.add_argument("--start", type=float, help="debut de la portion a analyser (s)")
    ap.add_argument("--end", type=float, help="fin de la portion a analyser (s)")
    ap.add_argument("--reuse", action="store_true", help="reutilise raw.json (pas de GPU)")
    ap.add_argument("--reclick", action="store_true", help="recliquer les coins du terrain")
    ap.add_argument("--no-video", action="store_true", help="ne pas generer la video annotee")
    args = ap.parse_args()

    out_dir = Path(args.out) / (args.name or Path(args.video).stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = Status(out_dir / "status.json")
    try:
        run(args, out_dir, status)
    except BaseException as e:
        status.finish("error", str(e) or type(e).__name__)
        raise
    status.finish("done")


if __name__ == "__main__":
    main()
