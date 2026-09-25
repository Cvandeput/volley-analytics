"""
Analyse d'une video de volley : joueurs (pose + tracking), balle, touches, sauts, possessions.

Premiere analyse (GPU) :
    python analyze.py <video> --ball <poids_balle.pt>
Reanalyse sans GPU apres modification des seuils CFG (quelques secondes) :
    python analyze.py <video> --reuse
Recliquer les coins du terrain :
    python analyze.py <video> --reuse --reclick

Sorties dans results/<nom_video>/ : results.json, annotated.mp4, corners.json, raw.json
"""
import argparse
import json
import math
import shutil
import subprocess
import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# Seuils : modifie-les puis relance avec --reuse
CFG = {
    "court_margin_px": 60,      # marge autour du terrain (defense hors limites)
    "ball_max_jump": 0.08,      # deplacement max balle entre 2 frames (fraction largeur image)
    "ball_max_gap": 6,          # trous de detection interpoles (frames)
    "touch_k": 3,               # fenetre (frames) des vitesses avant/apres
    "touch_min_speed": 0.004,   # vitesse mini balle (fraction largeur / frame)
    "touch_strike_ratio": 2.0,  # vitesse apres > ratio * vitesse avant = frappe
    "touch_min_gap_s": 0.30,    # ecart mini entre 2 touches
    "touch_reach": 0.6,         # distance balle-mains max, en hauteurs de joueur
    "jump_rise": 0.12,          # montee mini des chevilles (fraction hauteur joueur)
    "jump_air": 0.03,           # seuil "en l'air" pour le temps de vol
    "jump_max_air_s": 1.2,
    "jump_min_gap_s": 0.5,
    "jump_land_tol": 0.35,      # ecart max sol avant/apres (fraction hauteur)
    "rally_gap_s": 4.0,         # 4 s sans touche = nouveau rallye
    "min_track_s": 1.0,         # pistes plus courtes ignorees dans le tableau joueurs
}

KP_CONF = 0.3
L_WRIST, R_WRIST, L_ANKLE, R_ANKLE = 9, 10, 15, 16
TEAM_BGR = {"proche": (222, 111, 47), "loin": (0, 168, 224)}
BALL_BGR = (0, 230, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- terrain

class Court:
    """Coins dans l'ordre : fond gauche, fond droit, avant droit, avant gauche."""

    def __init__(self, corners, margin):
        self.poly = np.array(corners, np.float32)
        self.margin = margin
        a = (self.poly[0] + self.poly[3]) / 2
        b = (self.poly[1] + self.poly[2]) / 2
        self.net = (a, b)
        near_mid = (self.poly[2] + self.poly[3]) / 2
        self.near_sign = np.sign(self._cross(near_mid))

    def _cross(self, p):
        a, b = self.net
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

    def contains(self, x, y):
        return cv2.pointPolygonTest(self.poly.reshape(-1, 1, 2), (float(x), float(y)), True) >= -self.margin

    def side(self, x, y):
        return "proche" if np.sign(self._cross((x, y))) == self.near_sign else "loin"


def pick_corners(video, path, reclick=False):
    if path.exists() and not reclick:
        return json.loads(path.read_text())
    cap = cv2.VideoCapture(str(video))
    ok, base = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Impossible de lire {video}")
    labels = ["fond gauche", "fond droit", "avant droit", "avant gauche"]
    pts = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    win = "Coins du terrain"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)
    while True:
        view = base.copy()
        for i, p in enumerate(pts):
            cv2.circle(view, p, 7, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (p[0] + 10, p[1] - 10), FONT, 0.8, (0, 0, 255), 2)
        if len(pts) > 1:
            cv2.polylines(view, [np.array(pts, np.int32)], len(pts) == 4, (0, 255, 255), 2)
        msg = f"Clique : {labels[len(pts)]}" if len(pts) < 4 else "Entree : valider   R : recommencer"
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
    path.write_text(json.dumps(pts))
    return pts


# ---------------------------------------------------------------- inference (GPU)

def run_inference(video, ball_weights, args):
    from ultralytics import YOLO

    pose = YOLO(args.pose)
    ball = YOLO(ball_weights)
    ball_cls = [i for i, n in ball.names.items() if "ball" in n.lower()] or None
    print(f"Classes balle : {[ball.names[i] for i in ball_cls] if ball_cls else 'toutes'}")

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frames = []
    stream = pose.track(str(video), tracker="botsort.yaml", imgsz=args.imgsz, conf=0.2,
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
        if i % 100 == 0:
            print(f"  frame {i}/{total}")
    return {"fps": fps, "W": W, "H": H, "frames": frames}


# ---------------------------------------------------------------- balle

def select_ball(frames, W, cfg):
    """Une position de balle par frame : meilleure confiance, avec continuite spatiale."""
    n = len(frames)
    track = np.full((n, 2), np.nan)
    last, last_i = None, -10_000
    for i, fr in enumerate(frames):
        cands = fr["ball"]
        if not cands:
            continue
        if last is not None and i - last_i <= 10:
            limit = cfg["ball_max_jump"] * W * (i - last_i)
            cands = [c for c in cands if math.hypot(c[0] - last[0], c[1] - last[1]) <= limit]
            if not cands:
                continue
        best = max(cands, key=lambda c: c[2])
        track[i] = best[:2]
        last, last_i = best, i
    return track


def interpolate(track, max_gap):
    out = track.copy()
    valid = np.where(~np.isnan(track[:, 0]))[0]
    for a, b in zip(valid[:-1], valid[1:]):
        if 1 < b - a <= max_gap + 1:
            for k in range(1, b - a):
                w = k / (b - a)
                out[a + k] = track[a] * (1 - w) + track[b] * w
    return out


# ---------------------------------------------------------------- touches

def nearest_player(players, bp, reach):
    best, bd = None, float("inf")
    for p in players:
        x1, y1, x2, y2 = p["box"]
        h = max(1.0, y2 - y1)
        pts = []
        if p["kpts"]:
            pts = [(p["kpts"][k][0], p["kpts"][k][1]) for k in (L_WRIST, R_WRIST) if p["kpts"][k][2] >= KP_CONF]
        if not pts:
            pts = [((x1 + x2) / 2, y1 + 0.3 * h)]
        d = min(math.hypot(px - bp[0], py - bp[1]) for px, py in pts) / h
        if d < bd:
            best, bd = p, d
    return (best, bd) if bd <= reach else (None, bd)


def detect_touches(ball, frames, fps, W, cfg):
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

    touches, orphans = [], 0
    for t, _, kind in kept:
        best, bd = None, float("inf")
        for dt in (-1, 0, 1):
            if 0 <= t + dt < len(frames) and not np.isnan(ball[t + dt]).any():
                p, d = nearest_player(frames[t + dt]["players"], ball[t + dt], cfg["touch_reach"])
                if p is not None and d < bd:
                    best, bd = p, d
        if best is None:
            orphans += 1  # sol, filet, ou joueur non detecte
            continue
        touches.append({"frame": int(t), "t": round(t / fps, 2), "player": best["id"],
                        "team": best["team"], "kind": kind,
                        "x": round(float(ball[t][0]), 1), "y": round(float(ball[t][1]), 1)})
    return touches, orphans


def build_possessions(touches, fps, cfg):
    poss, rally, last = [], 0, None
    for tc in touches:
        new_rally = last is None or (tc["frame"] - last["frame"]) / fps > cfg["rally_gap_s"]
        if new_rally:
            rally += 1
        if new_rally or tc["team"] != poss[-1]["team"]:
            poss.append({"rally": rally, "team": tc["team"], "start_s": tc["t"], "end_s": tc["t"],
                         "touches": 0, "players": []})
        cur = poss[-1]
        cur["touches"] += 1
        cur["end_s"] = tc["t"]
        cur["players"].append(tc["player"])
        tc["rally"], tc["possession"] = rally, len(poss)
        last = tc
    return poss, rally


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
            s = series.setdefault(p["id"], {"f": [], "y": [], "h": [], "team": []})
            s["f"].append(i)
            s["y"].append(ankle_y(p))
            s["h"].append(p["box"][3] - p["box"][1])
            s["team"].append(p["team"])
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
        if b - a + 1 > max_air:
            continue
        t_air = (b - a + 1) / fps
        start = int(f[0] + a)
        if jumps and start - jumps[-1]["end"] < min_gap:
            continue
        jumps.append({"start": start, "end": int(f[0] + b), "peak": int(f[0] + peak),
                      "t": round(float(f[0] + peak) / fps, 2), "air_s": round(t_air, 2),
                      "height_cm": int(round(min(1.2, 9.81 * t_air ** 2 / 8) * 100))})
    return jumps


# ---------------------------------------------------------------- rendu video

def render(video, frames, ball, touches, jumps_by_id, court, fps, out_dir):
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
    flash = int(0.6 * fps)
    trail = deque(maxlen=int(fps * 0.5))
    poly = court.poly.astype(np.int32)
    na, nb = (tuple(int(v) for v in p) for p in court.net)

    i = 0
    while i < len(frames):
        ok, img = cap.read()
        if not ok:
            break
        cv2.polylines(img, [poly], True, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(img, na, nb, (255, 255, 255), 2, cv2.LINE_AA)
        for p in frames[i]["players"]:
            x1, y1, x2, y2 = map(int, p["box"])
            col = TEAM_BGR[p["team"]]
            air = p["id"] in airborne.get(i, ())
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if air else 1)
            label = f"{p['id']} saut" if air else str(p["id"])
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
            txt = f"Touche {idx} : joueur {tc['player']}"
            cv2.putText(img, txt, (c[0] + 26, c[1]), FONT, 0.7, (0, 0, 0), 5)
            cv2.putText(img, txt, (c[0] + 26, c[1]), FONT, 0.7, (255, 255, 255), 2)

        n_j = int(np.searchsorted(j_starts, i, side="right"))
        rally = touches[idx - 1]["rally"] if idx else 0
        overlay = img.copy()
        cv2.rectangle(overlay, (12, 12), (470, 62), (79, 49, 19), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.putText(img, f"Touches {idx}   Sauts {n_j}   Rallye {rally}", (26, 47), FONT, 0.85,
                    (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(img)
        i += 1
    cap.release()
    vw.release()

    out = out_dir / "annotated.mp4"
    ff = shutil.which("ffmpeg")
    if ff:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
        raw.unlink()
    else:
        raw.replace(out)
        print("ffmpeg introuvable : la video annotee risque de ne pas se lire dans le navigateur")
    return out


# ---------------------------------------------------------------- main

def analyze(raw, court, cfg):
    fps, W = raw["fps"], raw["W"]
    frames = []
    for fr in raw["frames"]:
        players = []
        for p in fr["players"]:
            if court.contains(*p["feet"]):
                players.append({**p, "team": court.side(*p["feet"])})
        frames.append({"players": players, "ball": fr["ball"]})

    ball_raw = select_ball(frames, W, cfg)
    ball = interpolate(ball_raw, cfg["ball_max_gap"])
    touches, orphans = detect_touches(ball, frames, fps, W, cfg)
    poss, n_rallies = build_possessions(touches, fps, cfg)

    series = player_series(frames)
    jumps_by_id = {pid: detect_jumps(s["f"], s["y"], s["h"], fps, cfg) for pid, s in series.items()}
    jumps_by_id = {pid: js for pid, js in jumps_by_id.items() if js}

    rows = []
    for pid, s in series.items():
        vis = len(s["f"]) / fps
        if vis < cfg["min_track_s"]:
            continue
        js = jumps_by_id.get(pid, [])
        hs = [j["height_cm"] for j in js]
        rows.append({
            "id": pid,
            "team": max(set(s["team"]), key=s["team"].count),
            "visible_s": round(vis, 1),
            "touches": sum(1 for t in touches if t["player"] == pid),
            "jumps": len(js),
            "max_jump_cm": max(hs) if hs else None,
            "avg_jump_cm": int(round(float(np.mean(hs)))) if hs else None,
        })
    rows.sort(key=lambda r: (-r["touches"], -r["jumps"], r["id"]))

    jumps = sorted(({"player": pid, "team": series[pid]["team"][0], **j}
                    for pid, js in jumps_by_id.items() for j in js), key=lambda j: j["start"])
    summary = {
        "duration_s": round(len(frames) / fps, 1),
        "touches_total": len(touches),
        "touches_by_team": {tm: sum(1 for t in touches if t["team"] == tm) for tm in ("proche", "loin")},
        "contacts_sans_joueur": orphans,
        "jumps_total": len(jumps),
        "rallies": n_rallies,
        "possessions": len(poss),
        "touches_per_possession": round(float(np.mean([p["touches"] for p in poss])), 2) if poss else 0,
        "possessions_suspectes": sum(1 for p in poss if p["touches"] > 3),
        "ball_detection_rate": round(float(np.mean(~np.isnan(ball_raw[:, 0]))), 3) if len(frames) else 0,
        "players_tracked": len(rows),
    }
    step = max(1, int(fps // 10))
    ball_track = [[round(i / fps, 2), round(float(b[0]), 1), round(float(b[1]), 1)]
                  for i, b in enumerate(ball) if i % step == 0 and not np.isnan(b).any()]
    results = {"fps": fps, "width": W, "height": raw["H"], "summary": summary, "players": rows,
               "touches": touches, "jumps": jumps, "possessions": poss, "ball_track": ball_track,
               "config": cfg}
    return results, frames, ball, touches, jumps_by_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ball", help="poids du modele balle (.pt)")
    ap.add_argument("--pose", default="yolo26m-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ball-imgsz", type=int, default=1280)
    ap.add_argument("--ball-conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reuse", action="store_true", help="reutilise raw.json (pas de GPU)")
    ap.add_argument("--reclick", action="store_true", help="recliquer les coins du terrain")
    ap.add_argument("--no-video", action="store_true", help="ne pas generer la video annotee")
    args = ap.parse_args()

    video = Path(args.video)
    out_dir = Path(args.out) / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    court = Court(pick_corners(video, out_dir / "corners.json", args.reclick), CFG["court_margin_px"])

    raw_path = out_dir / "raw.json"
    if args.reuse and raw_path.exists():
        raw = json.loads(raw_path.read_text())
        print("raw.json reutilise")
    else:
        if not args.ball:
            raise SystemExit("--ball <poids.pt> requis pour la premiere analyse")
        print("Inference joueurs + balle...")
        raw = run_inference(video, args.ball, args)
        raw_path.write_text(json.dumps(raw))

    results, frames, ball, touches, jumps_by_id = analyze(raw, court, CFG)
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    if not args.no_video:
        print("Rendu video...")
        render(video, frames, ball, touches, jumps_by_id, court, raw["fps"], out_dir)

    s = results["summary"]
    print(f"\nTouches {s['touches_total']} (proche {s['touches_by_team']['proche']}, "
          f"loin {s['touches_by_team']['loin']}), sauts {s['jumps_total']}, rallyes {s['rallies']}, "
          f"balle detectee {s['ball_detection_rate']:.0%}")
    print(f"Resultats : {out_dir}")


if __name__ == "__main__":
    main()
