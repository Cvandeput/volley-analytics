import cv2, numpy as np
from ultralytics import YOLO

SRC = r"VolleyVision\data\Tracking\back_view.mp4"
OUT = "filtered.mp4"
MARGIN = 60  # px autour du terrain

cap = cv2.VideoCapture(SRC); ok, first = cap.read(); cap.release()
pts = []
def on_click(e, x, y, *_):
    if e == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append((x, y)); cv2.circle(first, (x, y), 6, (0, 0, 255), -1)
cv2.namedWindow("coins"); cv2.setMouseCallback("coins", on_click)
while True:
    cv2.imshow("coins", first)
    if cv2.waitKey(20) == 13 and len(pts) == 4: break
cv2.destroyAllWindows()
poly = np.array(pts, np.int32)

def in_court(x, y):
    return cv2.pointPolygonTest(poly, (float(x), float(y)), True) >= -MARGIN

model = YOLO("yolo26m.pt")
writer, counts = None, []
for r in model.track(SRC, tracker="botsort.yaml", classes=[0], device=0, stream=True, verbose=False):
    frame = r.orig_img.copy()
    if writer is None:
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    cv2.polylines(frame, [poly], True, (0, 255, 255), 2)
    n = 0
    if r.boxes.id is not None:
        for (x1, y1, x2, y2), tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().tolist()):
            if in_court((x1 + x2) / 2, y2):
                n += 1
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, str(tid), (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    counts.append(n)
    writer.write(frame)
writer.release()
print(f"joueurs/frame : min {min(counts)}, moy {np.mean(counts):.1f}, max {max(counts)}")
