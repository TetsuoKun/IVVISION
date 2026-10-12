import cv2
import pandas as pd
import subprocess

# ---------------- Inputs ----------------
VIDEO_PATH = "videos/bbg.mp4"
BOXES_FILE = "tnt_boxes.xlsx"       # or "tnt_boxes.csv" (much faster to load)
ID_COLUMN = "id"                    # "id" = corrected ids, "raw_id" = tracker's original ids
BOXED_VIDEO = "tnt_boxed.mp4"
FINAL_OUTPUT = "tnt_boxed_with_audio.mp4"

# One distinct colour per id (BGR)
PALETTE = [
    (255, 56, 56), (56, 56, 255), (56, 255, 56), (0, 200, 255), (255, 0, 200),
    (255, 200, 0), (128, 0, 255), (0, 128, 255), (200, 255, 0), (255, 128, 128),
]


def colour_for(track_id):
    if pd.isna(track_id):
        return (180, 180, 180)      # grey for boxes with no track id
    return PALETTE[int(track_id) % len(PALETTE)]


# ---------------- Load boxes ----------------
if BOXES_FILE.endswith(".csv"):
    df = pd.read_csv(BOXES_FILE)
else:
    df = pd.read_excel(BOXES_FILE, sheet_name="boxes")

# frame number -> that frame's rows, so each lookup in the loop is instant
boxes_by_frame = {frame: rows for frame, rows in df.groupby("frame")}

# ---------------- Draw onto the raw video ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(BOXED_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    rows = boxes_by_frame.get(frame_idx)
    if rows is not None:
        for _, r in rows.iterrows():
            colour = colour_for(r[ID_COLUMN])
            tl = (int(r.tl_x), int(r.tl_y))
            br = (int(r.br_x), int(r.br_y))
            mid = (int(r.mid_x), int(r.mid_y))

            cv2.rectangle(frame, tl, br, colour, 2)
            cv2.circle(frame, mid, 4, colour, -1)

            label = "id ?" if pd.isna(r[ID_COLUMN]) else f"id {int(r[ID_COLUMN])}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y_text = max(tl[1], th + 10)
            cv2.rectangle(frame, (tl[0], y_text - th - 10), (tl[0] + tw + 6, y_text), colour, -1)
            cv2.putText(frame, label, (tl[0] + 3, y_text - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"Drew boxes on {frame_idx} frames -> {BOXED_VIDEO}")

# ---------------- Add the original audio back ----------------
subprocess.run([
    "ffmpeg", "-y",
    "-i", BOXED_VIDEO,
    "-i", VIDEO_PATH,
    "-map", "0:v:0",
    "-map", "1:a:0?",       # "?" = don't fail if the source has no audio
    "-c:v", "libx264",
    "-c:a", "aac",
    "-shortest",
    FINAL_OUTPUT,
], check=True)
