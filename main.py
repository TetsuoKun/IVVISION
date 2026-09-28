from ultralytics import YOLO 
import cv2
import numpy as np
import pathlib as Path
import pandas as pd

def remap_ids(df, n_people=4):
    df = df.sort_values("Frame").copy()
    df["cx"] = (df.X_Min + df.X_Max) / 2
    df["cy"] = (df.Y_Min + df.Y_Max) / 2
    canonical = list(df.Track_ID.drop_duplicates()[:n_people])
    mapping = {i: i for i in canonical}
    last_pos = {}

    for frame, grp in df.groupby("Frame"):
        present = {mapping[t] for t in grp.Track_ID if t in mapping}
        for _, row in grp.iterrows():
            tid = row.Track_ID
            if tid not in mapping:
                missing = [c for c in canonical if c not in present and c in last_pos]
                if missing:
                    best = min(missing, key=lambda c: np.hypot(row.cx - last_pos[c][0],
                                                                 row.cy - last_pos[c][1]))
                    mapping[tid] = best
                    present.add(best)
                else:
                    mapping[tid] = tid  # can't place it; leave for manual review
            last_pos[mapping[tid]] = (row.cx, row.cy)

    df["Person_ID"] = df.Track_ID.map(mapping)
    return df.drop(columns=["cx", "cy"])


directory = input("Path of the video you want to track?")
model = YOLO("yolo26m-pose.pt") # Change the YAML such that we can get hte occlusion a little bit better, and id's matching the number of people. 
results = model.track(directory, save=True, stream=True, conf=0.2, iou=0.7, show=True,tracker="botsort_occlusion.yaml", persist=True) 

cap = cv2.VideoCapture(directory)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.release()

writer = None
tracking_records = []

# Enumerate allows us to keep track of the frame index
for frame_idx, result in enumerate(results):
   

    blank = np.zeros_like(result.orig_img)   # black canvas, same H/W/dtype as the frame
    annotated = result.plot(img=blank)        # draws boxes + pose skeleton + track IDs onto it

    if writer is None:
        newName = "track_only_" + directory[str(directory).rindex("/"):]
        h, w = annotated.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(newName, fourcc, fps, (w, h))

    writer.write(annotated)

    bbox_list = result.boxes.xyxy.tolist()

    boxes = result.boxes
    # Validate that tracking IDs exist in the current frame
    if boxes.id is not None:
        track_ids = boxes.id.cpu().numpy().astype(int)
        coords = boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

        for idx, (track_id, box) in enumerate(zip(track_ids, coords)):
            row_data = {"Frame": frame_idx + 1, "Track_ID": track_id,
                        "X_Min": float(box[0]), "Y_Min": float(box[1]),
                        "X_Max": float(box[2]), "Y_Max": float(box[3])}
            if keypoints is not None:
                for kp_idx in range(17):
                    row_data[f"KP_{kp_idx}_X"] = float(keypoints[idx][kp_idx][0])
                    row_data[f"KP_{kp_idx}_Y"] = float(keypoints[idx][kp_idx][1])
            tracking_records.append(row_data)

# 4. Convert the full list of dicts to a Pandas DataFrame
#df = pd.DataFrame(tracking_records)
df = remap_ids(pd.DataFrame(tracking_records))

# 5. Export directly to an Excel sheet
# Make sure you have 'openpyxl' installed (pip install openpyxl)
output_filename = "yolo_tracking_data.xlsx"
df.to_excel(output_filename, index=False)

writer.release()

orig_path = "videos/tnt.mp4"
annotated_path = r"C:\Users\conra\OneDrive\Documents\IVVISION\runs\pose\track\tnt.avi"   # the one YOLO's save=True produced
out_path = "tnt.mp4"

cap_orig = cv2.VideoCapture(orig_path)
cap_ann = cv2.VideoCapture(annotated_path)

fps = cap_orig.get(cv2.CAP_PROP_FPS) or 30
w = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

threshold = 20  # how much a pixel must change to count as "overlay"

while True:
    ret1, orig_frame = cap_orig.read()
    ret2, ann_frame = cap_ann.read()
    if not (ret1 and ret2):
        break

    diff = cv2.absdiff(orig_frame, ann_frame)
    mask = np.any(diff > threshold, axis=2)

    blank = np.zeros_like(orig_frame)
    blank[mask] = ann_frame[mask]   # keep the annotated pixel wherever it changed
    writer.write(blank)

cap_orig.release()
cap_ann.release()
writer.release()

import subprocess

video_only = r"C:\Users\conra\OneDrive\Documents\IVVISION\runs\pose\track\tnt.avi"   # what your cv2 loop produced (no audio)
audio_source = directory             # your original clip — still has its audio track
final_output = "tracks_only_with_audio.mp4"

# subprocess.run([
#     "ffmpeg", "-y",
#     "-i", video_only,
#     "-i", audio_source,
#     "-map", "0:v:0",   # take video from the tracked-only file
#     "-map", "1:a:0",   # take audio from the original
#     "-c:v", "copy",    # don't re-encode video — just repackage it
#     "-c:a", "aac",     # re-encode audio to a safely compatible codec
#     "-shortest",       # trim to the shorter of the two, avoids trailing silence/black frames
#     final_output
# ], check=True)

subprocess.run([
    "ffmpeg", "-y",
    "-i", r"C:\Users\conra\OneDrive\Documents\IVVISION\runs\pose\track\tnt.avi",
    "-i", "videos/tnt.mp4",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "libx264",   # decode wmv2, re-encode to h264 — mp4 fully supports this
    "-c:a", "aac",
    "-shortest",
    "tracks_only_with_audio.mp4"
], check=True)

