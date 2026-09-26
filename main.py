from ultralytics import YOLO 
import cv2
import numpy as np

model = YOLO("yolo26m-pose.pt") # Change the YAML such that we can get hte occlusion a little bit better, and id's matching the number of people. 

results = model.track("videos/tnt.mp4", save=True, show=True, tracker = "dance_botsort.yaml")

cap = cv2.VideoCapture("videos/tnt.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.release()


writer = None
for result in results:
    blank = np.zeros_like(result.orig_img)   # black canvas, same H/W/dtype as the frame
    annotated = result.plot(img=blank)        # draws boxes + pose skeleton + track IDs onto it

    if writer is None:
        h, w = annotated.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("tracks_only_tnt.mp4", fourcc, fps, (w, h))

    writer.write(annotated)

    bbox_list = result.boxes.xyxy.tolist()

writer.release()

orig_path = "videos/tnt.mp4"
annotated_path = r"C:\Users\conra\OneDrive\Documents\IVVISION\runs\pose\track-5\tnt.avi"   # the one YOLO's save=True produced
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

video_only = r"C:\Users\conra\OneDrive\Documents\IVVISION\runs\pose\track-5\tnt.avi"   # what your cv2 loop produced (no audio)
audio_source = "videos/tnt.mp4"             # your original clip — still has its audio track
final_output = "tracks_only_with_audio4.mp4"

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
    "tracks_only_with_audio5.mp4"
], check=True)

