from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import shutil
import subprocess

# ---------------- Inputs ----------------
VIDEO_PATH = "videos/bbg.mp4"
NUM_PEOPLE = 5                     # predetermined number of people in the video
TRACKER_CFG = "dance_botsort.yaml"
TRACKS_VIDEO = "tracks_only_tnt.mp4"
EXCEL_OUT = "tnt_boxes.xlsx"
FINAL_OUTPUT = "tracks_only_with_audio.mp4"

model = YOLO("yolo26m-pose.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.release()

# stream=True yields one frame at a time instead of holding the whole video in memory
results = model.track(VIDEO_PATH, stream=True, show=True, tracker=TRACKER_CFG)


# ---------------- ID correction ----------------
# Matching a NEW tracker id (> NUM_PEOPLE) to someone who lost theirs
MAX_MATCH_DIST = 1.5  # in box heights: a new id only takes a missing person's id if it
                      # first appears this close to where that person was last seen
MAX_SIZE_RATIO = 1.5  # ...and if its box is no more than this many times wider/taller
                      # (or narrower/shorter) than that person's last box
SIZE_WEIGHT = 1.0     # how much box size counts vs. distance when ranking candidates
GRACE_FRAMES = 15     # an id seen this recently is probably just a missed detection,
                      # so a far-away new id is never matched to it

# Appearance: clothing colour and body proportions
COLOUR_WEIGHT = 3.0   # how much clothing colour counts when comparing people
BODY_WEIGHT = 3.0     # how much body proportions count when comparing people
COLOUR_ALPHA = 0.05   # how fast a person's colour profile adapts to new frames
OVERLAP_IOU = 0.15    # a box overlapping another this much shows a mix of two people,
                      # so it isn't used to update anyone's colour/body profile
KP_CONF = 0.5         # keypoints below this confidence are ignored
BODY_HISTORY = 90     # frames of body measurements remembered per person
MIN_BODY_PARTS = 4    # need at least this many measurable bones to compare two bodies

# Motion: predicting where each person should be from how they were moving
MOTION_WEIGHT = 3.0   # how much being off the predicted path counts when comparing people
VEL_WINDOW = 10       # recent clean positions used to estimate each person's velocity
MAX_PREDICT_FRAMES = 45  # don't extrapolate further than this (dancers change direction)
POS_HISTORY = 150     # clean positions remembered per person

# Fixing swaps the TRACKER makes between two on-screen people (incl. ids 1..NUM_PEOPLE)
FIX_SWAPS = True
SWAP_RATIO = 0.6      # swap two ids only if the swapped labels fit at most this fraction
                      # as badly as the current ones...
SWAP_MARGIN = 0.3     # ...and better by at least this much in absolute terms...
SWAP_CONFIRM = 8      # ...for this many frames in a row
SWAP_WINDOW = 90      # only check pairs that were close together within this many frames
LOOKS_MARGIN = 0.15   # when both people stayed visible, looks (colour/body) must favour
                      # the swap by at least this much before motion is allowed to count
MIN_PROFILE_FRAMES = 20  # profiles need this many clean frames before swap checks trust them

# Occlusion: when someone vanishes (e.g. passes behind another person), the tracker may
# hand their id to someone nearby. Those nearby people's profiles/positions are frozen
# until it's sorted out, so the wrong person can't overwrite the vanished person's history.
OCCLUSION_RADIUS = 1.5   # in box heights: people this close to someone who vanishes are suspects
OCCLUSION_HOLD = 120     # stop waiting for a vanished person after this many frames

# Bones measured from the COCO pose keypoints. Left/right pairs measure the same
# bone; the longer of the two is used, because a limb pointing towards the camera
# only ever looks shorter than it is, never longer.
BONES = {
    "upper_arm": [(5, 7), (6, 8)],
    "forearm":   [(7, 9), (8, 10)],
    "torso":     [(5, 11), (6, 12)],
    "thigh":     [(11, 13), (12, 14)],
    "shin":      [(13, 15), (14, 16)],
    "shoulders": [(5, 6)],
    "hips":      [(11, 12)],
}

id_map = {}            # raw tracker id -> corrected id, for every raw id that was relabelled
                       # (raw ids 1..NUM_PEOPLE not in here keep their own number)
last_box = {}          # corrected id -> ((x, y) midpoint, (w, h) size) of its most recent box
last_seen = {}         # corrected id -> frame it was last on screen
first_box = {}         # raw id -> ((x, y) midpoint, (w, h) size) where it was first captured
first_frame = {}       # raw id -> frame it was first captured
body_hist = {}         # corrected id -> recent body signatures
raw_body_hist = {}     # raw id -> recent body signatures (used while a new id is being matched)
colour_prof = {}       # corrected id -> [torso histogram, legs histogram] (running average)
raw_colour_prof = {}   # raw id -> same, for new ids being matched
profile_frames = {}    # corrected id -> number of clean frames its profiles were built from
pos_hist = {}          # corrected id -> [(frame, x, y, w, h)] from clean, trusted frames
pair_close = {}        # frozenset of two corrected ids -> last frame their boxes were close
pair_overlap = {}      # frozenset of two corrected ids -> (first, last) frame of their latest overlap
last_absent = {}       # corrected id -> last frame it was off screen (after first being seen)
swap_streak = {}       # frozenset of two corrected ids -> (frames in a row a swap fit better, close frame)
occluded = {}          # vanished corrected id -> (frame it vanished, ids that were near it)
remap_log = []         # one row per relabelling decision, for the Excel sheet


# ----- body proportions -----
def body_signature(kp_xy, kp_conf):
    """Log bone lengths with overall body scale removed, NaN where a bone isn't visible.

    Taking logs and subtracting their mean turns lengths into proportions, so the
    same person gives the same signature near or far from the camera.
    """
    logs = np.full(len(BONES), np.nan)
    for j, pairs in enumerate(BONES.values()):
        lengths = [np.hypot(*(kp_xy[p] - kp_xy[q])) for p, q in pairs
                   if kp_conf[p] >= KP_CONF and kp_conf[q] >= KP_CONF]
        if lengths and max(lengths) > 1:
            logs[j] = np.log(max(lengths))
    known = ~np.isnan(logs)
    if known.sum() < MIN_BODY_PARTS:
        return None
    logs[known] -= logs[known].mean()
    return logs


def body_profile(history):
    """Per-bone median over recent frames, so one odd pose doesn't throw it off."""
    if not history:
        return None
    arr = np.array(history)
    out = np.full(arr.shape[1], np.nan)
    for j in range(arr.shape[1]):
        col = arr[:, j][~np.isnan(arr[:, j])]
        if len(col):
            out[j] = np.median(col)
    return out


def body_diff(a, b):
    """Average difference in proportions over the bones both bodies have (0 = identical).

    Roughly: 0.1 means the bones differ by about 10% relative to each other.
    Returns None if there isn't enough to compare.
    """
    if a is None or b is None:
        return None
    common = ~np.isnan(a) & ~np.isnan(b)
    if common.sum() < MIN_BODY_PARTS:
        return None
    da, db = a[common] - a[common].mean(), b[common] - b[common].mean()
    return float(np.mean(np.abs(da - db)))


def remember(hist, key, sig):
    if sig is not None:
        hist.setdefault(key, []).append(sig)
        del hist[key][:-BODY_HISTORY]


# ----- clothing colour -----
def _region_hist(hsv, x1, y1, x2, y2):
    h, w = hsv.shape[:2]
    x1, x2 = int(max(0, x1)), int(min(w, x2))
    y1, y2 = int(max(0, y1)), int(min(h, y2))
    if (x2 - x1) * (y2 - y1) < 64:
        return None
    hist = cv2.calcHist([hsv[y1:y2, x1:x2]], [0, 1, 2], None, [12, 3, 3], [0, 180, 0, 256, 0, 256])
    return (hist / max(hist.sum(), 1)).astype(np.float32)


def colour_signature(hsv, box, kp_xy=None, kp_conf=None):
    """[torso histogram, legs histogram] of the clothing inside a box (either may be None).

    With confident keypoints the torso is the shoulder-to-hip area and the legs are
    hip-to-knee/ankle, so background and neighbours are mostly left out. Otherwise
    fixed central strips of the box are used.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    torso = legs = None
    if kp_xy is not None:
        ok = lambda *ids: all(kp_conf[i] >= KP_CONF for i in ids)
        if ok(5, 6, 11, 12):
            xs, ys = kp_xy[[5, 6, 11, 12], 0], kp_xy[[5, 6, 11, 12], 1]
            pad = 0.15 * (xs.max() - xs.min())
            torso = (xs.min() + pad, ys[:2].mean(), xs.max() - pad, ys[2:].mean())
            low = [i for i in (15, 16, 13, 14) if kp_conf[i] >= KP_CONF]
            if low:
                legs = (xs[2:].min(), ys[2:].mean(), xs[2:].max(), kp_xy[low, 1].max())
    if torso is None:
        torso = (x1 + 0.3 * w, y1 + 0.2 * h, x2 - 0.3 * w, y1 + 0.5 * h)
    if legs is None:
        legs = (x1 + 0.3 * w, y1 + 0.55 * h, x2 - 0.3 * w, y1 + 0.9 * h)
    return [_region_hist(hsv, *torso), _region_hist(hsv, *legs)]


def colour_diff(a, b):
    """0 = identical colours, 1 = nothing in common (Bhattacharyya distance), averaged
    over the parts (torso, legs) both have. None if nothing to compare."""
    if a is None or b is None:
        return None
    ds = [cv2.compareHist(p, q, cv2.HISTCMP_BHATTACHARYYA)
          for p, q in zip(a, b) if p is not None and q is not None]
    return float(np.mean(ds)) if ds else None


def blend_colour(prof, key, sig, alpha):
    """Running average of a colour profile; parts missing this frame are left as they were."""
    if sig is None:
        return
    old = prof.get(key)
    if old is None:
        prof[key] = [None if s is None else s.copy() for s in sig]
        return
    prof[key] = [s if o is None else o if s is None else (1 - alpha) * o + alpha * s
                 for o, s in zip(old, sig)]


# ----- motion -----
def remember_position(cid, frame_idx, x, y, w, h):
    hist = pos_hist.setdefault(cid, [])
    hist.append((frame_idx, x, y, w, h))
    del hist[:-POS_HISTORY]


def predict(cid, frame_idx):
    """Where person `cid` should be at frame_idx: ((x, y), (w, h)), or None if unknown.

    Fits a straight line through their last VEL_WINDOW clean positions and carries
    it forward from the last one (at most MAX_PREDICT_FRAMES ahead). Size is kept
    as last seen.
    """
    hist = pos_hist.get(cid)
    if not hist:
        return None
    recent = np.array(hist[-VEL_WINDOW:], dtype=float)
    f_last, x, y, w, h = recent[-1]
    vx = vy = 0.0
    if len(recent) >= 3:
        vx = np.polyfit(recent[:, 0], recent[:, 1], 1)[0]
        vy = np.polyfit(recent[:, 0], recent[:, 2], 1)[0]
    dt = min(frame_idx - f_last, MAX_PREDICT_FRAMES)
    return (x + vx * dt, y + vy * dt), (w, h)


def last_position_before(cid, frame_idx):
    """(frame, x, y) of person `cid`'s last clean position before frame_idx, or None."""
    for f, x, y, _, _ in reversed(pos_hist.get(cid, [])):
        if f < frame_idx:
            return f, x, y
    return None


def person_cost(det, cid, frame_idx, use_looks=True, use_motion=True):
    """How unlike person `cid` a detection is: colour + body (looks) and/or distance
    from their predicted position (motion). None if there is nothing to compare."""
    _, (x, y), (_, h), body, colour = det
    parts = []
    if use_looks and profile_frames.get(cid, 0) >= MIN_PROFILE_FRAMES:
        cd = colour_diff(colour, colour_prof.get(cid))
        bd = body_diff(body, body_profile(body_hist.get(cid)))
        if cd is not None:
            parts.append(COLOUR_WEIGHT * cd)
        if bd is not None:
            parts.append(BODY_WEIGHT * bd)
    pred = predict(cid, frame_idx) if use_motion else None
    if pred is not None:
        (px, py), _ = pred
        parts.append(MOTION_WEIGHT * np.hypot(x - px, y - py) / h)
    return sum(parts) if parts else None


# ----- matching new ids -----
def match_score(box, prev_box, cid, body=None, colour=None):
    """How different a new detection is from person `cid`.

    prev_box is where that person is predicted to be now (or their last box).
    Returns (score, distance_px, size_ratio, body_diff, colour_diff). Distance is in
    box heights so it means the same near and far from the camera; size uses log
    ratios so 2x bigger and 2x smaller count the same. Body and colour are added
    when there is enough to compare.
    """
    (x, y), (w, h) = box
    (px, py), (pw, ph) = prev_box
    dist = np.hypot(x - px, y - py)
    log_w, log_h = abs(np.log(w / pw)), abs(np.log(h / ph))
    score = dist / h + SIZE_WEIGHT * (log_w + log_h)
    bd = body_diff(body, body_profile(body_hist.get(cid)))
    cd = colour_diff(colour, colour_prof.get(cid))
    if bd is not None:
        score += BODY_WEIGHT * bd
    if cd is not None:
        score += COLOUR_WEIGHT * cd
    return score, dist, float(np.exp(max(log_w, log_h))), bd, cd


def _r(v, n):
    return None if v is None else round(float(v), n)


def log_remap(frame_idx, raw_id, new_id, reason, dist=None, size_ratio=None, bd=None, cd=None):
    remap_log.append({"frame": frame_idx, "raw_id": raw_id, "matched_id": new_id, "reason": reason,
                      "distance_px": _r(dist, 2), "size_ratio": _r(size_ratio, 2),
                      "body_diff": _r(bd, 3), "colour_diff": _r(cd, 3)})
    print(f"[frame {frame_idx}] id {raw_id} -> {new_id}: {reason}")


def assign_ids(dets, frame_idx):
    """Corrected id for each detection, in the same order as dets.

    dets is a list of (raw_id, (mid_x, mid_y), (w, h), body_sig, colour_sig).
    No two detections in a frame ever get the same corrected id:
      1. Raw ids 1..NUM_PEOPLE keep their own number (unless a swap fix relabelled them).
      2. A raw id relabelled earlier keeps that label, unless it's already on
         screen - then the label was wrong, so it is dropped and redone.
      3. Anything left (normally a new raw id > NUM_PEOPLE) gets, in order of preference:
         a. the best-matching off-screen id (position, size, body and colour), if it
            appeared within MAX_MATCH_DIST and MAX_SIZE_RATIO of that person's last
            box (someone who lost their id);
         b. an id 1..NUM_PEOPLE that hasn't been used yet (a new person walking in);
         c. the best-matching id that has been gone for more than GRACE_FRAMES;
         d. otherwise it keeps its raw id for now (if free) and is retried next frame.
    """
    ids = [None] * len(dets)
    taken = set()

    # Raw ids 1..N that still carry their own number go first
    for k, (rid, *_) in enumerate(dets):
        if rid is not None and rid <= NUM_PEOPLE and id_map.get(rid, rid) == rid:
            ids[k] = rid
            taken.add(rid)

    pending = []
    for k, (rid, center, size, *_) in enumerate(dets):
        if rid is None or ids[k] is not None:
            continue
        cid = id_map.get(rid)
        if cid is not None and cid not in taken:
            ids[k] = cid
            taken.add(cid)
            continue
        if cid is not None:
            del id_map[rid]
            first_box[rid] = (center, size)   # re-match from where it is now
            first_frame[rid] = frame_idx
            log_remap(frame_idx, rid, None, f"unmapped: id {cid} is already on screen")
        pending.append(k)

    for k in pending:
        rid, center, size, _, _ = dets[k]
        first_box.setdefault(rid, (center, size))
        first_frame.setdefault(rid, frame_idx)
        body = body_profile(raw_body_hist.get(rid))
        colour = raw_colour_prof.get(rid)

        # Compare with where each missing person should be by the time this id
        # appeared (following their last movement), not just where they vanished.
        # (score, distance_px, size_ratio, body_diff, colour_diff, id), best match first
        missing = sorted(
            (*match_score(first_box[rid], predict(cid, first_frame[rid]) or prev, cid, body, colour), cid)
            for cid, prev in last_box.items()
            if cid not in taken
        )
        close = [m for m in missing
                 if m[1] <= MAX_MATCH_DIST * first_box[rid][1][1] and m[2] <= MAX_SIZE_RATIO]
        unused = [cid for cid in range(1, NUM_PEOPLE + 1)
                  if cid not in last_box and cid not in taken]
        stale = [m for m in missing if frame_idx - last_seen[m[5]] > GRACE_FRAMES]

        if close:
            _, dist, ratio, bd, cd, choice = close[0]
            reason = "best-matching missing id (position + size + body + colour)"
        elif unused:
            dist, ratio, bd, cd, choice = None, None, None, None, unused[0]
            reason = "new person: unused id"
        elif stale:
            _, dist, ratio, bd, cd, choice = stale[0]
            reason = "best-matching long-missing id (outside match limits)"
        else:
            ids[k] = rid if rid not in taken else None
            continue

        id_map[rid] = choice
        taken.add(choice)
        ids[k] = choice
        log_remap(frame_idx, rid, choice, reason, dist, ratio, bd, cd)

    return ids


# ----- fixing swaps between on-screen people -----
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def relabel_along_paths(rows, a, b, from_frame, frame_idx, end_a, end_b):
    """Re-decide ids a/b in rows from from_frame up to (not incl.) frame_idx by motion.

    Each person's path is drawn as a straight line from their last clean position
    before from_frame to where they are now (end_a / end_b), and every row labelled
    a or b goes to whichever path it sits on. This also sorts out the frames where
    the two overlapped or one was hidden behind the other. Returns False (nothing
    changed) if either person has no position from before from_frame.
    """
    start_a, start_b = last_position_before(a, from_frame), last_position_before(b, from_frame)
    if start_a is None or start_b is None:
        return False

    def on_path(start, end, f):
        f0, x0, y0 = start
        t = (f - f0) / max(frame_idx - f0, 1)
        return x0 + t * (end[0] - x0), y0 + t * (end[1] - y0)

    by_frame = {}
    for row in reversed(rows):
        if row["frame"] < from_frame:
            break
        if row["id"] in (a, b):
            by_frame.setdefault(row["frame"], []).append(row)

    for f, frame_rows in by_frame.items():
        pa, pb = on_path(start_a, end_a, f), on_path(start_b, end_b, f)
        da = [np.hypot(r["mid_x"] - pa[0], r["mid_y"] - pa[1]) for r in frame_rows]
        db = [np.hypot(r["mid_x"] - pb[0], r["mid_y"] - pb[1]) for r in frame_rows]
        if len(frame_rows) == 1:
            frame_rows[0]["id"] = a if da[0] <= db[0] else b
        else:   # two rows: pick the pairing with the smaller total distance
            straight = da[0] + db[1]
            crossed = db[0] + da[1]
            frame_rows[0]["id"], frame_rows[1]["id"] = (a, b) if straight <= crossed else (b, a)
    return True


def fix_swaps(dets, ids, boxes, clean, frame_idx, rows):
    """Swap two on-screen ids back if each person clearly fits the other's id better.

    Trackers swap ids when two people cross or one passes behind the other.
    Afterwards each person is on the OTHER id's predicted path, and their clothes
    and body match the OTHER id's profile. When that holds for SWAP_CONFIRM frames
    in a row (and the two were close together recently), their labels are swapped
    from then on, and rows already written since the overlap began are re-decided
    by which person's path they sit on.

    Motion alone can't tell "crossed over" from "met and turned back" when two
    look-alike people fully overlap, so it is only trusted on its own when one of
    them actually vanished behind the other. If both stayed visible the whole time
    (the tracker could see them too), looks must also favour the swap.

    Returns (ids, frozen): frozen ids are mid-check, so their profiles and positions
    aren't updated this frame (otherwise a swapped person would overwrite them).
    """
    frozen = set()
    on_screen = [k for k, cid in enumerate(ids) if cid is not None and cid <= NUM_PEOPLE]
    visible = {ids[k] for k in on_screen}
    for cid in last_seen:
        if cid not in visible:
            last_absent[cid] = frame_idx

    for i, k1 in enumerate(on_screen):
        for k2 in on_screen[i + 1:]:
            b1, b2 = boxes[k1], boxes[k2]
            key = frozenset((ids[k1], ids[k2]))
            gap = np.hypot((b1[0] + b1[2] - b2[0] - b2[2]) / 2, (b1[1] + b1[3] - b2[1] - b2[3]) / 2)
            if iou(b1, b2) > 0 or gap < max(b1[3] - b1[1], b2[3] - b2[1]):
                pair_close[key] = frame_idx
            if iou(b1, b2) > 0:
                # Still the same overlap if it ran until last frame, or if one of them
                # was hidden (off screen) in the gap - i.e. passing behind the other
                first, last = pair_overlap.get(key, (frame_idx, frame_idx))
                hidden_since = max(last_absent.get(ids[k1], -1), last_absent.get(ids[k2], -1)) > last
                pair_overlap[key] = (first if last >= frame_idx - 1 or hidden_since else frame_idx,
                                     frame_idx)

    used = set()
    for i, k1 in enumerate(on_screen):
        for k2 in on_screen[i + 1:]:
            if k1 in used or k2 in used:
                continue
            a, b = ids[k1], ids[k2]
            key = frozenset((a, b))
            if frame_idx - pair_close.get(key, -10**9) > SWAP_WINDOW:
                continue
            if not (clean[k1] and clean[k2]):
                continue

            pairs = ((k1, a), (k2, b), (k1, b), (k2, a))
            costs = [person_cost(dets[k], cid, frame_idx) for k, cid in pairs]
            if None in costs:
                continue
            keep, swap = costs[0] + costs[1], costs[2] + costs[3]

            # One of them hid behind the other - or vanished while the other was nearby
            # (e.g. passed behind a third person), so the tracker may have swapped them
            overlap = pair_overlap.get(key)
            hid_behind = (overlap is not None and max(
                last_absent.get(a, -1), last_absent.get(b, -1)) >= overlap[0]) or any(
                x in occluded and y in occluded[x][1] for x, y in ((a, b), (b, a)))
            if not hid_behind:
                looks = [person_cost(dets[k], cid, frame_idx, use_motion=False) for k, cid in pairs]
                looks_agree = (None not in looks and
                               (looks[0] + looks[1]) - (looks[2] + looks[3]) > LOOKS_MARGIN)
            else:
                looks_agree = True

            if looks_agree and swap < SWAP_RATIO * keep and keep - swap > SWAP_MARGIN:
                n, since = swap_streak.get(key, (0, frame_idx))
                swap_streak[key] = (n + 1, since)
                frozen.update((a, b))
            else:
                swap_streak.pop(key, None)
                continue

            if swap_streak[key][0] < SWAP_CONFIRM:
                continue

            # Confirmed: swap labels for good...
            since = swap_streak.pop(key)[1]
            r1, r2 = dets[k1][0], dets[k2][0]
            id_map[r1], id_map[r2] = b, a
            ids[k1], ids[k2] = b, a
            used.update((k1, k2))

            # ...then fix the rows already written, back to when they started overlapping
            overlap = pair_overlap.get(key)
            from_frame = overlap[0] if overlap and frame_idx - overlap[1] <= SWAP_WINDOW else since
            end_a, end_b = dets[k2][1], dets[k1][1]   # where a and b really are now
            if not relabel_along_paths(rows, a, b, from_frame, frame_idx, end_a, end_b):
                from_frame = since
                for row in reversed(rows):
                    if row["frame"] < since:
                        break
                    if row["raw_id"] in (r1, r2) and row["id"] in (a, b):
                        row["id"] = b if row["id"] == a else a
            reason = f"swap fix: tracker swapped ids {a} and {b} (rows from frame {from_frame} re-checked)"
            log_remap(frame_idx, r1, b, reason)
            log_remap(frame_idx, r2, a, reason)

    return ids, frozen


# ---------------- Main loop ----------------
rows = []
writer = None

for frame_idx, result in enumerate(results):
    boxes = result.boxes
    xyxy = boxes.xyxy.tolist()
    raw_ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(xyxy)
    confs = boxes.conf.tolist()

    n = len(xyxy)

    kp_xy = kp_conf = None
    if result.keypoints is not None and n:
        kp_xy = result.keypoints.xy.cpu().numpy()
        kp_conf = (result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None
                   else np.ones(kp_xy.shape[:2]))

    hsv = cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2HSV)
    bodies = [body_signature(kp_xy[k], kp_conf[k]) if kp_xy is not None else None for k in range(n)]
    colours = [colour_signature(hsv, xyxy[k], None if kp_xy is None else kp_xy[k],
                                None if kp_xy is None else kp_conf[k]) for k in range(n)]
    # A box overlapping another shows a mix of two people, so it doesn't update profiles
    clean = [all(iou(xyxy[k], xyxy[j]) < OVERLAP_IOU for j in range(n) if j != k) for k in range(n)]

    for rid, body, colour, ok in zip(raw_ids, bodies, colours, clean):
        if rid is not None and ok:
            remember(raw_body_hist, rid, body)
            blend_colour(raw_colour_prof, rid, colour, 0.3)

    dets = [(rid, ((x1 + x2) / 2, (y1 + y2) / 2), (max(x2 - x1, 1), max(y2 - y1, 1)), body, colour)
            for (x1, y1, x2, y2), rid, body, colour in zip(xyxy, raw_ids, bodies, colours)]
    new_ids = assign_ids(dets, frame_idx)
    frozen = set()
    if FIX_SWAPS:
        new_ids, frozen = fix_swaps(dets, new_ids, xyxy, clean, frame_idx, rows)

    visible = {cid: dets[k] for k, cid in enumerate(new_ids) if cid is not None and cid <= NUM_PEOPLE}
    # Someone just vanished: anyone near where they were might take their id, so don't let
    # their profiles/positions update until it's sorted out
    for cid in last_seen:
        if cid not in visible and cid not in occluded and last_seen[cid] == frame_idx - 1:
            (x, y), (_, h) = last_box[cid]
            near = {o for o, d in visible.items()
                    if np.hypot(d[1][0] - x, d[1][1] - y) < OCCLUSION_RADIUS * h}
            occluded[cid] = (frame_idx, near)
    # Release once they've been back long enough for fix_swaps to decide, or after OCCLUSION_HOLD
    for cid, (since, near) in list(occluded.items()):
        back_for = frame_idx - last_absent.get(cid, frame_idx)
        if (cid in visible and back_for > 2 * SWAP_CONFIRM) or frame_idx - since > OCCLUSION_HOLD:
            del occluded[cid]
        else:
            frozen |= near | {cid}

    frame_boxes = []
    for (x1, y1, x2, y2), raw_id, conf, new_id, body, colour, ok in zip(
            xyxy, raw_ids, confs, new_ids, bodies, colours, clean):
        tl, tr, bl, br = (x1, y1), (x2, y1), (x1, y2), (x2, y2)
        mid_x = (tl[0] + tr[0]) / 2   # midpoint of top-left & top-right
        mid_y = (tl[1] + bl[1]) / 2   # midpoint of top-left & bottom-left

        if new_id is not None and new_id <= NUM_PEOPLE:
            last_box[new_id] = ((mid_x, mid_y), (max(x2 - x1, 1), max(y2 - y1, 1)))
            last_seen[new_id] = frame_idx
            if ok and new_id not in frozen:
                count = profile_frames.get(new_id, 0)
                remember(body_hist, new_id, body)
                blend_colour(colour_prof, new_id, colour, max(COLOUR_ALPHA, 1 / (count + 1)))
                profile_frames[new_id] = count + 1
                remember_position(new_id, frame_idx, mid_x, mid_y, x2 - x1, y2 - y1)

        rows.append({
            "frame": frame_idx, "raw_id": raw_id, "id": new_id, "conf": round(conf, 3),
            "tl_x": tl[0], "tl_y": tl[1], "tr_x": tr[0], "tr_y": tr[1],
            "bl_x": bl[0], "bl_y": bl[1], "br_x": br[0], "br_y": br[1],
            "mid_x": mid_x, "mid_y": mid_y,
        })
        frame_boxes.append(((int(x1), int(y1), int(x2), int(y2)), new_id))

    # Draw skeletons on a black canvas, then our own boxes with the CORRECTED ids
    # (result.plot would show the raw tracker ids)
    blank = np.zeros_like(result.orig_img)
    annotated = result.plot(img=blank, boxes=False)
    for (x1, y1, x2, y2), new_id in frame_boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, f"id {new_id}", (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if writer is None:
        h, w = annotated.shape[:2]
        writer = cv2.VideoWriter(TRACKS_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    writer.write(annotated)

if writer is not None:
    writer.release()


# ---------------- Excel export ----------------
df = pd.DataFrame(rows)
remap_df = pd.DataFrame(remap_log, columns=[
    "frame", "raw_id", "matched_id", "reason", "distance_px", "size_ratio", "body_diff",
    "colour_diff"])

with pd.ExcelWriter(EXCEL_OUT) as xl:            # needs: pip install openpyxl
    df.to_excel(xl, sheet_name="boxes", index=False)
    remap_df.to_excel(xl, sheet_name="id_remaps", index=False)
df.to_csv(EXCEL_OUT.replace(".xlsx", ".csv"), index=False)
print(f"Wrote {len(df)} box rows and {len(remap_df)} remaps to {EXCEL_OUT}")



def find_ffmpeg():
    """ffmpeg on PATH, else the copy bundled with `pip install imageio-ffmpeg`, else None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


# ---------------- Add the original audio back ----------------
FFMPEG = find_ffmpeg()
if FFMPEG is None:
    print(f"ffmpeg not found - skipped adding audio. Silent video is at {TRACKS_VIDEO}.")
    print("Fix: pip install imageio-ffmpeg")
else:
    subprocess.run([
        FFMPEG, "-y",
        "-i", TRACKS_VIDEO,
        "-i", VIDEO_PATH,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        FINAL_OUTPUT,
    ], check=True)
