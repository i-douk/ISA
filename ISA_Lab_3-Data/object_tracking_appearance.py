import os
import cv2
import pickle
import numpy as np
import random
from scipy.optimize import linear_sum_assignment

# Tracking by detecton: Spatial distance + HOG appearance features

def compute_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])

# we use cosine distance as a metric of comparaison of HOG features
# cosine similarity is defined as a.b/||a||.||b||
def cosine_distance(a, b, eps=1e-6):
    dot = np.dot(a, b)
    # norms of vectors
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    # we add eps to avoids division by zero
    return 1.0 - dot / (norm + eps)

# extract HOG features from a bounding box
def extract_hog(image, box, win_size=(64, 128)):
    # Convert box coordinates to integers
    x1, y1, x2, y2 = map(int, box)
    # Clamp coordinates to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1] - 1, x2)
    y2 = min(image.shape[0] - 1, y2)
    # Crop the image patch corresponding to the detection
    patch = image[y1:y2, x1:x2]

    # If the patch is empty, return a zero vector
    if patch.size == 0:
        return np.zeros(3780)  # Default HOG feature length

    # Resize patch to fixed size for consistent HOG dimensions
    patch = cv2.resize(patch, win_size)

    # Convert to grayscale (HOG works on intensity gradients)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    # Create a HOG descriptor with OpenCV defaults
    hog = cv2.HOGDescriptor()

    # Compute HOG features and flatten into 1D vector
    feat = hog.compute(gray).flatten()

    # Normalize feature vector to unit length
    feat = feat / (np.linalg.norm(feat) + 1e-6)

    return feat

# Combines spatial distance and HOG appearance
# Uses greedy one-to-one assignment (closest matches first)

def track_by_spatial_and_hog(
    predictions,        # dict: frame_idx -> list of bounding boxes
    images,             # list of video frames
    num_frames,         # total number of frames
    tau,                # max allowed frames_with_no_detection frames before deleting a track
    alpha=0.5,          # weight between spatial and appearance costs
    max_spatial_dist=100.0,
    max_appearance_dist=0.7
):
    # tracks dictionary:
    # key   = track_id
    # value = {
    #   'bbox'  : last bounding box,
    #   'feat'  : last HOG feature,
    #   'frames_with_no_detection': number of consecutive frames without detection
    # }
    tracks = {}

    # Store trajectory points for visualization
    trajectories = {}

    # Counter to assign unique IDs
    next_track_id = 0

    # Process frames sequentially
    for frame_idx in range(num_frames):

        detections = predictions.get(frame_idx, [])
        image = images[frame_idx]

        # No detections
        if len(detections) == 0:
            for t in list(tracks.keys()):
                tracks[t]['frames_with_no_detection'] += 1
                if tracks[t]['frames_with_no_detection'] >= tau:
                    del tracks[t]
            continue

        # Prepare detections
        det_boxes = detections
        det_centers = np.array([compute_center(b) for b in det_boxes])
        det_feats = [extract_hog(image, b) for b in det_boxes]

        track_ids = list(tracks.keys())

        assigned_tracks = set()      # tracks already matched
        assigned_dets = set()        # detections already matched

        # Build all possible (track, detection) pairs
        pairs = []
        # Each entry: (cost, track_id, detection_index)

        for track_id in track_ids:
            track_center = compute_center(tracks[track_id]['bbox'])
            track_feat = tracks[track_id]['feat']

            for j, (dc, df) in enumerate(zip(det_centers, det_feats)):
                spatial_cost = np.linalg.norm(track_center - dc)
                appearance_cost = cosine_distance(track_feat, df)

                # Apply gating constraints
                if spatial_cost > max_spatial_dist or appearance_cost > max_appearance_dist:
                    continue

                # Combined cost
                cost = alpha * spatial_cost + (1.0 - alpha) * appearance_cost

                pairs.append((cost, track_id, j))

        # Greedy assignment: sort by cost (lowest first)
        pairs.sort(key=lambda x: x[0])

        for cost, track_id, det_idx in pairs:
            # Enforce one-to-one matching
            if track_id in assigned_tracks:
                continue
            if det_idx in assigned_dets:
                continue

            # Assign detection to track
            tracks[track_id]['bbox'] = det_boxes[det_idx]
            tracks[track_id]['feat'] = det_feats[det_idx]
            tracks[track_id]['frames_with_no_detection'] = 0

            trajectories.setdefault(track_id, []).append(det_centers[det_idx])

            assigned_tracks.add(track_id)
            assigned_dets.add(det_idx)

        # Handle unmatched tracks
        for t in list(tracks.keys()):
            if t not in assigned_tracks:
                tracks[t]['frames_with_no_detection'] += 1
                if tracks[t]['frames_with_no_detection'] >= tau:
                    del tracks[t]

        # Create new tracks for unmatched detections
        for i in range(len(det_boxes)):
            if i not in assigned_dets:
                tracks[next_track_id] = {
                    'bbox': det_boxes[i],
                    'feat': det_feats[i],
                    'frames_with_no_detection': 0
                }
                trajectories[next_track_id] = [det_centers[i]]
                next_track_id += 1

    return trajectories


# Visualization

def draw_trajectories(image, trajectories, max_jump=100):
    img = image.copy()

    for track_id, points in trajectories.items():
        # Need at least two points to draw a line
        if len(points) < 2:
            continue

        # Random color per track
        color = tuple(random.randint(0, 255) for _ in range(3))

        for i in range(len(points) - 1):
            # Skip unrealistic jumps
            if np.linalg.norm(points[i + 1] - points[i]) > max_jump:
                continue

            p1 = tuple(points[i].astype(int))
            p2 = tuple(points[i + 1].astype(int))
            cv2.line(img, p1, p2, color, 3)

    return img

# ---------------- MAIN ----------------

image_sequence_path = '../sequence'
predictions_file_path = 'predictions.pickle'
# Track stop thresholds
taus = [2, 5, 10]

# Load detections
with open(predictions_file_path, 'rb') as f:
    predictions = pickle.load(f)

images = []
for fname in sorted(os.listdir(image_sequence_path)):
    images.append(cv2.imread(os.path.join(image_sequence_path, fname)))

num_frames = len(predictions)

first_frame = images[0]

for tau in taus:
    trajectories = track_by_spatial_and_hog(
        predictions,
        images,
        num_frames,
        tau,
        alpha=0.5
    )
    vis = draw_trajectories(first_frame, trajectories)
    cv2.imwrite(f'tracks_by_appearance_tau_{tau}.png', vis)
    print(f"τ = {tau}, total tracks: {len(trajectories)}")
