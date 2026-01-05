import os
import cv2
import pickle
import numpy as np
import random

# Task 2 : Tracking by detection using spatial distance alone

def compute_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def track_by_distance(predictions, num_frames, tau, max_dist=50):
    tracks = {}        # track_id -> {'bbox': np.array, 'frames_with_no_detection': int}
    trajectories = {}  # track_id -> list of (x, y) points
    next_track_id = 0

    for frame_idx in range(num_frames):
        detections = np.array(predictions.get(frame_idx, []), dtype=np.float32)

        # compute detection bounding box centers
        det_centers = np.array([compute_center(box) for box in detections])

        assigned_tracks = set()
        assigned_detections = set()

        # build list of all possible track-detection pairings: (distance, track_id, detection_idx)
        distances = []
        for track_id, track in tracks.items():
            track_center = compute_center(track['bbox'])
            for i, det_center in enumerate(det_centers):
                d = np.linalg.norm(track_center - det_center)
                distances.append((d, track_id, i))

        # greedy assignement ( closest matches first )
        distances.sort(key=lambda x: x[0])
        for distance, track_id, detection_idx in distances:
            # skip if detection is too far from track, likely a new object
            if distance > max_dist:
                continue
            # skip if track or detection already matched
            if track_id in assigned_tracks or detection_idx in assigned_detections:
                continue
            
            # update track with matched detection
            tracks[track_id]['bbox'] = detections[detection_idx]
            tracks[track_id]['frames_with_no_detection'] = 0
            
            # record detection center in trajectory history
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append(det_centers[detection_idx])

            #mark as assigned to avoid matching multiple times
            assigned_tracks.add(track_id)
            assigned_detections.add(detection_idx)

        # handle unmatched tracks
        # creates a copy to allow deletion durin iteration
        for track_id in list(tracks.keys()):
            if track_id not in assigned_tracks:
                tracks[track_id]['frames_with_no_detection'] += 1
                if tracks[track_id]['frames_with_no_detection'] >= tau:
                    del tracks[track_id]

        # create new track
        for detection_idx in range(len(detections)):
            if detection_idx not in assigned_detections:
                # initialize new track with current detection
                tracks[next_track_id] = {
                    'bbox': detections[detection_idx],
                    'frames_with_no_detection': 0
                }

                # inialize trajectory with first position
                trajectories[next_track_id] = [det_centers[detection_idx]]
                next_track_id += 1

    return trajectories

# Visualization

def draw_trajectories(image, trajectories, max_dist=50):
    img = image.copy()
    
    for track_id, points in trajectories.items():
        if len(points) < 2:  # Need at least 2 points to draw a line
            continue
        
        # Random color for this track
        color = tuple(random.randint(0, 255) for _ in range(3))
        
        # Draw lines between consecutive points
        for i in range(len(points) - 1):
            current_point = points[i]
            next_point = points[i + 1]
            
            # Skip if points are too far apart (likely tracking error/jump)
            distance = np.linalg.norm(next_point - current_point)
            if distance > max_dist:
                continue
            
            # Convert to integer coordinates and draw line
            p1 = tuple(current_point.astype(int))
            p2 = tuple(next_point.astype(int))
            cv2.line(img, p1, p2, color, thickness=3)
    
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
    trajectories = track_by_distance(predictions, num_frames, tau)
    vis = draw_trajectories(first_frame, trajectories)
    cv2.imwrite(f"tracks_by_distance_tau_{tau}.png", vis)
    print(f"τ = {tau}, total tracks: {len(trajectories)}")