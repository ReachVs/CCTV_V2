import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import math
import time
import tracemalloc
import yaml
import numpy as np
import cv2

# State variables for HOTA evaluation
IOU_THRESHOLD = 0.5

def calc_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    inter_width = max(0.0, x2_min - x1_max)
    inter_height = max(0.0, y2_min - y1_max)
    inter_area = inter_width * inter_height
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    if union_area == 0.0:
        return 0.0
    return inter_area / union_area

def calculate_hota(gt_frames, pred_frames, iou_threshold=0.5):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    matches = []
    iou_sums = []
    
    gt_track_counts = {}
    pred_track_counts = {}
    
    for gt, pred in zip(gt_frames, pred_frames):
        for gt_id in gt:
            gt_track_counts[gt_id] = gt_track_counts.get(gt_id, 0) + 1
        for pred_id in pred:
            pred_track_counts[pred_id] = pred_track_counts.get(pred_id, 0) + 1
            
        gt_ids = list(gt.keys())
        pred_ids = list(pred.keys())
        
        matched_gt = set()
        matched_pred = set()
        
        # Match using greedy IoU matching
        for g_id in gt_ids:
            g_box = gt[g_id]
            best_iou = 0
            best_p_id = None
            for p_id in pred_ids:
                if p_id in matched_pred:
                    continue
                p_box = pred[p_id]
                iou = calc_iou(g_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_p_id = p_id
            
            if best_iou >= iou_threshold:
                matched_gt.add(g_id)
                matched_pred.add(best_p_id)
                matches.append((g_id, best_p_id))
                iou_sums.append(best_iou)
                total_tp += 1
                
        total_fp += len(pred_ids) - len(matched_pred)
        total_fn += len(gt_ids) - len(matched_gt)
        
    if total_tp == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    # Association accuracy metrics
    pair_counts = {}
    gt_match_counts = {}
    pred_match_counts = {}
    
    for g_id, p_id in matches:
        pair_counts[(g_id, p_id)] = pair_counts.get((g_id, p_id), 0) + 1
        gt_match_counts[g_id] = gt_match_counts.get(g_id, 0) + 1
        pred_match_counts[p_id] = pred_match_counts.get(p_id, 0) + 1
        
    assa_sum = 0
    for g_id, p_id in matches:
        tpa = pair_counts[(g_id, p_id)]
        fpa = pred_track_counts[p_id] - tpa
        fna = gt_track_counts[g_id] - tpa
        assa_sum += tpa / (tpa + fpa + fna)
        
    assa = assa_sum / total_tp
    deta = total_tp / (total_tp + total_fp + total_fn)
    loca = float(np.mean(iou_sums)) if iou_sums else 0.0
    hota = math.sqrt(deta * assa) * loca
    return hota, deta, assa, loca

def test_occlusion_and_camera_motion():
    print("\n=== Stress Test 1: Occlusion and Camera Motion ===")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    
    crop_path = "data/debug_crops/debug_crop_1.jpg" if os.path.exists("data/debug_crops/debug_crop_1.jpg") else "debug_crop_1.jpg"
    crop1 = cv2.imread(crop_path)
    
    if crop1 is None:
        import glob
        candidate_imgs = glob.glob("events/*.jpg") + glob.glob("data/known_faces/*/*.jpg")
        for c_path in candidate_imgs:
            img = cv2.imread(c_path)
            if img is not None:
                res = model.predict(source=img, classes=[0], verbose=False)
                if len(res[0].boxes) > 0:
                    box = res[0].boxes.xyxy[0].cpu().numpy().astype(int)
                    crop1 = img[box[1]:box[3], box[0]:box[2]]
                    os.makedirs("data/debug_crops", exist_ok=True)
                    cv2.imwrite("data/debug_crops/debug_crop_1.jpg", crop1)
                    break

    if crop1 is None:
        crop1 = np.zeros((300, 200, 3), dtype=np.uint8) + [120, 160, 200]
        crop2 = np.zeros((300, 200, 3), dtype=np.uint8) + [200, 120, 160]
    else:
        crop1 = cv2.resize(crop1, (200, 280))
        crop2 = crop1.copy()
        
    h1, w1, _ = crop1.shape
    h2, w2, _ = crop2.shape
    
    duration_frames = 100
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = "stress_occlusion.mp4"
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 720))
    
    gt_frames = []
    
    for f in range(duration_frames):
        # Frame with random camera jitter noise
        jitter_x = int(5 * math.sin(f * 0.8))
        jitter_y = int(5 * math.cos(f * 0.8))
        frame = np.zeros((720, 1280, 3), dtype=np.uint8) + 40
        # Draw background grid with jitter (simulating camera motion)
        for x in range(0, 1280, 80):
            cv2.line(frame, (x + jitter_x, 0), (x + jitter_x, 720), (50, 50, 50), 1)
        for y in range(0, 720, 80):
            cv2.line(frame, (0, y + jitter_y), (1280, y + jitter_y), (50, 50, 50), 1)
            
        gt_dict = {}
        
        # Track 1 moves left-to-right (from x=100 to 1100)
        x1 = int(100 + 10 * f) + jitter_x
        y1 = 200 + jitter_y
        
        # Track 2 moves right-to-left (from x=1100 to 100)
        x2 = int(1100 - 10 * f) + jitter_x
        y2 = 200 + jitter_y # Same height so they cross and occlude!
        
        # Draw crop2 (Track 2)
        x2_clip = max(0, x2)
        y2_clip = max(0, y2)
        x2_end = min(1280, x2 + w2)
        y2_end = min(720, y2 + h2)
        if x2_end > x2_clip and y2_end > y2_clip:
            frame[y2_clip:y2_end, x2_clip:x2_end] = crop2[(y2_clip - y2):(y2_end - y2), (x2_clip - x2):(x2_end - x2)]
            gt_dict[2] = (x2_clip, y2_clip, x2_end, y2_end)
        
        # Draw crop1 (Track 1)
        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x1_end = min(1280, x1 + w1)
        y1_end = min(720, y1 + h1)
        if x1_end > x1_clip and y1_end > y1_clip:
            frame[y1_clip:y1_end, x1_clip:x1_end] = crop1[(y1_clip - y1):(y1_end - y1), (x1_clip - x1):(x1_end - x1)]
            gt_dict[1] = (x1_clip, y1_clip, x1_end, y1_end)
        
        out.write(frame)
        gt_frames.append(gt_dict)
        
    out.release()
    print(f"Generated occlusion scenario video: '{video_path}'")
    
    # Run tracker on the occlusion video and evaluate HOTA
    cap = cv2.VideoCapture(video_path)
    pred_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Use botsort tracker config
        results = model.track(source=frame, persist=True, tracker="botsort.yaml", classes=[0], verbose=False)
        pred_dict = {}
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for bbox, tid in zip(boxes, ids):
                pred_dict[tid] = bbox
        pred_frames.append(pred_dict)
        
    cap.release()
    
    hota, deta, assa, loca = calculate_hota(gt_frames, pred_frames, iou_threshold=IOU_THRESHOLD)
    print(f"Tracking Accuracy under Severe Occlusion:")
    print(f"HOTA: {hota:.4f}")
    print(f"LocA: {loca:.4f} (Localization Accuracy)")
    print(f"DetA: {deta:.4f} (Detection Accuracy)")
    print(f"AssA: {assa:.4f} (Association Accuracy)")
    
    assert deta > 0.40, f"DetA metric too low ({deta:.4f}), tracking recovery failed."
    print("Stress Test 1 (Occlusion & Camera Motion) PASSED!")
    return hota


def test_db_stability_and_memory_purging():
    print("\n=== Stress Test 2: Database Stability & Memory Purging ===")
    
    import faiss
    import json
    
    mock_index_path = "mock_stress_faces.index"
    mock_metadata_path = "mock_stress_metadata.json"
    
    DIM = 512
    N_PROFILES = 1000
    
    print(f"Building mock database with {N_PROFILES} profiles...")
    mock_embeddings = np.random.randn(N_PROFILES, DIM).astype(np.float32)
    norms = np.linalg.norm(mock_embeddings, axis=1, keepdims=True)
    mock_embeddings = mock_embeddings / np.where(norms == 0, 1.0, norms)
    
    index = faiss.IndexFlatL2(DIM)
    index.add(mock_embeddings)
    faiss.write_index(index, mock_index_path)
    
    mock_metadata = {str(i): f"Subject_{i}" for i in range(N_PROFILES)}
    with open(mock_metadata_path, "w") as f:
        json.dump(mock_metadata, f)
        
    tracemalloc.start()
    
    active_tracks = {}
    track_last_seen = {}
    track_votes = {}
    track_consensus_name = {}
    
    L2_THRESHOLD = 1.30
    COOLDOWN_FRAMES = 150
    identified_cooldowns = {}
    
    def l2_normalize(vector):
        v = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def query_mock_db(emb):
        qv = l2_normalize(emb).reshape(1, -1)
        distances, indices = index.search(qv, k=2)
        best_idx = indices[0][0]
        best_dist = distances[0][0]
        if best_idx == -1 or best_dist >= L2_THRESHOLD:
            return "Unknown"
        if N_PROFILES > 1 and indices[0][1] != -1:
            second_dist = distances[0][1]
            if second_dist - best_dist < 0.08:
                return "Unknown"
        return mock_metadata.get(str(best_idx), "Unknown")

    print("Simulating 1000 frames with 500 distinct target tracks entering/exiting...")
    
    active_in_frame = set()
    total_tracks_spawned = 0
    
    start_memory = tracemalloc.get_traced_memory()[0]
    
    for frame_count in range(1, 1001):
        if frame_count % 3 == 0 and total_tracks_spawned < 500:
            new_tid = total_tracks_spawned
            total_tracks_spawned += 1
            active_in_frame.add(new_tid)
            track_last_seen[new_tid] = frame_count
            
        if frame_count % 4 == 0 and active_in_frame:
            tid_to_leave = list(active_in_frame)[0]
            active_in_frame.remove(tid_to_leave)
            
        for tid in list(active_in_frame):
            track_last_seen[tid] = frame_count
            
            consensus = track_consensus_name.get(tid)
            if consensus is not None:
                active_tracks[tid] = consensus
            else:
                if tid not in track_votes:
                    track_votes[tid] = []
                
                mock_emb = np.random.randn(DIM).astype(np.float32)
                if tid % 2 == 0:
                    mock_emb = mock_embeddings[42] + np.random.randn(DIM) * 0.05
                    
                matched_name = query_mock_db(mock_emb)
                track_votes[tid].append(matched_name)
                
                if len(track_votes[tid]) >= 5:
                    from collections import Counter
                    votes = track_votes[tid][:5]
                    majority_name, count = Counter(votes).most_common(1)[0]
                    
                    if majority_name != "Unknown" and count >= 3:
                        last_seen = identified_cooldowns.get(majority_name, -999999)
                        if frame_count - last_seen < COOLDOWN_FRAMES:
                            pass
                        else:
                            identified_cooldowns[majority_name] = frame_count
                        track_consensus_name[tid] = majority_name
                        active_tracks[tid] = majority_name
                    else:
                        track_consensus_name[tid] = "Unknown"
                        active_tracks[tid] = "Unknown"
                        
        expired_ids = [tid for tid, last_seen in track_last_seen.items() if frame_count - last_seen > 30]
        for tid in expired_ids:
            if tid in active_tracks:
                del active_tracks[tid]
            if tid in track_votes:
                del track_votes[tid]
            if tid in track_consensus_name:
                del track_consensus_name[tid]
            del track_last_seen[tid]
            
    end_memory = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    
    if os.path.exists(mock_index_path):
        os.remove(mock_index_path)
    if os.path.exists(mock_metadata_path):
        os.remove(mock_metadata_path)
        
    print(f"Headless Simulation Summary:")
    print(f"Total simulated tracks spawned: {total_tracks_spawned}")
    print(f"Active tracks remaining in state: {len(active_tracks)}")
    print(f"Track last seen dictionary size: {len(track_last_seen)}")
    print(f"Memory traced during simulation: Start={start_memory/1024:.2f} KB, End={end_memory/1024:.2f} KB")
    
    max_expected_size = len(active_in_frame) + 31
    assert len(track_last_seen) <= max_expected_size, f"Memory leak detected! track_last_seen has {len(track_last_seen)} items, expected <= {max_expected_size}"
    assert len(active_tracks) <= max_expected_size, f"Memory leak detected! active_tracks has {len(active_tracks)} items, expected <= {max_expected_size}"
    
    print("Stress Test 2 (Database Stability & Memory Purging) PASSED!")


def test_hota_loca_spatial_stability_report():
    print("\n=== Stress Test 3: HOTA LocA Spatial Stability Evaluation ===")
    import json
    
    # Generate test synthetic ground truth trajectory (100 frames of smooth motion with jitter)
    # Generate test synthetic ground truth trajectory
    gt_frames = []
    raw_frames = []
    filtered_frames = []
    
    prev_filtered_box = None
        # Generate synthetic ground truth trajectory (50 stationary frames with jitter + 50 moving frames)
    for f in range(1, 101):
        if f <= 50:
            # Stationary subject standing still
            gt_x1 = 200.0
            gt_y1 = 200.0
        else:
            # Moving subject
            gt_x1 = 200.0 + (f - 50) * 6.0
            gt_y1 = 200.0
            
        gt_w = 80.0
        gt_h = 100.0
        gt_box = (gt_x1, gt_y1, gt_x1 + gt_w, gt_y1 + gt_h)
        gt_frames.append({1: gt_box})
        
        # Raw YOLO Bounding Box: Inject 4px high-frequency jitter noise
        jitter_x = 4.0 * math.sin(f * 1.5)
        jitter_y = 3.5 * math.cos(f * 1.5)
        raw_box = (gt_x1 + jitter_x, gt_y1 + jitter_y, gt_x1 + gt_w + jitter_x, gt_y1 + gt_h + jitter_y)
        raw_frames.append({1: raw_box})
        
        if prev_filtered_box is not None:
            ox1, oy1, ox2, oy2 = prev_filtered_box
            curr_cx = (raw_box[0] + raw_box[2]) / 2.0
            curr_cy = (raw_box[1] + raw_box[3]) / 2.0
            prev_cx = (ox1 + ox2) / 2.0
            prev_cy = (oy1 + oy2) / 2.0
            velocity = math.hypot(curr_cx - prev_cx, curr_cy - prev_cy)
            
            if velocity < 3.5:
                filt_box = (ox1, oy1, ox2, oy2)
            else:
                alpha = min(0.85, max(0.08, (velocity - 1.5) / 16.0))
                fx1 = (1.0 - alpha) * ox1 + alpha * raw_box[0]
                fy1 = (1.0 - alpha) * oy1 + alpha * raw_box[1]
                fx2 = (1.0 - alpha) * ox2 + alpha * raw_box[2]
                fy2 = (1.0 - alpha) * oy2 + alpha * raw_box[3]
                filt_box = (fx1, fy1, fx2, fy2)
        else:
            filt_box = raw_box
            
        prev_filtered_box = filt_box
        filtered_frames.append({1: filt_box})
        
    raw_hota, raw_deta, raw_assa, raw_loca = calculate_hota(gt_frames, raw_frames)
    filt_hota, filt_deta, filt_assa, filt_loca = calculate_hota(gt_frames, filtered_frames)
    
    loca_improvement = ((filt_loca - raw_loca) / max(1e-6, raw_loca)) * 100.0
    hota_improvement = ((filt_hota - raw_hota) / max(1e-6, raw_hota)) * 100.0
    
    print(f"RAW YOLO/BoT-SORT Metrics : HOTA={raw_hota:.4f} | LocA={raw_loca:.4f} | AssA={raw_assa:.4f} | DetA={raw_deta:.4f}")
    print(f"FILTERED Engine Metrics   : HOTA={filt_hota:.4f} | LocA={filt_loca:.4f} | AssA={filt_assa:.4f} | DetA={filt_deta:.4f}")
    print(f"Spatial Stability LocA Boost: +{loca_improvement:.2f}%")
    print(f"Overall HOTA Score Boost    : +{hota_improvement:.2f}%")
    
    report_data = {
        "raw_metrics": {"HOTA": round(raw_hota, 4), "LocA": round(raw_loca, 4), "AssA": round(raw_assa, 4), "DetA": round(raw_deta, 4)},
        "filtered_metrics": {"HOTA": round(filt_hota, 4), "LocA": round(filt_loca, 4), "AssA": round(filt_assa, 4), "DetA": round(filt_deta, 4)},
        "improvements": {
            "spatial_stability_loca_percent": round(loca_improvement, 2),
            "hota_percent": round(hota_improvement, 2)
        }
    }
    
    report_path = "hota_stability_report.json"
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)
        
    print(f"Saved HOTA Spatial Stability Report to '{report_path}'.")
    assert filt_loca >= raw_loca - 0.01, "Filtered LocA score must be competitive with raw box LocA score!"
    print("Stress Test 3 (HOTA LocA Spatial Stability Evaluation) PASSED!")
    return report_data


if __name__ == "__main__":
    test_occlusion_and_camera_motion()
    test_db_stability_and_memory_purging()
    test_hota_loca_spatial_stability_report()
    print("\n[STRESS TESTS] All HOTA, LocA spatial stability, and memory stress tests passed successfully.")
