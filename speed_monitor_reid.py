import os
import cv2
import time
import logging
import sqlite3
import requests
import warnings
import numpy as np
import torch
from datetime import datetime
from typing import Tuple, Optional, Dict, List, Any, Deque
from collections import deque
from ultralytics import YOLO

# 1. Logging and Warnings Configuration
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2. Helper Modules Import
try:
    from sort import Sort
    from db_utils import save_car, compare_embedding, ensure_db, update_car_embedding
    from reid_utils import get_embedding
except ImportError as e:
    logger.error(f"Missing helper files ({e}). Ensure sort.py, db_utils.py, and reid_utils.py exist.")
    exit(1)

# 3. System Configuration
class Config:
    """System-wide configuration settings."""
    # Telegram Bot Settings
    TOKEN: str = "8593853276:AAG1m5IacokzJ7kA-7w7cqDSA_X987jb_j"
    COOLDOWN_SECONDS: int = 300
    RADAR_SITE: str = "Baghdad - Highway"

    # Model & Detection Settings
    MODEL_PATH: str = "yolov8n.pt"
    VEHICLE_CLASSES: List[int] = [2, 5, 7]  # 2: Car, 5: Bus, 7: Truck
    CLASS_NAMES = {2: "Car", 5: "Bus", 7: "Truck"}
    
    # Thresholds & Parameters
    DETECTION_CONF: float = 0.4
    MIN_BOX_AREA: int = 1800
    REID_THRESHOLD: float = 0.9
    GALLERY_UPDATE_THRESHOLD: float = 0.95
    TRAIL_LENGTH: int = 50 

    # Visual Settings (BGR)
    COLORS = {
        2: (0, 165, 255),  # Car: Orange
        5: (255, 0, 0),    # Bus: Blue
        7: (0, 255, 255),  # Truck: Yellow
        'default': (200, 200, 200),
        'text': (255, 255, 255)
    }

# 4. Telegram Notification System
class TelegramNotifier:
    """Handles sending speed violation alerts via Telegram."""
    def __init__(self, token: str):
        self.base_url = f"https://api.telegram.org/bot{token}/sendPhoto"

    def send_violation_alert(self, owner_info: Tuple[str, str, str], speed: float, car_image_path: str):
        """Sends a markdown-formatted alert to the vehicle owner."""
        if not owner_info: return
        name, telegram_id, plate = owner_info

        message = (
            "🚨 **Speed Violation Alert** 🚨\n\n"
            f"👤 *Owner*: {name}\n"
            f"🚗 *Plate*: {plate}\n"
            f"🚀 *Detected Speed*: {speed:.2f} km/h\n\n"
            f"📍 *Location*: {Config.RADAR_SITE}\n"
            f"⏰ *Time*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "Please adhere to speed limits for everyone's safety. ❤️"
        )

        try:
            with open(car_image_path, 'rb') as photo:
                payload = {'chat_id': telegram_id, 'caption': message, 'parse_mode': 'Markdown'}
                files = {'photo': photo}
                response = requests.post(self.base_url, data=payload, files=files, timeout=5)
                response.raise_for_status()
                logger.info(f"Alert sent to owner: {name}")
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
        finally:
            if os.path.exists(car_image_path):
                try: os.remove(car_image_path)
                except: pass

    def save_alert_image(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], car_id: int) -> str:
        """Crops the vehicle from the frame and saves it as a temporary JPEG."""
        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape
        pad = 50
        crop = frame[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
        
        if not os.path.exists("violations"): os.makedirs("violations")
        filename = f"violations/violation_{car_id}_{int(time.time())}.jpg"
        cv2.imwrite(filename, crop)
        return filename

# 5. Main Traffic Monitoring System
class TrafficMonitor:
    """Orchestrates detection, tracking, Re-ID, and speed calculation."""
    def __init__(self, speed_limit: float, delay_sec: float):
        self.speed_limit = speed_limit
        self.delay_sec = delay_sec
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.notifier = TelegramNotifier(Config.TOKEN)
        self.distance_km = 0.0
        
        # Internal State
        self.car_history: Dict[int, Dict[str, float]] = {}
        self.cooldowns: Dict[int, float] = {}
        self.trails: Dict[int, Deque[Tuple[int, int]]] = {}
        self.track_db_map: Dict[int, int] = {}
        self.track_sim_map: Dict[int, float] = {}

        self._init_models()

    def _init_models(self):
        """Initializes YOLO and SORT tracker."""
        if not os.path.exists(Config.MODEL_PATH):
             Config.MODEL_PATH = 'yolov8n.pt'
        self.detector = YOLO(Config.MODEL_PATH).to(self.device)
        self.tracker = Sort(max_age=15, min_hits=5, iou_threshold=0.3)

    def get_owner_info(self, car_id: int) -> Optional[Tuple[str, str, str]]:
        """Queries the owners.db for vehicle owner details."""
        try:
            with sqlite3.connect("owners.db") as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT full_name, telegram_id, plate_number FROM owners WHERE car_id=?", (car_id,))
                return cursor.fetchone()
        except:
            return None

    def process_video(self, video_path: str, mode: str = 'entry'):
        """Main loop for processing a video file."""
        logger.info(f"Processing {mode} video: '{video_path}'")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_dir = "Final_Output"
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        save_name = "FINAL_OUTPUT.mp4" if mode == 'exit' else "PROCESSED_ENTRY.mp4"
        save_path = os.path.join(output_dir, save_name)
        out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # Reset components for new video session
        self.tracker = Sort(max_age=15, min_hits=5, iou_threshold=0.3)
        self.track_db_map, self.track_sim_map = {}, {}

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_idx += 1
            timestamp = (frame_idx / fps) + (self.delay_sec if mode == 'exit' else 0)

            # 1. Detection & Tracking
            tracks, classes_map = self._detect_and_track(frame)
            
            # 2. Sequential Processing of each tracked vehicle
            for x1, y1, x2, y2, track_id, _ in tracks:
                bbox = (int(x1), int(y1), int(x2), int(y2))
                cls_id = classes_map.get(track_id, 2)
                
                # Re-Identify Vehicle
                global_car_id, sim = self._resolve_vehicle_id(track_id, frame, bbox, frame_idx, video_path)
                
                # Logic & Visuals
                speed_info = ""
                if mode == 'entry':
                    self._handle_entry_logic(global_car_id, timestamp)
                else:
                    speed_info = self._handle_exit_logic(global_car_id, timestamp, frame, bbox)

                self._draw_visuals(frame, bbox, global_car_id, track_id, cls_id, speed_info, sim)

            out.write(frame)

        cap.release()
        out.release()
        logger.info(f"Finished {mode}. Output saved to: {save_path}")

    def _detect_and_track(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[int, int]]:
        """Detects vehicles using YOLO and updates the SORT tracker."""
        results = self.detector(frame, classes=Config.VEHICLE_CLASSES, verbose=False)
        
        raw_detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf, cls = box.conf[0].item(), int(box.cls[0].item())
                
                # Filters
                w, h = x2 - x1, y2 - y1
                if (w * h) < Config.MIN_BOX_AREA or conf < Config.DETECTION_CONF:
                    continue
                
                raw_detections.append({'bbox': [x1, y1, x2, y2], 'conf': conf, 'cls': cls, 'area': w * h})

        # 1. Box-in-Box Suppression (Fixes truck-part detection issues)
        raw_detections.sort(key=lambda x: x['area'], reverse=True)
        keep_detections = []
        for det_a in raw_detections:
            is_inside = False
            for det_b in keep_detections:
                box_a, box_b = det_a['bbox'], det_b['bbox']
                # Calc Intersection
                ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
                ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                
                if (inter_area / det_a['area']) > 0.60:
                    is_inside = True; break
            if not is_inside: keep_detections.append(det_a)

        # 2. Update Tracker
        dets_array = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['conf']] for d in keep_detections])
        tracks = self.tracker.update(dets_array if dets_array.size > 0 else np.empty((0, 5)))
        
        # 3. Associate Class IDs to Track IDs
        track_classes = {}
        for x1, y1, x2, y2, tid, _ in tracks:
            tcx, tcy = (x1+x2)/2, (y1+y2)/2
            best_cls, min_dist = 2, 100
            for d in keep_detections:
                dcx, dcy = (d['bbox'][0]+d['bbox'][2])/2, (d['bbox'][1]+d['bbox'][3])/2
                dist = np.linalg.norm([tcx-dcx, tcy-dcy])
                if dist < min_dist:
                    min_dist, best_cls = dist, d['cls']
            track_classes[tid] = best_cls
            
        return tracks, track_classes

    def _resolve_vehicle_id(self, track_id: int, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                            frame_idx: int, video_path: str) -> Tuple[int, float]:
        """Resolves a local tracking ID to a global car_id using Re-ID embeddings."""
        if track_id in self.track_db_map:
            return self.track_db_map[track_id], self.track_sim_map.get(track_id, 0.0)

        try:
            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10: return -1, 0.0
            
            emb = get_embedding(crop)
            match_id, sim = compare_embedding(emb, threshold=Config.REID_THRESHOLD)

            if match_id:
                if sim > Config.GALLERY_UPDATE_THRESHOLD:
                    try: update_car_embedding(match_id, emb)
                    except: pass
                final_id = match_id
            else:
                final_id = save_car(video_path, frame_idx, bbox, emb)
                sim = 1.0 
            
            self.track_db_map[track_id] = final_id
            self.track_sim_map[track_id] = sim
            return final_id, sim
        except:
            return -1, 0.0

    def _handle_entry_logic(self, car_id: int, timestamp: float):
        """Records the entry time for a vehicle."""
        if car_id != -1 and 'entry' not in self.car_history.get(car_id, {}):
            if car_id not in self.car_history: self.car_history[car_id] = {}
            self.car_history[car_id]['entry'] = timestamp
            logger.info(f"Entry Log: Car {car_id} at {timestamp:.2f}s")
    
    def _handle_exit_logic(self, car_id: int, timestamp: float, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """Calculates average speed and triggers alerts if necessary."""
        entry_data = self.car_history.get(car_id)
        if not entry_data or 'entry' not in entry_data: return "No Entry Record"
        
        # Lock Exit time on first detection
        if 'exit_first' not in entry_data:
            entry_data['exit_first'] = timestamp
            duration = (timestamp - entry_data['entry']) / 3600.0
            if duration > 0.0001:
                locked_speed = self.distance_km / duration
                logger.info(f"Speed Locked: ID {car_id} -> {locked_speed:.2f} km/h")
        
        duration_h = (entry_data['exit_first'] - entry_data['entry']) / 3600.0
        if duration_h <= 0.0001: return "Timing Error"
        
        speed = self.distance_km / duration_h
        if speed > self.speed_limit:
            self._trigger_alert_if_needed(car_id, speed, frame, bbox)
            return f"VIOLATION {speed:.1f} km/h"
        return f"OK {speed:.1f} km/h"

    def _trigger_alert_if_needed(self, car_id: int, speed: float, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Triggers a Telegram notification if not in cooldown."""
        if (time.time() - self.cooldowns.get(car_id, 0)) > Config.COOLDOWN_SECONDS:
            owner = self.get_owner_info(car_id)
            if owner:
                path = self.notifier.save_alert_image(frame, bbox, car_id)
                self.notifier.send_violation_alert(owner, speed, path)
                logger.warning(f"ALERT: Violation sent for ID {car_id}")
            self.cooldowns[car_id] = time.time()

    def _draw_visuals(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], car_id: int, 
                      track_id: int, cls_id: int, info_text: str, sim: float):
        """Draws bounding boxes, trails, IDs, and speed info on the frame."""
        x1, y1, x2, y2 = bbox
        
        # 1. Trail Management
        if track_id not in self.trails: self.trails[track_id] = deque(maxlen=Config.TRAIL_LENGTH)
        self.trails[track_id].append(((x1+x2)//2, y2))
        
        # 2. Assign Color (ID-based for stability)
        np.random.seed(car_id if car_id != -1 else track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        
        # 3. Draw Trail
        pts = list(self.trails[track_id])
        for i in range(1, len(pts)):
            thickness = int((i/len(pts)) * 10) + 2
            cv2.line(frame, pts[i-1], pts[i], color, thickness)
        if pts: cv2.circle(frame, pts[-1], 6, color, -1)

        # 4. Draw BBox & Labels
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        sim_txt = f"Sim:{sim*100:.1f}%" if sim > 0 else "New"
        cls_name = Config.CLASS_NAMES.get(cls_id, "Car")
        label = f"ID:{car_id} {cls_name} {info_text if info_text else sim_txt}"
        text_color = (0, 255, 255) if "VIOLATION" in info_text else (255, 255, 255)
        
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1-25), (x1+tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

# -------------------------------
# Main Execution Entry Point
# -------------------------------
if __name__ == "__main__":
    ensure_db()
    print("\n========= 🚔 Smart Traffic Speed Monitor (Enhanced) 🚔 =========\n")

    v_in = input("📂 Enter Entry Video Path: ").strip().replace('"', '')
    v_out = input("📂 Enter Exit Video Path: ").strip().replace('"', '')
    
    try:
        dist = float(input("📏 Real Distance between cameras (km) [Default 1]: ") or 1.0)
        limit = float(input("🚀 Speed Limit (km/h) [Default 100]: ") or 100.0)
        delay = float(input("⏱ Time Delay between cameras (seconds) [Default 0]: ") or 0.0)
        site = input(f"📍 Radar Location [Default '{Config.RADAR_SITE}']: ").strip()
        if site: Config.RADAR_SITE = site
    except:
        print("⚠️ Invalid Input! Using default parameters.")
        dist, limit, delay = 1.0, 100.0, 0.0

    print("\n⚙️ Loading System...\n")
    system = TrafficMonitor(speed_limit=limit, delay_sec=delay)
    system.distance_km = dist
    
    start_time = time.time()
    if os.path.exists(v_in): system.process_video(v_in, mode='entry')
    if os.path.exists(v_out):
        system.process_video(v_out, mode='exit')
        print(f"\n📂 Final Video Saved: Final_Output/FINAL_OUTPUT.mp4")
    
    print(f"\n✅ All tasks finished in {time.time()-start_time:.2f} seconds.")
