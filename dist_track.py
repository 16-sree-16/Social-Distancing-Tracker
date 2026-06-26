# social_distancing_tracker.py
import cv2
import numpy as np
import time
from typing import List, Tuple, Optional
import math
from scipy.spatial import distance
from collections import deque

class PersonDetector:
    
    def __init__(self, model_path: str = None):

        self.model = None
        self.initialized = False
        
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        if model_path:
            self.load_yolo_model(model_path)
        else:
            self.detector = cv2.createBackgroundSubtractorMOG2()
        
        self.min_distance = 150  
        self.violation_threshold = 60  
        
        self.trackers = []
        self.person_id_counter = 0
        self.person_history = {}
        self.violation_history = {}
        
    def load_yolo_model(self, model_path: str):

        try:
            self.net = cv2.dnn.readNet(model_path + ".weights", model_path + ".cfg")
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.initialized = True
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.initialized = False
    
    def detect_people_hog(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:

        boxes, weights = self.hog.detectMultiScale(frame, winStride=(4, 4), 
                                                   padding=(8, 8), scale=1.05)
        
        detections = []
        for box, weight in zip(boxes, weights):
            if weight > 1.0: 
                x, y, w, h = box
                detections.append((x, y, w, h))
        
        return detections
    
    def detect_people_yolo(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:

        if not self.initialized:
            return self.detect_people_hog(frame)
        
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and class_id == 0: 
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    detections.append((x, y, w, h))
        
        return detections
    
    def track_people(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int]]):

        tracked_people = []
        current_persons = []
        
        for tracker in self.trackers:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                current_persons.append((x, y, w, h))
        
        for detection in detections:
            x, y, w, h = detection
            best_match = None
            best_distance = float('inf')
            
            for i, (cx, cy, cw, ch) in enumerate(current_persons):
                center1 = (x + w//2, y + h//2)
                center2 = (cx + cw//2, cy + ch//2)
                dist = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if dist < 50 and dist < best_distance: 
                    best_distance = dist
                    best_match = i
            
            if best_match is not None:
                self.trackers[best_match] = tracker
            else:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x, y, w, h))
                self.trackers.append(tracker)
                self.person_id_counter += 1
                self.person_history[self.person_id_counter] = {
                    'last_position': (x, y, w, h),
                    'first_seen': time.time()
                }
        
        return current_persons

class SocialDistancingTracker:
    
    def __init__(self, video_source: int = 0, 
                 min_distance: float = 150,
                 detection_method: str = 'hog'):

        self.video_source = video_source
        self.min_distance = min_distance
        self.detection_method = detection_method
        
        self.detector = PersonDetector()
        
        self.total_violations = 0
        self.violation_frames = {}
        self.person_positions = {}
        self.frame_count = 0
        
        self.colors = [
            (0, 255, 0),   
            (0, 0, 255),   
            (255, 0, 0),  
            (255, 255, 0)  
        ]
        
        self.fps = 0
        self.prev_time = time.time()
        
    def calculate_distance(self, p1: Tuple[float, float], 
                           p2: Tuple[float, float]) -> float:

        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def detect_violations(self, positions: List[Tuple[float, float]]) -> List[set]:

        violations = []
        violation_pairs = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = self.calculate_distance(positions[i], positions[j])
                if dist < self.min_distance:
                    violations.append((i, j))
                    violation_pairs.append({i, j})
        
        return violation_pairs
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:

        self.frame_count += 1
        
        if self.detection_method == 'hog':
            detections = self.detector.detect_people_hog(frame)
        else:
            detections = self.detector.detect_people_yolo(frame)
        
        centers = []
        boxes = []
        for (x, y, w, h) in detections:
            center_x = x + w // 2
            center_y = y + h // 2
            centers.append((center_x, center_y))
            boxes.append((x, y, w, h))
        
        violation_pairs = self.detect_violations(centers)
        
        if violation_pairs:
            self.total_violations += 1
            
            for i in range(len(detections)):
                if any(i in pair for pair in violation_pairs):
                    if i not in self.violation_frames:
                        self.violation_frames[i] = 0
                    self.violation_frames[i] += 1
        
        frame = self.draw_results(frame, detections, centers, violation_pairs)
        
        self.fps = 1.0 / (time.time() - self.prev_time)
        self.prev_time = time.time()
        
        self.display_stats(frame)
        
        return frame
    
    def draw_results(self, frame: np.ndarray, 
                     detections: List[Tuple[int, int, int, int]],
                     centers: List[Tuple[float, float]],
                     violation_pairs: List[set]) -> np.ndarray:

        for i, ((x, y, w, h), center) in enumerate(zip(detections, centers)):
            is_violating = any(i in pair for pair in violation_pairs)
            color = self.colors[1] if is_violating else self.colors[0]
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            cv2.circle(frame, (int(center[0]), int(center[1])), 5, color, -1)
            
            cv2.putText(frame, f"P{i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for pair in violation_pairs:
            idx1, idx2 = list(pair)
            if idx1 < len(centers) and idx2 < len(centers):
                cv2.line(frame, 
                        (int(centers[idx1][0]), int(centers[idx1][1])),
                        (int(centers[idx2][0]), int(centers[idx2][1])),
                        self.colors[1], 2)
                
                dist = self.calculate_distance(centers[idx1], centers[idx2])
                mid_x = (centers[idx1][0] + centers[idx2][0]) / 2
                mid_y = (centers[idx1][1] + centers[idx2][1]) / 2
                cv2.putText(frame, f"{dist:.1f}px", 
                           (int(mid_x), int(mid_y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[1], 2)
        
        return frame
    
    def display_stats(self, frame: np.ndarray):

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        y = 20
        cv2.putText(frame, f"Social Distancing Tracker", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y += 25
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, f"People Detected: {len(self.detector.trackers)}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        cv2.putText(frame, f"Violations: {self.total_violations}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        y += 20
        cv2.putText(frame, f"Min Distance: {self.min_distance}px", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y += 20
        active_violations = sum(1 for v in self.violation_frames.values() if v > 0)
        cv2.putText(frame, f"Active Violations: {active_violations}", (10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):

        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Social Distancing Tracker started...")
        print("Press 'q' to quit")
        print("Press 'r' to reset statistics")
        print("Press 'd' to change detection method")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            
            cv2.imshow("Social Distancing Tracker", processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.total_violations = 0
                self.violation_frames.clear()
                print("Statistics reset")
            elif key == ord('d'):
                if self.detection_method == 'hog':
                    self.detection_method = 'yolo'
                else:
                    self.detection_method = 'hog'
                print(f"Detection method changed to: {self.detection_method}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def analyze_video_file(self, video_path: str, output_path: str = None):

        print(f"Analyzing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            processed_frame = self.process_frame(frame)
            
            if output_path:
                out.write(processed_frame)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
                print(f"Total violations: {self.total_violations}")
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"Analysis complete. Total frames: {frame_count}")
        print(f"Total violations detected: {self.total_violations}")
        
        self.generate_report()

class SocialDistancingTrackerGUI:
    
    def __init__(self):
        self.tracker = None
        self.video_source = 0
        self.min_distance = 150
        self.detection_method = 'hog'
        self.running = False
        
    def run_gui(self):

        print("="*50)
        print("SOCIAL DISTANCING TRACKER")
        print("="*50)
        
        while True:
            print("\nOptions:")
            print("1. Start Real-time Tracking (Webcam)")
            print("2. Analyze Video File")
            print("3. Configure Settings")
            print("4. View Statistics")
            print("5. Generate Report")
            print("6. Exit")
            
            choice = input("\nEnter choice: ")
            
            if choice == '1':
                self.start_realtime()
            elif choice == '2':
                self.analyze_video()
            elif choice == '3':
                self.configure_settings()
            elif choice == '4':
                self.view_statistics()
            elif choice == '5':
                self.generate_report()
            elif choice == '6':
                print("Thank you for using Social Distancing Tracker!")
                break
    
    def start_realtime(self):

        self.tracker = SocialDistancingTracker(
            video_source=self.video_source,
            min_distance=self.min_distance,
            detection_method=self.detection_method
        )
        self.tracker.run()
    
    def analyze_video(self):

        video_path = input("Enter video file path: ")
        if not video_path:
            print("No file provided")
            return
        
        self.tracker = SocialDistancingTracker(
            min_distance=self.min_distance,
            detection_method=self.detection_method
        )
        self.tracker.analyze_video_file(video_path)
    
    def configure_settings(self):

        print("\nCurrent Settings:")
        print(f"Video Source: {self.video_source}")
        print(f"Min Distance: {self.min_distance}px")
        print(f"Detection Method: {self.detection_method}")
        
        print("\nNew Settings (press Enter to keep current):")
        
        source = input(f"Video Source (0 for webcam, or file path) [{self.video_source}]: ")
        if source:
            try:
                self.video_source = int(source)
            except ValueError:
                self.video_source = source
        
        distance = input(f"Minimum Distance (pixels) [{self.min_distance}]: ")
        if distance:
            self.min_distance = float(distance)
        
        method = input(f"Detection Method (hog/yolo) [{self.detection_method}]: ")
        if method in ['hog', 'yolo']:
            self.detection_method = method
        
        print("\nSettings updated!")
    
    def view_statistics(self):

        if not self.tracker:
            print("No tracking data available. Start tracking first.")
            return
        
        print("\nCurrent Statistics:")
        print(f"Total Violations: {self.tracker.total_violations}")
        print(f"Active Violations: {len([v for v in self.tracker.violation_frames.values() if v > 0])}")
        print(f"Frames Processed: {self.tracker.frame_count}")
        print(f"Detection Method: {self.tracker.detection_method}")
    
    def generate_report(self):

        if not self.tracker:
            print("No tracking data available. Start tracking first.")
            return
        
        print("\n" + "="*50)
        print("SOCIAL DISTANCING TRACKER REPORT")
        print("="*50)
        
        print(f"\nDetection Method: {self.tracker.detection_method}")
        print(f"Minimum Distance: {self.tracker.min_distance}px")
        print(f"Total Frames Processed: {self.tracker.frame_count}")
        print(f"Total Violations: {self.tracker.total_violations}")
        
        if self.tracker.frame_count > 0:
            violation_rate = (self.tracker.total_violations / self.tracker.frame_count) * 100
            print(f"Violation Rate: {violation_rate:.2f}%")
        
        print("\nViolation Duration Statistics:")
        if self.tracker.violation_frames:
            durations = list(self.tracker.violation_frames.values())
            print(f"  Average Duration: {np.mean(durations):.1f} frames")
            print(f"  Max Duration: {np.max(durations)} frames")
            print(f"  Min Duration: {np.min(durations)} frames")
        else:
            print("  No violations detected")

def social_distancing_demo():
    
    print("=============================================================")
    print("SOCIAL DISTANCING TRACKER")
    print("=============================================================")
    
    print("\nSelect tracking mode:\n1. Real-time tracking (uses webcam)\n2. Video file analysis")
    
    mode = input("\nEnter choice (1 or 2): ")
    
    if mode == '1':
        print("\nStarting real-time social distancing tracking...\nPress 'q' to quit, 'r' to reset stats, 'd' to change detection method")
        
        tracker = SocialDistancingTracker(video_source=0, min_distance=150)
        tracker.run()
    
    elif mode == '2':
        video_path = input("Enter video file path: ")
        if video_path:
            print(f"\nAnalyzing video: {video_path}")
            tracker = SocialDistancingTracker(min_distance=150)
            tracker.analyze_video_file(video_path)
        else:
            print("No video file provided.")
    
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    social_distancing_demo()