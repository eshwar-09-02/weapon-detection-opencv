import torch
import cv2
import numpy as np
from pathlib import Path
from contextlib import contextmanager
import pathlib
import time
from collections import deque

# Define the context manager to temporarily replace PosixPath with WindowsPath
@contextmanager
def set_posix_windows():
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup

class ObjectDetector:
    def __init__(self, model_path, conf_thresh=0.25, iou_thresh=0.45):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model with optimizations
        with set_posix_windows():
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        
        # Apply optimizations
        self.model.to(self.device)
        self.model.conf = conf_thresh  # Confidence threshold
        self.model.iou = iou_thresh    # NMS IOU threshold
        self.model.agnostic = False    # NMS class-agnostic
        self.model.multi_label = False # NMS multiple labels per box
        self.model.max_det = 1000      # Maximum number of detections
        
        # Enable model optimizations
        if self.device.type != 'cpu':
            self.model.half()  # Convert to FP16 for faster inference
        
        # Initialize tracking
        self.prev_boxes = None
        self.smooth_factor = 0.5
        self.track_history = {}
        self.max_track_history = 30
        
        # Performance monitoring
        self.fps_buffer = deque(maxlen=30)
        
    def preprocess_frame(self, frame):
        # Enhanced preprocessing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.ascontiguousarray(frame)  # Ensure memory is contiguous
        return frame
        
    def smooth_detections(self, current_boxes):
        if self.prev_boxes is None:
            self.prev_boxes = current_boxes
            return current_boxes
            
        # Apply temporal smoothing
        smoothed_boxes = []
        for curr_box in current_boxes:
            best_iou = 0
            matched_prev = None
            
            for prev_box in self.prev_boxes:
                iou = self.calculate_iou(curr_box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    matched_prev = prev_box
                    
            if matched_prev is not None and best_iou > 0.5:
                # Smooth the detection
                smoothed_box = self.smooth_box(curr_box, matched_prev)
                smoothed_boxes.append(smoothed_box)
            else:
                smoothed_boxes.append(curr_box)
                
        self.prev_boxes = smoothed_boxes
        return smoothed_boxes
    
    @staticmethod
    def calculate_iou(box1, box2):
        # Calculate intersection over union of two boxes
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
    
    def smooth_box(self, curr_box, prev_box):
        # Apply exponential smoothing to box coordinates
        return [
            curr_box[0] * (1 - self.smooth_factor) + prev_box[0] * self.smooth_factor,
            curr_box[1] * (1 - self.smooth_factor) + prev_box[1] * self.smooth_factor,
            curr_box[2] * (1 - self.smooth_factor) + prev_box[2] * self.smooth_factor,
            curr_box[3] * (1 - self.smooth_factor) + prev_box[3] * self.smooth_factor,
            curr_box[4],  # confidence
            curr_box[5]   # class
        ]
    
    def draw_detections(self, frame, detections):
        annotated_frame = frame.copy()
        
        for det in detections:
            if len(det) == 0:
                continue
                
            *xyxy, conf, cls = det
            
            # Convert coordinates to integers
            xyxy = [int(x) for x in xyxy]
            
            # Get class name and color
            label = f"{self.model.names[int(cls)]} {conf:.2f}"
            color = self.get_color(int(cls))
            
            # Draw box with thickness based on confidence
            thickness = max(2, int(conf * 3))
            cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, thickness)
            
            # Add label with confidence
            cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw tracking history
            self.draw_tracking_history(annotated_frame, xyxy, int(cls))
            
        return annotated_frame
    
    def get_color(self, class_id):
        # Generate unique color for each class
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
    
    def draw_tracking_history(self, frame, xyxy, class_id):
        # Get center point of detection
        center_x = int((xyxy[0] + xyxy[2]) // 2)
        center_y = int((xyxy[1] + xyxy[3]) // 2)
        center = (center_x, center_y)
        
        # Update tracking history
        if class_id not in self.track_history:
            self.track_history[class_id] = deque(maxlen=self.max_track_history)
        self.track_history[class_id].append(center)
        
        # Draw tracking lines
        points = list(self.track_history[class_id])
        for i in range(1, len(points)):
            pt1 = (int(points[i-1][0]), int(points[i-1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))
            cv2.line(frame, pt1, pt2, self.get_color(class_id), 2)
    
    def process_frame(self, frame):
        start_time = time.time()
        
        # Preprocess
        processed_frame = self.preprocess_frame(frame)
        
        # Inference
        with torch.no_grad():
            results = self.model(processed_frame)
        
        # Get detections
        pred = results.xyxy[0].cpu().numpy()
        
        # Apply smoothing
        smoothed_pred = self.smooth_detections(pred)
        
        # Draw detections
        annotated_frame = self.draw_detections(frame, smoothed_pred)
        
        # Calculate and display FPS
        elapsed_time = time.time () - start_time 
        fps = 1 / (elapsed_time + 1e-6)
        self.fps_buffer.append(fps)
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
        cv2.putText(annotated_frame, f'FPS: {avg_fps:.1f}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated_frame, smoothed_pred

def main():
    # Initialize detector
    model_path = Path('C:/Users/nagas/Documents/project/yolov5/runs/train/exp/weights/best.pt')
    detector = ObjectDetector(model_path, conf_thresh=0.25, iou_thresh=0.45)
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Increased resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture video.")
                break
            
            # Process frame
            annotated_frame, detections = detector.process_frame(frame)
            
            # Display
            cv2.imshow("YOLOv5 Enhanced Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()