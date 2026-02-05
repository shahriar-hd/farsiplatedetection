import cv2
import time
import sys
from picamzero import Camera  # New library for RPi Camera
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_PATH = "best_ncnn_model_640"
CLASS_NAMES = [
    "plate", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
    "alef", "be", "pe", "te", "se", "jim", "dal", "ze", "sin", 
    "shin", "sad", "ta", "za", "ein", "fe", "ghaf", "kaf", 
    "gaf", "lam", "mim", "noon", "vav", "he", "ye", "zhe", 
    "disabled", "protocol", "S", "D",
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def is_inside(box_inner, box_outer):
    c_x = (box_inner[0] + box_inner[2]) / 2
    c_y = (box_inner[1] + box_inner[3]) / 2
    return (box_outer[0] < c_x < box_outer[2]) and (box_outer[1] < c_y < box_outer[3])

def format_plate_text(chars_list):
    return "".join(chars_list)

# ==========================================
# 3. MAIN PROCESS
# ==========================================

def main():
    print("--- LPR with PiCamZero ---")
    
    # 1. Initialize Camera
    try:
        cam = Camera()
        # Set resolution for better performance
        # Note: picamzero handles the internal libcamera complex setup
        print("Camera initialized successfully.")
    except Exception as e:
        print(f"FATAL: Could not initialize camera. Error: {e}")
        return

    # 2. Load Model
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH, task='detect')
    except Exception as e:
        print(f"FATAL: Model load error: {e}")
        return

    print("Starting Detection Loop. Press Ctrl+C to stop.")

    prev_time = time.time()

    try:
        while True:
            # 3. Capture Frame using picamzero
            # .np generates a numpy array (OpenCV compatible)
            frame = cam.capture_array() 

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time

            # 4. YOLO Inference
            results = model(frame, verbose=False, conf=0.5)[0]

            detections = []
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'class': CLASS_NAMES[int(class_id)]
                })

            # 5. Process Detection Logic
            plates = [d for d in detections if d['class'] == 'plate']
            characters = [d for d in detections if d['class'] != 'plate']

            if plates:
                for plate in plates:
                    plate_chars = [c for c in characters if is_inside(c['box'], plate['box'])]
                    # Sort characters by X coordinate (left to right)
                    plate_chars.sort(key=lambda x: x['box'][0])
                    
                    text = format_plate_text([c['class'] for c in plate_chars])
                    if text:
                        print(f"[{time.strftime('%H:%M:%S')}] PLATE DETECTED: {text} (FPS: {fps:.1f})")

    except KeyboardInterrupt:
        print("\nStopping script...")
    finally:
        # Camera is automatically released by picamzero object's destruction
        print("Done.")

if __name__ == "__main__":
    main()