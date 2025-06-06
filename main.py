import cv2
from ultralytics import YOLO
import numpy as np
import pyttsx3
import time
from threading import Thread
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize text-to-speech engine
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.9)
except Exception as e:
    logger.error(f"Failed to initialize text-to-speech: {e}")
    engine = None

def speak(text):
    """Thread-safe text-to-speech function."""
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"Text-to-speech error: {e}")

def async_speak(text):
    """Run speech in a separate thread to avoid blocking."""
    Thread(target=speak, args=(text,)).start()

def initialize_model(model_path="yolov8s.pt"):
    """Initialize YOLO model with error handling."""
    try:
        model = YOLO(model_path)
        logger.info(f"Loaded YOLO model: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None

def initialize_video(video_path):
    """Initialize video capture with error handling."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        logger.info(f"Loaded video: {video_path}")
        return cap
    except Exception as e:
        logger.error(f"Failed to load video: {e}")
        return None

def estimate_distance(box, frame_height):
    """Estimate object distance based on bounding box size (simplified)."""
    _, y1, _, y2 = map(int, box.xyxy[0])
    box_height = y2 - y1
    return max(1, frame_height / (box_height + 1e-5))  # Avoid division by zero

def draw_gui(frame, speed, steering, decision, collision_risk, stopped):
    """Draw GUI overlay on the frame."""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    status = "STOPPED" if stopped else f"Speed: {speed:.1f} km/h"
    cv2.putText(frame, status, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Steering: {steering}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Decision: {decision}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if collision_risk:
        cv2.putText(frame, "⚠️ COLLISION WARNING ⚠️", (width//4, height//2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

def process_frame(frame, model, target_classes, speed, width, height):
    """Process a single frame and update driving parameters."""
    decision = "Cruising"
    steering = "↑"
    collision_risk = False
    stopped = False

    try:
        results = model(frame, verbose=False, conf=0.5)[0]
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        return decision, steering, speed, collision_risk, stopped

    critical_objects = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        conf = float(box.conf[0])

        if conf < 0.5 or label not in target_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        distance = estimate_distance(box, height)
        critical_objects.append((label, cx, distance))

    # Sort objects by distance (closest first) to prioritize critical ones
    critical_objects.sort(key=lambda x: x[2])

    for label, cx, distance in critical_objects:
        if label in {"car", "bus", "truck", "person"}:
            if distance < 8:  # Very close: stop
                decision = f"Stopping for {label}"
                speed = 0
                stopped = True
                collision_risk = True
                break
            elif distance < 15 and abs(cx - width // 2) < width * 0.2:  # Close: slow down
                decision = f"Slowing for {label}"
                speed = max(15, speed - np.random.uniform(3, 7))
                steering = "→" if cx < width // 2 else "←"
            elif distance < 25 and abs(cx - width // 2) < width * 0.25:  # Moderate: lane change
                decision = f"Obstacle ahead. Change to {'right' if cx < width // 2 else 'left'} lane"
                speed = max(20, speed - np.random.uniform(2, 5))
                steering = "→" if cx < width // 2 else "←"
                collision_risk = distance < 10  # Only trigger collision risk for very close objects
            else:
                decision = "Monitoring obstacle"
                speed = max(25, speed - np.random.uniform(0, 3))
                steering = "→" if cx < width // 2 else "←"
        elif label in {"stop sign", "traffic light"}:
            if distance < 10:
                decision = f"Stopping for {label}"
                speed = 0
                stopped = True
            else:
                decision = f"Slowing for {label}"
                speed = max(15, speed - np.random.uniform(3, 7))

        if stopped:
            break  # Prioritize stopping over other actions

    return decision, steering, speed, collision_risk, stopped

def main():
    # Configuration
    video_path = r"D:\Downloads\Untitled video - Made with Clipchamp (1).mp4"
    target_classes = {"car", "bus", "truck", "person", "stop sign", "traffic light"}
    speed = 40.0
    last_spoken = ""
    last_warning_time = 0
    warning_cooldown = 3.0  # Increased cooldown for less frequent warnings

    # Initialize components
    model = initialize_model()
    if not model:
        return

    cap = initialize_video(video_path)
    if not cap:
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video reached")
                break

            height, width = frame.shape[:2]

            # Process frame
            decision, steering, speed, collision_risk, stopped = process_frame(
                frame, model, target_classes, speed, width, height
            )

            # Update speed for cruising
            if decision == "Cruising" and not stopped:
                speed = min(60, speed + 0.5)

            # Voice feedback (only speak if decision changes significantly)
            if decision != last_spoken and decision not in {"Monitoring obstacle"}:
                async_speak(decision)
                last_spoken = decision

            # Collision warning (only for critical situations)
            if collision_risk and time.time() - last_warning_time > warning_cooldown:
                async_speak("Warning! Collision risk ahead.")
                last_warning_time = time.time()

            # Draw GUI
            draw_gui(frame, speed, steering, decision, collision_risk, stopped)

            # Resize and display
            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            cv2.imshow("🚗 Self-Driving Car Simulation", frame)

            # Handle exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User terminated simulation")
                break

    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Simulation ended, resources released")

if __name__ == "__main__":
    main()