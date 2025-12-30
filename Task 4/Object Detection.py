import cv2
import time
from ultralytics import YOLO

# --- Task 4: Real-time Object Tracking System ---
# Project by: Malay Patel (CodeAlpha Internship 2025)

def run_vision_system():
    # Load the YOLOv8 model (Nano version for optimized performance)
    print("🚀 Initializing Malay's Vision Engine...")
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Initialize Webcam
    video_stream = cv2.VideoCapture(0)
    
    # Set Resolution for better quality
    video_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not video_stream.isOpened():
        print("❌ System Error: Webcam access denied or not found.")
        return

    prev_time = 0
    print("✅ Tracking System Active!")
    print("💡 Instruction: Press 'Q' or 'ESC' to exit the application.")

    while True:
        ret, frame = video_stream.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # AI Tracking Logic
        # persist=True ensures objects keep their ID across frames
        results = model.track(frame, persist=True, conf=0.45, verbose=False)

        # Generate visual bounding boxes
        output_frame = results[0].plot()

        # Calculate FPS (Performance Monitoring)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # --- Professional UI Overlay ---
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 120), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, output_frame, 0.4, 0, output_frame)

        # Display Name and Project Title
        cv2.putText(output_frame, "MALAY PATEL - CODEALPHA TASK 4", (20, 40), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        
        # Display Real-time Performance (FPS)
        cv2.putText(output_frame, f"System Status: ONLINE | FPS: {int(fps)}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # Display Live Tracking Count
        if results[0].boxes is not None and results[0].boxes.id is not None:
            obj_count = len(results[0].boxes.id)
            cv2.putText(output_frame, f"Objects in Frame: {obj_count}", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(output_frame, "Objects in Frame: 0", (20, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # Show the final processed window
        cv2.imshow("Malay's AI Vision Interface", output_frame)

        # Keyboard Interrupt Handling
        key = cv2.waitKey(1)
        if key & 0xFF == ord("q") or key == 27:
            print("👋 Exiting system...")
            break

    # Proper Shutdown Sequence
    video_stream.release()
    cv2.destroyAllWindows()
    print("🔒 System offline. (Developed by Malay Patel)")

# --- FIXED: Corrected the double underscore syntax here ---
if __name__ == "__main__":
    run_vision_system()