import os
import time
import cv2
import tempfile
from datetime import datetime
from object_detector import ObjectDetector
from sound_player import SoundPlayer
from db_handler import OCRDatabase
from ocr_module.ocr_wrapper import predict_text


os.makedirs("tracking_images", exist_ok=True)

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def process_videostream(detector, sound_player, db, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    state_counter = {"open": 0, "closed": 0, "sliding": 0}
    state_threshold = 1
    sharpest_image = None
    max_sharpness = 0
    ocr_sent = False
    detection_start_time = None
    last_door_state = None
    
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            bbox, labels, confidences = detector.detect_objects(frame)
            current_state = next((state for state in ["open", "closed", "sliding"] if state in labels), None)
            if current_state:
                last_door_state = current_state
                state_counter[current_state] += 1
                for state in state_counter:
                    if state != current_state:
                        state_counter[state] = 0
            # print('current_state:',current_state)
            
            for state, count in state_counter.items():
                if count >= state_threshold:
                    sound_player.play_transition_sound(state)
                    break

            number_location_found = False
            for i, label in enumerate(labels):
                if label == "number_location":
                    number_location_found = True
                    x, y, w, h = bbox[i]
                    cropped = frame[y:y+h, x:x+w]
                    if cropped.size == 0:
                        continue
                    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                    sharpness = variance_of_laplacian(gray_crop)
                    if sharpness > max_sharpness:
                        max_sharpness = sharpness
                        sharpest_image = cropped.copy()
                    if detection_start_time is None:
                        detection_start_time = time.time()

            if not number_location_found:
                detection_start_time = None
                sharpest_image = None
                max_sharpness = 0
                ocr_sent = False

            if detection_start_time and not ocr_sent:
                if time.time() - detection_start_time >= 0.5 and sharpest_image is not None:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    cv2.imwrite(temp_file.name, sharpest_image)
                    ocr_result = predict_text(temp_file.name)
                    print("Forklift:", ocr_result)
                    print("Last door state:", last_door_state)
                    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    save_frame = detector.draw_bbox(frame, bbox, labels, confidences)
                    cv2.imwrite(f"tracking_images/{timestamp}.jpg", save_frame)
                    db.save_ocr_result(ocr_result, timestamp, last_door_state)
                    ocr_sent = True

            frame = detector.draw_bbox(frame, bbox, labels, confidences)
            end_time = time.time()
            inference_time = end_time - start_time
            fps = 1 / inference_time if inference_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 25, 0), 2)
            cv2.imshow("Object Detection - Video File", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

def main():
    detector = ObjectDetector(
        weights="/home/elevatorpi4/project/EMS-phase2-deploy/models/yolov4_det_250407.weights",
        config="/home/elevatorpi4/project/EMS-phase2-deploy/config.cfg",
        labels_path="/home/elevatorpi4/project/EMS-phase2-deploy/classes.txt",
        confidence_threshold=0.6
    )
    sound_player = SoundPlayer(
        sliding_open_sound="/home/elevatorpi4/project/EMS-phase2-deploy/audio/SLIDING.mp3",
        sliding_close_sound="/home/elevatorpi4/project/EMS-phase2-deploy/audio/SLIDING.mp3"
    )
    db = OCRDatabase()
    video_path = '/home/elevatorpi4/project/EMS-phase2-deploy/video_data/ppart1.avi'
    process_videostream(detector, sound_player, db, video_path)

if __name__ == "__main__":
    main()
