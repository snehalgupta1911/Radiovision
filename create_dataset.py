import cv2
import os
from hand_tracking import HandDetector

# -------------------------------
# CONFIG
# -------------------------------
SAVE_PATH = "dataset"
IMG_SIZE = 224
MAX_IMAGES_PER_CLASS = 300

GESTURES = {
    "0": "palm",
    "1": "fist",
    "2": "peace",
    "3": "thumbs_up",
    "4": "ok"
}

# -------------------------------
# CREATE DATASET FOLDERS
# -------------------------------
os.makedirs(SAVE_PATH, exist_ok=True)
for gesture in GESTURES.values():
    os.makedirs(os.path.join(SAVE_PATH, gesture), exist_ok=True)

# -------------------------------
# IMAGE COUNTERS
# -------------------------------
image_counts = {}
for gesture in GESTURES.values():
    image_counts[gesture] = len(os.listdir(os.path.join(SAVE_PATH, gesture)))

current_label = None

# -------------------------------
# START WEBCAM & DETECTOR
# -------------------------------
cap = cv2.VideoCapture(0) # Using default, can add cv2.CAP_AVFOUNDATION if warnings occur
detector = HandDetector(max_hands=1, detection_confidence=0.7)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n=== DATASET COLLECTION STARTED ===")
print("Press these keys to choose gesture class:")
for key, value in GESTURES.items():
    print(f"  {key} -> {value}")
print("Press 's' to save image")
print("Press 'q' to quit\n")

offset = 40  # Padding around the hand

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)        # Mirror the frame
    frame_clean = frame.copy()        # Keep a clean copy for cropping
    
    # Run MediaPipe hand tracking natively
    frame = detector.find_hands(frame, draw=False)
    lm_list = detector.find_position(frame, draw=False)
    
    hand_crop = None
    
    if len(lm_list) != 0:
        # Dynamic ROI based on hand landmarks
        x_list = [lm[1] for lm in lm_list]
        y_list = [lm[2] for lm in lm_list]
        
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)
        
        # Add offset, making sure we don't go outside image boundaries
        h, w, _ = frame.shape
        y1, y2 = max(0, y_min - offset), min(h, y_max + offset)
        x1, x2 = max(0, x_min - offset), min(w, x_max + offset)
        
        # Crop from the clean frame without any drawings
        roi = frame_clean[y1:y2, x1:x2]
        
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            hand_crop = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
            
            # Draw tracking box on the display frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow("Hand Crop", hand_crop)

    # Display current class info
    if current_label is not None:
        cv2.putText(frame, f"Current Class: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Saved: {image_counts[current_label]}/{MAX_IMAGES_PER_CLASS}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Select class: 0=palm 1=fist 2=peace 3=thumbs_up 4=ok", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    if len(lm_list) == 0:
        cv2.putText(frame, "No hand detected! Bring hand into view.", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------------------------------
    # CLASS SELECTION
    # -------------------------------
    try:
        pressed_key = chr(key)
    except:
        pressed_key = ""

    if pressed_key in GESTURES:
        current_label = GESTURES[pressed_key]
        print(f"Selected class: {current_label}")

    # -------------------------------
    # SAVE IMAGE
    # -------------------------------
    elif key == ord('s'):
        if current_label is None:
            print("Please press a number key (0-4) to select a class first.")
        elif image_counts[current_label] >= MAX_IMAGES_PER_CLASS:
            print(f"Reached max images for class: {current_label}")
        elif hand_crop is None:
            print("No hand detected. Cannot save image!")
        else:
            save_dir = os.path.join(SAVE_PATH, current_label)
            filename = f"{current_label}_{image_counts[current_label]:04d}.jpg"
            filepath = os.path.join(save_dir, filename)
            cv2.imwrite(filepath, hand_crop)
            image_counts[current_label] += 1
            print(f"Saved: {filepath}")

    # -------------------------------
    # QUIT
    # -------------------------------
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Dataset collection finished.")
