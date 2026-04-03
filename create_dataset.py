import cv2
import os
import time
from hand_tracking import HandDetector

def main():
    print("=== Hand Gesture Dataset Creator ===")
    gesture_name = input("Enter the name of the gesture you want to capture (e.g., 'thumbs_up'): ")
    save_path = os.path.join("dataset", gesture_name)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    else:
        print(f"Directory {save_path} already exists. Images will be added here.")
    
    cap = cv2.VideoCapture(0)
    # We only need 1 hand for standard gesture datasets usually
    detector = HandDetector(max_hands=1, detection_confidence=0.7)
    
    offset = 40 # Padding around the hand for the crop
    
    count = 0 
    
    print("\nInstructions:")
    print(" - Press 's' to manually save an image of your hand.")
    print(" - Press 'q' to quit.")
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        img_output = img.copy()
        img = detector.find_hands(img, draw=False) # We don't want landmarks drawn on the data images
        lm_list = detector.find_position(img, draw=False)
        
        if len(lm_list) != 0:
            # Finding the bounding box coordinates
            x_list = [lm[1] for lm in lm_list]
            y_list = [lm[2] for lm in lm_list]
            
            x_min, x_max = min(x_list), max(x_list)
            y_min, y_max = min(y_list), max(y_list)
            
            # Crop the hand image safely
            h, w, c = img.shape
            y1, y2 = max(0, y_min - offset), min(h, y_max + offset)
            x1, x2 = max(0, x_min - offset), min(w, x_max + offset)
            
            # Crop the hand image cleanly BEFORE drawing the bounding box
            img_crop = img_output[y1:y2, x1:x2].copy()
            
            # Draw bounding box on the output window so user knows what's being cropped
            cv2.rectangle(img_output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Check if crop size is valid before displaying or saving
            if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
                cv2.imshow("Cropped Hand Preview", img_crop)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    count += 1
                    # Save the raw crop without landmarks written on it
                    file_name = f'{save_path}/Image_{time.time()}_{count}.jpg'
                    cv2.imwrite(file_name, img_crop)
                    print(f"Saved {file_name} (Total for {gesture_name}: {count})")
        
        cv2.imshow("Dataset Creator", img_output)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
