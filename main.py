import cv2
import time
from hand_tracking import HandDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detection_confidence=0.7)
    
    p_time = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to access the webcam. Please ensure it's connected and accessible.")
            break
            
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, draw=False)
        
        if len(lm_list) != 0:
            fingers = detector.fingers_up()
            total_fingers = fingers.count(1)
            
            # Display the number of fingers
            cv2.rectangle(img, (20, 20), (150, 100), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(total_fingers), (45, 85), cv2.FONT_HERSHEY_PLAIN,
                        5, (255, 0, 0), 5)
        
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (10, 140), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        
        cv2.imshow("Hand Gesture Recognition", img)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
