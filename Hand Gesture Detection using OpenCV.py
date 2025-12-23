import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]

# Function to determine finger states (up or down)
def get_finger_states(hand_landmarks):
    fingers = []
    # Thumb: Check if thumb tip is to the left of its base (for right hand, flipped image)
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers: Check if finger tip is above its base
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Function to check thumb orientation for Thumbs Down
def is_thumbs_down(hand_landmarks):
    wrist = hand_landmarks.landmark[0]
    thumb_tip = hand_landmarks.landmark[tip_ids[0]]
    # Calculate angle between wrist and thumb tip
    angle = math.degrees(math.atan2(thumb_tip.y - wrist.y, thumb_tip.x - wrist.x))
    # Thumbs Down typically has a downward tilt (angle > 45 degrees)
    return angle > 45

# Function to detect specific gestures
def detect_gesture(fingers, hand_landmarks, w, h):
    # Hi: All fingers up (open hand)
    if fingers == [1, 1, 1, 1, 1]:
        return "Hi"
    # Thumbs Up: Thumb up, other fingers down, not tilted downward
    elif fingers == [1, 0, 0, 0, 0] and not is_thumbs_down(hand_landmarks):
        return "Thumbs Up"
    # Thumbs Down: Thumb up, other fingers down, tilted downward
    elif fingers == [1, 0, 0, 0, 0] and is_thumbs_down(hand_landmarks):
        return "Thumbs Down"
    # Live Long: Index, Middle, and Thumb up, others down
    elif fingers == [1, 1, 1, 0, 0]:
        return "Live Long"
    # Victory: Index and Middle up, others down
    elif fingers == [0, 1, 1, 0, 0]:
        return "Victory"
    # Peace: Ring and Pinky up, others down
    elif fingers == [0, 0, 0, 1, 1]:
        return "Peace"
    # Point Up: Index up, others down
    elif fingers == [0, 1, 0, 0, 0]:
        return "Point Up"
    # Fist: All fingers down
    elif fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 0, 0, 0, 1]:
        return "Maara Jayega..."
    elif fingers == [1, 1, 0, 0, 1]:
        return "Nai"
    elif fingers == [0, 0, 0, 0, 1]:
        return "Nature's Call !!!"
    return ""

# Initialize video capture with debugging
print("Attempting to open webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for Windows; remove or change for macOS/Linux
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error: Could not open webcam with index 0. Trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam with index 1. Please check webcam connection or try another index.")
        exit()

# Log webcam properties
print("Webcam opened successfully.")
print("Resolution:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Backend:", cap.get(cv2.CAP_PROP_BACKEND))

last_action_time = time.time()

while True:
    print("Capturing frame...")
    success, img = cap.read()
    if not success or img is None:
        print("Error: Failed to capture image from webcam.")
        continue
    print("Frame captured successfully.")
    img = cv2.flip(img, 1)  # Flip horizontally for mirror effect
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture_text = ""

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        h, w, _ = img.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        # Calculate bounding box
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
        # Draw rectangle around hand
        cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)

        # Detect fingers and gesture
        fingers = get_finger_states(hand_landmarks)
        gesture_text = detect_gesture(fingers, hand_landmarks, w, h)

        # Display gesture text with a semi-transparent background box
        if gesture_text:
            # Define text properties
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.2
            font_thickness = 3
            text_color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background
            # Get text size
            (text_w, text_h), _ = cv2.getTextSize(gesture_text, font, font_scale, font_thickness)
            # Center the text above the rectangle
            text_x = x_min + (x_max - x_min - text_w) // 2
            text_y = y_min - 30
            # Define background rectangle coordinates
            bg_x_min = text_x - 10
            bg_y_min = text_y - text_h - 10
            bg_x_max = text_x + text_w + 10
            bg_y_max = text_y + 5
            # Create a semi-transparent overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (bg_x_min, bg_y_min), (bg_x_max, bg_y_max), bg_color, -1)
            alpha = 0.6  # Transparency factor
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            # Draw text
            cv2.putText(img, gesture_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # Display the frame
    print("Displaying frame...")
    cv2.imshow("Hand Gesture Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam released and windows closed.")  