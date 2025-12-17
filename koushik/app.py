# import mediapipe
# print("loading mediapipe from:", mediapipe.__file__)
# import mediapipe.python.solutions as solutions # Force check

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# --- DEBUG IMPORT ---
import mediapipe as mp
try:
    from mediapipe.python.solutions import hands as mp_hands_debug
    print("Success: Hands module loaded!")
except ImportError as e:
    print(f"\nCRITICAL ERROR DETAILS: {e}\n")
    raise # Crash the program so we see the full trace
# --------------------

from collections import deque
import math

# --- Configuration ---
# Screen settings
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Sensitivity & Thresholds
MOUSE_SENSITIVITY = 1.5  # Higher = faster mouse
SCROLL_SENSITIVITY = 50
ZOOM_THRESHOLD_DISTANCE = 0.05  # Distance between thumb and index to trigger zoom
GESTURE_COOLDOWN = 0.5 # Seconds to wait between discrete actions (like clicks)
MOVE_THRESHOLD = 0.002 # Ignore tiny movements to reduce jitter

# Margin for mouse control (keeps hand in center of frame)
FRAME_MARGIN = 100 

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.8, 
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# State Management
last_action_time = 0
prev_location_x, prev_location_y = 0, 0
curr_location_x, curr_location_y = 0, 0
smoothing_factor = 5 # Higher = smoother mouse, more lag

# Gesture State Tracking
prev_index_y = None
prev_peace_x = None
prev_peace_y = None
prev_pinch_dist = None

# Initialize Camera
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

def calculate_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_fingers_status(landmarks):
    """
    Returns a list of booleans [Thumb, Index, Middle, Ring, Pinky]
    True = Open/Extended, False = Closed/Curled
    """
    fingers = []
    
    # Thumb (compare x position relative to other joints depending on hand side)
    # Assuming right hand for simplicity, or checking distance to palm center
    # Simpler check: Tip is higher (lower y value) than IP joint is unreliable for thumb.
    # We use x-axis for thumb usually, but let's use a simpler geometry check:
    # Check if thumb tip is far from pinky base
    if landmarks[4].x < landmarks[3].x: # Adjust based on hand laterality if needed
        fingers.append(True)
    else:
        fingers.append(False) # This is a basic check, might need tuning

    # Other 4 fingers (Tip y < PIP y) - Note: Y increases downwards
    fingers.append(landmarks[8].y < landmarks[6].y)  # Index
    fingers.append(landmarks[12].y < landmarks[10].y) # Middle
    fingers.append(landmarks[16].y < landmarks[14].y) # Ring
    fingers.append(landmarks[20].y < landmarks[18].y) # Pinky
    return fingers

def detect_gesture(landmarks):
    """
    Classifies the current hand pose into a string state.
    """
    # Get coordinates of tips
    # thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Check which fingers are up
    # Since thumb logic can vary by rotation, we often rely on the 4 fingers for main gestures
    
    index_up = index_tip.y < landmarks[6].y
    middle_up = middle_tip.y < landmarks[10].y
    ring_up = ring_tip.y < landmarks[14].y
    pinky_up = pinky_tip.y < landmarks[18].y
    
    # 1. FIST: All fingers closed
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "FIST"

    # 2. FIVE: All fingers open
    if index_up and middle_up and ring_up and pinky_up:
        return "FIVE"

    # 3. PEACE: Index and Middle open, Ring and Pinky closed
    if index_up and middle_up and not ring_up and not pinky_up:
        return "PEACE"

    # 4. PINCH / ZOOM: Index and Thumb are the primary actors.
    # To avoid conflict with INDEX, we check if Index and Thumb are clearly interacting
    # or if it's strictly just Index up.
    # Let's define PINCH as: Index Up, others down, but we calculate distance separately.
    # Actually, let's define INDEX ONLY first.
    
    # 5. INDEX: Index up, Middle/Ring/Pinky closed
    if index_up and not middle_up and not ring_up and not pinky_up:
        # If thumb is also "out", it might be a pinch.
        # We handle pinch logic inside the main loop if gesture is INDEX or similar.
        return "INDEX"

    return "UNKNOWN"

print("System Ready. Hand Control Active.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)
    img_h, img_w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    action_text = "Scanning..."
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            landmarks = hand_landmarks.landmark
            gesture = detect_gesture(landmarks)
            
            # --- 1. MOVE MOUSE (FIVE POSE) ---
            if gesture == "FIVE":
                action_text = "MOUSE MOVE"
                
                # Get Index Finger Tip coordinates
                x1 = landmarks[8].x * img_w
                y1 = landmarks[8].y * img_h
                
                # Convert coordinates with margin (interpolation)
                x3 = np.interp(x1, (FRAME_MARGIN, img_w - FRAME_MARGIN), (0, SCREEN_WIDTH))
                y3 = np.interp(y1, (FRAME_MARGIN, img_h - FRAME_MARGIN), (0, SCREEN_HEIGHT))
                
                # Smooth the movement
                curr_location_x = prev_location_x + (x3 - prev_location_x) / smoothing_factor
                curr_location_y = prev_location_y + (y3 - prev_location_y) / smoothing_factor
                
                pyautogui.moveTo(curr_location_x, curr_location_y)
                prev_location_x, prev_location_y = curr_location_x, curr_location_y

                # Reset other states
                prev_index_y = None
                prev_peace_x, prev_peace_y = None, None

            # --- 2. CLICK (FIST POSE) ---
            elif gesture == "FIST":
                if time.time() - last_action_time > GESTURE_COOLDOWN:
                    action_text = "CLICK!"
                    pyautogui.click()
                    last_action_time = time.time()
                else:
                    action_text = "Click Cooldown"

            # --- 3. SCROLL & SEEK (PEACE POSE) ---
            elif gesture == "PEACE":
                # Average position of index and middle finger
                current_peace_x = (landmarks[8].x + landmarks[12].x) / 2
                current_peace_y = (landmarks[8].y + landmarks[12].y) / 2
                
                if prev_peace_x is not None and prev_peace_y is not None:
                    dx = current_peace_x - prev_peace_x
                    dy = current_peace_y - prev_peace_y
                    
                    # Determine dominant axis
                    if abs(dy) > abs(dx) and abs(dy) > MOVE_THRESHOLD:
                        # Vertical Movement -> Scroll
                        action_text = "SCROLL"
                        scroll_amount = int(dy * SCROLL_SENSITIVITY * -100) # -1 to flip direction
                        pyautogui.scroll(scroll_amount)
                        
                    elif abs(dx) > abs(dy) and abs(dx) > MOVE_THRESHOLD:
                        # Horizontal Movement -> Seek
                        if time.time() - last_action_time > 0.5: # Slower cooldown for seek
                            if dx > 0: # Moved Right
                                action_text = "FF >>"
                                pyautogui.press('right')
                            else: # Moved Left
                                action_text = "RW <<"
                                pyautogui.press('left')
                            last_action_time = time.time()
                
                prev_peace_x, prev_peace_y = current_peace_x, current_peace_y
                prev_index_y = None # Reset volume state

            # --- 4. VOLUME & ZOOM (INDEX POSE + PINCH CHECK) ---
            elif gesture == "INDEX":
                # Check for Pinch (Zoom) first
                # Calculate distance between Index Tip (8) and Thumb Tip (4)
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                dist = calculate_distance(thumb_tip, index_tip)
                
                # Check if other fingers are definitely curled (handled by detect_gesture)
                
                # If Thumb is close to Index, treat as PINCH/ZOOM logic
                # We use a relative check. If distance changes significantly.
                if prev_pinch_dist is not None:
                    delta_dist = dist - prev_pinch_dist
                    
                    # If thumb and index are close enough to be considering a pinch interaction
                    if dist < 0.2: 
                        if abs(delta_dist) > 0.01: # Threshold for zoom jitter
                            if delta_dist > 0: # Moving apart
                                action_text = "ZOOM IN (+)"
                                pyautogui.hotkey('ctrl', '+')
                            else: # Moving together
                                action_text = "ZOOM OUT (-)"
                                pyautogui.hotkey('ctrl', '-')
                            # Small sleep to prevent zoom flying too fast
                            # time.sleep(0.1) 
                            
                        prev_pinch_dist = dist
                    else:
                        # If fingers are far apart, assume VOLUME Mode (Index Up Vertical)
                        current_y = landmarks[8].y
                        if prev_index_y is not None:
                            dy = current_y - prev_index_y
                            if abs(dy) > MOVE_THRESHOLD:
                                if dy < 0: # Moved Up (Y decreases going up)
                                    action_text = "VOL UP"
                                    pyautogui.press('volumeup')
                                else:
                                    action_text = "VOL DOWN"
                                    pyautogui.press('volumedown')
                        prev_index_y = current_y
                        prev_pinch_dist = None # Reset pinch
                else:
                    prev_pinch_dist = dist
                    prev_index_y = landmarks[8].y

            else:
                action_text = "No Gesture"
                # Reset tracking variables to avoid jumps when re-entering a gesture
                prev_index_y = None
                prev_peace_x = None
                prev_pinch_dist = None

    # UI Feedback
    cv2.rectangle(frame, (0,0), (640, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"Mode: {action_text}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Hand Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()