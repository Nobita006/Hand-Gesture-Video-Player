import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
import threading
import math

# --- Configuration & Sensitivity Settings ---
# Screen settings: Get the actual size of your monitor
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Camera Resolution: Lower resolution (640x480) is faster and sufficient for gestures
CAM_WIDTH, CAM_HEIGHT = 640, 480

# Sensitivity Factors
MOUSE_SENSITIVITY = 1.5   # Multiplier for mouse speed (Higher = Faster)
SCROLL_SENSITIVITY = 50   # Multiplier for scrolling speed
FRAME_MARGIN = 100        # Creates a "dead zone" at the edges of the camera view. 
                          # Moving your hand to the edge of this box = edge of your screen.

# Thresholds to prevent accidental triggers
GESTURE_COOLDOWN = 0.5    # Seconds to wait between discrete actions (like clicks) to prevent double-clicks
MOVE_THRESHOLD = 0.002    # Minimum movement required to register an action (reduces jitter/shaking)
ZOOM_THRESHOLD = 0.2      # Distance between thumb and index to detect a "Pinch"

class HandGestureController:
    def __init__(self, root):
        """
        Initialize the main GUI window and state variables.
        """
        self.root = root
        self.root.title("Gesture Control Center")
        self.root.geometry("400x280")
        self.root.resizable(False, False)
        
        # --- State Management Flags ---
        self.is_running = False  # Is the camera currently active?
        self.thread = None       # Placeholder for the background thread

        # Build the User Interface
        self.setup_ui()

    def setup_ui(self):
        """
        Creates the buttons and labels for the application window.
        """
        # Title
        title_label = tk.Label(self.root, text="Hand Gesture Controller", font=("Segoe UI", 16, "bold"))
        title_label.pack(pady=20)

        # Instructions
        instructions = (
            "Instructions:\n"
            "• FIVE (Open Hand): Move Mouse\n"
            "• FIST (Closed Hand): Left Click\n"
            "• PEACE Sign: Scroll (Up/Down) or Seek (Left/Right)\n"
            "• INDEX Finger: Volume (Up/Down)\n"
            "• PINCH (Index+Thumb): Zoom In/Out"
        )
        info_label = tk.Label(self.root, text=instructions, font=("Segoe UI", 9), justify="left")
        info_label.pack(pady=5)

        # Buttons Container
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)

        # Start Button
        self.btn_start = tk.Button(btn_frame, text="Start Tracker", bg="#4CAF50", fg="white", 
                                   font=("Segoe UI", 11, "bold"), width=15, command=self.start_tracking)
        self.btn_start.grid(row=0, column=0, padx=10)

        # Stop Button (Disabled initially)
        self.btn_stop = tk.Button(btn_frame, text="Stop / Exit", bg="#f44336", fg="white", 
                                  font=("Segoe UI", 11, "bold"), width=15, command=self.stop_tracking, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=1, padx=10)

        # Status Bar
        self.status_label = tk.Label(self.root, text="Status: Ready", fg="gray", font=("Segoe UI", 9))
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def start_tracking(self):
        """
        Minimizes the window and starts the Computer Vision logic in a separate thread.
        We use a thread so the GUI doesn't freeze while the camera is running.
        """
        if not self.is_running:
            self.is_running = True
            self.status_label.config(text="Status: Camera Running...", fg="green")
            
            # Update Button States
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            
            # Minimize the main settings window to get it out of the way
            self.root.iconify()

            # Start the heavy lifting in a background thread
            self.thread = threading.Thread(target=self.run_cv_logic)
            self.thread.daemon = True  # Daemon threads close automatically if the app crashes
            self.thread.start()

    def stop_tracking(self):
        """
        Signals the thread to stop and restores the UI.
        """
        if self.is_running:
            self.is_running = False
            self.status_label.config(text="Status: Stopping...", fg="orange")
            # The run_cv_logic loop checks 'self.is_running', so it will break naturally.

    def run_cv_logic(self):
        """
        The main logic loop. This contains all the OpenCV and MediaPipe code.
        """
        try:
            # --- Initialize MediaPipe ---
            mp_hands = mp.solutions.hands
            # max_num_hands=1 ensures we don't get confused by two hands
            hands = mp_hands.Hands(
                max_num_hands=1, 
                min_detection_confidence=0.8, 
                min_tracking_confidence=0.8
            )
            mp_draw = mp.solutions.drawing_utils

            # --- Initialize Camera ---
            cap = cv2.VideoCapture(0)
            cap.set(3, CAM_WIDTH)
            cap.set(4, CAM_HEIGHT)

            # --- Motion Smoothing Variables ---
            # We store previous positions to calculate smooth movement (interpolation)
            prev_location_x, prev_location_y = 0, 0
            curr_location_x, curr_location_y = 0, 0
            smoothing_factor = 5 # Higher number = smoother but slightly more "laggy" feel

            # --- Gesture State Variables ---
            last_action_time = 0
            prev_index_y = None       # For tracking volume gestures
            prev_peace_x = None       # For tracking fast forward/rewind
            prev_peace_y = None       # For tracking scrolling
            prev_pinch_dist = None    # For tracking zoom

            print("Camera Thread Started.")

            while self.is_running and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    print("Camera not found.")
                    break

                # Flip frame horizontally for a "mirror" effect (more natural interaction)
                frame = cv2.flip(frame, 1)
                img_h, img_w, _ = frame.shape
                
                # MediaPipe requires RGB images
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                action_text = "Scanning..." # Default status text

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw the skeleton on the hand
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = hand_landmarks.landmark
                        
                        # --- GESTURE DETECTION ---
                        # Get specific landmarks for fingertips
                        index_tip = landmarks[8]
                        middle_tip = landmarks[12]
                        ring_tip = landmarks[16]
                        pinky_tip = landmarks[20]
                        thumb_tip = landmarks[4]

                        # Check which fingers are extended (Up)
                        # Note: In OpenCV 'y' coordinates, 0 is top, 1 is bottom. 
                        # So, tip.y < joint.y means the finger is UP.
                        index_up = index_tip.y < landmarks[6].y
                        middle_up = middle_tip.y < landmarks[10].y
                        ring_up = ring_tip.y < landmarks[14].y
                        pinky_up = pinky_tip.y < landmarks[18].y
                        
                        # Determine the Gesture Mode
                        gesture = "UNKNOWN"
                        if not index_up and not middle_up and not ring_up and not pinky_up:
                            gesture = "FIST"
                        elif index_up and middle_up and ring_up and pinky_up:
                            gesture = "FIVE"
                        elif index_up and middle_up and not ring_up and not pinky_up:
                            gesture = "PEACE"
                        elif index_up and not middle_up and not ring_up and not pinky_up:
                            gesture = "INDEX"

                        # --- MODE 1: MOUSE MOVE (Hand Open / FIVE) ---
                        if gesture == "FIVE":
                            action_text = "MOUSE MOVE"
                            
                            # 1. Get raw coordinates of index finger
                            x1 = index_tip.x * img_w
                            y1 = index_tip.y * img_h
                            
                            # 2. Map coordinates from Camera box to Screen size
                            # np.interp allows us to map a range (e.g. 100 to 540) to another range (0 to 1920)
                            x3 = np.interp(x1, (FRAME_MARGIN, img_w - FRAME_MARGIN), (0, SCREEN_WIDTH))
                            y3 = np.interp(y1, (FRAME_MARGIN, img_h - FRAME_MARGIN), (0, SCREEN_HEIGHT))
                            
                            # 3. Smooth the values to reduce jitter
                            curr_location_x = prev_location_x + (x3 - prev_location_x) / smoothing_factor
                            curr_location_y = prev_location_y + (y3 - prev_location_y) / smoothing_factor
                            
                            # 4. Move Mouse
                            pyautogui.moveTo(curr_location_x, curr_location_y)
                            
                            # Update previous location
                            prev_location_x, prev_location_y = curr_location_x, curr_location_y
                            
                            # Reset other gesture states so they don't trigger accidentally
                            prev_index_y, prev_peace_x, prev_peace_y, prev_pinch_dist = None, None, None, None

                        # --- MODE 2: CLICK (Closed Fist) ---
                        elif gesture == "FIST":
                            # Use cooldown to prevent rapid-fire clicking
                            if time.time() - last_action_time > GESTURE_COOLDOWN:
                                action_text = "CLICK"
                                pyautogui.click()
                                last_action_time = time.time()
                            else:
                                action_text = "Wait..."

                        # --- MODE 3: SCROLL & SEEK (Peace Sign) ---
                        elif gesture == "PEACE":
                            # Calculate the center point between Index and Middle finger
                            current_peace_x = (landmarks[8].x + landmarks[12].x) / 2
                            current_peace_y = (landmarks[8].y + landmarks[12].y) / 2
                            
                            if prev_peace_x is not None:
                                dx = current_peace_x - prev_peace_x
                                dy = current_peace_y - prev_peace_y
                                
                                # Check if movement is mostly Vertical (Scroll) or Horizontal (Seek)
                                if abs(dy) > abs(dx) and abs(dy) > MOVE_THRESHOLD:
                                    # Vertical Movement
                                    action_text = "SCROLL"
                                    # Multiply by -100 because scroll direction is often inverted
                                    pyautogui.scroll(int(dy * SCROLL_SENSITIVITY * -100))
                                    
                                elif abs(dx) > abs(dy) and abs(dx) > MOVE_THRESHOLD:
                                    # Horizontal Movement (Wait for cooldown so we don't skip too much video)
                                    if time.time() - last_action_time > 0.5:
                                        if dx > 0: 
                                            action_text = "FF >>"
                                            pyautogui.press('right')
                                        else: 
                                            action_text = "<< RW"
                                            pyautogui.press('left')
                                        last_action_time = time.time()
                                        
                            # Update tracking variables
                            prev_peace_x, prev_peace_y = current_peace_x, current_peace_y
                            prev_index_y, prev_pinch_dist = None, None

                        # --- MODE 4: ZOOM & VOLUME (Index Finger) ---
                        elif gesture == "INDEX":
                            # Calculate distance between Thumb and Index tip for pinching
                            dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
                            
                            # Is it a Pinch? (Distance is small)
                            if prev_pinch_dist is not None and dist < ZOOM_THRESHOLD:
                                delta_dist = dist - prev_pinch_dist
                                if abs(delta_dist) > 0.01: # Small threshold to ignore tiny movements
                                    if delta_dist > 0:
                                        action_text = "ZOOM IN"
                                        pyautogui.hotkey('ctrl', '+')
                                    else:
                                        action_text = "ZOOM OUT"
                                        pyautogui.hotkey('ctrl', '-')
                                    # Optional: Add small sleep here if zoom is too fast
                                prev_pinch_dist = dist
                                
                            # If not pinching, it's VOLUME (Distance is large)
                            elif dist >= ZOOM_THRESHOLD:
                                if prev_index_y is not None:
                                    dy = landmarks[8].y - prev_index_y
                                    if abs(dy) > MOVE_THRESHOLD:
                                        if dy < 0:
                                            action_text = "VOL UP"
                                            pyautogui.press('volumeup')
                                        else:
                                            action_text = "VOL DOWN"
                                            pyautogui.press('volumedown')
                                prev_index_y = landmarks[8].y
                                prev_pinch_dist = None # Reset pinch tracking
                            else:
                                prev_pinch_dist = dist

                # --- Draw Feedback on Camera Window ---
                # Draw a black box at the top for text
                cv2.rectangle(frame, (0,0), (640, 60), (0, 0, 0), -1)
                # Draw the status text (e.g., "MOUSE MOVE")
                cv2.putText(frame, f"Mode: {action_text}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the window
                cv2.imshow("Gesture Camera View", frame)
                
                # Allow exiting by pressing 'Q' on the camera window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_tracking()

            # --- Cleanup Phase ---
            cap.release()
            cv2.destroyAllWindows()
            self.reset_ui()

        except Exception as e:
            # If anything crashes (like MediaPipe missing), show a nice error box
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.stop_tracking()

    def reset_ui(self):
        """
        Reset buttons and bring the main window back up.
        """
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped", fg="red")
        self.root.deiconify() # Restore the main window

# --- Main Application Entry Point ---
if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureController(root)
    root.mainloop()