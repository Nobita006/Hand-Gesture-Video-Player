---

# **Hand Gesture Controlled Video Player**

## **Project Overview**

This project demonstrates a hand gesture-controlled video player using a webcam. The system uses hand gestures to perform various video control actions, such as play/pause, adjusting volume, fast forwarding, and rewinding. The gestures are recognized using Mediapipe's hand tracking module, and corresponding actions are triggered by simulating key presses using PyAutoGUI.

## Demonstration - https://youtu.be/1MytSf-vbFE


https://github.com/user-attachments/assets/f5178045-24f9-4ad4-b994-922066390aa3


## **Functionality**

### **1. Open Palm Gesture**
- **Function**: Plays or pauses the video.
- **Detection**: The system detects when the hand is open, with all fingers extended.
- **Trigger**: The spacebar is pressed to toggle play/pause.

### **2. Closed Fist Gesture**
- **Function**: Adjusts the volume based on fist movement.
- **Detection**: The system detects when the hand is closed into a fist.
- **Trigger**:
  - **Fist Moving Left**: Decreases the volume.
  - **Fist Moving Right**: Increases the volume.
  - The volume keys are simulated to control the volume.

### **3. Index Finger Pointing Up**
- **Function**: Fast forwards the video.
- **Detection**: The system detects when the index finger is extended upward while all other fingers are closed.
- **Trigger**: The right arrow key is pressed to skip forward a few seconds.

### **4. Thumb Pointing Down**
- **Function**: Rewinds the video.
- **Detection**: The system detects when the thumb is pointing downward while all other fingers are extended.
- **Trigger**: The left arrow key is pressed to rewind a few seconds.

## **Challenges and Solutions**

### **1. Gesture Differentiation**
**Challenge**: 
One of the main challenges was accurately differentiating between similar hand gestures, such as distinguishing between an open palm and a thumbs-up or a fist and index finger pointing up.

**Solution**:
- **Gesture Specific Landmarks**: We overcame this by fine-tuning the conditions that define each gesture. For instance, an open palm was identified by ensuring all fingers were extended and significantly higher than their respective knuckles. Similarly, a thumbs-up was identified by ensuring the thumb was higher than the other fingers, while all other fingers were curled.
- **Smoothing Function**: A smoothing function was implemented to average the positions of hand landmarks over a few frames, reducing noise and making gesture recognition more reliable.

### **2. Real-Time Performance**
**Challenge**:
Ensuring that gesture recognition happened in real-time without significant delays was crucial for a good user experience.

**Solution**:
- **Frame Rate Control**: We limited the frame rate to ensure the system processed frames efficiently, balancing responsiveness with processing load.
- **Efficient Landmark Processing**: We processed only the necessary hand landmarks to reduce computational overhead, allowing the system to maintain real-time performance.

### **3. False Positives in Gesture Detection**
**Challenge**:
Initially, the system sometimes triggered the wrong action due to false positives, such as detecting a thumbs-up gesture when only a fist was shown.

**Solution**:
- **Gesture Delay Management**: We introduced delays between gesture recognitions to prevent multiple actions from being triggered by a single gesture.
- **Enhanced Conditional Checks**: We refined the conditions used to detect gestures, such as ensuring that the thumb's position relative to other fingers was checked to avoid confusion between gestures.

## **Conclusion**

This project demonstrates the potential of using computer vision and machine learning techniques for intuitive, hands-free control of media applications. By carefully addressing the challenges of gesture differentiation, real-time performance, and false positives, the system provides a robust and user-friendly interface for controlling video playback using hand gestures.

This approach can be extended to control other applications or adapted for accessibility solutions, where touch-free control is essential.

---
