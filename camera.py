import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Webcam accessed successfully.")
else:
    print("Failed to access webcam.")
cap.release()
