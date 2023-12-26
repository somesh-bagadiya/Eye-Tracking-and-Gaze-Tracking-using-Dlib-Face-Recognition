import cv2
import dlib
import face_recognition
import numpy as np
import pyautogui


def eye_aspect_ratio(eye_landmarks):
    # Extract (x, y) coordinates from eye landmarks
    left_eye_x = [eye_landmarks[i][0] for i in range(6)]
    left_eye_y = [eye_landmarks[i][1] for i in range(6)]
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(np.array([left_eye_x[1] - left_eye_x[5], left_eye_y[1] - left_eye_y[5]]))
    B = np.linalg.norm(np.array([left_eye_x[2] - left_eye_x[4], left_eye_y[2] - left_eye_y[4]]))
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(np.array([left_eye_x[0] - left_eye_x[3], left_eye_y[0] - left_eye_y[3]]))
    # Compute the aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear


# Load the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Open a video capture object
cap = cv2.VideoCapture(0)

# Set the desired width and height for resizing
new_width = 480
new_height = 360

# Set the screen width and height (adjust based on your screen resolution)
screen_width, screen_height = pyautogui.size()

ear_threshold = 0.25

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (new_width, new_height))

    # Detect faces in the frame
    faces = detector(frame_resized)

    for face in faces:
        landmarks = predictor(frame_resized, face)
        landmarks_np = face_recognition.face_landmarks(frame_resized)

        # Extract coordinates of the left and right eyes
        left_eye = landmarks_np[0]["left_eye"]
        right_eye = landmarks_np[0]["right_eye"]

        # Calculate the center of each eye
        left_eye_center = np.mean(left_eye, axis=0).astype("int")
        right_eye_center = np.mean(right_eye, axis=0).astype("int")

        # Draw circles around the eyes
        cv2.circle(frame_resized, tuple(left_eye_center), 5, (0, 255, 0), -1)
        cv2.circle(frame_resized, tuple(right_eye_center), 5, (0, 255, 0), -1)

        # Calculate the midpoint between the eyes
        eyes_midpoint = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

        # Draw a circle at the midpoint
        cv2.circle(frame_resized, eyes_midpoint, 5, (0, 0, 255), -1)

        # Normalize the gaze coordinates to the screen size
        gaze_x = int((eyes_midpoint[0] / new_width) * screen_width)
        gaze_y = int((eyes_midpoint[1] / new_height) * screen_height)

        left_eye_ear = eye_aspect_ratio(landmarks_np[0]["left_eye"])

        if left_eye_ear < ear_threshold:
            cv2.putText(frame, "Left Eye Click", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            pyautogui.click()
            print("click")
            # Control the mouse position
        pyautogui.moveTo(gaze_x, gaze_y)

    cv2.imshow('Gaze Tracking and Mouse Control', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
