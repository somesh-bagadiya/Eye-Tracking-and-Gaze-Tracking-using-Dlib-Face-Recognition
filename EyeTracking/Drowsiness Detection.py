import cv2
import face_recognition
import numpy as np
import time
import winsound


def countdown(t):
    while t:
        mins, secs = divmod(t, 60)
        timer = '{:02d}:{:02d}'.format(mins, secs)
        print(timer, end="\r")
        time.sleep(1)
        t -= 1


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


cap = cv2.VideoCapture(0)

new_width = 480
new_height = 360
ear_threshold = 0.19
closed_flag = False

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (new_width, new_height))

    # Find face locations and landmarks in the resized frame
    face_locations = face_recognition.face_locations(frame)
    face_landmarks_list = face_recognition.face_landmarks(frame)

    # Create separate frames for left and right eyes
    left_eye_frame = np.zeros_like(frame)
    right_eye_frame = np.zeros_like(frame)

    # Draw rectangles around the detected faces and boxes around the eyes
    for (top, right, bottom, left), face_landmarks in zip(face_locations, face_landmarks_list):
        # Scale back the face location to the original resolution
        top = int(top * (new_height / frame.shape[0]))
        right = int(right * (new_width / frame.shape[1]))
        bottom = int(bottom * (new_height / frame.shape[0]))
        left = int(left * (new_width / frame.shape[1]))

        # Draw face rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

        landmarks_np = face_recognition.face_landmarks(frame)

        left_eye = landmarks_np[0]["left_eye"]
        right_eye = landmarks_np[0]["right_eye"]

        # Calculate the center of each eye
        left_eye_center = np.mean(left_eye, axis=0).astype("int")
        right_eye_center = np.mean(right_eye, axis=0).astype("int")

        # Draw circles around the eyes
        cv2.circle(frame, tuple(left_eye_center), 5, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_eye_center), 5, (0, 255, 0), -1)

        # Calculate the midpoint between the eyes
        eyes_midpoint = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

        # Draw a circle at the midpoint
        cv2.circle(frame, eyes_midpoint, 5, (0, 0, 255), -1)

        left_eye_x_start = int(left + 0.175 * (right - left))
        left_eye_x_end = int(left + 0.375 * (right - left))
        left_eye_y_start = int(top + 0.175 * (bottom - top))
        left_eye_y_end = int(top + 0.375 * (bottom - top))

        right_eye_x_start = int(left + 0.675 * (right - left))
        right_eye_x_end = int(left + 0.875 * (right - left))
        right_eye_y_start = int(top + 0.175 * (bottom - top))
        right_eye_y_end = int(top + 0.375 * (bottom - top))

        # Extract regions of interest for left and right eyes
        left_eye_roi = frame[left_eye_y_start:left_eye_y_end, left_eye_x_start:left_eye_x_end]
        right_eye_roi = frame[right_eye_y_start:right_eye_y_end, right_eye_x_start:right_eye_x_end]

        # Convert eye regions to grayscale
        left_eye_gray = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
        right_eye_gray = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)

        # Draw rectangles around the eyes on the original frame
        cv2.rectangle(frame, (left_eye_x_start, left_eye_y_start), (left_eye_x_end, left_eye_y_end), (255, 0, 0), 1)
        cv2.rectangle(frame, (right_eye_x_start, right_eye_y_start), (right_eye_x_end, right_eye_y_end), (255, 0, 0), 1)

        # Extract regions of interest for left and right eyes
        left_eye_roi = frame[left_eye_y_start: left_eye_y_end, left_eye_x_start: left_eye_x_end]
        right_eye_roi = frame[right_eye_y_start: right_eye_y_end, right_eye_x_start: right_eye_x_end]

        # Convert eye regions to grayscale
        left_eye_frame = cv2.cvtColor(left_eye_roi, cv2.COLOR_BGR2GRAY)
        right_eye_frame = cv2.cvtColor(right_eye_roi, cv2.COLOR_BGR2GRAY)

        # Perform pupil detection (example using HoughCircles)
        left_pupils = cv2.HoughCircles(left_eye_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=10, param2=10, minRadius=5, maxRadius=20)
        right_pupils = cv2.HoughCircles(right_eye_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=10, param2=10, minRadius=5, maxRadius=20)

        # Draw the detected pupils
        if left_pupils is not None:
            left_pupils = np.round(left_pupils[0, :]).astype("int")
            for (x, y, r) in left_pupils:
                cv2.circle(left_eye_frame, (x + left_eye_x_start, y + left_eye_y_start), r, (0, 255, 0), 1)

        if right_pupils is not None:
            right_pupils = np.round(right_pupils[0, :]).astype("int")
            for (x, y, r) in right_pupils:
                cv2.circle(right_eye_frame, (x + right_eye_x_start, y + right_eye_y_start), r, (0, 255, 0), 1)

        left_eye_ear = eye_aspect_ratio(face_landmarks["left_eye"])
        right_eye_ear = eye_aspect_ratio(face_landmarks["right_eye"])

        if left_eye_ear < ear_threshold and right_eye_ear < ear_threshold:
            cv2.putText(frame, "Both Eyes are closed", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            closed_flag = True
            eyes_closed_start_time = time.time()
            print("both")
        elif left_eye_ear < ear_threshold:
            cv2.putText(frame, "Left Eye Closed", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # print("left")
        elif right_eye_ear < ear_threshold:
            cv2.putText(frame, "Right Eye Closed", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            # print("right")

        if closed_flag == True:
            eyes_closed_duration = eyes_closed_duration + time.time() - eyes_closed_start_time
            print(eyes_closed_duration)
            if eyes_closed_duration > 0.3:
                cv2.putText(frame, "Drowsiness Detected", (left, top - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                            1)
                print("Drowsiness Detection")
                duration = 1000  # milliseconds
                freq = 440  # Hz
                winsound.Beep(freq, duration)

        if left_eye_ear > ear_threshold and right_eye_ear > ear_threshold:
            closed_flag = False
            eyes_closed_duration = 0

    # Display the frame with the detected faces
    cv2.imshow('Live Face Detection', frame)

    # Display frames with the detected pupils
    left_eye_frame = cv2.resize(left_eye_frame, (new_width - 267, new_height - 200))
    right_eye_frame = cv2.resize(right_eye_frame, (new_width - 267, new_height - 200))

    cv2.imshow('Left Eye Pupils', left_eye_frame)
    cv2.imshow('Right Eye Pupils', right_eye_frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
