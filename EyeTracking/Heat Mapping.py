import cv2
import face_recognition
import numpy as np
import time
import winsound
import copy
import pygame
from matplotlib import pyplot as plt
import autopy
import pyautogui

cap = cv2.VideoCapture(0)

new_width = 480
new_height = 360
ear_threshold = 0.19
closed_flag = False

eye_x_positions = []
eye_y_positions = []

k = 1
sumx = 0
sumy = 0
orn = 2

screen_width, screen_height = pyautogui.size()

print(screen_width, screen_height)

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

        # If face detected fetch Eye Landmarks
        if len(landmarks_np) != 0:
            left_eye = landmarks_np[0]["left_eye"]
            right_eye = landmarks_np[0]["right_eye"]

            # Calculate the center of each eye
            left_eye_center = np.mean(left_eye, axis=0).astype("int")
            right_eye_center = np.mean(right_eye, axis=0).astype("int")

            frame_pupil = frame.copy()

            # Draw circles around the eyes
            cv2.circle(frame_pupil, tuple(left_eye_center), 5, (0, 255, 0), -1)
            cv2.circle(frame_pupil, tuple(right_eye_center), 5, (0, 255, 0), -1)

            # Calculate the midpoint between the eyes
            eyes_midpoint = (
            (left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)

            # Draw a circle at the midpoint
            cv2.circle(frame_pupil, eyes_midpoint, 5, (0, 0, 255), -1)

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

            eye_left_blur = left_eye_gray
            eye_right_blur = right_eye_gray

            eye_left_blur = cv2.resize(eye_left_blur, (new_width, new_height))
            eye_right_blur = cv2.resize(eye_right_blur, (new_width, new_height))

            img_left_blur = cv2.resize(eye_left_blur, (new_width, new_height))
            # img_left_blur = cv2.bilateralFilter(eye_left_blur, 10, 195, 195)
            cv2.imshow("img_left_blur", img_left_blur)

            img_right_blur = cv2.resize(eye_right_blur, (new_width, new_height))
            cv2.imshow("img_right_blur", img_right_blur)

            # Draw rectangles around the eyes on the original frame
            cv2.rectangle(frame, (left_eye_x_start, left_eye_y_start), (left_eye_x_end, left_eye_y_end), (255, 0, 0), 1)
            cv2.rectangle(frame, (right_eye_x_start, right_eye_y_start), (right_eye_x_end, right_eye_y_end),
                          (255, 0, 0), 1)

            # Perform pupil detection (example using HoughCircles)
            left_pupils = cv2.HoughCircles(left_eye_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=10, param2=10,
                                           minRadius=5, maxRadius=20)
            right_pupils = cv2.HoughCircles(right_eye_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=10, param2=10,
                                            minRadius=5, maxRadius=20)

            if left_pupils is not None:
                left_pupils = np.uint16(np.around(left_pupils))
                left_pupils = left_pupils[0][0]
                # x = left_pupils[0] * (1920 / 22)
                # y = left_pupils[1] * (1080 / 22)
                # print(x, y)
                # eye_x_p = x
                # eye_y_p = y
                frame_height, frame_width = left_eye_gray.shape
                eye_movement_x = left_pupils[0]
                eye_movement_y = left_pupils[1]

                # Define the range of eye movements in the frame
                min_eye_movement_x, max_eye_movement_x = 0, frame_width
                min_eye_movement_y, max_eye_movement_y = 0, frame_height

                # Define the screen coordinates (screen resolution)
                min_screen_x, max_screen_x = 0, screen_width
                min_screen_y, max_screen_y = 0, screen_height

                # Map eye movements to screen coordinates
                mapped_screen_x = (eye_movement_x - min_eye_movement_x) / (max_eye_movement_x - min_eye_movement_x) * (
                            max_screen_x - min_screen_x) + min_screen_x
                mapped_screen_y = (eye_movement_y - min_eye_movement_y) / (max_eye_movement_y - min_eye_movement_y) * (
                            max_screen_y - min_screen_y) + min_screen_y

                # Ensure coordinates stay within screen boundaries
                eye_x_p = max(min_screen_x, min(mapped_screen_x, max_screen_x))
                eye_y_p = max(min_screen_y, min(mapped_screen_y, max_screen_y))

                pyautogui.moveTo(eye_x_p, eye_y_p)
                eye_x_positions.append(eye_x_p)
                eye_y_positions.append(eye_y_p)

                if len(eye_x_positions) > 50:
                    print("inside for loop")
                    data_all = list(zip(eye_x_positions, eye_y_positions))
                    print(min(eye_x_positions), max(eye_x_positions))
                    print(min(eye_y_positions), max(eye_y_positions))
                    print(data_all)
                    # plt.axis([0, 480, 270, 0])
                    plt.scatter(eye_x_positions, eye_y_positions, color="blue")
                    plt.title("Eye position")
                    plt.xlabel("X position")
                    plt.ylabel("Y position")
                    plt.show()
                    break

        face_roi = frame[top:bottom, left:right]
        cv2.imshow("Face ROI", face_roi)

    cv2.imshow('Live Face Detection', frame)
    cv2.imshow('Live pupil Detection', frame_pupil)

    # Exit when 'q' is pressed
    if (cv2.waitKey(1) & 0xFF == ord('q')) and len(eye_x_positions) == 10:
        break

data_all = list(zip(eye_x_positions, eye_y_positions))
print(data_all)
plt.scatter(eye_x_positions, eye_y_positions, color="blue")
plt.title("Eye position")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.axis([0, 150, 55, 0])
plt.show()

cap.release()
cv2.destroyAllWindows()
