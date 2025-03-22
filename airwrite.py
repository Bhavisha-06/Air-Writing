import os
import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from google.cloud import vision
from google.cloud.vision_v1 import types

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hand writing recognition using Mediapipe and Google Cloud Vision')
    parser.add_argument('--api_key', type=str, required=True, help='Path to Google Cloud API key JSON file')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set the Google Cloud credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.api_key

    # Initialize Mediapipe Hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    # Initialize the drawing path
    points = []

    # Initialize the Mediapipe Hands model
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

    # Initialize Google Cloud Vision client
    client = vision.ImageAnnotatorClient()

    # Threshold angle for detecting writing (finger fully extended)
    WRITING_ANGLE_THRESHOLD = 160  # in degrees

    # Time threshold for detecting pause (5 seconds)
    PAUSE_THRESHOLD = 5  # in seconds

    # Function to calculate the angle between three points (using law of cosines)
    def calculate_angle(a, b, c):
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ac = np.linalg.norm(a - c)
        angle = np.degrees(np.arccos((ab**2 + bc**2 - ac**2) / (2 * ab * bc)))
        return angle

    def recognize_text_from_image(image):
        # Convert the image to a format suitable for Google Vision API
        _, encoded_image = cv2.imencode('.png', image)
        content = encoded_image.tobytes()
        vision_image = types.Image(content=content)

        # Perform text detection using Google Vision API
        response = client.text_detection(image=vision_image)

        # Extract and return only the recognized text
        texts = response.text_annotations
        if texts:
            return texts[0].description.strip()
        return ""

    cap = cv2.VideoCapture(0)
    last_activity_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get coordinates of the index finger joints
                tip_index = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
                pip_index = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y])
                mcp_index = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y])

                # Calculate the angle between the joints of the index finger
                angle = calculate_angle(tip_index, pip_index, mcp_index)

                # Convert coordinates to image space
                x, y = int(tip_index[0] * frame.shape[1]), int(tip_index[1] * frame.shape[0])

                # Consider the fingertip to be writing if the angle is above the threshold
                if angle > WRITING_ANGLE_THRESHOLD:
                    points.append((x, y))
                    last_activity_time = time.time()
                else:
                    # Add a None point to signify pausing the writing
                    points.append(None)
                # Draw the hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Create a blank canvas to draw strokes
        canvas = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(canvas, points[i - 1], points[i], (0, 0, 0), 2)

        # Check for pause and recognize text
        if time.time() - last_activity_time > PAUSE_THRESHOLD:
            if points:
                # Recognize text from the canvas
                text = recognize_text_from_image(canvas)
                print(f'Recognized text: {text}')
                points.clear()  # Clear the drawing path
                last_activity_time = time.time()  # Reset the timer

        # Draw the path on the frame
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            cv2.line(frame, points[i - 1], points[i], (255, 0, 0), 8)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()