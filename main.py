import cv2
import numpy as np
import time
import os
from datetime import datetime
from deepface import DeepFace  # Import DeepFace for emotion detection

# Cooldown timer
last_captured_time = 0
cooldown_seconds = 10
output_folder = "captured_faces"

# Create folder to save captured images
os.makedirs(output_folder, exist_ok=True)

# Open or create the emotions.txt file to write emotions
emotions_file = os.path.join(output_folder, "emotions.txt")
if not os.path.exists(emotions_file):
    with open(emotions_file, 'w') as f:
        f.write("Face Emotion Records:\n")  # Write header to the file

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    current_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Use DeepFace to detect emotion in the frame
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        score = analysis[0]['emotion'][emotion]
    except Exception as e:
        print(f"Error during emotion prediction: {e}")
        emotion = "Unknown"
        score = 0

    # If faces are detected and cooldown has passed
    if emotion and current_time - last_captured_time >= cooldown_seconds:
        # Get face bbox (using OpenCV's default face detection)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Crop the face tightly from the frame
            face_image = frame[y:y + h, x:x + w]

            # Resize the cropped face image to create a thumbnail (e.g., 100x100)
            thumbnail_size = (100, 100)
            face_thumbnail = cv2.resize(face_image, thumbnail_size)

            # Create the image file name with emotion
            image_filename = f"face_thumbnail_{timestamp}_{emotion}.jpg"
            thumbnail_path = os.path.join(output_folder, image_filename)

            # Save the thumbnail
            cv2.imwrite(thumbnail_path, face_thumbnail)
            print(f"Face captured and saved as thumbnail at {thumbnail_path}")

            # Write the emotion and timestamp to emotions.txt
            with open(emotions_file, 'a') as f:
                f.write(f"{timestamp} - {emotion} (Confidence: {score})\n")

            # Update the cooldown timer after saving the image and writing the emotion
            last_captured_time = current_time

    # Display the webcam feed
    cv2.imshow("Webcam", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()