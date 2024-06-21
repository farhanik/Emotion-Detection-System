import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('fer2013_mini_XCEPTION.110-0.65.hdf5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
   
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]

        # Resize and normalize face ROI for model input
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = np.expand_dims(np.expand_dims(face_roi, -1), 0) / 255.0

        # Predict facial expression
        predictions = model.predict(face_roi)
        emotion_label = emotion_labels[np.argmax(predictions)]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Facial Expression Recognition', frame)

    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
