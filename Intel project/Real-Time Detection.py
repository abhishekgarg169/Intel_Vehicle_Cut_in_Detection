import cv2
import numpy as np

def preprocess_image_for_inference(image):
    resized_image = cv2.resize(image, (224, 224))
    normalized_image = resized_image / 255.0
    return normalized_image

# Load trained model
model = tf.keras.models.load_model('vehicle_cutin_detection_model.h5')

# Initialize video capture
cap = cv2.VideoCapture('path/to/video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    preprocessed_frame = preprocess_image_for_inference(frame)
    
    # Predict cut-in event
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    
    # Display result
    if prediction > 0.5:
        cv2.putText(frame, 'Cut-in Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
