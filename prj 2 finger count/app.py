import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(r'D:\luminar\DL Projects\dlprj\prj 2 finger count\prj2model.h5')
input_size=(200,200)
recent_predictions = []
max_frames = 10

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, input_size)  # Resize image to 200x200
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (200, 200, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, 200, 200, 1)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image

video=cv2.VideoCapture(0)
while True:
    s,img=video.read()
    img1=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    process_img=preprocess_image(img1)
    prediction=model.predict(process_img)
    confidence = np.max(prediction)
    p = prediction.argmax(axis=-1)[0]

    if confidence > 0.7:  # Use a confidence threshold
        recent_predictions.append(p)
        if len(recent_predictions) > max_frames:
            recent_predictions.pop(0)

        final_prediction = max(set(recent_predictions), key=recent_predictions.count)
        g = str(final_prediction)
        if g=='6':
            g = "Uncertain -cannot identify"
    else:
        g = "Uncertain-adjust your hand"
    cv2.putText(img,g,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    cv2.imshow('FINGER COUNT',img)
    if cv2.waitKey(1) & 0XFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows()