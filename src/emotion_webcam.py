import cv2
import numpy as np
from tensorflow.keras import models
from mtcnn.mtcnn import MTCNN

# Load the model
trained_model = models.load_model('../trained_models/trained_vggface.h5', compile=False)
cv2.ocl.setUseOpenCL(False)

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}
detector = MTCNN()

def draw_radar_plot(frame, prediction, center=(200, 250), size=100):
    num_vars = len(prediction)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    pred = np.concatenate((prediction, prediction[:1]))

    # Draw each axis line and labels
    for i, angle in enumerate(angles[:-1]):
        x_axis = int(center[0] + size * np.cos(angle))
        y_axis = int(center[1] + size * np.sin(angle))
        cv2.line(frame, center, (x_axis, y_axis), (255, 255, 255), 1)

        # Adding labels
        label_x = int(center[0] + (size + 10) * np.cos(angle))
        label_y = int(center[1] + (size + 10) * np.sin(angle))
        cv2.putText(frame, list(emotion_dict.values())[i], (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw the polygon
    pts = []
    for i in range(num_vars):
        x = int(center[0] + pred[i] * size * np.cos(angles[i]))
        y = int(center[1] + pred[i] * size * np.sin(angles[i]))
        pts.append([x, y])
    pts = np.array([pts], np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.fillPoly(frame, [pts], color=(255, 0, 0, 0.1))

    # Draw circle border
    cv2.circle(frame, center, size, (255, 255, 255), 1)

    return frame

def process_frame(frame):
    results = detector.detect_faces(frame)
    if results:
        x1, y1, width, height = results[0]['box']
        face = frame[y1:y1+height, x1:x1+width]
        cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (255, 0, 0), 2)
        cropped_img = cv2.resize(face, (96, 96)).astype(float)
        prediction = trained_model.predict(np.expand_dims(cropped_img, axis=0))
        return prediction[0], (x1, y1)
    return np.zeros(len(emotion_dict)), None

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    prediction, (x1, y1) = process_frame(frame)
    maxindex = int(np.argmax(prediction))
    accuracy = prediction[maxindex] * 100
    emotion_text = f"{emotion_dict[maxindex]}: {accuracy:.2f}%"
    cv2.putText(frame, emotion_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    frame = draw_radar_plot(frame, prediction)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
