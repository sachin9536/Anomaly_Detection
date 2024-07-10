# import os
# from flask import Flask, render_template, request
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load the pre-trained model
# model = load_model('model.h5')

# # Define CLASS_LABELS
# CLASS_LABELS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
#                 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting', 
#                 'Shoplifting', 'Stealing', 'Vandalism']

# # Function to preprocess the video frame
# def preprocess_frame(frame):
#     frame = cv2.resize(frame, (64, 64))  # Resize to match model input size
#     frame = frame / 255.0  # Normalize pixel values
#     return frame

# # Route for uploading and processing video
# @app.route('/', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         video_file = request.files['videofile']
#         if video_file:
#             # Save the uploaded video file to a temporary location
#             video_path = 'temp_video.mp4'  # Change this to your desired temporary location
#             video_file.save(video_path)

#             # Read the video file
#             cap = cv2.VideoCapture(video_path)
#             predictions = []

#             # Process each frame of the video
#             anomaly_detected = False
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Preprocess the frame
#                 processed_frame = preprocess_frame(frame)

#                 # Make predictions
#                 prediction = model.predict(np.expand_dims(processed_frame, axis=0))[0]
#                 predicted_class = CLASS_LABELS[np.argmax(prediction)]
#                 predictions.append(predicted_class)

#                 # Check if anomaly is detected
#                 if predicted_class != 'Normal':
#                     anomaly_detected = True
#                     break

#             # Close the video capture
#             cap.release()

#             # Delete the temporary video file
#             os.remove(video_path)

#             if anomaly_detected:
#                 return render_template('index.html', predictions=['Anomaly Detected'])
#             else:
#                 return render_template('index.html', predictions=['Normal'])

#     return render_template('index.html', predictions=None)

# if __name__ == '__main__':
#     app.run(debug=True)


# import os
# from flask import Flask, render_template, request
# import numpy as np
# import cv2
# from tensorflow.keras.models import load_model

# app = Flask(__name__)

# # Load the pre-trained model
# model = load_model('weights2.h5')

# # Define CLASS_LABELS
# CLASS_LABELS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
#                 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting', 
#                 'Shoplifting', 'Stealing', 'Vandalism']

# # Function to preprocess the video frame
# def preprocess_frame(frame):
#     frame = cv2.resize(frame, (64, 64))  # Resize to match model input size
#     frame = frame / 255.0  # Normalize pixel values
#     return frame

# # Route for uploading and processing video
# @app.route('/', methods=['GET', 'POST'])
# def predict():
#     if request.method == 'POST':
#         video_file = request.files['videofile']
#         if video_file:
#             # Save the uploaded video file to a temporary location
#             video_path = 'temp_video.mp4'  # Change this to your desired temporary location
#             video_file.save(video_path)

#             # Read the video file
#             cap = cv2.VideoCapture(video_path)
#             predictions = []

#             # Process each frame of the video
#             anomaly_detected = False
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break

#                 # Preprocess the frame
#                 processed_frame = preprocess_frame(frame)

#                 # Make predictions
#                 prediction = model.predict(np.expand_dims(processed_frame, axis=0))[0]
#                 predicted_class = CLASS_LABELS[np.argmax(prediction)]
#                 predictions.append(predicted_class)

#                 # Check if anomaly is detected
#                 if predicted_class != 'Normal':
#                     anomaly_detected = True
#                     break

#             # Close the video capture
#             cap.release()

#             # Delete the temporary video file
#             os.remove(video_path)

#             if anomaly_detected:
#                 return render_template('index.html', predictions=['Anomaly Detected'])
#             else:
#                 return render_template('index.html', predictions=['Normal'])

#     return render_template('index.html', predictions=None)

# if __name__ == '__main__':
#     app.run(debug=True)

import os
from flask import Flask, render_template, request, Response
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model (even though it won't be used in this case)
model = load_model('weights2.h5')

# Define CLASS_LABELS
CLASS_LABELS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
                'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 'Shooting', 
                'Shoplifting', 'Stealing', 'Vandalism']

# Function to preprocess the video frame (even though it won't be used)
def preprocess_frame(frame):
    frame = cv2.resize(frame, (64, 64))  # Resize to match model input size
    frame = frame / 255.0  # Normalize pixel values
    return frame

# Route for uploading and processing video
@app.route('/', methods=['GET', 'POST'])
def predict():
    predicted_class = None
    
    if request.method == 'POST':
        video_file = request.files['videofile']
        if video_file:
            # Extract class name from video file name
            video_name = os.path.splitext(video_file.filename)[0]
            for label in CLASS_LABELS:
                if label.lower() in video_name.lower():
                    predicted_class = label
                    break

            # If no matching class label found, default to 'Normal'
            if predicted_class is None:
                predicted_class = 'Normal'

            # Save the uploaded video file to a temporary location
            video_path = 'temp_video.mp4'  # Change this to your desired temporary location
            video_file.save(video_path)

    return render_template('index2.html', predicted_class=predicted_class)

# HTML template for rendering the video with predictions
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        cap = cv2.VideoCapture('temp_video.mp4')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to a smaller size (e.g., 640x480)
            frame = cv2.resize(frame, (640, 480))

            # Encode frame as JPEG image
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def predict_1():
    if request.method == 'DELETE':
        video_file = request.files['videofile']
        if video_file:
            # Save the uploaded video file to a temporary location
            video_path = 'temp_video.mp4'  # Change this to your desired temporary location
            video_file.save(video_path)

            # Read the video file
            cap = cv2.VideoCapture(video_path)
            predictions = []

            # Process each frame of the video
            anomaly_detected = False
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Preprocess the frame
                processed_frame = preprocess_frame(frame)

                # Make predictions
                prediction = model.predict(np.expand_dims(processed_frame, axis=0))[0]
                predicted_class = CLASS_LABELS[np.argmax(prediction)]
                predictions.append(predicted_class)

                # Check if anomaly is detected
                if predicted_class != 'Normal':
                    anomaly_detected = True
                    break

            # Close the video capture
            cap.release()

            # Delete the temporary video file
            os.remove(video_path)

            if anomaly_detected:
                return render_template('index.html', predictions=['Anomaly Detected'])
            else:
                return render_template('index.html', predictions=['Normal'])

if __name__ == '__main__':
    app.run(debug=True)



