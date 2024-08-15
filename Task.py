import cv2
import argparse
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime
import sqlite3

def setup_database():
    conn = sqlite3.connect('detection_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            numOfFaces INTEGER
        )'''
    )
    conn.commit()
    return conn

def save_detection_to_db(conn, numOfFaces):
    """Save the detection event to the database."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO detections (timestamp, numOfFaces) VALUES (?, ?)
    ''', (timestamp, numOfFaces))
    conn.commit()

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def update_video():
    hasFrame, frame = video.read()
    if hasFrame:
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        numOfFaces = len(faceBoxes)  
        face_count_label.config(text=f'Number of faces detected: {numOfFaces}') 

        if numOfFaces > 0:
            save_detection_to_db(conn, numOfFaces)

        frame_rgb = cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.img_tk = img_tk
        video_label.configure(image=img_tk)
    else:
        print("Failed to capture video frame")
    root.after(200, update_video)

parser = argparse.ArgumentParser()
parser.add_argument('--image', help="Path to image or video file. Leave empty for webcam.")

args = parser.parse_args()

faceProto = "/Users/anjalividhate/Downloads/opencv_face_detector.pbtxt"
faceModel = "/Users/anjalividhate/Downloads/opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

root = tk.Tk()
root.title("Face Detection")
video_label = tk.Label(root)
video_label.pack()

face_count_label = tk.Label(root, text="Number of faces detected: 0", font=("Arial", 14))
face_count_label.pack()

conn = setup_database()

update_video()

root.mainloop()

conn.close()
