# USAGE
# python "drowsiness detection.py" -p shape_predictor_68_face_landmarks.dat -c COM3

from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import argparse
import time
import dlib
import cv2
import serial

# --- Helper Functions ---
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_mar(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar

def get_head_pose(shape, frame_shape):
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    image_points = np.array([
        (shape[30][0], shape[30][1]),
        (shape[8][0], shape[8][1]),
        (shape[36][0], shape[36][1]),
        (shape[45][0], shape[45][1]),
        (shape[48][0], shape[48][1]),
        (shape[54][0], shape[54][1])
    ], dtype="double")

    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    rmat, _ = cv2.Rodrigues(rotation_vector)
    pitch = np.arcsin(-rmat[1, 2]) * 180 / np.pi
    return pitch

# --- Argument Parser ---
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-c", "--port", type=str, required=True, help="Serial port for Arduino")
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# --- Serial Port Setup ---
ARDUINO_PORT = args["port"]
try:
    ser = serial.Serial(ARDUINO_PORT, 9600, timeout=1)
    print(f"Connected to Arduino on {ARDUINO_PORT}")
    time.sleep(2) # Give Arduino time to reset
except Exception as e:
    ser = None
    print(f"Warning: Could not connect to Arduino on {ARDUINO_PORT}. {e}")

# --- Counters ---
eye_counter = 0
yawn_counter = 0
nod_counter = 0

# --- dlib Setup ---
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# --- Start Video Stream (Standard CV2) ---
print("[INFO] Starting video stream... (NEW CODE)")
cap = cv2.VideoCapture(args["webcam"])

if not cap.isOpened():
    print("Error: Could not open webcam. Try changing the index with -w 1")
    exit()

# --- Tuning Window ---
def nothing(x): pass
cv2.namedWindow("Tuning")
cv2.resizeWindow("Tuning", 400, 400)
cv2.createTrackbar("EAR_Thresh", "Tuning", 25, 100, nothing) 
cv2.createTrackbar("EAR_Frames", "Tuning", 20, 100, nothing)
cv2.createTrackbar("MAR_Thresh", "Tuning", 60, 100, nothing)
cv2.createTrackbar("MAR_Frames", "Tuning", 25, 100, nothing)
cv2.createTrackbar("Pitch_Thresh", "Tuning", 30, 90, nothing)
cv2.createTrackbar("Pitch_Frames", "Tuning", 20, 100, nothing)

# --- Main Loop ---
while True:
    # 1. Read Frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break
    
    # 2. Resize and Convert
    frame = cv2.resize(frame, (640, 480))
    if frame is None:
        print("Invalid frame received. Skipping...")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    frame_shape = frame.shape
    
    # Update Thresholds from Trackbars
    EAR_THRESHOLD = cv2.getTrackbarPos("EAR_Thresh", "Tuning") / 100.0
    EAR_CONSEC_FRAMES = cv2.getTrackbarPos("EAR_Frames", "Tuning")
    MAR_THRESHOLD = cv2.getTrackbarPos("MAR_Thresh", "Tuning") / 100.0
    MAR_CONSEC_FRAMES = cv2.getTrackbarPos("MAR_Frames", "Tuning")
    PITCH_THRESHOLD = cv2.getTrackbarPos("Pitch_Thresh", "Tuning")
    PITCH_CONSEC_FRAMES = cv2.getTrackbarPos("Pitch_Frames", "Tuning")

    drowsy_alert = False 
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Eye Logic
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = calculate_ear(leftEye)
        rightEAR = calculate_ear(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        if ear < EAR_THRESHOLD:
            eye_counter += 1
            if eye_counter >= EAR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                drowsy_alert = True
        else:
            eye_counter = 0

        # Mouth Logic
        mouth = shape[mStart:mEnd]
        mar = calculate_mar(mouth)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

        if mar > MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= MAR_CONSEC_FRAMES:
                cv2.putText(frame, "YAWN DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                drowsy_alert = True
        else:
            yawn_counter = 0

        # Head Nod Logic
        pitch = get_head_pose(shape, frame_shape)
        if pitch > PITCH_THRESHOLD:
            nod_counter += 1
            if nod_counter >= PITCH_CONSEC_FRAMES:
                cv2.putText(frame, "HEAD NOD", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                drowsy_alert = True
        else:
            nod_counter = 0
            
        # Display Stats
        cv2.putText(frame, f"EAR: {ear:.2f}", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (480, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # --- Send Signal to Arduino ---
    if drowsy_alert and ser is not None:
        try:
            ser.write(b'D')
            print("Sent 'D' to Arduino")
        except:
            pass

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if ser is not None:
    ser.close()
