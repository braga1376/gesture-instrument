import cv2
import mediapipe as mp
import time
import numpy as np
import threading
from pyo import *

INDEX_FINGER_IDX = 8
THUMB_IDX = 4

C_MAJOR_SCALE = [
    1046.50, 987.77, 880.00, 783.99, 698.46, 659.25, 587.33,  # C6 to B5
    523.25, 493.88, 440.00, 392.00, 349.23, 329.63, 293.66,   # C5 to B4
    261.63, 246.94  # C4 to B3
]

C_MAJOR_SCALE_DICT = {
    "1046.50": "C6",
    "987.77": "B5",
    "880.00": "A5",
    "783.99": "G5",
    "698.46": "F5",
    "659.25": "E5",
    "587.33": "D5",
    "523.25": "C5",
    "493.88": "B4",
    "440.00": "A4",
    "392.00": "G4",
    "349.23": "F4",
    "329.63": "E4",
    "293.66": "D4",
    "261.63": "C4",
    "246.94": "B3"
}


current_freq1 = C_MAJOR_SCALE[0]
target_freq1 = C_MAJOR_SCALE[0]
is_playing1 = False

current_freq2 = C_MAJOR_SCALE[0]
target_freq2 = C_MAJOR_SCALE[0]
is_playing2 = False

current_lpfreq = 1000
lpfreq = 1000

s = Server().boot()
s.start()

fm1 = FM(carrier=current_freq1, ratio=[4, 2, 1, 0.5], index=5, mul=0.5).out()
fm2 = FM(carrier=current_freq2, ratio=[4, 2, 1, 0.5], index=5, mul=0.5).out()

lp1 = ButLP(fm1, freq=current_lpfreq).out()
lp2 = ButLP(fm2, freq=current_lpfreq).out()

reverb1 = Freeverb(lp1, size=0.8, damp=0.5, bal=0.5).out()
reverb2 = Freeverb(lp2, size=0.8, damp=0.5, bal=0.5).out()

def update_frequency():
    global current_freq1, target_freq1, is_playing1
    global current_freq2, target_freq2, is_playing2
    global current_lpfreq, lpfreq
    while True:
        if is_playing1:
            current_freq1 = float(target_freq1)
            fm1.carrier = current_freq1
            fm1.mul = 0.5
            fm1.play()
        else:
            fm1.stop()

        if is_playing2:
            current_freq2 = float(target_freq2)
            fm2.carrier = current_freq2
            fm2.mul = 0.5
            fm2.play()
        else:
            fm2.stop()

        current_lpfreq = float(lpfreq)
        lp1.freq = current_lpfreq
        lp2.freq = current_lpfreq

        time.sleep(0.005) 

freq_thread = threading.Thread(target=update_frequency)
freq_thread.daemon = True
freq_thread.start()

cap = cv2.VideoCapture(0)
handSolution = mp.solutions.hands
hands = handSolution.Hands()

faceLandmarker = mp.solutions.face_mesh
face = faceLandmarker.FaceMesh()


def get_closest_scale_freq(center_y):
    # Map the center_y value to the closest note in the C major scale
    num_notes = len(C_MAJOR_SCALE)
    note_index = int(center_y * num_notes)
    note_index = min(max(note_index, 0), num_notes - 1)
    return C_MAJOR_SCALE[note_index]

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
            
    recHands = hands.process(img)
    recFace = face.process(img)
    if recHands.multi_hand_landmarks:
        for i, hand in enumerate(recHands.multi_hand_landmarks):
            for datapoint_id, point in enumerate(hand.landmark):
                h, w, c = img.shape
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)
            
            thumb = (hand.landmark[THUMB_IDX].x, hand.landmark[THUMB_IDX].y)
            index = (hand.landmark[INDEX_FINGER_IDX].x, hand.landmark[INDEX_FINGER_IDX].y) 
            distance = np.linalg.norm(np.array(thumb) - np.array(index))

            is_hand_open = distance > 0.07 

            # Calculate the center of the hand
            center_y = np.mean([point.y for point in hand.landmark])

            if i == 0:  # First hand
                target_freq1 = get_closest_scale_freq(center_y)
                if is_hand_open:
                    is_playing1 = True
                    cv2.putText(img, f'Hand 1 Open: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f'Note: {C_MAJOR_SCALE_DICT["{:.2f}".format(target_freq1)]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    is_playing1 = False
                    cv2.putText(img, f'Hand 1 Closed: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(img, f'Note: {C_MAJOR_SCALE_DICT["{:.2f}".format(target_freq1)]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif i == 1:  # Second hand
                target_freq2 = get_closest_scale_freq(center_y)
                if is_hand_open:
                    is_playing2 = True
                    cv2.putText(img, f'Hand 2 Open: {distance:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f'Note: {C_MAJOR_SCALE_DICT["{:.2f}".format(target_freq2)]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    is_playing2 = False
                    cv2.putText(img, f'Hand 2 Closed: {distance:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(img, f'Note: {C_MAJOR_SCALE_DICT["{:.2f}".format(target_freq2)]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if recFace.multi_face_landmarks:
            for face_landmark in recFace.multi_face_landmarks:
                for i, landmark in enumerate(face_landmark.landmark):
                    h, w, c = img.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img, (x, y), 3, (0, 255, 0), cv2.FILLED)

         
            upper_lip = face_landmark.landmark[13] 
            lower_lip = face_landmark.landmark[14]

            lip_distance = np.linalg.norm(np.array([upper_lip.x, upper_lip.y]) - np.array([lower_lip.x, lower_lip.y]))

            lpfreq = 400 + lip_distance * 13000 
            
            cv2.putText(img, f'Lip Distance: {lip_distance:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'LP Freq: {lpfreq:.2f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    else:
        is_playing1 = False
        is_playing2 = False
        
    cv2.imshow("CamOutput", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
s.stop()