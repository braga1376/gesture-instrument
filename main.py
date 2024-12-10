import cv2
import mediapipe as mp
import time
import numpy as np
import threading
from pyo import *

INDEX_FINGER_IDX = 8
THUMB_IDX = 4

MIN_FREQ = 200
MAX_FREQ = 800

current_freq1 = MIN_FREQ
target_freq1 = MIN_FREQ
is_playing1 = False

current_freq2 = MIN_FREQ
target_freq2 = MIN_FREQ
is_playing2 = False

s = Server().boot()
s.start()

fm1 = FM(carrier=current_freq1, ratio=[1.2, 1.21], index=5, mul=0.5).out()
fm2 = FM(carrier=current_freq2, ratio=[1.2, 1.21], index=5, mul=0.5).out()

lp1 = ButLP(fm1, freq=1000).out()
lp2 = ButLP(fm2, freq=1000).out()

reverb1 = Freeverb(lp1, size=0.8, damp=0.5, bal=0.5).out()
reverb2 = Freeverb(lp2, size=0.8, damp=0.5, bal=0.5).out()

def update_frequency():
    global current_freq1, target_freq1, is_playing1
    global current_freq2, target_freq2, is_playing2
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

        time.sleep(0.005) 

freq_thread = threading.Thread(target=update_frequency)
freq_thread.daemon = True
freq_thread.start()

cap = cv2.VideoCapture(0)
handSolution = mp.solutions.hands
hands = handSolution.Hands()

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
            
    recHands = hands.process(img)
    if recHands.multi_hand_landmarks:
        for i, hand in enumerate(recHands.multi_hand_landmarks):
            for datapoint_id, point in enumerate(hand.landmark):
                h, w, c = img.shape
                x, y = int(point.x * w), int(point.y * h)
                cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)

            thumb = (hand.landmark[THUMB_IDX].x, hand.landmark[THUMB_IDX].y)
            index = (hand.landmark[INDEX_FINGER_IDX].x, hand.landmark[INDEX_FINGER_IDX].y) 
            distance = np.linalg.norm(np.array(thumb) - np.array(index))

            is_hand_open = distance > 0.1 

            center_y = np.mean([point.y for point in hand.landmark])

            if i == 0: 
                if is_hand_open:
                    target_freq1 = MIN_FREQ + (1 - center_y) * (MAX_FREQ - MIN_FREQ)
                    is_playing1 = True
                    cv2.putText(img, f'Hand 1 Open: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    is_playing1 = False
                    cv2.putText(img, f'Hand 1 Closed: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif i == 1: 
                if is_hand_open:
                    target_freq2 = MIN_FREQ + (1 - center_y) * (MAX_FREQ - MIN_FREQ)
                    is_playing2 = True
                    cv2.putText(img, f'Hand 2 Open: {distance:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    is_playing2 = False
                    cv2.putText(img, f'Hand 2 Closed: {distance:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        is_playing1 = False
        is_playing2 = False
        
    cv2.imshow("CamOutput", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
s.stop()