import cv2
import numpy as np
import mediapipe as mp

class FaceTracker:
    def __init__(self, sound_manager):
        self.sound_manager = sound_manager
        self.face = mp.solutions.face_mesh.FaceMesh()

    def process(self, img):
        return self.face.process(img)

    def update_face(self, recFace, img, h, w):
        if recFace.multi_face_landmarks:
            for face_landmark in recFace.multi_face_landmarks:
                for i, landmark in enumerate(face_landmark.landmark):
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img, (x, y), 3, (0, 255, 0), cv2.FILLED)

                upper_lip = face_landmark.landmark[13]
                lower_lip = face_landmark.landmark[14]

                lip_distance = np.linalg.norm(np.array([upper_lip.x, upper_lip.y]) - np.array([lower_lip.x, lower_lip.y]))

                lpfreq = lip_distance * 17000
                self.sound_manager.set_lpfreq(lpfreq)

                cv2.putText(img, f'Lip Distance: {lip_distance:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(img, f'LP Freq: {lpfreq:.2f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)