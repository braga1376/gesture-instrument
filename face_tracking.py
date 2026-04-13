import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceTracker:
    def __init__(self, sound_manager, model_path='face_landmarker.task'):
        self.sound_manager = sound_manager

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self._start_time = time.time()

    def process(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def update_face(self, recFace, img, h, w, marks):
        if recFace.face_landmarks:
            for face_landmark in recFace.face_landmarks:
                # In the new API each face is a plain list of landmarks, not a protobuf,
                # so face_landmark[idx] replaces the old face_landmark.landmark[idx]
                if marks:
                    for landmark in face_landmark:
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(img, (x, y), 3, (0, 255, 0), cv2.FILLED)

                upper_lip = face_landmark[13]
                lower_lip = face_landmark[14]

                lip_distance = np.linalg.norm(
                    np.array([upper_lip.x, upper_lip.y]) - np.array([lower_lip.x, lower_lip.y])
                )

                lpfreq = lip_distance * 17000
                self.sound_manager.set_lpfreq(lpfreq)

                if marks:
                    cv2.putText(img, f'Lip Distance: {lip_distance:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, f'LP Freq: {lpfreq:.2f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)