import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    INDEX_FINGER_IDX = 8
    THUMB_IDX = 4

    def __init__(self, sound_manager, model_path='hand_landmarker.task'):
        self.sound_manager = sound_manager

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def process(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    def update_hands(self, recHands, img, h, w, sound, marks, lines, distance_threshold):
        num_notes = self.sound_manager.get_num_notes()

        if recHands.hand_landmarks:
            n_hands = len(recHands.hand_landmarks)
            cv2.putText(img, f'Num Hands: {n_hands}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if n_hands == 1:
                self.sound_manager.set_is_playing2(False)

            for i, hand in enumerate(recHands.hand_landmarks):
                # In the new API each hand is a plain list of landmarks, not a protobuf,
                # so hand[idx] replaces the old hand.landmark[idx]
                if marks:
                    for point in hand:
                        x, y = int(point.x * w), int(point.y * h)
                        cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)

                thumb = (hand[self.THUMB_IDX].x, hand[self.THUMB_IDX].y)
                index = (hand[self.INDEX_FINGER_IDX].x, hand[self.INDEX_FINGER_IDX].y)
                distance = np.linalg.norm(np.array(thumb) - np.array(index))

                is_hand_open = distance > distance_threshold

                center_y = np.mean([point.y for point in hand])

                if i == 0:
                    target_freq1 = self.sound_manager.get_closest_scale_freq(center_y)
                    self.sound_manager.set_target_freq1(target_freq1)
                    if is_hand_open:
                        if sound:
                            self.sound_manager.set_is_playing1(True)
                        if marks:
                            cv2.putText(img, f'Hand 1 Open: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(img, f'Note: {self.sound_manager.scale.scale_dict["{:.2f}".format(target_freq1)]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        self.sound_manager.set_is_playing1(False)
                        if marks:
                            cv2.putText(img, f'Hand 1 Closed: {distance:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(img, f'Note: {self.sound_manager.scale.scale_dict["{:.2f}".format(target_freq1)]}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif i == 1:
                    target_freq2 = self.sound_manager.get_closest_scale_freq(center_y)
                    self.sound_manager.set_target_freq2(target_freq2)
                    if is_hand_open:
                        if sound:
                            self.sound_manager.set_is_playing2(True)
                        if marks:
                            cv2.putText(img, f'Hand 2 Open: {distance:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(img, f'Note: {self.sound_manager.scale.scale_dict["{:.2f}".format(target_freq2)]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        self.sound_manager.set_is_playing2(False)
                        if marks:
                            cv2.putText(img, f'Hand 2 Closed: {distance:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.putText(img, f'Note: {self.sound_manager.scale.scale_dict["{:.2f}".format(target_freq2)]}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if lines:
                selected_note_index1 = self.sound_manager.scale.scale.index(self.sound_manager.get_target_freq1())
                y1 = int(selected_note_index1 * h / num_notes)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, y1), (w, y1 + int(h / num_notes)), (0, 255, 0), -1)
                if n_hands > 1:
                    selected_note_index2 = self.sound_manager.scale.scale.index(self.sound_manager.get_target_freq2())
                    y2 = int(selected_note_index2 * h / num_notes)
                    cv2.rectangle(overlay, (0, y2), (w, y2 + int(h / num_notes)), (0, 255, 0), -1)

                alpha = 0.1
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        else:
            self.sound_manager.set_is_playing1(False)
            self.sound_manager.set_is_playing2(False)