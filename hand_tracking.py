import cv2
import numpy as np
import mediapipe as mp

class HandTracker:
    INDEX_FINGER_IDX = 8
    THUMB_IDX = 4

    def __init__(self, sound_manager):
        self.sound_manager = sound_manager
        self.hands = mp.solutions.hands.Hands()

    def process(self, img):
        return self.hands.process(img)

    def update_hands(self, recHands, img, h, w, marks, lines):
        num_notes = self.sound_manager.get_num_notes()

        if recHands.multi_hand_landmarks:
            for i, hand in enumerate(recHands.multi_hand_landmarks):
                if marks:
                    for datapoint_id, point in enumerate(hand.landmark):
                        x, y = int(point.x * w), int(point.y * h)
                        cv2.circle(img, (x, y), 10, (255, 0, 255), cv2.FILLED)

                thumb = (hand.landmark[self.THUMB_IDX].x, hand.landmark[self.THUMB_IDX].y)
                index = (hand.landmark[self.INDEX_FINGER_IDX].x, hand.landmark[self.INDEX_FINGER_IDX].y)
                distance = np.linalg.norm(np.array(thumb) - np.array(index))

                is_hand_open = distance > 0.07

                center_y = np.mean([point.y for point in hand.landmark])

                if i == 0: 
                    target_freq1 = self.sound_manager.get_closest_scale_freq(center_y)
                    self.sound_manager.set_target_freq1(target_freq1)
                    if is_hand_open:
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
                selected_note_index2 = self.sound_manager.scale.scale.index(self.sound_manager.get_target_freq2())
                y1 = int(selected_note_index1 * h / num_notes)
                y2 = int(selected_note_index2 * h / num_notes)
                overlay = img.copy()
                cv2.rectangle(overlay, (0, y1), (w, y1 + int(h / num_notes)), (0, 255, 0), -1)
                cv2.rectangle(overlay, (0, y2), (w, y2 + int(h / num_notes)), (0, 255, 0), -1)
                alpha = 0.1
                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        else:
            self.sound_manager.set_is_playing1(False)
            self.sound_manager.set_is_playing2(False)