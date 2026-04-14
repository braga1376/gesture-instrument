import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------------------------------------------------------------------
# VelocityTracker
# ---------------------------------------------------------------------------

class VelocityTracker:
    """
    Tracks the smoothed velocity of a single hand center over a short
    rolling window.

    Stores the last HISTORY_LEN (x, y, timestamp) samples and computes
    raw velocity as total displacement divided by window duration.
    An exponential moving average then smooths out frame-to-frame jitter.
    """
    HISTORY_LEN  = 3      # shorter window catches fast peaks better
    SMOOTHING    = 0.65   # more responsive to quick changes

    def __init__(self):
        self._history = deque(maxlen=self.HISTORY_LEN)
        self.vx    = 0.0
        self.vy    = 0.0
        self.speed = 0.0

    def update(self, x, y):
        """Add a new normalised (x, y) position sample and recompute velocity."""
        self._history.append((x, y, time.time()))
        if len(self._history) < 2:
            return

        oldest, newest = self._history[0], self._history[-1]
        dt = newest[2] - oldest[2]
        if dt < 1e-6:
            return

        raw_vx = (newest[0] - oldest[0]) / dt
        raw_vy = (newest[1] - oldest[1]) / dt

        a = self.SMOOTHING
        self.vx    = a * raw_vx    + (1 - a) * self.vx
        self.vy    = a * raw_vy    + (1 - a) * self.vy
        self.speed = np.sqrt(self.vx**2 + self.vy**2)

    def reset(self):
        """Clear history and zero velocity — called when the hand disappears."""
        self._history.clear()
        self.vx    = 0.0
        self.vy    = 0.0
        self.speed = 0.0


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------

class HandTracker:
    INDEX_FINGER_IDX = 8
    THUMB_IDX        = 4
    GRACE_FRAMES     = 8  # frames of absence before a voice is deactivated
                          # (~250ms at 30fps) — prevents restart on brief tracking loss

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
        self.detector    = vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

        # One velocity tracker per hand slot
        self.vel_trackers = [VelocityTracker(), VelocityTracker()]

        # Whether each hand was visible in the previous frame
        self._prev_open = [False, False]

        # Consecutive frames each hand has been absent — used for grace period
        self._lost_frames = [0, 0]

    # ------------------------------------------------------------------
    # MediaPipe processing
    # ------------------------------------------------------------------

    def process(self, img):
        img_rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        return self.detector.detect_for_video(mp_image, timestamp_ms)

    # ------------------------------------------------------------------
    # Main update — called every video frame
    # ------------------------------------------------------------------

    def update_hands(self, recHands, img, h, w, sound, marks, lines, distance_threshold):
        num_notes = self.sound_manager.get_num_notes()
        mode      = self.sound_manager.mode

        if recHands.hand_landmarks:
            n_hands = len(recHands.hand_landmarks)
            cv2.putText(img, f'Num Hands: {n_hands}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for i, hand in enumerate(recHands.hand_landmarks):

                # --- landmark dots (both modes) ---
                if marks:
                    for point in hand:
                        cv2.circle(img, (int(point.x * w), int(point.y * h)),
                                   10, (255, 0, 255), cv2.FILLED)

                # --- open/closed detection (both modes) ---
                thumb    = (hand[self.THUMB_IDX].x,        hand[self.THUMB_IDX].y)
                index    = (hand[self.INDEX_FINGER_IDX].x, hand[self.INDEX_FINGER_IDX].y)
                distance = np.linalg.norm(np.array(thumb) - np.array(index))
                is_hand_open = distance > distance_threshold

                # --- hand centre (both modes) ---
                center_x = np.mean([p.x for p in hand])
                center_y = np.mean([p.y for p in hand])

                # --- always update velocity tracker ---
                self.vel_trackers[i].update(center_x, center_y)
                vt = self.vel_trackers[i]

                # ==============================================================
                # SIMPLE MODE — original behaviour, completely unchanged
                # ==============================================================
                if mode == "simple":
                    if i == 0:
                        target_freq = self.sound_manager.get_closest_scale_freq(center_y)
                        self.sound_manager.set_target_freq1(target_freq)
                        if is_hand_open:
                            if sound:
                                self.sound_manager.set_is_playing1(True)
                            if marks:
                                cv2.putText(img, f'Hand 1 Open: {distance:.2f}', (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(img, f'Note: {self.sound_manager.scale.scale_dict["{:.2f}".format(target_freq)]}',
                                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            self.sound_manager.set_is_playing1(False)
                            if marks:
                                cv2.putText(img, f'Hand 1 Closed: {distance:.2f}', (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(img, f'Note: {self.sound_manager.scale.scale_dict["{:.2f}".format(target_freq)]}',
                                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    elif i == 1:
                        target_freq = self.sound_manager.get_closest_scale_freq(center_y)
                        self.sound_manager.set_target_freq2(target_freq)
                        if is_hand_open:
                            if sound:
                                self.sound_manager.set_is_playing2(True)
                            if marks:
                                cv2.putText(img, f'Hand 2 Open: {distance:.2f}', (10, 70),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(img, f'Note: {self.sound_manager.scale.scale_dict["{:.2f}".format(target_freq)]}',
                                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        else:
                            self.sound_manager.set_is_playing2(False)
                            if marks:
                                cv2.putText(img, f'Hand 2 Closed: {distance:.2f}', (10, 70),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(img, f'Note: {self.sound_manager.scale.scale_dict["{:.2f}".format(target_freq)]}',
                                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # ==============================================================
                # GESTURE MODE — velocity drives melodic movement.
                # Open/closed hand is ignored; the voice is active whenever
                # the hand is visible. _prev_open[i] here means "was this
                # hand visible last frame" — used only to detect first
                # appearance so we can set the home note.
                # ==============================================================
                else:
                    home_index = self.sound_manager.get_home_index(center_y)

                    if sound:
                        # Decide whether this is a genuine fresh start or a
                        # reappearance within the grace period.
                        # Grace period: hand was briefly lost but cursor/home
                        # are preserved — resume without resetting.
                        # Fresh start: hand was absent long enough to deactivate.
                        is_fresh = (not self._prev_open[i] and
                                    self._lost_frames[i] >= self.GRACE_FRAMES)
                        is_resume = (not self._prev_open[i] and
                                     self._lost_frames[i] < self.GRACE_FRAMES and
                                     self._lost_frames[i] > 0)

                        if is_fresh or (not self._prev_open[i] and self._lost_frames[i] == 0):
                            # Genuine first appearance — set home note and reset cursor
                            if i == 0:
                                self.sound_manager.activate_gesture1(home_index)
                            else:
                                self.sound_manager.activate_gesture2(home_index)
                        elif is_resume:
                            # Reappeared within grace window — reactivate without
                            # resetting the cursor so the phrase continues naturally
                            if i == 0:
                                self.sound_manager.gesture_voice1.is_active = True
                            else:
                                self.sound_manager.gesture_voice2.is_active = True

                        # Every frame: push current velocity to the voice
                        if i == 0:
                            self.sound_manager.set_gesture_velocity1(vt.vx, vt.vy, vt.speed)
                        else:
                            self.sound_manager.set_gesture_velocity2(vt.vx, vt.vy, vt.speed)

                    # Hand is visible — reset its lost-frames counter
                    self._lost_frames[i] = 0

                    if marks:
                        cv2.putText(img, f'Hand {i+1}: spd={vt.speed:.3f}',
                                    (10, 30 if i == 0 else 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Active as long as sound is on and hand is visible
                    self._prev_open[i] = sound

            # ------------------------------------------------------------------
            # Post-loop: handle any hand slot that was active but is now absent.
            # ------------------------------------------------------------------
            for i in range(2):
                if i >= n_hands:
                    if mode == "simple":
                        if i == 0:
                            self.sound_manager.set_is_playing1(False)
                        else:
                            self.sound_manager.set_is_playing2(False)
                    else:
                        if self._prev_open[i]:
                            self._lost_frames[i] += 1
                            if self._lost_frames[i] >= self.GRACE_FRAMES:
                                if i == 0:
                                    self.sound_manager.deactivate_gesture1()
                                else:
                                    self.sound_manager.deactivate_gesture2()
                                self.vel_trackers[i].reset()
                                self._prev_open[i] = False

            # ------------------------------------------------------------------
            # Lines overlay — shows the active note band for each hand
            # ------------------------------------------------------------------
            if lines:
                overlay = img.copy()
                if mode == "simple":
                    idx1 = self.sound_manager.scale.scale.index(
                        self.sound_manager.get_target_freq1())
                else:
                    idx1 = self.sound_manager.get_gesture_note_index1()

                y1 = int(idx1 * h / num_notes)
                cv2.rectangle(overlay, (0, y1), (w, y1 + int(h / num_notes)),
                              (0, 255, 0), -1)

                if n_hands > 1:
                    if mode == "simple":
                        idx2 = self.sound_manager.scale.scale.index(
                            self.sound_manager.get_target_freq2())
                    else:
                        idx2 = self.sound_manager.get_gesture_note_index2()

                    y2 = int(idx2 * h / num_notes)
                    cv2.rectangle(overlay, (0, y2), (w, y2 + int(h / num_notes)),
                                  (0, 255, 0), -1)

                cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)

        else:
            # No hands detected at all
            if mode == "simple":
                self.sound_manager.set_is_playing1(False)
                self.sound_manager.set_is_playing2(False)
                self._prev_open = [False, False]
            else:
                for i in range(2):
                    if self._prev_open[i]:
                        self._lost_frames[i] += 1
                        if self._lost_frames[i] >= self.GRACE_FRAMES:
                            if i == 0:
                                self.sound_manager.deactivate_gesture1()
                            else:
                                self.sound_manager.deactivate_gesture2()
                            self.vel_trackers[i].reset()
                            self._prev_open[i] = False