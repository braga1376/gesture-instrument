import time
import numpy as np
from pyo import *

# ---------------------------------------------------------------------------
# Scale data — extended to ~4 octaves (C3 to C7) so the gesture cursor
# has plenty of room to drift without hitting the boundary in practice
# ---------------------------------------------------------------------------

C_PENTATONIC_SCALE = [
    2093.00, 1760.00, 1567.98, 1318.51, 1174.66,  # C7 to D6
    1046.50,  880.00,  783.99,  659.25,  587.33,  # C6 to D5
     523.25,  440.00,  392.00,  329.63,  293.66,  # C5 to D4
     261.63,  220.00,  196.00,  164.81,  146.83,  # C4 to D3
     130.81,                                       # C3
]
C_PENTATONIC_SCALE_DICT = {
    "2093.00": "C7", "1760.00": "A6", "1567.98": "G6",
    "1318.51": "E6", "1174.66": "D6",
    "1046.50": "C6",  "880.00": "A5",  "783.99": "G5",
     "659.25": "E5",  "587.33": "D5",
     "523.25": "C5",  "440.00": "A4",  "392.00": "G4",
     "329.63": "E4",  "293.66": "D4",
     "261.63": "C4",  "220.00": "A3",  "196.00": "G3",
     "164.81": "E3",  "146.83": "D3",
     "130.81": "C3",
}

C_MAJOR_SCALE = [
    2093.00, 1975.53, 1760.00, 1567.98, 1396.91, 1318.51, 1174.66,  # C7–D6
    1046.50,  987.77,  880.00,  783.99,  698.46,  659.25,  587.33,  # C6–D5
     523.25,  493.88,  440.00,  392.00,  349.23,  329.63,  293.66,  # C5–D4
     261.63,  246.94,  220.00,  196.00,  174.61,  164.81,  146.83,  # C4–D3
     130.81,                                                          # C3
]
C_MAJOR_SCALE_DICT = {
    "2093.00": "C7", "1975.53": "B6", "1760.00": "A6",
    "1567.98": "G6", "1396.91": "F6", "1318.51": "E6", "1174.66": "D6",
    "1046.50": "C6",  "987.77": "B5",  "880.00": "A5",
     "783.99": "G5",  "698.46": "F5",  "659.25": "E5",  "587.33": "D5",
     "523.25": "C5",  "493.88": "B4",  "440.00": "A4",
     "392.00": "G4",  "349.23": "F4",  "329.63": "E4",  "293.66": "D4",
     "261.63": "C4",  "246.94": "B3",  "220.00": "A3",
     "196.00": "G3",  "174.61": "F3",  "164.81": "E3",  "146.83": "D3",
     "130.81": "C3",
}


# ---------------------------------------------------------------------------
# Scale class — unchanged from before
# ---------------------------------------------------------------------------

class scale:
    def __init__(self, name="pentatonic"):
        self.name = name
        self._apply(name)

    def _apply(self, name):
        if name in ("pentatonic", "Pentatonic"):
            self.scale = C_PENTATONIC_SCALE
            self.scale_dict = C_PENTATONIC_SCALE_DICT
        elif name in ("diatonic", "Diatonic"):
            self.scale = C_MAJOR_SCALE
            self.scale_dict = C_MAJOR_SCALE_DICT
        else:
            raise ValueError("Scale not recognized. Use 'pentatonic' or 'diatonic'.")
        self.num_notes = len(self.scale)

    def get_scale(self):
        return self.scale

    def get_scale_dict(self):
        return self.scale_dict

    def get_num_notes(self):
        return self.num_notes

    def set_scale(self, name):
        self.name = name
        self._apply(name)


# ---------------------------------------------------------------------------
# GestureVoice — owns the cursor, velocity state, and note scheduler
# for one hand in gesture mode
# ---------------------------------------------------------------------------

class GestureVoice:
    # Tuning constants — calibrated against observed speed range
    # (slow movement ≈ 0.3–0.4 normalised coords/sec)
    SPEED_THRESHOLD = 0.08    # below this the hand is considered still → silence
    MAX_SPEED       = 1.5     # speed at which note rate hits its ceiling
    MIN_INTERVAL    = 0.09    # seconds between notes at max speed (~11 notes/sec)
    MAX_INTERVAL    = 0.55    # seconds between notes at min speed (~2 notes/sec)
    MAX_STEP        = 3       # largest scale-step jump per note (at full vertical speed)

    def __init__(self, scale_obj):
        self.scale      = scale_obj
        self.cursor     = 0
        self.home_index = len(scale_obj.scale) // 2  # start in the middle of the scale
        self.is_active  = False

        # Velocity — set from the video thread, read by tick() on the audio thread
        self.vx    = 0.0
        self.vy    = 0.0
        self.speed = 0.0

        self._last_note_time = 0.0
        self._current_freq   = None   # held between ticks so note sustains

    def set_scale(self, scale_obj):
        """Called when the user switches scale so the voice stays in sync."""
        self.scale      = scale_obj
        self.home_index = min(self.home_index, len(scale_obj.scale) - 1)
        self.cursor     = 0

    def activate(self, home_index):
        """
        Called when the hand reopens. Sets the home note from the hand's
        Y position and resets the cursor so the phrase starts cleanly.
        """
        self.home_index      = home_index
        self.cursor          = 0
        self.is_active       = True
        self._last_note_time = 0.0   # fire a note immediately on activation
        self._current_freq   = None

    def deactivate(self):
        """Called when the hand closes or leaves the frame."""
        self.is_active     = False
        self._current_freq = None

    def set_velocity(self, vx, vy, speed):
        """Called from the video thread each frame with the smoothed velocity."""
        self.vx    = vx
        self.vy    = vy
        self.speed = speed

    def tick(self):
        """
        Called from the audio thread every loop iteration.
        Returns the frequency to play, or None for silence.

        The hand is still  → None (silence)
        Between note beats → _current_freq (hold the current note)
        On a note beat     → new frequency computed from cursor + step
        """
        if not self.is_active or self.speed < self.SPEED_THRESHOLD:
            self._current_freq = None
            return None

        # Note interval shrinks as speed grows
        speed_norm = min(self.speed / self.MAX_SPEED, 1.0)
        interval   = self.MAX_INTERVAL - (self.MAX_INTERVAL - self.MIN_INTERVAL) * speed_norm

        now = time.time()
        if now - self._last_note_time < interval:
            return self._current_freq  # sustain current note until next beat

        # --- it is time for a new note ---
        self._last_note_time = now

        # vy_norm is the vertical component of the unit velocity vector.
        # Image Y increases downward, so negate to get musical direction:
        # upward hand movement → negative vy → positive vy_norm → ascending.
        # The scale array is ordered high-to-low, so we negate the step again
        # so that a positive vy_norm produces a negative cursor step (lower index
        # = higher frequency).
        vy_norm = (-self.vy / self.speed) if self.speed > 0 else 0.0
        step    = round(-vy_norm * self.MAX_STEP)

        # Advance the cursor freely — it is only clamped at the scale boundary
        self.cursor += step
        scale_len   = len(self.scale.scale)
        self.cursor = max(-self.home_index, min(scale_len - 1 - self.home_index, self.cursor))

        note_index         = max(0, min(scale_len - 1, self.home_index + self.cursor))
        self._current_freq = self.scale.scale[note_index]
        return self._current_freq


# ---------------------------------------------------------------------------
# SoundManager
# ---------------------------------------------------------------------------

class SoundManager:
    def __init__(self):
        self.scale = scale("pentatonic")
        self.mode  = "simple"   # "simple" | "gesture"

        # --- simple mode state (unchanged) ---
        self.current_freq1 = self.scale.scale[0]
        self.target_freq1  = self.scale.scale[0]
        self.is_playing1   = False
        self._was_playing1 = False

        self.current_freq2 = self.scale.scale[0]
        self.target_freq2  = self.scale.scale[0]
        self.is_playing2   = False
        self._was_playing2 = False

        self.current_lpfreq = 1000
        self.lpfreq         = 1000
        self._running       = False

        # --- gesture mode state ---
        self.gesture_voice1 = GestureVoice(self.scale)
        self.gesture_voice2 = GestureVoice(self.scale)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_sound(self):
        self.s = Server(nchnls=2, duplex=0).boot()
        self.s.start()

        lfo1 = Sine(0.1).range(0.1, 0.75)
        self.synth1 = SuperSaw(freq=self.current_freq1, detune=lfo1, bal=0.7, mul=0.5)
        self.lp1    = ButLP(self.synth1, freq=self.current_lpfreq)
        self.panned1 = Pan(self.lp1, outs=2, pan=0.5).out()

        lfo2 = Sine(0.1).range(0.1, 0.75)
        self.synth2 = SuperSaw(freq=self.current_freq2, detune=lfo2, bal=0.7, mul=0.5)
        self.lp2    = ButLP(self.synth2, freq=self.current_lpfreq)
        self.panned2 = Pan(self.lp2, outs=2, pan=0.5).out()

        self.reverb1 = Freeverb(self.lp1, size=0.6, damp=0.5, bal=0.5)
        self.reverb2 = Freeverb(self.lp2, size=0.6, damp=0.5, bal=0.5)
        self.limiter = Compress(
            self.panned1 + self.panned2 + self.reverb1 + self.reverb2,
            thresh=-20, ratio=4
        ).out()

        # pyo objects start playing automatically on creation — stop them
        # so the initial state matches _was_playing = False
        self.synth1.stop()
        self.synth2.stop()

    # ------------------------------------------------------------------
    # Audio thread
    # ------------------------------------------------------------------

    def update_frequency(self):
        self._running = True
        while self._running:

            if self.mode == "simple":
                # --- simple mode: existing behaviour, completely unchanged ---
                if self.is_playing1:
                    self.current_freq1 = float(self.target_freq1)
                    self.synth1.setFreq(self.current_freq1)
                    if not self._was_playing1:
                        self.synth1.mul = 0.5
                        self.synth1.play()
                        self._was_playing1 = True
                else:
                    if self._was_playing1:
                        self.synth1.stop()
                        self._was_playing1 = False

                if self.is_playing2:
                    self.current_freq2 = float(self.target_freq2)
                    self.synth2.setFreq(self.current_freq2)
                    if not self._was_playing2:
                        self.synth2.mul = 0.5
                        self.synth2.play()
                        self._was_playing2 = True
                else:
                    if self._was_playing2:
                        self.synth2.stop()
                        self._was_playing2 = False

            else:
                # --- gesture mode: tick the voice schedulers ---
                freq1 = self.gesture_voice1.tick()
                if freq1 is not None:
                    self.synth1.setFreq(float(freq1))
                    if not self._was_playing1:
                        self.synth1.mul = 0.5
                        self.synth1.play()
                        self._was_playing1 = True
                else:
                    if self._was_playing1:
                        self.synth1.stop()
                        self._was_playing1 = False

                freq2 = self.gesture_voice2.tick()
                if freq2 is not None:
                    self.synth2.setFreq(float(freq2))
                    if not self._was_playing2:
                        self.synth2.mul = 0.5
                        self.synth2.play()
                        self._was_playing2 = True
                else:
                    if self._was_playing2:
                        self.synth2.stop()
                        self._was_playing2 = False

            # LP filter applies in both modes
            self.current_lpfreq = float(self.lpfreq)
            self.lp1.setFreq(self.current_lpfreq)
            self.lp2.setFreq(self.current_lpfreq)

            time.sleep(0.005)

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_mode(self, mode):
        """Switch between 'simple' and 'gesture', cleaning up state."""
        self.is_playing1 = False
        self.is_playing2 = False
        self.gesture_voice1.deactivate()
        self.gesture_voice2.deactivate()
        self.mode = mode

    # ------------------------------------------------------------------
    # Simple mode API (unchanged)
    # ------------------------------------------------------------------

    def get_closest_scale_freq(self, center_y):
        num_notes  = self.scale.num_notes
        note_index = int(center_y * num_notes)
        note_index = min(max(note_index, 0), num_notes - 1)
        return self.scale.scale[note_index]

    def get_num_notes(self):
        return self.scale.num_notes

    def get_target_freq1(self):
        return self.target_freq1

    def set_target_freq1(self, freq):
        self.target_freq1 = freq

    def get_target_freq2(self):
        return self.target_freq2

    def set_target_freq2(self, freq):
        self.target_freq2 = freq

    def set_is_playing1(self, state):
        self.is_playing1 = state

    def set_is_playing2(self, state):
        self.is_playing2 = state

    def set_lpfreq(self, freq):
        self.lpfreq = freq

    # ------------------------------------------------------------------
    # Gesture mode API
    # ------------------------------------------------------------------

    def get_home_index(self, center_y):
        """
        Maps the hand's Y position to a scale index to use as the home note.
        Returns an index rather than a frequency — gesture mode works in
        index space so the cursor can step through the scale naturally.
        """
        num_notes  = self.scale.num_notes
        note_index = int(center_y * num_notes)
        return min(max(note_index, 0), num_notes - 1)

    def activate_gesture1(self, home_index):
        self.gesture_voice1.activate(home_index)

    def activate_gesture2(self, home_index):
        self.gesture_voice2.activate(home_index)

    def deactivate_gesture1(self):
        self.gesture_voice1.deactivate()

    def deactivate_gesture2(self):
        self.gesture_voice2.deactivate()

    def set_gesture_velocity1(self, vx, vy, speed):
        self.gesture_voice1.set_velocity(vx, vy, speed)

    def set_gesture_velocity2(self, vx, vy, speed):
        self.gesture_voice2.set_velocity(vx, vy, speed)

    def get_gesture_note_index1(self):
        """Current scale index being played by gesture voice 1 (for lines display)."""
        idx = self.gesture_voice1.home_index + self.gesture_voice1.cursor
        return max(0, min(len(self.scale.scale) - 1, idx))

    def get_gesture_note_index2(self):
        """Current scale index being played by gesture voice 2 (for lines display)."""
        idx = self.gesture_voice2.home_index + self.gesture_voice2.cursor
        return max(0, min(len(self.scale.scale) - 1, idx))

    # ------------------------------------------------------------------
    # Scale switching — updates both voices so they stay in sync
    # ------------------------------------------------------------------

    def set_scale(self, name):
        self.scale.set_scale(name)
        self.gesture_voice1.set_scale(self.scale)
        self.gesture_voice2.set_scale(self.scale)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        self._running = False
        time.sleep(0.02)
        self.s.stop()