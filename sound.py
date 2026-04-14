import time
import numpy as np
from pyo import *

# ---------------------------------------------------------------------------
# Scale data — extended to ~4 octaves (C3 to C7)
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

TIMBRES = ['Synth', 'Pad', 'Marimba']


# ---------------------------------------------------------------------------
# Scale class
# ---------------------------------------------------------------------------

class scale:
    def __init__(self, name="pentatonic"):
        self.name = name
        self._apply(name)

    def _apply(self, name):
        if name in ("pentatonic", "Pentatonic"):
            self.scale      = C_PENTATONIC_SCALE
            self.scale_dict = C_PENTATONIC_SCALE_DICT
        elif name in ("diatonic", "Diatonic"):
            self.scale      = C_MAJOR_SCALE
            self.scale_dict = C_MAJOR_SCALE_DICT
        else:
            raise ValueError("Scale not recognized. Use 'pentatonic' or 'diatonic'.")
        self.num_notes = len(self.scale)

    def get_scale(self):       return self.scale
    def get_scale_dict(self):  return self.scale_dict
    def get_num_notes(self):   return self.num_notes

    def set_scale(self, name):
        self.name = name
        self._apply(name)


# ---------------------------------------------------------------------------
# Voice — wraps one synth channel (oscillator + effects chain).
#
# Three timbres:
#   Synth   — SuperSaw with LFO detune.  Rich and buzzy.
#   Pad     — Sine through a Chorus.     Warm and lush.
#   Marimba — Sine with ADSR envelope.  Percussive; envelope retriggered
#             on each new note so repeated notes sound distinct.
# ---------------------------------------------------------------------------

class Voice:
    def __init__(self, timbre, freq, lpfreq):
        self.timbre       = timbre
        self._freq        = freq
        self._was_playing = False
        self._build(freq, lpfreq)

    def _build(self, freq, lpfreq):
        # Fader controls amplitude with a short fade-in and fade-out,
        # preventing the click/scratch that happens when audio is cut abruptly
        self._fader = Fader(fadein=0.01, fadeout=0.03, dur=0, mul=1)

        if self.timbre == 'Synth':
            lfo       = Sine(0.1).range(0.1, 0.75)
            self._osc = SuperSaw(freq=freq, detune=lfo, bal=0.7, mul=self._fader * 0.5)
            self._env = None
            out       = self._osc

        elif self.timbre == 'Pad':
            self._osc    = Sine(freq=freq, mul=self._fader * 0.45)
            self._chorus = Chorus(self._osc, depth=1.5, feedback=0.25, bal=0.7)
            self._env    = None
            out          = self._chorus

        elif self.timbre == 'Marimba':
            self._env = Adsr(attack=0.008, decay=0.45, sustain=0.0,
                             release=0.08, dur=0.7, mul=self._fader * 0.5)
            self._osc = Sine(freq=freq, mul=self._env)
            out       = self._osc

        self.lp     = ButLP(out, freq=lpfreq)
        self.pan    = Pan(self.lp, outs=2, pan=0.5).out()
        self.reverb = Freeverb(self.lp, size=0.6, damp=0.5, bal=0.5)

        # Keep the oscillator running at all times — amplitude is gated
        # entirely by the Fader, which starts at zero
        self._osc.play()

    def play(self, freq):
        """Fade in at the given frequency."""
        self._freq = freq
        self._osc.setFreq(freq)
        if self.timbre == 'Marimba':
            self._env.play()
        self._fader.play()
        self._was_playing = True

    def stop(self):
        """Fade out gracefully — no abrupt cut."""
        self._fader.stop()
        self._was_playing = False

    def set_freq(self, freq):
        """
        Update frequency. For Marimba, also retriggeres the envelope when
        the note changes so each new pitch has a fresh attack.
        """
        if freq == self._freq:
            return
        self._freq = freq
        self._osc.setFreq(freq)
        if self.timbre == 'Marimba' and self._was_playing:
            self._env.play()

    def retrigger(self, freq):
        """
        Force a new note attack at the given frequency regardless of whether
        the pitch changed. Used by gesture mode so each beat fires a fresh
        envelope on Marimba, while Synth/Pad just update their frequency.
        """
        self._freq = freq
        self._osc.setFreq(freq)
        if self.timbre == 'Marimba' and self._was_playing:
            self._env.play()

    def set_lp_freq(self, freq):
        self.lp.setFreq(freq)

    @property
    def was_playing(self):
        return self._was_playing


# ---------------------------------------------------------------------------
# GestureVoice — note scheduler for one hand in gesture mode
# ---------------------------------------------------------------------------

class GestureVoice:
    SPEED_THRESHOLD = 0.08    # below this the hand is considered still → silence
    MAX_SPEED       = 2.0     # speed at which note rate hits its ceiling
    MIN_INTERVAL    = 0.1    # seconds between notes at max speed (~14 notes/sec)
    MAX_INTERVAL    = 0.3     # seconds between notes at min speed (~1.7 notes/sec)
    MAX_STEP        = 3       # largest scale-step jump per note at full vertical speed

    def __init__(self, scale_obj):
        self.scale      = scale_obj
        self.cursor     = 0
        self.home_index = len(scale_obj.scale) // 2
        self.is_active  = False
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self._last_note_time = 0.0
        self._current_freq   = None

    def set_scale(self, scale_obj):
        self.scale      = scale_obj
        self.home_index = min(self.home_index, len(scale_obj.scale) - 1)
        self.cursor     = 0

    def activate(self, home_index):
        self.home_index      = home_index
        self.cursor          = 0
        self.is_active       = True
        self._last_note_time = 0.0
        self._current_freq   = None

    def deactivate(self):
        self.is_active     = False
        self._current_freq = None

    def set_velocity(self, vx, vy, speed):
        self.vx    = vx
        self.vy    = vy
        self.speed = speed

    def tick(self):
        """
        Called from the audio thread each loop iteration.

        Returns (freq, is_new_beat) where:
          freq         — frequency to play, or None for silence
          is_new_beat  — True only when the scheduler fires a new note this tick.
                         False when sustaining the current note or silent.

        The is_new_beat flag lets the audio thread retrigger Marimba envelopes
        on each beat without retriggering for every sustain tick.
        """
        if not self.is_active or self.speed < self.SPEED_THRESHOLD:
            self._current_freq = None
            return (None, False)

        speed_norm = min(self.speed / self.MAX_SPEED, 1.0)
        # Square root curve: spreads the slow end of the range wider so the
        # instrument feels responsive across its full dynamic range, rather
        # than everything collapsing into the fast end of the interval scale
        interval   = self.MAX_INTERVAL - (self.MAX_INTERVAL - self.MIN_INTERVAL) * np.sqrt(speed_norm)

        now = time.time()
        if now - self._last_note_time < interval:
            return (self._current_freq, False)   # sustain current note

        # --- new beat ---
        self._last_note_time = now

        vy_norm = (-self.vy / self.speed) if self.speed > 0 else 0.0
        step    = round(-vy_norm * self.MAX_STEP)

        self.cursor += step
        scale_len   = len(self.scale.scale)
        self.cursor = max(-self.home_index, min(scale_len - 1 - self.home_index, self.cursor))

        note_index         = max(0, min(scale_len - 1, self.home_index + self.cursor))
        self._current_freq = self.scale.scale[note_index]
        return (self._current_freq, True)


# ---------------------------------------------------------------------------
# SoundManager
# ---------------------------------------------------------------------------

class SoundManager:
    def __init__(self):
        self.scale  = scale("pentatonic")
        self.mode   = "simple"
        self.timbre = "Synth"

        # Simple mode frequency targets
        self.current_freq1 = self.scale.scale[0]
        self.target_freq1  = self.scale.scale[0]
        self.is_playing1   = False

        self.current_freq2 = self.scale.scale[0]
        self.target_freq2  = self.scale.scale[0]
        self.is_playing2   = False

        self.current_lpfreq = 1000
        self.lpfreq         = 1000
        self._running       = False

        # Gesture voices
        self.gesture_voice1 = GestureVoice(self.scale)
        self.gesture_voice2 = GestureVoice(self.scale)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_sound(self):
        self.s = Server(nchnls=2, duplex=0).boot()
        self.s.start()
        self._build_chain()

    def _build_chain(self):
        """Build (or rebuild) the full signal chain for the current timbre."""
        self.voice1 = Voice(self.timbre, self.current_freq1, self.current_lpfreq)
        self.voice2 = Voice(self.timbre, self.current_freq2, self.current_lpfreq)
        self.limiter = Compress(
            self.voice1.pan + self.voice2.pan +
            self.voice1.reverb + self.voice2.reverb,
            thresh=-20, ratio=4
        ).out()

    # ------------------------------------------------------------------
    # Audio thread
    # ------------------------------------------------------------------

    def update_frequency(self):
        self._running = True
        while self._running:

            if self.mode == "simple":
                # Voice 1
                if self.is_playing1:
                    freq = float(self.target_freq1)
                    if not self.voice1.was_playing:
                        self.voice1.play(freq)
                    else:
                        self.voice1.set_freq(freq)
                    self.current_freq1 = self.voice1._freq
                else:
                    if self.voice1.was_playing:
                        self.voice1.stop()

                # Voice 2
                if self.is_playing2:
                    freq = float(self.target_freq2)
                    if not self.voice2.was_playing:
                        self.voice2.play(freq)
                    else:
                        self.voice2.set_freq(freq)
                    self.current_freq2 = self.voice2._freq
                else:
                    if self.voice2.was_playing:
                        self.voice2.stop()

            else:
                # Gesture mode — tick() returns (freq, is_new_beat)
                freq1, beat1 = self.gesture_voice1.tick()
                if freq1 is not None:
                    if not self.voice1.was_playing:
                        self.voice1.play(float(freq1))
                    elif beat1:
                        # New scheduled beat: retrigger (matters for Marimba)
                        self.voice1.retrigger(float(freq1))
                    else:
                        self.voice1.set_freq(float(freq1))
                else:
                    if self.voice1.was_playing:
                        self.voice1.stop()

                freq2, beat2 = self.gesture_voice2.tick()
                if freq2 is not None:
                    if not self.voice2.was_playing:
                        self.voice2.play(float(freq2))
                    elif beat2:
                        self.voice2.retrigger(float(freq2))
                    else:
                        self.voice2.set_freq(float(freq2))
                else:
                    if self.voice2.was_playing:
                        self.voice2.stop()

            # LP filter applies in both modes
            lp_freq = float(self.lpfreq)
            self.voice1.set_lp_freq(lp_freq)
            self.voice2.set_lp_freq(lp_freq)

            time.sleep(0.005)

    # ------------------------------------------------------------------
    # Mode and timbre switching
    # ------------------------------------------------------------------

    def set_mode(self, mode):
        """Switch between 'simple' and 'gesture', stopping all voices cleanly."""
        self.is_playing1 = False
        self.is_playing2 = False
        self.gesture_voice1.deactivate()
        self.gesture_voice2.deactivate()
        self.mode = mode

    def set_timbre(self, timbre):
        """
        Switch to a different timbre. Rebuilds the signal chain so there is
        a brief silence during the switch — intentional and clean.
        """
        # Stop everything first
        self.voice1.stop()
        self.voice2.stop()
        self.is_playing1 = False
        self.is_playing2 = False
        self.gesture_voice1.deactivate()
        self.gesture_voice2.deactivate()

        # Tear down limiter before rebuilding (stops output)
        self.limiter.stop()

        self.timbre = timbre
        self._build_chain()

    # ------------------------------------------------------------------
    # Scale switching — updates gesture voices so they stay in sync
    # ------------------------------------------------------------------

    def set_scale(self, name):
        self.scale.set_scale(name)
        self.gesture_voice1.set_scale(self.scale)
        self.gesture_voice2.set_scale(self.scale)

    # ------------------------------------------------------------------
    # Simple mode API
    # ------------------------------------------------------------------

    def get_closest_scale_freq(self, center_y):
        num_notes  = self.scale.num_notes
        note_index = int(center_y * num_notes)
        return self.scale.scale[min(max(note_index, 0), num_notes - 1)]

    def get_num_notes(self):   return self.scale.num_notes
    def get_target_freq1(self): return self.target_freq1
    def get_target_freq2(self): return self.target_freq2
    def set_target_freq1(self, freq): self.target_freq1 = freq
    def set_target_freq2(self, freq): self.target_freq2 = freq
    def set_is_playing1(self, state): self.is_playing1 = state
    def set_is_playing2(self, state): self.is_playing2 = state
    def set_lpfreq(self, freq):       self.lpfreq = freq

    # ------------------------------------------------------------------
    # Gesture mode API
    # ------------------------------------------------------------------

    def get_home_index(self, center_y):
        num_notes  = self.scale.num_notes
        note_index = int(center_y * num_notes)
        return min(max(note_index, 0), num_notes - 1)

    def activate_gesture1(self, home_index):  self.gesture_voice1.activate(home_index)
    def activate_gesture2(self, home_index):  self.gesture_voice2.activate(home_index)
    def deactivate_gesture1(self):            self.gesture_voice1.deactivate()
    def deactivate_gesture2(self):            self.gesture_voice2.deactivate()

    def set_gesture_velocity1(self, vx, vy, speed):
        self.gesture_voice1.set_velocity(vx, vy, speed)

    def set_gesture_velocity2(self, vx, vy, speed):
        self.gesture_voice2.set_velocity(vx, vy, speed)

    def get_gesture_note_index1(self):
        idx = self.gesture_voice1.home_index + self.gesture_voice1.cursor
        return max(0, min(len(self.scale.scale) - 1, idx))

    def get_gesture_note_index2(self):
        idx = self.gesture_voice2.home_index + self.gesture_voice2.cursor
        return max(0, min(len(self.scale.scale) - 1, idx))

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        self._running = False
        time.sleep(0.02)
        self.s.stop()