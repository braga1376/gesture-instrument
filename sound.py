import numpy as np
from pyo import *

C_PENTATONIC_SCALE = [
    1046.50, 880.00, 783.99, 659.25, 587.33,  # C6 to B5
    523.25, 440.00, 392.00, 329.63, 293.66,   # C5 to B4
    261.63 # C4 to B3
]
C_PENTATONIC_SCALE_DICT = {
    "1046.50": "C6",
    "880.00": "A5",
    "783.99": "G5",
    "659.25": "E5",
    "587.33": "D5",
    "523.25": "C5",
    "440.00": "A4",
    "392.00": "G4",
    "329.63": "E4",
    "293.66": "D4",
    "261.63": "C4",
}

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

class scale:
    def __init__(self, name="diatonic"):
        self.name = name
        if name == "pentatonic" or name == "Pentatonic":
            self.scale = C_PENTATONIC_SCALE
            self.scale_dict = C_PENTATONIC_SCALE_DICT
        elif name == "diatonic" or name == "Diatonic":
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
        if name == "pentatonic":
            self.scale = C_PENTATONIC_SCALE
            self.scale_dict = C_PENTATONIC_SCALE_DICT
        elif name == "diatonic":
            self.scale = C_MAJOR_SCALE
            self.scale_dict = C_MAJOR_SCALE_DICT
        else:
            raise ValueError("Scale not recognized. Use 'pentatonic' or 'diatonic'.")
        self.num_notes = len(self.scale)

class SoundManager:
    def __init__(self):
        self.scale = scale("pentatonic")

        self.current_freq1 = self.scale.scale[0]
        self.target_freq1 = self.scale.scale[0]
        self.is_playing1 = False

        self.current_freq2 = self.scale.scale[0]
        self.target_freq2 = self.scale.scale[0]
        self.is_playing2 = False

        self.current_lpfreq = 1000
        self.lpfreq = 1000

        self.num_notes = self.scale.num_notes

    def init_sound(self):
        self.s = Server(nchnls=2, duplex=0).boot()
        self.s.start()

        # Synth 1 with stereo panning
        lfo1 = Sine(0.1).range(0.1, 0.75)
        self.synth1 = SuperSaw(freq=self.current_freq1, detune=lfo1, bal=0.7, mul=0.5)
        self.lp1 = ButLP(self.synth1, freq=self.current_lpfreq)
        self.panned1 = Pan(self.lp1, outs=2, pan=0.5).out()  

        # Synth 2 with stereo panning
        lfo2 = Sine(0.1).range(0.1, 0.75)
        self.synth2 = SuperSaw(freq=self.current_freq2, detune=lfo2, bal=0.7, mul=0.5)
        self.lp2 = ButLP(self.synth2, freq=self.current_lpfreq)
        self.panned2 = Pan(self.lp2, outs=2, pan=0.5).out() 

        # Reverb and limiter
        self.reverb1 = Freeverb(self.lp1, size=0.6, damp=0.5, bal=0.5)
        self.reverb2 = Freeverb(self.lp2, size=0.6, damp=0.5, bal=0.5)
        self.limiter = Compress(self.panned1 + self.panned2 + self.reverb1 + self.reverb2, thresh=-20, ratio=4).out()

    def update_frequency(self):
        while True:
            if self.is_playing1:
                self.current_freq1 = float(self.target_freq1)
                self.synth1.setFreq(self.current_freq1)
                self.synth1.mul = 0.5
                self.synth1.play()
            else:
                self.synth1.stop()

            if self.is_playing2:
                self.current_freq2 = float(self.target_freq2)
                self.synth2.setFreq(self.current_freq2)
                self.synth2.mul = 0.5
                self.synth2.play()
            else:
                self.synth2.stop()

            self.current_lpfreq = float(self.lpfreq)
            self.lp1.freq = self.current_lpfreq
            self.lp2.freq = self.current_lpfreq

            time.sleep(0.005)

    def get_closest_scale_freq(self, center_y):
        note_index = int(center_y * self.num_notes)
        note_index = min(max(note_index, 0), self.num_notes - 1)
        return self.scale.scale[note_index]
    
    def get_num_notes(self):
        return self.num_notes
    
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
    
    def stop_sound(self):
        self.s.stop()