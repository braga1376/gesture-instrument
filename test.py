from pyo import *
import tkinter as tk
from tkinter import ttk

s = Server(duplex=0).boot()
s.start()

freq = Sig(440)  # Frequency signal
amp = Sig(0.5)   # Amplitude signal
synth = SuperSaw(freq=freq, detune=0.5, bal=0.7, mul=amp).out()

def update_freq(val):
    freq.value = float(val)

def update_amp(val):
    amp.value = float(val)

root = tk.Tk()
root.title("Pyo Synth Controller")

freq_label = ttk.Label(root, text="Frequency")
freq_label.pack()
freq_slider = ttk.Scale(root, from_=220, to=880, orient="horizontal", command=update_freq)
freq_slider.set(440)
freq_slider.pack()

# Amplitude slider
amp_label = ttk.Label(root, text="Amplitude")
amp_label.pack()
amp_slider = ttk.Scale(root, from_=0, to=1, orient="horizontal", command=update_amp)
amp_slider.set(0.5)
amp_slider.pack()

# Run the Tkinter event loop
root.mainloop()

# Stop the server when the window is closed
s.stop()