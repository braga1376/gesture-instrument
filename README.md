# Gesture-Based Musical Instrument

This project uses a webcam to detect hand gestures and control sound frequencies. The project uses pyo for sound, OpenCV for video capture and MediaPipe for hand tracking.

## Requirements

- Python 3.9
- OpenCV
- MediaPipe
- NumPy
- pyo

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/gesture-instrument.git
    cd gesture-instrument
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python mediapipe numpy pyo
    ```

## Usage

1. Run the `main.py` script:
    ```sh
    python main.py
    ```

2. The script will open a webcam feed and display a GUI with the following controls:
- Sound On/Off: Toggle sound playback.
- Select Scale: Choose between Pentatonic and Diatonic scales.
- Marks On/Off: Toggle visual markers for hand tracking.
- Note Lines On/Off: Toggle horizontal lines representing musical notes.
- Lip Control On/Off: Toggle lip-controlled LowPass filter.
start detecting hand gestures. 
- Distance Threshold Slider: Adjust the threshold distance between the thumb and index finger to control sound playback.
3. Use hand gestures to control the sound:
- The thumb and index finger determine if the sound is played.
- The vertical position of the hand determines the frequency of the sound.

## How It Works

- The script captures video from the webcam using OpenCV.
- MediaPipe is used to detect hand landmarks.
- The distance between the thumb and index finger is calculated to determine if the sound should be played or not.
- The vertical position of the hand is used to control the frequency of the sound.
- The `pyo` library is used to generate and play the sound.
