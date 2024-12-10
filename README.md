# Gesture-Based Musical Instrument

This project uses a webcam to detect hand gestures and control sound frequencies using the `pyo` library. The project leverages OpenCV for video capture and MediaPipe for hand tracking.

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

2. The script will open a webcam feed and start detecting hand gestures. The frequency of the sound will change based on the position of your hands, the thumb and index finger determine if the sound is played.

## How It Works

- The script captures video from the webcam using OpenCV.
- MediaPipe is used to detect hand landmarks.
- The distance between the thumb and index finger is calculated to determine if the sound should be played or not.
- The vertical position of the hand is used to control the frequency of the sound.
- The `pyo` library is used to generate and play the sound.

## Code Overview

- `main.py`: The main script that captures video, processes hand gestures, and controls sound frequencies.

### Key Functions and Variables

- `update_frequency()`: Updates the sound frequency based on hand gestures.
- `INDEX_FINGER_IDX`, `THUMB_IDX`: Indices for the thumb and index finger landmarks.
- `MIN_FREQ`, `MAX_FREQ`: Minimum and maximum frequencies for the sound.
- `current_freq1`, `target_freq1`, `is_playing1`: Variables for the first hand's frequency and play state.
- `current_freq2`, `target_freq2`, `is_playing2`: Variables for the second hand's frequency and play state.