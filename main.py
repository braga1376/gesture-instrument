import cv2
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from hand_tracking import HandTracker
from face_tracking import FaceTracker
from sound import SoundManager

sound       = False
marks       = False
lines       = False
lip_control = False

# Shared state between the capture thread and the display thread.
# The capture thread writes; the display thread reads.
_latest_frame      = None
_latest_frame_lock = threading.Lock()


def capture_loop(cap, hand_tracker, face_tracker, sound_manager,
                 distance_threshold_var, running_flag):
    """
    Runs on a background thread. Reads frames from the camera, runs
    MediaPipe, applies tracking logic, and stores the annotated frame
    for the display thread to pick up.

    Decoupling this from the Tkinter main thread means display is never
    blocked waiting for MediaPipe, and MediaPipe is never slowed down
    by Tkinter rendering.
    """
    global _latest_frame, sound, marks, lines, lip_control

    while running_flag[0]:
        ret, img = cap.read()
        if not ret:
            continue

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        if lines:
            num_notes = sound_manager.scale.num_notes
            overlay   = img.copy()
            for i in range(num_notes):
                y = int(i * h / num_notes)
                cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 2)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

        # Downscale a copy for MediaPipe — faster inference without affecting
        # display quality. Landmarks come back as normalised 0–1 coordinates
        # so they map correctly onto the full-resolution frame.
        PROC_WIDTH  = 640
        PROC_HEIGHT = 480
        img_small = cv2.resize(img, (PROC_WIDTH, PROC_HEIGHT),
                               interpolation=cv2.INTER_AREA)

        recHands = hand_tracker.process(img_small)
        recFace  = face_tracker.process(img_small)

        # Pass full-res dimensions so landmark coordinates are scaled correctly
        # when drawing onto the full-res frame
        hand_tracker.update_hands(
            recHands, img, h, w, sound, marks, lines,
            distance_threshold_var[0]
        )
        if lip_control:
            face_tracker.update_face(recFace, img, h, w, marks)

        # Convert to RGB once here so the display thread doesn't have to
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with _latest_frame_lock:
            _latest_frame = img_rgb


def setup_tkinter_gui(sound_manager, cap, hand_tracker, face_tracker):
    global marks, lines, sound, lip_control

    root = tk.Tk()
    root.title("Gesture Instrument Controller")

    # Request a higher frame rate from the camera but leave resolution at
    # native — we downscale only the copy sent to MediaPipe (see capture_loop)
    cap.set(cv2.CAP_PROP_FPS, 60)

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        return
    video_height, video_width, _ = frame.shape
    video_aspect_ratio = video_width / video_height

    # Shared mutable value for distance threshold — written by the slider
    # callback (main thread), read by the capture thread
    distance_threshold_var = [0.07]

    control_frame = tk.Frame(root)
    control_frame.pack(pady=5)

    def toggle_sound():
        global sound
        sound = not sound
        sound_button.config(text=f"Sound: {'ON' if sound else 'OFF'}")
        print(f"Sound {'enabled' if sound else 'disabled'}")

    sound_button = ttk.Button(control_frame, text="Sound: OFF", command=toggle_sound)
    sound_button.pack(side=tk.LEFT, padx=5)

    scale_label = ttk.Label(control_frame, text="Scale:")
    scale_label.pack(side=tk.LEFT, padx=5)

    def update_scale(event):
        sound_manager.set_scale(scale_selector.get().lower())
        print(f"Scale changed to: {scale_selector.get()}")

    scale_selector = ttk.Combobox(control_frame, values=["Pentatonic", "Diatonic"],
                                  state="readonly", width=10)
    scale_selector.set("Pentatonic")
    scale_selector.bind("<<ComboboxSelected>>", update_scale)
    scale_selector.pack(side=tk.LEFT, padx=5)

    def toggle_marks():
        global marks
        marks = not marks
        marks_button.config(text=f"Marks: {'ON' if marks else 'OFF'}")

    marks_button = ttk.Button(control_frame, text="Marks: OFF", command=toggle_marks)
    marks_button.pack(side=tk.LEFT, padx=5)

    def toggle_lines():
        global lines
        lines = not lines
        lines_button.config(text=f"Note Lines: {'ON' if lines else 'OFF'}")

    lines_button = ttk.Button(control_frame, text="Note Lines: OFF", command=toggle_lines)
    lines_button.pack(side=tk.LEFT, padx=5)

    def toggle_lip_control():
        global lip_control
        lip_control = not lip_control
        lip_control_button.config(text=f"Lip Control: {'ON' if lip_control else 'OFF'}")

    lip_control_button = ttk.Button(control_frame, text="Lip Control: OFF",
                                    command=toggle_lip_control)
    lip_control_button.pack(side=tk.LEFT, padx=5)

    def toggle_mode():
        new_mode = "gesture" if sound_manager.mode == "simple" else "simple"
        sound_manager.set_mode(new_mode)
        mode_button.config(text=f"Mode: {'Gesture' if new_mode == 'gesture' else 'Simple'}")

    mode_button = ttk.Button(control_frame, text="Mode: Simple", command=toggle_mode)
    mode_button.pack(side=tk.LEFT, padx=5)

    timbre_label = ttk.Label(control_frame, text="Timbre:")
    timbre_label.pack(side=tk.LEFT, padx=5)

    def update_timbre(event):
        sound_manager.set_timbre(timbre_selector.get())

    timbre_selector = ttk.Combobox(control_frame, values=["Synth", "Pad", "Marimba"],
                                   state="readonly", width=9)
    timbre_selector.set("Synth")
    timbre_selector.bind("<<ComboboxSelected>>", update_timbre)
    timbre_selector.pack(side=tk.LEFT, padx=5)

    def update_distance_threshold(val):
        distance_threshold_var[0] = float(val)

    distance_slider_label = ttk.Label(control_frame, text="Threshold:")
    distance_slider_label.pack(side=tk.LEFT, padx=5)

    distance_threshold_tk = tk.DoubleVar(value=0.07)
    distance_slider = ttk.Scale(control_frame, from_=0.03, to=0.13,
                                orient="horizontal",
                                command=update_distance_threshold,
                                variable=distance_threshold_tk)
    distance_slider.pack(side=tk.LEFT, padx=5)

    video_canvas = tk.Canvas(root, width=video_width, height=video_height)
    video_canvas.pack(fill=tk.BOTH, expand=True)

    # Flag shared with the capture thread — set to False on close
    running_flag = [True]
    after_id     = None

    def on_closing():
        nonlocal after_id
        running_flag[0] = False
        if after_id is not None:
            root.after_cancel(after_id)
        sound_manager.set_is_playing1(False)
        sound_manager.set_is_playing2(False)
        cap.release()
        sound_manager.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start the capture + processing thread
    capture_thread = threading.Thread(
        target=capture_loop,
        args=(cap, hand_tracker, face_tracker, sound_manager,
              distance_threshold_var, running_flag),
        daemon=True
    )
    capture_thread.start()

    def update_display():
        """
        Runs on the main thread. Picks up the latest processed frame from
        the capture thread and renders it. Schedules itself with after(0)
        so it yields to Tkinter's event loop between frames without
        imposing a fixed delay.
        """
        nonlocal after_id
        global _latest_frame

        with _latest_frame_lock:
            frame = _latest_frame

        if frame is not None:
            canvas_width  = video_canvas.winfo_width()
            canvas_height = video_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                canvas_aspect = canvas_width / canvas_height
                if canvas_aspect > video_aspect_ratio:
                    new_h = canvas_height
                    new_w = int(new_h * video_aspect_ratio)
                else:
                    new_w = canvas_width
                    new_h = int(new_w / video_aspect_ratio)

                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            img     = Image.fromarray(frame)
            imgtk   = ImageTk.PhotoImage(image=img)
            video_canvas.delete("all")
            video_canvas.create_image(
                canvas_width // 2, canvas_height // 2,
                anchor=tk.CENTER, image=imgtk
            )
            video_canvas.imgtk = imgtk

        after_id = root.after(8, update_display)

    update_display()
    root.mainloop()


def main():
    sound_manager = SoundManager()
    sound_manager.init_sound()

    freq_thread = threading.Thread(target=sound_manager.update_frequency, daemon=True)
    freq_thread.start()

    cap = cv2.VideoCapture(0)

    hand_tracker = HandTracker(sound_manager)
    face_tracker = FaceTracker(sound_manager)

    setup_tkinter_gui(sound_manager, cap, hand_tracker, face_tracker)


if __name__ == "__main__":
    main()