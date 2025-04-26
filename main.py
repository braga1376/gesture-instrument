# import cv2
# import mediapipe as mp
# import threading
# import argparse
# from hand_tracking import HandTracker
# from face_tracking import FaceTracker
# from sound import SoundManager


# def main():
#     parser = argparse.ArgumentParser(description="Gesture-based instrument with optional lip-controlled LowPass filter.")
#     parser.add_argument('--lip_control', action='store_true', help="Enable lip-controlled LowPass filter")
#     args = parser.parse_args()

#     sound_manager = SoundManager()
#     sound_manager.init_sound()

#     freq_thread = threading.Thread(target=sound_manager.update_frequency)
#     freq_thread.daemon = True
#     freq_thread.start()

#     cap = cv2.VideoCapture(0)
#     cv2.namedWindow("CamOutput", cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty("CamOutput", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

#     hand_tracker = HandTracker(sound_manager)
#     face_tracker = FaceTracker(sound_manager)

#     while True:
#         ret, img = cap.read()
#         if not ret:
#             break

#         img = cv2.flip(img, 1)
#         h, w, c = img.shape

#         num_notes = sound_manager.scale.num_notes
#         for i in range(num_notes):
#             y = int(i * h / num_notes)
#             overlay = img.copy()
#             cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 2)
#             alpha = 0.3
#             cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

#         recHands = hand_tracker.process(img)
#         recFace = face_tracker.process(img)

#         hand_tracker.update_hands(recHands, img, h, w)
#         if args.lip_control:
#             face_tracker.update_face(recFace, img, h, w)

#         cv2.imshow("CamOutput", img)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     sound_manager.stop_sound()

# if __name__ == "__main__":
#     main()


import cv2
import mediapipe as mp
import threading
import argparse
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from hand_tracking import HandTracker
from face_tracking import FaceTracker
from sound import SoundManager

marks = False
lines = False
sound = False

def setup_tkinter_gui(sound_manager, cap, args):
    global marks
    global lines
    global sound

    root = tk.Tk()
    root.title("Gesture Instrument Controller")

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        return
    video_height, video_width, _ = frame.shape
    video_aspect_ratio = video_width / video_height

    def update_scale(event):
        selected_scale = scale_selector.get()
        sound_manager.scale.set_scale(selected_scale.lower())
        print(f"Scale changed to: {selected_scale}")

    control_frame = tk.Frame(root)
    control_frame.pack(pady=5)

    scale_label = ttk.Label(control_frame, text="Select Scale:")
    scale_label.pack(side=tk.LEFT, padx=5)

    scale_selector = ttk.Combobox(control_frame, values=["Pentatonic", "Diatonic"], state="readonly")
    scale_selector.set("Pentatonic")  # Default value
    scale_selector.bind("<<ComboboxSelected>>", update_scale)
    scale_selector.pack(side=tk.LEFT, padx=5)

    def toggle_lip_control():
        args.lip_control = not args.lip_control
        lip_control_button.config(text=f"Lip Control: {'ON' if args.lip_control else 'OFF'}")
        print(f"Lip Control {'enabled' if args.lip_control else 'disabled'}")

    lip_control_button = ttk.Button(control_frame, text="Lip Control: OFF", command=toggle_lip_control)
    lip_control_button.pack(side=tk.LEFT, padx=5)

    def toggle_marks():
        global marks
        marks = not marks
        marks_button.config(text=f"Marks: {'ON' if marks else 'OFF'}")
        print(f"Marks {'enabled' if marks else 'disabled'}")

    marks_button = ttk.Button(control_frame, text="Marks: OFF", command=toggle_marks)
    marks_button.pack(side=tk.LEFT, padx=5)

    def toggle_lines():
        global lines
        lines = not lines
        lines_button.config(text=f"Note Lines: {'ON' if lines else 'OFF'}")
        print(f"Note Lines {'enabled' if marks else 'disabled'}")

    lines_button = ttk.Button(control_frame, text="Note Lines: OFF", command=toggle_lines)
    lines_button.pack(side=tk.LEFT, padx=5)

    video_canvas = tk.Canvas(root, width=video_width, height=video_height)
    video_canvas.pack(fill=tk.BOTH, expand=True)

    def update_video():
        ret, img = cap.read()
        if ret:
            img = cv2.flip(img, 1)
            h, w, c = img.shape

            if lines:
                num_notes = sound_manager.scale.num_notes
                for i in range(num_notes):
                    y = int(i * h / num_notes)
                    overlay = img.copy()
                    cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 2)
                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            recHands = hand_tracker.process(img)
            recFace = face_tracker.process(img)

            hand_tracker.update_hands(recHands, img, h, w, marks, lines)
            if args.lip_control:
                face_tracker.update_face(recFace, img, h, w, marks)

            canvas_width = video_canvas.winfo_width()
            canvas_height = video_canvas.winfo_height()
            canvas_aspect_ratio = canvas_width / canvas_height
            if canvas_aspect_ratio > video_aspect_ratio:
                new_height = canvas_height
                new_width = int(new_height * video_aspect_ratio)
            else:
                new_width = canvas_width
                new_height = int(new_width / video_aspect_ratio)
            
            if canvas_width > 1 and canvas_height > 1:
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            video_canvas.delete("all")
            video_canvas.create_image(canvas_width // 2, canvas_height // 2, anchor=tk.CENTER, image=imgtk)
            video_canvas.imgtk = imgtk 

        root.after(10, update_video)

    update_video()
    root.mainloop()

def main():
    parser = argparse.ArgumentParser(description="Gesture-based instrument with optional lip-controlled LowPass filter.")
    parser.add_argument('--lip_control', action='store_true', help="Enable lip-controlled LowPass filter")
    args = parser.parse_args()

    sound_manager = SoundManager()
    sound_manager.init_sound()

    freq_thread = threading.Thread(target=sound_manager.update_frequency)
    freq_thread.daemon = True
    freq_thread.start()

    cap = cv2.VideoCapture(0)

    global hand_tracker, face_tracker
    hand_tracker = HandTracker(sound_manager)
    face_tracker = FaceTracker(sound_manager)

    setup_tkinter_gui(sound_manager, cap, args)

    cap.release()
    sound_manager.stop_sound()

if __name__ == "__main__":
    main()