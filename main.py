import cv2
import mediapipe as mp
import threading
import argparse
from hand_tracking import HandTracker
from face_tracking import FaceTracker
from sound import SoundManager


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
    cv2.namedWindow("CamOutput", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CamOutput", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    hand_tracker = HandTracker(sound_manager)
    face_tracker = FaceTracker(sound_manager)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.flip(img, 1)
        h, w, c = img.shape

        num_notes = sound_manager.scale.num_notes
        for i in range(num_notes):
            y = int(i * h / num_notes)
            overlay = img.copy()
            cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 2)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        recHands = hand_tracker.process(img)
        recFace = face_tracker.process(img)

        hand_tracker.update_hands(recHands, img, h, w)
        if args.lip_control:
            face_tracker.update_face(recFace, img, h, w)

        cv2.imshow("CamOutput", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    sound_manager.stop_sound()

if __name__ == "__main__":
    main()


# import cv2
# import mediapipe as mp
# import threading
# import argparse
# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# from hand_tracking import HandTracker
# from face_tracking import FaceTracker
# from sound import SoundManager

# def setup_tkinter_gui(sound_manager, cap, args):
#     """Sets up the Tkinter GUI with an embedded OpenCV video feed."""
#     root = tk.Tk()
#     root.title("Gesture Instrument Controller")

#     # Function to update the scale
#     def update_scale(event):
#         selected_scale = scale_selector.get()
#         sound_manager.scale.set_scale(selected_scale.lower())  # Update the scale in SoundManager
#         print(f"Scale changed to: {selected_scale}")

#     # Add a dropdown menu for selecting the scale
#     scale_label = ttk.Label(root, text="Select Scale:")
#     scale_label.pack(pady=5)

#     scale_selector = ttk.Combobox(root, values=["Pentatonic", "Diatonic"], state="readonly")
#     scale_selector.set("Pentatonic")  # Default value
#     scale_selector.bind("<<ComboboxSelected>>", update_scale)
#     scale_selector.pack(pady=5)

#     # Create a Canvas for the video feed
#     video_canvas = tk.Canvas(root, width=640, height=480)
#     video_canvas.pack()

#     # Function to update the video feed
#     def update_video():
#         ret, img = cap.read()
#         if ret:
#             img = cv2.flip(img, 1)
#             h, w, c = img.shape

#             # Process the video frame
#             num_notes = sound_manager.scale.num_notes
#             for i in range(num_notes):
#                 y = int(i * h / num_notes)
#                 overlay = img.copy()
#                 cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 2)
#                 alpha = 0.3
#                 cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

#             recHands = hand_tracker.process(img)
#             recFace = face_tracker.process(img)

#             hand_tracker.update_hands(recHands, img, h, w)
#             if args.lip_control:
#                 face_tracker.update_face(recFace, img, h, w)

#             # Convert the frame to a format compatible with Tkinter
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(img)
#             imgtk = ImageTk.PhotoImage(image=img)

#             # Update the Canvas with the new frame
#             video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
#             video_canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

#         # Schedule the next frame update
#         root.after(10, update_video)

#     # Start the video update loop
#     update_video()

#     # Run the Tkinter event loop
#     root.mainloop()

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

#     # Initialize trackers
#     global hand_tracker, face_tracker
#     hand_tracker = HandTracker(sound_manager)
#     face_tracker = FaceTracker(sound_manager)

#     # Run the Tkinter GUI with the OpenCV video feed
#     setup_tkinter_gui(sound_manager, cap, args)

#     # Cleanup
#     cap.release()
#     sound_manager.stop_sound()

# if __name__ == "__main__":
#     main()