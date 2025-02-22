import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from typing import Tuple
from RecognizeFace import recognize_face, load_encodings
import base64
from io import BytesIO

# Constants
OVAL_WIDTH_RATIO = 0.2
OVAL_HEIGHT_RATIO = 0.35
OVAL_COLOR = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def draw_oval_guide(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.zeros_like(frame)
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        axes = (int(width * OVAL_WIDTH_RATIO), int(height * OVAL_HEIGHT_RATIO))
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
        overlay = frame.copy()
        cv2.ellipse(overlay, center, axes, 0, 0, 360, OVAL_COLOR, 2)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        return frame, mask

    def draw_scanning_animation(self, frame: np.ndarray, progress: float) -> np.ndarray:
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        axes = (int(width * OVAL_WIDTH_RATIO), int(height * OVAL_HEIGHT_RATIO))

        start_angle = -90
        end_angle = -90 + (360 * progress)
        cv2.ellipse(frame, center, axes, 0, start_angle, end_angle, OVAL_COLOR, 2)

        text = f"{3 - int(progress * 3)}"
        text_size = cv2.getTextSize(text, FONT, 1.5, 2)[0]
        text_x = center[0] - (text_size[0] // 2)
        text_y = center[1] + (text_size[1] // 2)

        cv2.putText(frame, text, (text_x, text_y), FONT, 1.5, OVAL_COLOR, 2)
        return frame

    def check_face_in_oval(self, frame: np.ndarray, mask: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return False

        for x, y, w, h in faces:
            face_center = (x + w // 2, y + h // 2)
            if mask[face_center[1], face_center[0]].any():
                return True
        return False


def initialize_camera():
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main():
    st.title("Real-time Face Recognition")

    # Load face encodings
    load_encodings()

    detector = FaceDetector()

    # Create placeholders
    frame_placeholder = st.empty()
    result_placeholder = st.empty()
    status_placeholder = st.empty()

    # Session state for camera control
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = True
        st.session_state.cap = initialize_camera()

    # Create restart button
    if st.button("Restart Camera", key="restart_button"):
        if st.session_state.cap is not None and st.session_state.cap.isOpened():
            st.session_state.cap.release()
        st.session_state.cap = initialize_camera()
        st.session_state.camera_active = True

    # State variables
    countdown_active = False
    countdown_start = 0
    last_recognition_time = 0
    recognition_cooldown = 2
    face_detection_start = 0
    continuous_detection_required = 3
    last_face_detected_time = 0

    while st.session_state.camera_active:
        if not st.session_state.cap.isOpened():
            st.error("Camera disconnected")
            st.session_state.camera_active = False
            break

        ret, frame = st.session_state.cap.read()
        if not ret:
            st.error("Failed to grab frame")
            st.session_state.camera_active = False
            break

        try:
            frame = cv2.flip(frame, 1)
            frame_with_oval, mask = detector.draw_oval_guide(frame)
            face_in_oval = detector.check_face_in_oval(frame, mask)
            current_time = time.time()

            # Face detection logic
            if face_in_oval:
                if face_detection_start == 0:
                    face_detection_start = current_time
                    status_placeholder.info("Keep your face steady...")
                last_face_detected_time = current_time
            else:
                if current_time - last_face_detected_time > 2:
                    face_detection_start = 0
                    countdown_active = False
                    status_placeholder.warning("Face not detected in oval guide")

            # Start countdown only after continuous face detection
            if face_detection_start > 3 and not countdown_active:
                if current_time - face_detection_start >= continuous_detection_required:
                    if current_time - last_recognition_time > recognition_cooldown:
                        countdown_active = True
                        countdown_start = current_time

            if countdown_active:
                elapsed = current_time - countdown_start
                if elapsed > 1:
                    if st.session_state.cap.isOpened() and face_in_oval:
                        try:
                            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = recognize_face(rgb_frame)

                            # Parse the error detail from HTTPException
                            if isinstance(results, dict):
                                if (
                                    "detail" in results
                                ):  # Handle FastAPI HTTPException response
                                    error_message = results["detail"].lower()
                                    if "spoof" in error_message:
                                        status_placeholder.warning(
                                            "⚠️ Spoof detected! Please use a real face instead of an image."
                                        )
                                        result_placeholder.empty()
                                    else:
                                        status_placeholder.error(
                                            f"Recognition error: {results['detail']}"
                                        )
                                        result_placeholder.empty()
                                elif "name" in results:  # Successful recognition
                                    result_placeholder.success(
                                        f"Recognized: {results['name']} (Distance: {results['distance']:.2f})"
                                    )
                                    status_placeholder.empty()

                        except Exception as e:
                            error_text = str(e)
                            if hasattr(e, "detail"):  # FastAPI HTTPException
                                error_text = e.detail

                            if "spoof" in error_text.lower():
                                status_placeholder.warning(
                                    "⚠️ Spoof detected! Please use a real face instead of an image."
                                )
                                result_placeholder.empty()
                            elif "face could not be detected" in error_text.lower():
                                status_placeholder.warning(
                                    "Face not detected. Please stay within the oval guide."
                                )
                                result_placeholder.empty()
                            else:
                                status_placeholder.error(
                                    f"Recognition error: {error_text}"
                                )
                                result_placeholder.empty()

                    countdown_active = False
                    face_detection_start = 0
                    last_recognition_time = current_time
                else:
                    # pass
                    progress = elapsed / 3
                    frame_with_oval = detector.draw_scanning_animation(
                        frame_with_oval, progress
                    )

            frame_placeholder.image(
                cv2.cvtColor(frame_with_oval, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.session_state.camera_active = False
            break

        time.sleep(0.01)

    # Ensure camera is properly released
    if st.session_state.cap is not None and st.session_state.cap.isOpened():
        st.session_state.cap.release()


if __name__ == "__main__":
    main()
