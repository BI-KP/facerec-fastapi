import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from typing import Tuple
from server import register_new_face, recognize_face, load_encodings
import base64
from io import BytesIO
from datetime import datetime

# Constants
OVAL_WIDTH_RATIO = 0.2
OVAL_HEIGHT_RATIO = 0.35
OVAL_COLOR = (255, 255, 0)
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

        text = f"{2 - int(progress * 2)}"
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
        cap.set(cv2.CAP_PROP_FPS, 60)
    return cap


def main():
    st.title("Face Recognition System")

    # Add mode selection
    mode = st.sidebar.selectbox("Select Mode", ["Recognition", "Registration"])

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

    # Registration input field
    if mode == "Registration":
        face_id = st.text_input("Enter Name")

        # Add tabs for different registration methods
        tab1, tab2 = st.tabs(["Camera", "Upload File"])

        with tab1:
            register_button = st.button("Register Face from Camera")

        with tab2:
            uploaded_file = st.file_uploader(
                "Choose an image file", type=["jpg", "jpeg", "png"]
            )
            register_file_button = st.button("Register Face from File")

            if register_file_button and uploaded_file is not None and face_id:
                try:
                    # Convert uploaded file to numpy array
                    file_bytes = np.asarray(
                        bytearray(uploaded_file.read()), dtype=np.uint8
                    )
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Try to register the face
                    success = register_new_face(image_rgb, face_id)
                    if success:
                        st.success(f"Successfully registered {face_id} from file")
                        # Reload encodings
                        load_encodings()
                    else:
                        st.error("Registration failed")
                except Exception as e:
                    st.error(f"Registration error: {str(e)}")

    # State variables
    countdown_active = False
    countdown_start = 0
    last_recognition_time = 0
    recognition_cooldown = 2
    face_detection_start = 0
    continuous_detection_required = 2
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
                    if mode == "Recognition":
                        status_placeholder.info("Keep your face steady...")
                    else:
                        status_placeholder.info(
                            "Ready to register. Click 'Register Face' button."
                        )
                last_face_detected_time = current_time
            else:
                if current_time - last_face_detected_time > 2:
                    face_detection_start = 0
                    countdown_active = False
                    status_placeholder.warning("Face not detected in oval guide")

            # Handle registration from camera
            if mode == "Registration" and register_button and face_id and face_in_oval:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    success = register_new_face(rgb_frame, face_id)
                    if success:
                        status_placeholder.success(f"Successfully registered {face_id}")
                        # Reload encodings after successful registration
                        load_encodings()
                        time.sleep(2)  # Show success message for 2 seconds
                    else:
                        status_placeholder.error("Registration failed")
                except Exception as e:
                    status_placeholder.error(f"Registration error: {str(e)}")

            # Recognition logic (only in Recognition mode)
            if mode == "Recognition":
                # Start countdown only after continuous face detection
                if face_detection_start > 3 and not countdown_active:
                    if (
                        current_time - face_detection_start
                        >= continuous_detection_required
                    ):
                        if current_time - last_recognition_time > recognition_cooldown:
                            countdown_active = True
                            countdown_start = current_time

                if countdown_active:
                    elapsed = current_time - countdown_start
                    if elapsed >= 2:
                        if st.session_state.cap.isOpened() and face_in_oval:
                            try:
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = recognize_face(rgb_frame)

                                # Parse the error detail from HTTPException
                                if isinstance(results, dict):
                                    if "detail" in results:
                                        error_message = results["detail"].lower()
                                        if "spoof" in error_message:
                                            spoof_message = f"""
                                             **SPOOF DETECTED**
                                            - **Time**: {datetime.now().strftime('%H:%M:%S')}
                                            - **Status**: Spoofed image detected
                                            """
                                            # Show in Streamlit
                                            status_placeholder.error(spoof_message)
                                            result_placeholder.empty()
                                    elif "name" in results:
                                        result_placeholder.success(
                                            f"Recognized: {results['name']} (Distance: {results['distance']:.2f})"
                                        )
                                        status_placeholder.empty()

                            except Exception as e:
                                error_text = str(e)
                                if hasattr(e, "detail"):
                                    error_text = e.detail

                                if "spoof" in error_text.lower():
                                    st.markdown("# 🚨 SPOOF DETECTED")

                                    # Display detailed information
                                    st.markdown(
                                        f"""
                                    - **Time**: {datetime.now().strftime('%H:%M:%S')}
                                    - **Error**: {error_text}
                                    """
                                    )

                                    # Clear previous results
                                    result_placeholder.empty()
                                    status_placeholder.error(
                                        "Access Denied: Spoof Detected"
                                    )
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
                        # Modified progress calculation
                        progress = elapsed / 2
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
