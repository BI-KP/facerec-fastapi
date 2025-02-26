# Import required libraries
from deepface import DeepFace  # Face detection and recognition library
import cv2  # OpenCV for image processing
from PIL import Image  # Python Imaging Library for image operations

# Define path to test image
IMG_PATH = "test/mahsa.jpg"
# Load image using OpenCV
img = cv2.imread(IMG_PATH)

# Extract faces from the image using DeepFace with fastmtcnn detector
# enforce_detection=False allows processing even if a face isn't confidently detected
face = DeepFace.extract_faces(img, enforce_detection=False, detector_backend='fastmtcnn')

# Extract facial area coordinates from the first detected face
face_data = face[0]['facial_area']
print(face_data)  # Print the coordinates of the detected face

# Extract the face region from the image using the detected coordinates
x1, y1, width, height = face_data['x'], face_data['y'], face_data['w'], face_data['h']
cropped_image = img[y1:y1+height, x1:x1+width]  # Crop the face from the original image
print(cropped_image.dtype)  # Print the data type of the cropped image

# Convert the cropped image from RGB to BGR color format
# This is necessary because OpenCV uses BGR but DeepFace expects RGB
cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

# Generate face embedding using DeepFace with Facenet512 model
# The embedding is a numerical representation of facial features
encode = DeepFace.represent(cropped_image_bgr, model_name='Facenet512')
print(encode)  # Print the face embedding information

