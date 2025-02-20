from deepface import DeepFace
import cv2
from PIL import Image

IMG_PATH = "test/mahsa.jpg"
img = cv2.imread(IMG_PATH)

face = DeepFace.extract_faces(img, enforce_detection=False, detector_backend='fastmtcnn')

face_data = face[0]['facial_area']
print(face_data)
# Convert the face array to an 8-bit unsigned integer type
x1, y1, width, height= face_data['x'], face_data['y'], face_data['w'], face_data['h']
cropped_image = img[y1:y1+height, x1:x1+width]
print(cropped_image.dtype)

# Convert the cropped image to RGB format
cropped_image_bgr = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

# Represent the cropped image using DeepFace
encode = DeepFace.represent(cropped_image_bgr, model_name='Facenet512')
print(encode)

