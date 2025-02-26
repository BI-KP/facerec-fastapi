# Import required libraries
import cv2  # OpenCV for image processing
from deepface import DeepFace  # Face detection and recognition library
import numpy as np  # For numerical operations
import pickle  # For saving encodings to file
import os  # For file operations

# Set path to folder containing images to encode
folderPath = "ResizedImages"
# Get list of all image files in the folder
imgPathList = os.listdir(folderPath)
# Load all images using OpenCV
imgList = [cv2.imread(os.path.join(folderPath, path)) for path in imgPathList]

# Extract person IDs from filenames (assuming filename format is "personID.extension")
imgIdList = [os.path.splitext(path)[0] for path in imgPathList]

# Function to crop an image to just the face region
def crop_image_face(img):
    # Detect faces in the image using fastmtcnn detector
    face = DeepFace.extract_faces(img, detector_backend='fastmtcnn', enforce_detection=False, align=True)
    # Extract coordinates of the detected face
    face_data = face[0]['facial_area']
    x1, y1, width, height = face_data['x'], face_data['y'], face_data['w'], face_data['h']
    # Crop the image to the face region
    cropped_image = img[y1:y1+height, x1:x1+width]
    return cropped_image

# Function to generate face encodings for a list of images
def findEncoding(imgList):
    encodeList = []
    for img in imgList:
        # Convert image color format from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Crop the image to just the face region
        img = crop_image_face(img)
        # Convert back to RGB format for DeepFace
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Generate face embedding using Facenet512 model
        encode = DeepFace.represent(img, model_name='Facenet512', detector_backend='fastmtcnn')
        # Extract the embedding vector
        embedding = encode[0]['embedding']
        # Add the embedding to our list
        encodeList.append(embedding)
    return encodeList

# Initial call to findEncoding to ensure function works
findEncoding(imgList)
print("Encoding Start . . .")
# Generate encodings for all images
encodeListKnown = findEncoding(imgList)
# Create a list containing both encodings and their corresponding IDs
encodeListKnownDict = [encodeListKnown, imgIdList]
print("Encoding Complete")

# Save encodings to a binary file using pickle
file = open("encodeFile.p", "wb")
pickle.dump(encodeListKnownDict, file)
file.close()
print("File Created")


