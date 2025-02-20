import cv2
from deepface import DeepFace
import numpy as np
import pickle
import os

# Import Images Graphic
folderPath = "ResizedImages"
imgPathList = os.listdir(folderPath)
imgList = [cv2.imread(os.path.join(folderPath, path)) for path in imgPathList]

#Take the id from file name
imgIdList = [os.path.splitext(path)[0] for path in imgPathList]

def crop_image_face(img):
    face = DeepFace.extract_faces(img, detector_backend='fastmtcnn', enforce_detection=False, align=True)
    face_data = face[0]['facial_area']
    x1, y1, width, height= face_data['x'], face_data['y'], face_data['w'], face_data['h']
    cropped_image = img[y1:y1+height, x1:x1+width]
    return cropped_image

# Encode the images
def findEncoding(imgList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = crop_image_face(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode = DeepFace.represent(img, model_name='Facenet512', detector_backend='fastmtcnn')
        embedding = encode[0]['embedding']
        encodeList.append(embedding)
    return encodeList

findEncoding(imgList)
print("Encoding Start . . .")
encodeListKnown = findEncoding(imgList)
encodeListKnownDict = [encodeListKnown, imgIdList]
print("Encoding Complete")

file = open("encodeFile.p", "wb")

pickle.dump(encodeListKnownDict, file)
file.close()
print("File Created")


