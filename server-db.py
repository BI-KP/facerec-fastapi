from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from io import BytesIO
import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial import distance
from PIL import Image
import base64
from models import Data
from db import load_encodings, insert_encoding

app = FastAPI()

origins = [
    "http://localhost:3001",
    "http://localhost:3000",
    "localhost:3001",
    "localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initial load
encodeList, imgIds = load_encodings()

def register_new_face(image: np.ndarray, face_id: str):
    try:
        # Convert and get face encoding
        imgS = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = DeepFace.extract_faces(imgS, detector_backend='fastmtcnn', enforce_detection=True)
        if not face:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        encode = DeepFace.represent(imgS, model_name='Facenet512', detector_backend='fastmtcnn')
        new_encoding = encode[0]['embedding']
        
        # Insert new encoding into MongoDB
        if insert_encoding(face_id, new_encoding):
            # Reload encodings in memory
            global encodeList, imgIds
            encodeList, imgIds = load_encodings()
            return True
        else:
            raise HTTPException(status_code=500, detail="Failed to insert encoding")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/register/")
async def register_face(face_id: str, file: UploadFile = File(...)):
    if not face_id:
        raise HTTPException(status_code=400, detail="face_id is required")
    
    image = np.array(Image.open(BytesIO(await file.read())))
    success = register_new_face(image, face_id)
    
    return {"success": success, "face_id": face_id}

@app.post("/register-base64/")
async def register_face_base64(data: Data, face_id: str):
    if not face_id:
        raise HTTPException(status_code=400, detail="face_id is required")
    
    image_data = base64.b64decode(data.image)
    image = np.array(Image.open(BytesIO(image_data)))
    success = register_new_face(image, face_id)
    
    return {"success": success, "face_id": face_id}

def recognize_face(image: np.ndarray):
    imgS = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodeImage = DeepFace.represent(imgS, model_name='Facenet512', detector_backend='fastmtcnn')
    encodeImage = encodeImage[0]['embedding']

    results = []
    for encodeFace in encodeList:
        faceDist = distance.cosine(encodeImage, encodeFace)
        if faceDist < 0.6:  # Threshold for recognizing a face
            matchIndex = encodeList.index(encodeFace)
            name = imgIds[matchIndex]
            results.append({"name": name, "distance": faceDist})
        else:
            results.append({"name": "Unknown Person", "distance": faceDist})
    
    # Filter with most lowest distance
    results.sort(key=lambda x: x['distance'])
    return results[0]

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    image = np.array(Image.open(BytesIO(await file.read())))
    results = recognize_face(image)
    return {"results": results}

@app.post("/testimage/")
async def recognize_image(data: Data):
    print(data.image)
    image_data = base64.b64decode(data.image)
    image = np.array(Image.open(BytesIO(image_data)))
    results = recognize_face(image)
    return {"results": results}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


