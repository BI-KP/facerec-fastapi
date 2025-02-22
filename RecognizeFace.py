from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from io import BytesIO
import pickle
import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial import distance
from PIL import Image
import base64
from pydantic import BaseModel


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


class Data(BaseModel):
    image: str


# Move file loading to a function so we can reload it
def load_encodings():
    global encodeList, imgIds
    print("Loading Encoding File . . .")
    try:
        file = open("encodeFile.p", "rb")
        encodeListWithId = pickle.load(file)
        file.close()
        encodeList, imgIds = encodeListWithId
        print("Encoding File Loaded")
    except Exception as e:
        print(f"Error loading encodings: {e}")
        encodeList, imgIds = [], []


# Initial load
load_encodings()


def register_new_face(image: np.ndarray, face_id: str):
    try:
        # Convert and get face encoding
        imgS = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = DeepFace.extract_faces(
            imgS, detector_backend="fastmtcnn", enforce_detection=True
        )
        if not face:
            raise HTTPException(status_code=400, detail="No face detected in image")

        encode = DeepFace.represent(
            imgS, model_name="Facenet512", detector_backend="fastmtcnn"
        )
        new_encoding = encode[0]["embedding"]

        # Load current encodings
        file = open("encodeFile.p", "rb")
        current_data = pickle.load(file)
        file.close()

        current_encodings, current_ids = current_data

        # Add new encoding
        current_encodings.append(new_encoding)
        current_ids.append(face_id)

        # Save updated encodings
        with open("encodeFile.p", "wb") as file:
            pickle.dump([current_encodings, current_ids], file)

        # Reload encodings in memory
        load_encodings()
        return True
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def check_antispoofing(img_path):
    result = DeepFace.extract_faces(img_path, anti_spoofing=True)
    print("RESULT: ", result)
    print("SPOOF SCORE: ", result[0]["antispoof_score"])
    return (
        True if result[0]["antispoof_score"] > 0.7 and result[0]["is_real"] else False
    )


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
    # print(image.shape)
    # print(image.dtype)
    imgS = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodeImage = None
    try:
        encodeImage = DeepFace.represent(
            imgS,
            model_name="Facenet512",
            detector_backend="fastmtcnn",
            anti_spoofing=True,
        )
        # print("ENCODED: ", encodeImage)
        encodeImage = encodeImage[0]["embedding"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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
    results.sort(key=lambda x: x["distance"])
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
