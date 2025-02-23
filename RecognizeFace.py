from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from io import BytesIO
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import base64
from pydantic import BaseModel
from db import weaviate

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Weaviate class name
WEAVIATE_CLASS = "FaceEmbeddings"

# Ensure the class exists in Weaviate
def setup_weaviate_schema():
    schema = {
        "class": WEAVIATE_CLASS,
        "vectorizer": "none",  # We're using our own embeddings
        "properties": [
            {"name": "face_id", "dataType": ["string"]},
        ],
    }

    if not weaviate.schema.exists(WEAVIATE_CLASS):
        weaviate.schema.create_class(schema)

setup_weaviate_schema()

class Data(BaseModel):
    image: str

@app.post("/register/")
async def register_face(face_id: str, file: UploadFile = File(...)):
    if not face_id:
        raise HTTPException(status_code=400, detail="face_id is required")

    image = np.array(Image.open(BytesIO(await file.read())))

    try:
        # Extract and encode face
        imgS = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = DeepFace.extract_faces(imgS, detector_backend="fastmtcnn", enforce_detection=True)
        if not face:
            raise HTTPException(status_code=400, detail="No face detected in image")

        encode = DeepFace.represent(imgS, model_name="Facenet512", detector_backend="fastmtcnn")
        embedding = encode[0]["embedding"]

        # Store embedding in Weaviate
        weaviate.data_object.create(
            class_name=WEAVIATE_CLASS,
            data={"face_id": face_id},
            vector=embedding,
        )

        return {"success": True, "face_id": face_id}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    image = np.array(Image.open(BytesIO(await file.read())))

    try:
        # Extract and encode face
        imgS = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = DeepFace.represent(imgS, model_name="Facenet512", detector_backend="fastmtcnn")
        query_embedding = encode[0]["embedding"]

        # Search for the closest match in Weaviate
        response = weaviate.query.get(
            WEAVIATE_CLASS, ["face_id"]
        ).with_near_vector(query_embedding).with_limit(1).do()

        if response["data"]["Get"][WEAVIATE_CLASS]:
            match = response["data"]["Get"][WEAVIATE_CLASS][0]
            return {"name": match["face_id"], "distance": "similar"}  # Weaviate does similarity scoring
        else:
            return {"name": "Unknown Person", "distance": "far"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/register-base64/")
async def register_face_base64(data: Data, face_id: str):
    if not face_id:
        raise HTTPException(status_code=400, detail="face_id is required")

    image_data = base64.b64decode(data.image)
    image = np.array(Image.open(BytesIO(image_data)))

    return await register_face(face_id, UploadFile(filename="image.jpg", file=BytesIO(image_data)))

@app.post("/testimage/")
async def recognize_image(data: Data):
    image_data = base64.b64decode(data.image)
    image = np.array(Image.open(BytesIO(image_data)))
    return await recognize(UploadFile(filename="image.jpg", file=BytesIO(image_data)))
