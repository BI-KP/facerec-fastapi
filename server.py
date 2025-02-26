# Import necessary libraries
from fastapi import FastAPI, UploadFile, File, HTTPException  # Web framework and file handling
from starlette.middleware.cors import CORSMiddleware  # For handling Cross-Origin requests
from io import BytesIO  # For handling binary data
import pickle  # For loading/saving face encodings
import cv2  # OpenCV for image processing
import numpy as np  # For numerical operations
from deepface import DeepFace  # Face recognition library
from scipy.spatial import distance  # For calculating face similarity
from PIL import Image  # For image processing
from pydantic import BaseModel  # For data validation

# Initialize FastAPI application
app = FastAPI()

# Define allowed origins for CORS policy
origins = [
    "http://localhost:3001",
    "http://localhost:3000",
    "localhost:3001",
    "localhost:3000",
]

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Define data model for image input
class Data(BaseModel):
    image: str

# Function to load face encodings from file
def load_encodings():
    global encodeList, imgIds  # Define global variables to store encodings and IDs
    print("Loading Encoding File . . .")
    try:
        file = open("encodeFile.p", "rb")  # Open binary file containing encodings
        encodeListWithId = pickle.load(file)  # Load the encodings and IDs
        file.close()
        encodeList, imgIds = encodeListWithId  # Unpack into separate lists
        print("Encoding File Loaded")
    except Exception as e:
        print(f"Error loading encodings: {e}")
        encodeList, imgIds = [], []  # Initialize empty lists if loading fails

# Initial load of encodings when server starts
load_encodings()

# Function to register a new face in the system
def register_new_face(image: np.ndarray, face_id: str):
    try:
        # Convert image to RGB and detect face
        imgS = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = DeepFace.extract_faces(imgS, detector_backend='fastmtcnn', enforce_detection=True)
        if not face:
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        # Generate face embedding using Facenet512 model
        encode = DeepFace.represent(imgS, model_name='Facenet512', detector_backend='fastmtcnn')
        new_encoding = encode[0]['embedding']  # Extract embedding vector
        
        # Load current face encodings from file
        file = open("encodeFile.p", "rb")
        current_data = pickle.load(file)
        file.close()
        
        current_encodings, current_ids = current_data  # Unpack data
        
        # Add new face encoding and ID
        current_encodings.append(new_encoding)
        current_ids.append(face_id)
        
        # Save updated encodings back to file
        with open("encodeFile.p", "wb") as file:
            pickle.dump([current_encodings, current_ids], file)
        
        # Reload encodings into memory
        load_encodings()
        return True
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# API endpoint for registering a new face
@app.post("/register/")
async def register_face(face_name: str, file: UploadFile = File(...)):
    if not face_name:
        raise HTTPException(status_code=400, detail="face_name is required")
    
    # Convert uploaded file to numpy array
    image = np.array(Image.open(BytesIO(await file.read())))
    success = register_new_face(image, face_name)
    
    return {"success": success, "face_name": face_name}

# Function to recognize a face from an image
def recognize_face(image: np.ndarray):
    # Convert image to RGB for processing
    imgS = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodeImage = None
    try:
        # Generate face embedding with anti-spoofing check
        encodeImage = DeepFace.represent(imgS, model_name='Facenet512', detector_backend='fastmtcnn', anti_spoofing=True)
        encodeImage = encodeImage[0]['embedding']
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Compare with stored face encodings
    results = []
    for encodeFace in encodeList:
        # Calculate cosine distance (similarity measure)
        faceDist = distance.cosine(encodeImage, encodeFace)
        if faceDist < 0.3:  # Threshold untuk face recognition. semakin kecil semakin strict
            matchIndex = encodeList.index(encodeFace)
            name = imgIds[matchIndex]
            results.append({"name": name, "distance": faceDist})
        else:
            results.append({"name": "Unknown Person", "distance": faceDist})
    
    # Return the match with lowest distance (highest similarity)
    results.sort(key=lambda x: x['distance'])
    return results[0]

# API endpoint for recognizing a face
@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    # Convert uploaded file to numpy array
    image = np.array(Image.open(BytesIO(await file.read())))
    results = recognize_face(image)
    return {"results": results}

# Commented out server startup code - likely handled externally
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


