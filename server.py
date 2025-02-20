import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force TensorFlow to use CPU

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Explicitly disable GPU

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from FaceRecognition import recognize_face_with_embedding, connect_db, generate_image_encoding, check_antispoofing
# import uvicorn

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Welcome to Face Recognition API"}

@app.post("/spoof-check/")
async def spoof_check(file: UploadFile = File(...)):
    contents = await file.read()
    img_path = f"/tmp/{file.filename}"
    
    with open(img_path, "wb") as f:
        f.write(contents)
    
    if not check_antispoofing(img_path):
        result = False
    else:
        result = True
    return {"is_spoofing": result,
            "message": "Image is spoofing" if result else "Image is not spoofing",
            "status": status.HTTP_200_OK}

@app.post("/recognize-face/")
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()

    img_path = f"/tmp/{file.filename}"
    spoof_check = False
    
    with open(img_path, "wb") as f:
        f.write(contents)
    
    # if not check_antispoofing(img_path):
    #     spoof_check = True
    #     return {"message": "Image is spoofing", "is_spoofing": spoof_check}
    
    query_embedding = generate_image_encoding(img_path)
    conn = connect_db()
    match, score = recognize_face_with_embedding(query_embedding, conn, threshold=0.6, metric="cosine")
    conn.close()
    
    if match and score:
        return {"match": match, "score": score, "is_spoofing": spoof_check}
    else:
        raise HTTPException(status_code=404, detail="Face not found")
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
