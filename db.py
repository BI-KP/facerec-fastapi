from pymongo import MongoClient
import bson

# MongoDB setup
client = MongoClient("mongodb+srv://corneliusardensatwikahermawan:9pRkkb17xcE1xU99@face-db.nqg8u.mongodb.net/")
db = client["face_recognition"]
collection = db["encodings"]

def load_encodings():
    try:
        encodings = list(collection.find({}))
        encodeList = [bson.Binary(enc['encoding']) for enc in encodings]
        imgIds = [enc['face_id'] for enc in encodings]
        return encodeList, imgIds
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []

def insert_encoding(face_id: str, encoding: bytes):
    try:
        collection.insert_one({"face_id": face_id, "encoding": bson.Binary(encoding)})
        return True
    except Exception as e:
        print(f"Error inserting encoding: {e}")
        return False
