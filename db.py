# from pymongo import MongoClient
# import bson
#
# # MongoDB setup
# client = MongoClient("mongodb+srv://corneliusardensatwikahermawan:9pRkkb17xcE1xU99@face-db.nqg8u.mongodb.net/")
# db = client["face_recognition"]
# collection = db["encodings"]
#
# def load_encodings():
#     try:
#         encodings = list(collection.find({}))
#         encodeList = [bson.Binary(enc['encoding']) for enc in encodings]
#         imgIds = [enc['face_id'] for enc in encodings]
#         return encodeList, imgIds
#     except Exception as e:
#         print(f"Error loading encodings: {e}")
#         return [], []
#
# def insert_encoding(face_id: str, encoding: bytes):
#     try:
#         collection.insert_one({"face_id": face_id, "encoding": bson.Binary(encoding)})
#         return True
#     except Exception as e:
#         print(f"Error inserting encoding: {e}")
#         return False
import os
import weaviate
from weaviate import AuthApiKey
from dotenv import load_dotenv

load_dotenv()

weaviate_url: str = os.getenv("URL")
weaviate_api_key: AuthApiKey = AuthApiKey(os.getenv("API_KEY"))

weaviate = weaviate.Client(
    url=weaviate_url,
    auth_client_secret=weaviate_api_key
)
if weaviate.is_ready():
    print("Connected to Weaviate DB")
