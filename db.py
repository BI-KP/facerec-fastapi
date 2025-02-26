# MongoDB connection code (currently commented out)
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

# Import required libraries
import os  # For accessing environment variables
import weaviate  # Vector database for storing face embeddings
from weaviate import AuthApiKey  # Authentication method for Weaviate
from dotenv import load_dotenv  # For loading environment variables from .env file

# Load environment variables from .env file
load_dotenv()

# Get Weaviate connection details from environment variables
weaviate_url: str = os.getenv("URL")
weaviate_api_key: AuthApiKey = AuthApiKey(os.getenv("API_KEY"))

# Initialize Weaviate client with connection details
weaviate = weaviate.Client(
    url=weaviate_url,
    auth_client_secret=weaviate_api_key
)

# Check if connection to Weaviate database is successful
if weaviate.is_ready():
    print("Connected to Weaviate DB")

import os
import weaviate
from weaviate.classes.init import Auth

# Best practice: store your credentials in environment variables
weaviate_url: str = os.getenv("URL")
weaviate_api_key: str = os.getenv("API_KEY")

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

# print(client.is_ready())