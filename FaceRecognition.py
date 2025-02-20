import os
import time  # Import time module for runtime measurement

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Disables tensorflow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disables tensorflow warnings
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/cuda' # Disables tensorflow warnings
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force TensorFlow to use CPU

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Explicitly disable GPU

from deepface import DeepFace
import numpy as np
import json
import psycopg2

import psycopg2

def connect_db():
    conn = psycopg2.connect('postgres://avnadmin:AVNS_O2ErT6GM3NB0THc97cP@db-fr-bi-kp-bi.c.aivencloud.com:23176/defaultdb?sslmode=require')
    return conn

def insert_face(conn, name, encoding):
    encoding_json = json.dumps(encoding)  
    cursor = conn.cursor()
    cursor.execute("INSERT INTO face_encodings (name, embedding) VALUES (%s, %s)", (name, encoding_json))
    conn.commit()
    
def fetch_all_encodings(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM face_encodings")
    results = cursor.fetchall()
    cursor.close()
    return [{"name": row[0], "embedding": np.array(row[1])} for row in results]

def get_face_encodings(cursor):
    cursor.execute("SELECT * FROM face_encodings")
    results = cursor.fetchall()
    return [{"name": row[0], "embedding": row[1]} for row in results]

def generate_image_encoding(img):
    embedding = DeepFace.represent(img_path=img, model_name='SFace')
    return embedding[0]['embedding']

def check_antispoofing(img_path):
    result = DeepFace.extract_faces(img_path, anti_spoofing=True)
    return True if result[0]['antispoof_score'] > 0.7 else False

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def euclidean_distance(embedding1, embedding2):
    return np.linalg.norm(embedding1 - embedding2)

def recognize_face_with_embedding(query_embedding, conn, threshold=0.6, metric="euclidean"):
    start_time = time.time()  # Start time measurement

    cursor = conn.cursor()
    query_embedding_json = json.dumps(query_embedding)

    if metric == "cosine":
        cursor.execute("""
            SELECT name, cosine_similarity(embedding, %s) AS score
            FROM face_encodings
            ORDER BY score DESC
            LIMIT 1
        """, (query_embedding_json,))
    elif metric == "euclidean":
        cursor.execute("""
            SELECT name, (embedding <-> %s) AS score
            FROM face_encodings
            ORDER BY score ASC
            LIMIT 1
        """, (query_embedding_json,))
    else:
        raise ValueError("Unsupported metric: choose 'euclidean' or 'cosine'.")

    result = cursor.fetchone()
    cursor.close()

    best_match = result[0] if result else None
    best_score = result[1] if result else None

    end_time = time.time()  # End time measurement
    print(f"recognize_face_with_embedding runtime: {end_time - start_time} seconds")
    return best_match, best_score

def process_and_store_images(db_directory, conn):
    for filename in os.listdir(db_directory):
        if filename.endswith(".jpg"):
            name = filename.split('-')[0]
            img_path = os.path.join(db_directory, filename)
            encoding = generate_image_encoding(img_path)
            insert_face(conn, name, encoding)

# # Example usage
# if __name__ == "__main__":
#     img_path = "./test/mahsa-picek.jpg"
    
#     # Check for antispoofing
#     if not check_antispoofing(img_path):
#         print("Image is spoofing")
#     else:
#         query_embedding = generate_image_encoding(img_path)
#         conn = connect_db()
#         match, score = recognize_face_with_embedding(query_embedding, conn, threshold=0.6, metric="cosine")
#         if match:
#             print(f"Match found: {match} (Score: {score})")
#         else:
#             print("No match found.")
#         conn.close()
    



