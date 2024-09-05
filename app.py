
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import psycopg2
import boto3
from scipy.spatial.distance import cosine
import os

# Database and S3 settings
RDS_HOST = st.secrets["RDS_HOST"]
RDS_DBNAME = st.secrets["RDS_DBNAME"]
RDS_USER = st.secrets["RDS_USER"]
RDS_PASSWORD = st.secrets["RDS_PASSWORD"]
S3_BUCKET = st.secrets["S3_BUCKET"]
AWS_ACCESS_KEY = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY = st.secrets["AWS_SECRET_KEY"]

# Load the pretrained FaceNet model
facenet_model = InceptionResnetV1(pretrained='vggface2').eval()

# Load OpenCV's Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
                         0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Function to connect to RDS PostgreSQL database


def connect_to_db():
    conn = psycopg2.connect(
        host=RDS_HOST,
        database=RDS_DBNAME,
        user=RDS_USER,
        password=RDS_PASSWORD
    )
    return conn

# Function to fetch existing vectors from the RDS database


def fetch_vectors_from_db():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding, image_url FROM celebrities")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

# Function to download image from S3


def download_image_from_s3(image_id):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                      aws_secret_access_key=AWS_SECRET_KEY)

    # Construct the key based on the expected format
    image_key = f"celebrities/{image_id}.jpg"

    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=image_key)
        image = Image.open(obj['Body'])
        return image
    except s3.exceptions.NoSuchKey:
        st.error(f"Error: The specified key does not exist in S3: {image_key}")
        return None

# Function to calculate cosine similarity


def cosine_similarity(vector1, vector2):
    return 1 - cosine(vector1, vector2)


def dot_product_similarity(vector1, vector2):
    return np.dot(vector1, vector2)


def detect_face(image, zoom_out_factor=1.8, move_up_factor=0.05):
    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a face is detected, crop to the first detected face
    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Calculate new bounding box dimensions to zoom out
        x = int(max(0, x - (w * (zoom_out_factor - 1) / 2)))
        y = int(max(0, y - (h * (zoom_out_factor - 1) / 2)))
        w = int(min(image.width, w * zoom_out_factor))
        h = int(min(image.height, h * zoom_out_factor))

        # Adjust y-coordinate to move the face up
        y = int(max(0, y - h * move_up_factor))

        cropped_face = image.crop((x, y, x+w, y+h))
        return cropped_face
    else:
        # Return None if no face is detected
        return None


def preprocess_image(image):
    # Detect and crop the face
    face_image = detect_face(image)

    if face_image is not None:
        # Apply preprocessing to the cropped face image
        image_tensor = preprocess(face_image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor, face_image
    else:
        return None, None


def vectorize_image(image_tensor):
    # Get the embedding (vector) for the image
    with torch.no_grad():
        vector = facenet_model(image_tensor)
    return vector


def getCelebName(id):
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM celebrities WHERE id=%s", (id,))
    name = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return name


def main():
    st.title("Face Similarity App")
    st.write("Upload an image or take a photo with your camera")

    # Image upload or camera input
    image_file = st.file_uploader(
        "Upload an image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        image_file = st.camera_input("Take a photo")

    if image_file is not None:
        # Open the image
        image = Image.open(image_file)

        # Preprocess the image (detect and crop face)
        image_tensor, processed_image = preprocess_image(image)

        if processed_image is not None:
            # Display the processed image
            st.image(processed_image,
                     caption="Processed Face (Moved Up)", use_column_width=True)

            # Vectorize the image
            embedding = vectorize_image(image_tensor).numpy()

            # Fetch existing vectors from the database
            vectors = fetch_vectors_from_db()

            # Calculate similarities and find the top 3 matches
            similarities = []
            for row in vectors:
                id, vector, _ = row
                vector = np.array(vector)
                similarity = cosine_similarity(
                    embedding.flatten(), vector)
                similarities.append((id, similarity))

            # Sort by similarity and get top 3
            top_matches = sorted(
                similarities, key=lambda x: x[1], reverse=True)[:3]

            # Display the top 3 matches with similarity scores
            st.write("Top 3 matches:")
            for match in top_matches:
                id, similarity = match
                name = getCelebName(id)
                similar_image = download_image_from_s3(id)
                if similar_image:
                    normalized_similarity = similarity * 100  # Convert to percentage
                    st.image(
                        similar_image, caption=f"{name}, Similarity: {normalized_similarity:.2f}%", use_column_width=True)
        else:
            st.write("No face detected. Please upload a different image.")


if __name__ == "__main__":
    main()
