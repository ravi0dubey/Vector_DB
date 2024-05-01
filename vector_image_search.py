import boto3
import cv2
import numpy as np
from pinecone import Pinecone
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA


pinecone_connection = Pinecone(api_key="your API Key")
# index_name = pinecone_connection.Index("image-index1")
# print(pinecone_connection.list_indexes())
# print(index_name)

s3 = boto3.client('s3', aws_access_key_id='Access key ID', aws_secret_access_key='Secret access key')


def encode_image(image_bytes):
    image_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, (100, 100))  
    encoded_vector = resized_image.flatten().astype(float)
    return encoded_vector


def store_vectors_in_pinecone(image_vectors, image_keys):
    pinecone_index = pinecone_connection.Index("face-index2")
    chunk_size = 100  
    for i in range(0, len(image_vectors), chunk_size):
        vectors_chunk = image_vectors[i:i+chunk_size]
        keys_chunk = image_keys[i:i+chunk_size]
        vectors_to_upsert = [
            {"id": key, "values": vector.tolist()} for key, vector in zip(keys_chunk, vectors_chunk)
        ]
        pinecone_index.upsert(vectors=vectors_to_upsert, namespace="ns1")

def process_image(image_key):
    try:
        image_obj = s3.get_object(Bucket=bucket_name, Key=image_key)
        image_bytes = image_obj['Body'].read()
        encoded_vector = encode_image(image_bytes)
        return encoded_vector, image_key
    except Exception as e:
        print(f"Error processing image {image_key}: {e}")
        return None

def encode_and_store_images_from_s3(bucket_name):
    image_vectors = []
    image_keys = []
    objects = s3.list_objects_v2(Bucket=bucket_name)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_image, obj['Key']) for obj in objects.get('Contents', [])]
        for future in futures:
            result = future.result()
            if result:
                encoded_vector, image_key = result
                image_keys.append(image_key)
                image_vectors.append(encoded_vector)
    store_vectors_in_pinecone(image_vectors, image_keys)

def search_similar_images(image_path):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    search_vector = encode_image(image_bytes)
    query_vector =  search_vector.tolist()
    pinecone_index = pinecone_connection.Index("face-index2")
    results = pinecone_index.query(namespace="ns1",vector=query_vector, top_k=5,include_values=True)
    return results


def main():
    global bucket_name
    bucket_name = "faces-images-scraped"
    # encode_and_store_images_from_s3(bucket_name)
    new_image_path = "face_2.jpg"
    search_results = search_similar_images(new_image_path)
    data = search_results
    ids = [match['id'] for match in data['matches']]
    print("Similar images:")
    print(ids)

if __name__ == "__main__":
    bucket_name = "faces-images-scraped"
    training_data = []
    objects = s3.list_objects_v2(Bucket=bucket_name)
    for obj in objects.get('Contents', []):
        image_obj = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
        image_bytes = image_obj['Body'].read()
        encoded_vector = encode_image(image_bytes)
        training_data.append(encoded_vector) 
    pca = PCA(n_components=min(len(training_data), 18))  
    pca.fit(training_data)

    main()