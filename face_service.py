# import cv2
# import torch
# import numpy as np
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from PIL import Image

# # Initialize models
# mtcnn = MTCNN(keep_all=True, min_face_size=40, device='cpu') 
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to('cpu')

# def get_face_embeddings(image_path):
#     try:
#         # 1. Load image and convert to RGB (Fixes the "4 channels" error)
#         img = Image.open(image_path).convert('RGB')
        
#         # 2. Detect faces
#         boxes, probs = mtcnn.detect(img)
        
#         if boxes is None:
#             return [], []

#         # 3. Extract faces
#         faces_tensors = mtcnn(img)
        
#         if faces_tensors is None:
#             return [], []

#         # 4. Calculate embeddings
#         # .detach().numpy() gives us a NumPy array
#         embeddings_array = resnet(faces_tensors).detach().numpy()
        
#         # Convert NumPy array to standard Python List (Fixes the "Ambiguous truth value" error)
#         embeddings = embeddings_array.tolist()
        
#         # Format locations [top, right, bottom, left]
#         formatted_locations = []
#         for box in boxes:
#             box = [int(b) for b in box]
#             formatted_locations.append([box[1], box[2], box[3], box[0]])

#         return formatted_locations, embeddings

#     except Exception as e:
#         print(f"Error processing image: {e}")
#         return [], []

# def find_match(unknown_encoding, known_faces_list, tolerance=0.8):
#     if not known_faces_list:
#         return None

#     unknown_encoding = np.array(unknown_encoding)
#     best_dist = float('inf')
#     best_match = None

#     for face in known_faces_list:
#         known_encoding = np.array(face['embedding'])
#         dist = np.linalg.norm(known_encoding - unknown_encoding)
        
#         if dist < best_dist:
#             best_dist = dist
#             best_match = face

#     if best_dist < tolerance:
#         return best_match
    
#     return None

from deepface import DeepFace
import numpy as np

# You can choose: 'VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib'
# 'VGG-Face' is the default and quite accurate.
MODEL_NAME = "VGG-Face"
DETECTOR_BACKEND = "opencv" # Fast, but less accurate than 'retinaface' or 'mtcnn'

def get_face_embeddings(image_path):
    try:
        # DeepFace.represent finds all faces in the image and returns their embeddings
        # enforce_detection=False prevents crash if no face is found
        results = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        
        locations = []
        encodings = []

        for face in results:
            # DeepFace returns facial_area as {'x': int, 'y': int, 'w': int, 'h': int}
            area = face['facial_area']
            x, y, w, h = area['x'], area['y'], area['w'], area['h']
            
            # Convert to our API format: [top, right, bottom, left]
            # y = top, x+w = right, y+h = bottom, x = left
            locations.append([y, x+w, y+h, x])
            
            # Get the embedding list
            encodings.append(face['embedding'])

        return locations, encodings

    except Exception as e:
        print(f"Error in DeepFace: {e}")
        return [], []

def find_match(unknown_encoding, known_faces_list, threshold=0.4):
    """
    Compares the unknown encoding with known faces using Cosine Similarity.
    DeepFace typically uses Cosine similarity.
    Threshold: 0.4 is standard for VGG-Face (lower is stricter).
    """
    if not known_faces_list:
        return None

    # Convert to numpy for math operations
    unknown_encoding = np.array(unknown_encoding)
    
    best_score = float('inf') # For Cosine Distance, lower is better (0 = identical)
    best_match = None

    for face in known_faces_list:
        known_encoding = np.array(face['embedding'])
        
        # Calculate Cosine Distance
        # Formula: 1 - (A . B) / (||A|| * ||B||)
        dot_product = np.dot(unknown_encoding, known_encoding)
        norm_a = np.linalg.norm(unknown_encoding)
        norm_b = np.linalg.norm(known_encoding)
        
        cosine_distance = 1 - (dot_product / (norm_a * norm_b))
        
        if cosine_distance < best_score:
            best_score = cosine_distance
            best_match = face

    if best_score < threshold:
        return best_match
    
    return None