import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
import numpy as np
# from deepface import DeepFace
from scipy.spatial.distance import cosine, euclidean
import uuid
import datetime
from ip_finder import get_local_ip
import threading
import time
import certifi
from dotenv import load_dotenv
from waitress import serve
import hupper
from PIL import Image
import cv2
from insightface.app import FaceAnalysis
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# useless
final_ip="https://gallery.snorlax.codes"
my_ipp=get_local_ip()
# final_ip="http://{my_ip}:5069"



load_dotenv()

app = Flask(__name__)
CORS(app)

@app.errorhandler(ServerSelectionTimeoutError)
def handle_mongo_timeout(error):
    return jsonify({"error": "Database connection temporarily unavailable. The server will automatically retry."}), 503

@app.errorhandler(ConnectionFailure)
def handle_mongo_connection(error):
    return jsonify({"error": "Database connection failed. The server will automatically retry."}), 503


# ==============================================================================
# LOCAL TEXT EMBEDDING INITIALIZATION
# ==============================================================================
print("[INIT] Loading local text embedding model (all-MiniLM-L6-v2)...")
text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("[INIT] Text embedding model loaded.")

import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers
    return c * r

# --- FACE RECOGNITION FUNCTIONS ---


# ==============================================================================
# GLOBAL INITIALIZATION (Production Standard)
# Initialize the model once when the backend server starts to avoid cold-start 
# latency and memory leaks on every request.
# ==============================================================================
face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# ==============================================================================
# GEMINI INITIALIZATION (Production Standard)
# ==============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your_mongodb_connection_string_here")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[WARNING] GEMINI_API_KEY is not set in .env. Image analysis will be skipped.")


def sanitize_for_mongo(data):
    """
    Recursively strips NumPy data types from dictionaries and lists, 
    converting them to native Python scalars for MongoDB BSON compatibility.
    """
    if isinstance(data, dict):
        return {k: sanitize_for_mongo(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_mongo(v) for v in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return sanitize_for_mongo(data.tolist())
    else:
        return data
    

def get_face_embeddings(image_path):
    """
    Extract face locations and embeddings using InsightFace (buffalo_l)
    Returns: (locations, encodings) where locations are bounding boxes and encodings are embeddings
    """
    try:
        # OpenCV reads in BGR, which InsightFace natively expects
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"[FACE] Error: Could not read image at {image_path}")
            return [], []

        results = face_analyzer.get(img)
        
        locations = []
        encodings = []
        
        for face in results:
            # InsightFace bbox format is [left, top, right, bottom] -> [x1, y1, x2, y2]
            bbox = face.bbox.astype(int)
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            
            # Convert to the (top, right, bottom, left) format required by your pipeline
            box = (y1, x2, y2, x1)
            locations.append(box)
            
            # Extract embedding (This is automatically L2-normalized)
            embedding = face.normed_embedding
            encodings.append(embedding)
            
        return locations, encodings
    except Exception as e:
        print(f"[FACE] Error extracting face embeddings: {e}")
        return [], []


def faces_match(emb1, emb2):
    """
    Compare two ArcFace embeddings using Cosine Similarity.
    Returns True if the similarity indicates a match.
    """
    try:
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        # Optimization: Since both vectors are L2 normalized, 
        # the Dot Product is perfectly equivalent to Cosine Similarity.
        similarity = np.dot(emb1, emb2)
        
        # InsightFace buffalo_l Thresholds (Cosine Similarity):
        # 0.40 - 0.45 : Standard application (Trade-off between False Accept and False Reject)
        # 0.50 - 0.55 : High security (e.g., Financial KYC, Access Control)
        SIMILARITY_THRESHOLD = 0.45 
        
        is_match = similarity >= SIMILARITY_THRESHOLD
        
        print(f"[FACE] Comparison: cos_sim={similarity:.4f} | Threshold: {SIMILARITY_THRESHOLD} -> {'MATCH' if is_match else 'NO MATCH'}")
        
        # Cast numpy boolean to standard Python bool for JSON serializability
        return bool(is_match)
        
    except Exception as e:
        print(f"[FACE] Error comparing embeddings: {e}")
        return False


def find_match(encoding, user_list):
    """
    Find if encoding matches any user/face in the list.
    Returns the matching document or None.
    """
    for user_doc in user_list:
        stored_embedding = user_doc.get('embedding', [])
        if stored_embedding is not None and len(stored_embedding) > 0:
            if faces_match(encoding, stored_embedding):
                return user_doc
    return None

# --- CONFIG ---
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MONGO_URI = os.getenv("MONGO_URI", "your_mongodb_connection_string_here")

# --- DB SETUP ---
# print("connection string is: ",MONGO_URI)
# client = MongoClient(MONGO_URI)
client = MongoClient(
    MONGO_URI,
    tlsCAFile=certifi.where(),       
    maxIdleTimeMS=45000,             
    connectTimeoutMS=60000,          # Increased to 60s for slow internet
    serverSelectionTimeoutMS=60000,  # Wait up to 60s for network to recover instead of throwing an error
    socketTimeoutMS=60000,           # Wait 60s for data to finish transmitting
    retryWrites=True                 
)
db = client.new
users_col = db.users
strangers_col = db.strangers
shared_queue_col = db.shared_queue
photos_col = db.photos  # New collection for all photos with processing status
alerts_col = db.alerts  # Geofence alerts collection
geofences_col = db.geofences # Geofence definitions

# --- BACKGROUND WORKER ---
processing_active = True

def extract_image_metadata(image_path):
    """
    Uses Gemini 2.5 Flash to analyze the image and return a strict JSON structure
    containing a description and search tags.
    """
    if not GEMINI_API_KEY:
        return {"description": "na", "tags": []}

    try:
        # gemini-2.5-flash is the industry standard for fast, cost-effective multimodal tasks
        model = genai.GenerativeModel(
            'gemini-3.1-flash-lite',
            generation_config={"response_mime_type": "application/json"}
        )
        
        # PIL Image is passed directly to the Gemini SDK
        with Image.open(image_path) as img:
            prompt = """
            Analyze this image for a photo gallery search engine. 
            Provide a brief, objective description of the scene, location, and activities.
            Provide an array of 10 to 15 highly relevant search tags/keywords (e.g., "beach", "sunset", "wedding", "crowd").
            
            Return ONLY a JSON object with this exact schema:
            {
                "description": "string",
                "tags": ["string", "string"]
            }
            """
            response = model.generate_content([prompt, img])
            
            # Parse the guaranteed JSON response
            metadata = json.loads(response.text)
            print(f"[GEMINI] Extracted {len(metadata.get('tags', []))} tags.")
            return metadata
            
    except Exception as e:
        print(f"[GEMINI] API Error: {e}")
        return {"description": "", "tags": []}

def background_photo_processor():
    """
    Background worker that continuously:
    1. Fetches photos with status='pending' from photos collection
    2. Processes them (face detection, matching)
    3. Compares with users and strangers
    4. Updates persons_present field
    5. Handles stranger-to-user conversion
    6. Adds photo to shared queues for matched users
    """
    print("[WORKER] Background photo processor started")
    
    while processing_active:
        try:
            # Get all pending photos
            pending_photos = list(photos_col.find({"status": "pending"}))
            
            if pending_photos:
                print(f"[WORKER] Found {len(pending_photos)} pending photos to process")
            
            for photo_doc in pending_photos:
                try:
                    photo_id = photo_doc['_id']
                    local_path = photo_doc.get('local_path')
                    owner_email = photo_doc.get('owner_email', 'anonymous')
                    
                    # Skip if file doesn't exist
                    if not os.path.exists(local_path):
                        print(f"[WORKER] File not found: {local_path}, skipping photo {photo_id}")
                        photos_col.update_one({"_id": photo_id}, {"$set": {"status": "error", "error": "File not found"}})
                        continue
                    
                    print(f"[WORKER] Processing photo {photo_id} from {owner_email}...")
                    
                    # Update status to processing
                    photos_col.update_one({"_id": photo_id}, {"$set": {"status": "processing"}})
                    
                    # AI Processing
                    locations, encodings = get_face_embeddings(local_path)
                    print(f"[WORKER] Found {len(encodings)} face(s) in photo {photo_id}")
                    
                    detected_people = []
                    persons_present_ids = []  # Track unique person IDs
                    
                    # Get all known users and strangers
                    all_users = list(users_col.find({}))
                    all_strangers = list(strangers_col.find({}))
                    
                    # DEBUG: Log how many users we're checking
                    users_with_embedding = sum(1 for u in all_users if u.get('embedding'))
                    users_without_embedding = len(all_users) - users_with_embedding
                    print(f"[WORKER] Users available: {len(all_users)} total ({users_with_embedding} with embedding, {users_without_embedding} without)")
                    print(f"[WORKER] Strangers available: {len(all_strangers)}")
                    with Image.open(local_path) as img:
                        img_w, img_h = img.size
                    for idx, (box, encoding) in enumerate(zip(locations, encodings)):
                        top, right, bottom, left = box
                        
                        # 1. Force strict native Python integers to prevent BSON serialization crashes
                        # Math operations must be done on the casted integers.
                        safe_y = int(top)
                        safe_x = int(right)
                        safe_h = int(bottom) - int(top)
                        safe_w = int(right) - int(left)
                        
                        bbox = {
                            "y": safe_y, 
                            "x": safe_x, 
                            "h": safe_h, 
                            "w": safe_w,
                            "img_w": int(img_w),
                            "img_h": int(img_h)
                        }
                        
                        print(f"bbox (sanitized): {bbox}")

                        person_id = None
                        person_type = None
                        person_email = None
                        
                        # A. Check if it's a Registered User
                        # ... (rest of your loop logic remains identical from here)
                        # A. Check if it's a Registered User
                        match_user = find_match(encoding, all_users)
                        if match_user:
                            person_id = match_user.get('user_id', match_user.get('email'))
                            person_type = "User"
                            email_matched = match_user.get('email')
                            person_email = email_matched
                            person_name = match_user.get('name', email_matched)
                            print(f"[WORKER] Face #{idx+1}: ✓ MATCHED registered user {email_matched} (user_id: {person_id})")
                            if person_id not in persons_present_ids:
                                persons_present_ids.append(person_id)
                            # Add to their shared queue
                            shared_queue_col.insert_one({
                                "recipient_email": email_matched,
                                "recipient_id": person_id,
                                "photo_id": str(photo_id),
                                "image_url": photo_doc.get('image_url'),
                                "metadata": photo_doc.get('location_data', {}),
                                "owner_email": owner_email,
                                "status": "pending",
                                "name": person_name,
                            })
                            print(f"[WORKER] Photos shared with user {email_matched}")
                        else:
                            # No user match - check strangers
                            # B. Check if it's a Known Stranger
                            match_stranger = find_match(encoding, all_strangers)
                            if match_stranger:
                                person_id = match_stranger['face_id']
                                person_type = "Stranger (Known)"
                                person_email = match_stranger.get('email')  # If you ever add email to strangers
                                print(f"[WORKER] Face #{idx+1}: ✓ MATCHED known stranger {person_id}")
                                if person_id not in persons_present_ids:
                                    persons_present_ids.append(person_id)
                            else:
                                # C. Create New Stranger
                                person_id = f"stranger_{str(uuid.uuid4())[:8]}"
                                person_name = f"Unknown_{person_id[-4:]}"
                                
                                # Wrap the entire dictionary in the sanitizer
                                stranger_payload = sanitize_for_mongo({
                                    "face_id": person_id,
                                    "name": person_name,
                                    "embedding": encoding.tolist(),
                                    "source_photo_id": str(photo_id),
                                    "image_url": photo_doc.get('image_url'),
                                    "local_path": local_path,
                                    "bbox": bbox,
                                    "created_at": datetime.datetime.now()
                                })
                                strangers_col.insert_one(stranger_payload)
                                all_strangers.append({"face_id": person_id, "embedding": encoding.tolist()})
                                person_type = "Stranger (New)"
                                print(f"[WORKER] Face #{idx+1}: ⚠ CREATED new stranger {person_id} (no user or known stranger matched)")
                                if person_id not in persons_present_ids:
                                    persons_present_ids.append(person_id)

                        detected_people.append({
                            "id": person_id,
                            "type": person_type,
                            "bbox": bbox,
                            "email": person_email
                        })
                    
                    # ==========================================
                    # NEW: GEMINI AI METADATA EXTRACTION
                    # ==========================================
                    print(f"[WORKER] Calling Gemini for image metadata on {photo_id}...")
                    image_metadata = extract_image_metadata(local_path)
                    
                    # Update photo with processing results, persons_present, AND image metadata
                    # ==========================================
                    # NEW: LOCAL TEXT EMBEDDING FOR SEARCH
                    # ==========================================
                    text_for_embedding = f"{image_metadata.get('description', '')} " + " ".join(image_metadata.get('tags', []))
                    text_embedding = []
                    
                    if text_for_embedding.strip():
                        try:
                            # .encode() returns a numpy array, we convert to list for MongoDB
                            text_embedding = text_embedder.encode(text_for_embedding).tolist()
                            print("[WORKER] Successfully generated local text embedding for search.")
                        except Exception as e:
                            print(f"[EMBEDDING] Local Embedding Error: {e}")

                    # Update photo with processing results
                    update_payload = sanitize_for_mongo({
                        "status": "done",
                        "faces_found": len(detected_people),
                        "ai_info": detected_people,
                        "persons_present": persons_present_ids,
                        "description": image_metadata.get("description", ""),
                        "tags": image_metadata.get("tags", []),
                        "text_embedding": text_embedding, # Saved locally
                        "processed_at": datetime.datetime.now()
                    })
                    
                    photos_col.update_one(
                        {"_id": photo_id},
                        {"$set": update_payload}
                    )
                    
                    print(f"[WORKER] Finished processing photo {photo_id}. Faces: {len(detected_people)}, Persons: {persons_present_ids}, Tags: {len(image_metadata.get('tags', []))}")
                    
                except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                    print(f"[WORKER] Database connection lost while processing photo. Will retry on next loop.")
                except Exception as e:
                    print(f"[WORKER] Error processing photo: {e}")
                    import traceback
                    traceback.print_exc()
                    try:
                        photos_col.update_one({"_id": photo_doc['_id']}, {"$set": {"status": "error", "error": str(e)}})
                    except (ConnectionFailure, ServerSelectionTimeoutError):
                        pass
            
            # Sleep before next check
            time.sleep(2)
        
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            print(f"[WORKER] Database connection lost. Retrying in 5 seconds... (Timeout)")
            time.sleep(2)
        except Exception as e:
            print(f"[WORKER] Background processor error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(2)

# Start background worker thread
worker_thread = threading.Thread(target=background_photo_processor, daemon=True)
worker_thread.start()
print("[APP] Background worker thread started")

# --- ROUTES ---

@app.route('/register', methods=['POST'])
def register():
    """
    User sends email + optional selfie to register.
    1. Check if user already exists (to reuse existing user_id)
    2. If image provided: Extract face embedding and save photo
    3. Save/update user to users collection
    4. If selfie provided, save it to photos collection for processing
    """
    # Handle both JSON and form data
    if request.is_json:
        # print(data)
        data = request.get_json()
        email = data.get('email')
        file = None
    else:
        email = request.form.get('email')
        file = request.files.get('file') if 'file' in request.files else None
    
    print(f"[REGISTER] Received registration for email: {email}")
    
    # CHECK IF USER ALREADY EXISTS - REUSE EXISTING user_id
    existing_user = users_col.find_one({"email": email})
    photo_id = None
        
    if existing_user:
        user_id = existing_user.get('user_id')
        print(f"[REGISTER] User {email} already exists with user_id: {user_id}")
        return jsonify({
            "message": f"User {email} already registered",
            "user_id": user_id,
            "photo_id": photo_id
        })
    else:
        # NEW USER - GENERATE user_id only once
        user_id = f"user_{str(uuid.uuid4())[:8]}"
        print(f"[REGISTER] New user {email}, generated user_id: {user_id}")
    
    
    # If image provided, process it
    if file and file.filename:
        filename = secure_filename(f"{email}_{uuid.uuid4()}.jpg")
        path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(path)
        print(f"[REGISTER] Saved file to {path}")
        my_ip = get_local_ip()


        # Get face embeddings
        locations, encodings = get_face_embeddings(path)
        print(f"[REGISTER] Found {len(encodings)} face(s) in registration photo")
        
        if len(encodings) > 0:
            # Extract and save embedding
            new_embedding = encodings[0].tolist()
            
            # Update user with embedding
            image_url = f"{final_ip}/uploads/{filename}"
            users_col.update_one(
                {"email": email}, 
                {"$set": {
                    "email": email, 
                    "user_id": user_id,
                    "embedding": new_embedding,
                    "image_path":image_url,
                    "registered_at": datetime.datetime.now()
                }},
                upsert=True
            )
            print(f"[REGISTER] Saved user {email} with embedding")
            
            # ===== STRANGER-TO-USER CONVERSION (NEW USER REGISTRATION) =====
            # Check if this new user's embedding matches any existing strangers
            print(f"[REGISTER] New user registration. Checking for matching strangers...")
            
            # Get all strangers
            all_strangers = list(strangers_col.find({}))
            
            # Find if new embedding matches any existing stranger
            matching_stranger_id = None
            for stranger_doc in all_strangers:
                stranger_embedding = stranger_doc.get('embedding')
                if stranger_embedding and faces_match(np.array(new_embedding), np.array(stranger_embedding)):
                    matching_stranger_id = stranger_doc.get('face_id')
                    print(f"[REGISTER] Found matching stranger: {matching_stranger_id}")
                    break
            
            # If we found a matching stranger, convert all photos with that stranger to this user
            if matching_stranger_id:
                print(f"[REGISTER] Converting stranger {matching_stranger_id} to user {user_id} in all photos...")
                
                # Update all photos that contain this stranger
                photos_with_stranger = photos_col.find({"persons_present": matching_stranger_id})
                conversion_count = 0
                
                for photo in photos_with_stranger:
                    # Replace stranger_id with user_id in persons_present
                    new_persons_present = photo.get('persons_present', [])
                    update_data = {}
                    
                    if matching_stranger_id in new_persons_present:
                        new_persons_present.remove(matching_stranger_id)
                        if user_id not in new_persons_present:
                            new_persons_present.append(user_id)
                        update_data["persons_present"] = new_persons_present
                    
                    # Also update ai_info to change stranger details to user details
                    # Also update ai_info to change stranger details to user details
                    new_ai_info = photo.get('ai_info', [])
                    for face_info in new_ai_info:
                        if face_info.get('id') == matching_stranger_id:
                            old_type = face_info.get('type')
                            face_info['id'] = user_id
                            face_info['type'] = 'User'
                            face_info['email'] = email  # <--- ADDED THIS LINE
                            print(f"[REGISTER] Updated ai_info for photo {photo['_id']}: changed face from '{old_type}' to 'User'")
                            break  # Only update this specific face

                    if new_ai_info:
                        update_data["ai_info"] = new_ai_info
                    
                    if update_data:
                        photos_col.update_one(
                            {"_id": photo['_id']},
                            {"$set": update_data}
                        )
                        conversion_count += 1
                        print(f"[REGISTER] Converted photo {photo['_id']}: {matching_stranger_id} -> {user_id}")
                
                print(f"[REGISTER] Converted {conversion_count} photos from stranger to user")
                
                # Delete the stranger entry since they're now identified as a user
                strangers_col.delete_one({"face_id": matching_stranger_id})
                print(f"[REGISTER] Deleted stranger entry {matching_stranger_id}")
            
            # Save selfie photo to photos collection for full processing
            
            photo_doc = {
                "filename": filename,
                "local_path": path,
                "image_url": image_url,
                "location_data": {"source": "registration"},
                "owner_email": email,
                "owner_id": user_id,
                "status": "pending",
                "faces_found": 0,
                "ai_info": [],
                "persons_present": [user_id],
                "created_at": datetime.datetime.now()
            }
            
            result = photos_col.insert_one(photo_doc)
            photo_id = str(result.inserted_id)
            print(f"[REGISTER] Created photo document {photo_id}")
        else:
            print("[REGISTER] No face detected, registering without embedding")
            # Still create user without embedding
            users_col.update_one(
                {"email": email}, 
                {"$set": {
                    "email": email, 
                    "user_id": user_id,
                    "embedding": None,
                    "registered_at": datetime.datetime.now()
                }},
                upsert=True
            )
    else:
        # No image provided, just create user without embedding
        print(f"[REGISTER] No image provided, creating user without embedding")
        users_col.update_one(
            {"email": email}, 
            {"$set": {
                "email": email, 
                "user_id": user_id,
                "embedding": None,
                "registered_at": datetime.datetime.now()
            }},
            upsert=True
        )

    return jsonify({
        "message": f"User {email} registered successfully",
        "user_id": user_id,
        "photo_id": photo_id
    })


# !!! check this route, why 0 persons detectediun the end

@app.route('/update-photo', methods=['POST'])
def update_photo():
    """
    Update/Add photo for an existing user + CONVERT STRANGERS TO USER.
    Receives: email + photo file
    1. Extract face embedding and update user
    2. Check if any existing strangers match this new embedding
    3. If match found, convert those strangers to this user_id in all photos
    4. Save new photo to photos collection for processing
    """
    email = request.form.get('email')
    print(f"[UPDATE-PHOTO] Received photo update for email: {email}")
    
    if 'file' not in request.files:
        print("[UPDATE-PHOTO] No file in request")
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    
    # Check if user exists
    user = users_col.find_one({"email": email})
    if not user:
        print(f"[UPDATE-PHOTO] User {email} not found")
        return jsonify({"error": "User not found. Please register first."}), 404
    
    user_id = user.get('user_id')
    old_embedding = user.get('embedding')
    
    # Save photo file
    filename = secure_filename(f"{email}_photo_{uuid.uuid4()}.jpg")
    my_ip = get_local_ip()
    image_url = f"{final_ip}/uploads/{filename}"

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    print(f"[UPDATE-PHOTO] Saved file to {path}")

    # Get face embeddings
    locations, encodings = get_face_embeddings(path)
    print(f"[UPDATE-PHOTO] Found {len(encodings)} face(s) in photo")
    
    if len(encodings) > 0:
        new_embedding = encodings[0].tolist()
        
        # Update user embedding
        users_col.update_one(
            {"email": email}, 
            {"$set": {
                "embedding": new_embedding,
                "image_path":image_url,
                "embedding_updated_at": datetime.datetime.now()
            }}
        )
        print(f"[UPDATE-PHOTO] Updated embedding for user {email} (user_id: {user_id})")
        
        # ===== STRANGER-TO-USER CONVERSION =====
        # If user didn't have embedding before, check for matching strangers in photos
        if old_embedding is None:
            print(f"[UPDATE-PHOTO] User had no embedding before. Checking for matching strangers...")
            
            # Get all strangers
            all_strangers = list(strangers_col.find({}))
            
            # Find if new embedding matches any existing stranger
            matching_stranger_id = None
            for stranger_doc in all_strangers:
                stranger_embedding = stranger_doc.get('embedding')
                if stranger_embedding and faces_match(np.array(new_embedding), np.array(stranger_embedding)):
                    matching_stranger_id = stranger_doc.get('face_id')
                    print(f"[UPDATE-PHOTO] Found matching stranger: {matching_stranger_id}")
                    break
            
            # If we found a matching stranger, convert all photos with that stranger to this user
            if matching_stranger_id:
                print(f"[UPDATE-PHOTO] Converting stranger {matching_stranger_id} to user {user_id} in all photos...")
                
                # Update all photos that contain this stranger
                photos_with_stranger = photos_col.find({"persons_present": matching_stranger_id})
                conversion_count = 0
                
                for photo in photos_with_stranger:
                    # Replace stranger_id with user_id in persons_present
                    new_persons_present = photo.get('persons_present', [])
                    update_data = {}
                    
                    if matching_stranger_id in new_persons_present:
                        new_persons_present.remove(matching_stranger_id)
                        if user_id not in new_persons_present:
                            new_persons_present.append(user_id)
                        update_data["persons_present"] = new_persons_present
                    
                    # Also update ai_info to change stranger details to user details
                    new_ai_info = photo.get('ai_info', [])
                    for face_info in new_ai_info:
                        if face_info.get('id') == matching_stranger_id:
                            old_type = face_info.get('type')
                            face_info['id'] = user_id
                            face_info['type'] = 'User'
                            face_info['email'] = email  # <--- ADD THIS LINE
                            print(f"[REGISTER/UPDATE] Updated ai_info for photo {photo['_id']}: changed face from '{old_type}' to 'User'")
                            break  # Only update this specific face
                    
                    if new_ai_info:
                        update_data["ai_info"] = new_ai_info
                    
                    if update_data:
                        photos_col.update_one(
                            {"_id": photo['_id']},
                            {"$set": update_data}
                        )
                        conversion_count += 1
                        print(f"[UPDATE-PHOTO] Converted photo {photo['_id']}: {matching_stranger_id} -> {user_id}")
                
                print(f"[UPDATE-PHOTO] Converted {conversion_count} photos from stranger to user")
                
                # Delete the stranger entry since they're now identified as a user
                strangers_col.delete_one({"face_id": matching_stranger_id})
                print(f"[UPDATE-PHOTO] Deleted stranger entry {matching_stranger_id}")
    
    # Save NEW photo to photos collection for full processing
    my_ip = get_local_ip()
    image_url = f"{final_ip}/uploads/{filename}"
    
    photo_doc = {
        "filename": filename,
        "local_path": path,
        "image_url": image_url,
        "location_data": {"source": "photo_update"},
        "owner_email": email,
        "owner_id": user_id,
        "status": "pending",
        "faces_found": 0,
        "ai_info": [],
        "persons_present": [user_id] if len(encodings) > 0 else [],
        "created_at": datetime.datetime.now()
    }
    
    result = photos_col.insert_one(photo_doc)
    photo_id = str(result.inserted_id)
    print(f"[UPDATE-PHOTO] Created photo document {photo_id}")

    return jsonify({
        "message": f"Photo updated successfully for {email}",
        "photo_id": photo_id,
        "user_id": user_id
    })

@app.route('/upload', methods=['POST'])
def upload_image():
    """
    Main endpoint: Receives image + location + owner_email -> Saves to MongoDB with status='pending'
    Processing happens via background worker
    """
    print("[UPLOAD] Received image upload request")
    if 'file' not in request.files:
        print("[UPLOAD] No file in request")
        return jsonify({"error": "No file"}), 400
    
    # Get metadata from request
    location_data = request.form.get('location_data')
    owner_email = request.form.get('owner_email')  # Email of person uploading
    
    if location_data:
        try:
            location_data = json.loads(location_data)
        except:
            location_data = {}
    else:
        location_data = {}
    
    if not owner_email:
        print("[UPLOAD] No owner email provided")
        owner_email = "anonymous"
    
    # 1. Save Image to disk
    file = request.files['file']
    unique_id = str(uuid.uuid4())
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
    filename = f"{unique_id}.{ext}"
    local_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(local_path)
    print(f"[UPLOAD] Saved file to {local_path}")

    # Generate accessible URL
    my_ip = get_local_ip()
    image_url = f"{final_ip}/uploads/{filename}"
    print(f"[UPLOAD] Image URL: {image_url}")

    # Get owner's user_id if they exist
    owner_user = users_col.find_one({"email": owner_email})
    owner_id = owner_user.get('user_id') if owner_user else None

    # 2. Create document in MongoDB with status='pending'
    photo_doc = {
        "filename": filename,
        "local_path": local_path,
        "image_url": image_url,
        "location_data": location_data,
        "owner_email": owner_email,  # Track who uploaded this
        "owner_id": owner_id,
        "status": "pending",  # Will be set to 'processing' then 'done' by background worker
        "faces_found": 0,
        "ai_info": [],
        "persons_present": [],  # Will be populated during processing
        "created_at": datetime.datetime.now()
    }
    
    result = photos_col.insert_one(photo_doc)
    photo_id = str(result.inserted_id)
    print(f"[UPLOAD] Created photo document {photo_id} in MongoDB")

    return jsonify({
        "status": "queued",
        "photo_id": photo_id,
        "image_url": image_url,
        "message": "Photo queued for processing"
    }), 200

@app.route('/shared-photos', methods=['GET'])
def get_shared_photos():
    """
    Client polls this to see if any photos were shared with them.
    Requires header: 'X-User-Email'
    """
    email = request.headers.get('X-User-Email')
    if not email:
        return jsonify([]), 200 # Return empty if no user identified

    # Find pending items
    items = list(shared_queue_col.find({"recipient_email": email, "status": "pending"}))
    
    results = []
    for item in items:
        results.append({
            "photo_id": item.get('photo_id'),
            "url": item['image_url'],
            "metadata": item.get('metadata', {})
        })
        # Mark as delivered so we don't send it again
        shared_queue_col.update_one({"_id": item["_id"]}, {"$set": {"status": "delivered"}})
        
    return jsonify(results)

# --- NEW ENDPOINTS FOR GALLERY ---

@app.route('/photos', methods=['GET'])
def get_all_photos():
    """
    Fetch all photos from MongoDB for gallery display.
    Can optionally filter by status: ?status=done
    """
    status_filter = request.args.get('status', 'done')
    
    try:
        query = {"status": status_filter} if status_filter else {}
        photos = list(photos_col.find(query).sort("created_at", -1))
        
        # Convert ObjectId to string for JSON serialization
        for photo in photos:
            photo['_id'] = str(photo['_id'])
            if 'created_at' in photo:
                photo['created_at'] = photo['created_at'].isoformat()
            if 'processed_at' in photo:
                photo['processed_at'] = photo['processed_at'].isoformat()
            
            # Debug logging for persons_present
            persons_present = photo.get('persons_present', [])
            owner_email = photo.get('owner_email', 'unknown')
            # print(f"[GET_PHOTOS] Photo {photo['_id']}: owner={owner_email}, persons_present={persons_present}")
        
        print(f"[GET_PHOTOS] Returning {len(photos)} photos with status={status_filter}")
        return jsonify(photos), 200
    except Exception as e:
        print(f"[GET_PHOTOS] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/photos/<photo_id>', methods=['GET'])
def get_photo_details(photo_id):
    """
    Fetch details of a single photo by ID
    """
    try:
        photo = photos_col.find_one({"_id": ObjectId(photo_id)})
        if not photo:
            return jsonify({"error": "Photo not found"}), 404
        
        photo['_id'] = str(photo['_id'])
        if 'created_at' in photo:
            photo['created_at'] = photo['created_at'].isoformat()
        if 'processed_at' in photo:
            photo['processed_at'] = photo['processed_at'].isoformat()
        
        return jsonify(photo), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/photos/<photo_id>', methods=['DELETE'])
def delete_photo(photo_id):
    """
    Delete a photo from MongoDB and file system
    """
    try:
        # Find the photo document
        photo = photos_col.find_one({"_id": ObjectId(photo_id)})
        if not photo:
            return jsonify({"error": "Photo not found"}), 404
        
        # Delete from file system
        if 'local_path' in photo and os.path.exists(photo['local_path']):
            os.remove(photo['local_path'])
            print(f"[DELETE] Deleted file: {photo['local_path']}")
        
        # Delete from MongoDB
        photos_col.delete_one({"_id": ObjectId(photo_id)})
        print(f"[DELETE] Deleted photo document: {photo_id}")
        
        return jsonify({"status": "deleted"}), 200
    except Exception as e:
        print(f"[DELETE] Error: {e}")
        return jsonify({"error": str(e)}), 500

# Serve static files (the images)
@app.route('/uploads/<path:filename>')
def serve_image(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    if os.path.exists(filepath):
        # Calculate size in Megabytes
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[NETWORK] Serving: {filename} | Size: {size_mb:.2f} MB")
    else:
        print(f"[NETWORK] File not found: {filename}")
        
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/search', methods=['POST'])
def search_photos():
    """
    Search photos using Semantic Text Matching (Gemini Embeddings) 
    and Geographic Radius Filtering (Haversine).
    """
    data = request.json or {}
    query_text = data.get('query', '').strip()
    target_lat = data.get('lat')
    target_lon = data.get('lon')
    radius_km = data.get('radius', 50.0) # Default 50km radius
    user_id = data.get('user_id')

    # Base query: Only return fully processed photos
    # You can restrict to photos owned by or shared with the user here if needed
    db_query = {"status": "done"}
    # if user_id:
    #     db_query["persons_present"] = user_id

    all_photos = list(photos_col.find(db_query))
    filtered_photos = []

    # 1. GEOGRAPHIC FILTERING (If coordinates provided)
    if target_lat is not None and target_lon is not None:
        for photo in all_photos:
            loc = photo.get('location_data', {})
            loc1 = loc.get('coords', {})
            # print(photo)
            # Handle different common coordinate key names
            p_lat = loc1.get('latitude') or loc.get('lat')
            p_lon = loc1.get('longitude') or loc.get('lon')
            if(not p_lat):
                continue
            print(str(p_lat) + " | " + str(p_lon))
            
            if p_lat is not None and p_lon is not None:
                try:
                    distance = haversine(float(target_lat), float(target_lon), float(p_lat), float(p_lon))
                    if distance <= radius_km:
                        photo['distance_km'] = distance
                        filtered_photos.append(photo)
                except Exception as e:
                    print(f"[SEARCH] Geo parse error for photo {photo['_id']}: {e}")
    else:
        filtered_photos = all_photos

    # 2. SEMANTIC TEXT FILTERING (If text query provided)
    if query_text:
        try:
            # Generate embedding for the search query locally
            query_embedding = text_embedder.encode(query_text).tolist()

            scored_photos = []
            for photo in filtered_photos:
                doc_embedding = photo.get('text_embedding', [])
                
                if doc_embedding and len(doc_embedding) > 0:
                    # 1 - cosine distance = cosine similarity
                    similarity = 1 - cosine(query_embedding, doc_embedding)
                    photo['relevance'] = float(similarity)
                else:
                    photo['relevance'] = 0.0
                
                scored_photos.append(photo)
            
            # Sort by highest relevance score
            filtered_photos = sorted(scored_photos, key=lambda x: x['relevance'], reverse=True)
            
            # Optional: Filter out low relevance matches (e.g., threshold < 0.2 for MiniLM)
            # filtered_photos = [p for p in filtered_photos if p['relevance'] > 0.2]

        except Exception as e:
            print(f"[SEARCH] Semantic search error: {e}")
            return jsonify({"error": "Failed to perform semantic search"}), 500

    # Format the payload for JSON response
    for photo in filtered_photos:
        photo['_id'] = str(photo['_id'])
        if 'created_at' in photo:
            photo['created_at'] = photo['created_at'].isoformat()
        if 'processed_at' in photo:
            photo['processed_at'] = photo['processed_at'].isoformat()
        # Strip out the heavy embedding array before sending to frontend
        photo.pop('text_embedding', None) 

    print(f"[SEARCH] Returning {len(filtered_photos)} results.")
    return jsonify(filtered_photos), 200


# ==============================================================================
# SEARCH BY FACE  — finds all photos containing a matching face (stranger-aware)
# ==============================================================================
@app.route('/search-by-face', methods=['POST'])
def search_by_face():
    """
    Receive an uploaded face image and return all processed photos that contain
    a face matching the uploaded one.

    Strategy:
      1. Extract embedding from the query image.
      2. Compare against every face stored in photos (ai_info[].id) by
         looking up embeddings from users_col and strangers_col.
      3. Return the matching photos with a 'matched_person' annotation.

    Strangers that match are still returned — labelled 'Stranger'.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filename = secure_filename(f"query_{uuid.uuid4()}.jpg")
    tmp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(tmp_path)
    print(f"[SEARCH-FACE] Saved query image to {tmp_path}")

    try:
        locations, encodings = get_face_embeddings(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        return jsonify({"error": f"Face extraction failed: {e}"}), 500

    if not encodings:
        os.remove(tmp_path)
        return jsonify({"error": "No face detected in the uploaded image"}), 400

    query_embedding = encodings[0]  # Use the first (largest) face
    print(f"[SEARCH-FACE] Extracted query embedding successfully")

    # --- Build a lookup: person_id -> embedding ---
    all_users = list(users_col.find({}))
    all_strangers = list(strangers_col.find({}))

    person_embedding_map = {}
    person_label_map = {}  # person_id -> display label

    for u in all_users:
        pid = u.get('user_id') or u.get('email')
        emb = u.get('embedding')
        if pid and emb:
            person_embedding_map[pid] = emb
            person_label_map[pid] = u.get('name') or u.get('email') or pid

    for s in all_strangers:
        pid = s.get('face_id')
        emb = s.get('embedding')
        if pid and emb:
            person_embedding_map[pid] = emb
            person_label_map[pid] = "Stranger"

    # --- Find matching person IDs ---
    matched_person_ids = []
    for pid, emb in person_embedding_map.items():
        if faces_match(query_embedding, np.array(emb)):
            matched_person_ids.append(pid)
            label = person_label_map.get(pid, "Stranger")
            print(f"[SEARCH-FACE] Match found: {pid} ({label})")

    # If no person matched but we still want to brute-force check photos directly
    # (for photos whose strangers may not have their own DB entry), we also do a
    # direct photo-level embedding comparison using stored ai_info bboxes approach.
    # For now the person_embedding_map covers all stored identities.

    if not matched_person_ids:
        # Try direct comparison against every photo's ai_info embeddings
        # This is a fallback for edge cases
        os.remove(tmp_path)
        return jsonify([]), 200

    # --- Fetch all photos that contain matched person IDs ---
    matching_photos = list(photos_col.find({
        "status": "done",
        "persons_present": {"$in": matched_person_ids}
    }).sort("created_at", -1))

    # Annotate and serialize
    results = []
    for photo in matching_photos:
        photo['_id'] = str(photo['_id'])
        if 'created_at' in photo:
            photo['created_at'] = photo['created_at'].isoformat()
        if 'processed_at' in photo:
            photo['processed_at'] = photo['processed_at'].isoformat()
        photo.pop('text_embedding', None)

        # Attach which matched person(s) are in this photo
        matched_in_photo = [pid for pid in matched_person_ids if pid in photo.get('persons_present', [])]
        photo['matched_persons'] = [
            {"id": pid, "label": person_label_map.get(pid, "Stranger")}
            for pid in matched_in_photo
        ]
        results.append(photo)

    os.remove(tmp_path)
    print(f"[SEARCH-FACE] Returning {len(results)} photos for {len(matched_person_ids)} matched person(s)")
    return jsonify(results), 200


# ==============================================================================
# CO-OCCURRENCE GRAPH  — who appears together most often?
# ==============================================================================
@app.route('/co-occurrence', methods=['GET'])
def co_occurrence():
    """
    Build a co-occurrence graph from all processed photos.
    Returns nodes (people) and edges (shared-photo counts).
    Strangers are included with their face_id as label 'Stranger'.
    """
    all_photos = list(photos_col.find({"status": "done"}, {"persons_present": 1, "ai_info": 1}))

    edge_counts = {}   # frozenset({id1, id2}) -> count
    node_ids = set()

    for photo in all_photos:
        persons = photo.get('persons_present', [])
        if len(persons) < 2:
            continue
        node_ids.update(persons)
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                key = tuple(sorted([persons[i], persons[j]]))
                edge_counts[key] = edge_counts.get(key, 0) + 1

    # Build label lookup
    all_users = {u.get('user_id'): (u.get('name') or u.get('email') or u.get('user_id'))
                 for u in users_col.find({}, {"user_id": 1, "name": 1, "email": 1})}
    all_strangers = {s.get('face_id'): "Stranger" for s in strangers_col.find({}, {"face_id": 1})}

    def get_label(pid):
        if pid in all_users:
            return all_users[pid]
        if pid in all_strangers:
            return f"Stranger ({pid[-4:]})"
        return pid

    nodes = [{"id": pid, "label": get_label(pid), "isStranger": pid.startswith("stranger_")}
             for pid in node_ids]

    edges = [{"source": k[0], "target": k[1], "weight": v}
             for k, v in edge_counts.items()]

    print(f"[CO-OCCUR] {len(nodes)} nodes, {len(edges)} edges")
    return jsonify({"nodes": nodes, "edges": edges}), 200


# ==============================================================================
# HEATMAP  — GPS density from all processed photos
# ==============================================================================
@app.route('/heatmap', methods=['GET'])
def heatmap():
    """
    Aggregate all GPS coordinates from processed photos.
    Returns a list of {lat, lon, weight} points for density rendering.
    """
    photos = list(photos_col.find({"status": "done"}, {"location_data": 1, "created_at": 1}))

    points = []
    for photo in photos:
        loc = photo.get('location_data', {})
        coords = loc.get('coords', {})
        lat = coords.get('latitude') or coords.get('lat') or loc.get('lat')
        lon = coords.get('longitude') or coords.get('lon') or loc.get('lon')
        if lat is not None and lon is not None:
            try:
                points.append({"lat": float(lat), "lon": float(lon), "weight": 1})
            except (ValueError, TypeError):
                pass

    print(f"[HEATMAP] Returning {len(points)} GPS points")
    return jsonify(points), 200


# ==============================================================================
# PERSONS LIST  — all users + strangers for geofence selection UI
# ==============================================================================
@app.route('/persons-list', methods=['GET'])
def persons_list():
    """
    Return all known persons (users + strangers) with their display image
    for the geofence alert setup UI.
    """
    users = list(users_col.find({}, {"_id": 0, "user_id": 1, "name": 1, "email": 1, "image_path": 1}))
    strangers = list(strangers_col.find({}, {"_id": 0, "face_id": 1, "name": 1, "image_url": 1, "bbox": 1, "source_photo_id": 1}))

    result_users = [
        {
            "id": u.get("user_id"),
            "label": u.get("name") or u.get("email") or u.get("user_id"),
            "image": u.get("image_path"),
            "type": "user"
        }
        for u in users if u.get("user_id")
    ]

    result_strangers = [
        {
            "id": s.get("face_id"),
            "label": f"Stranger ({s.get('face_id', '')[-4:]})",
            "image": s.get("image_url"),
            "type": "stranger"
        }
        for s in strangers if s.get("face_id")
    ]

    return jsonify(result_users + result_strangers), 200


# ==============================================================================
# GEOFENCES & ALERTS
# ==============================================================================
@app.route('/geofences', methods=['GET'])
def get_geofences():
    geofences = list(geofences_col.find({}))
    for g in geofences:
        g['_id'] = str(g['_id'])
        if 'created_at' in g:
            g['created_at'] = g['created_at'].isoformat()
    return jsonify(geofences), 200

@app.route('/geofences', methods=['POST'])
def create_geofence():
    data = request.json or {}
    name = data.get('name')
    person_id = data.get('person_id')
    polygon = data.get('polygon')
    
    if not name or not person_id or not polygon:
        return jsonify({"error": "name, person_id, and polygon are required"}), 400
        
    gf_doc = {
        "name": name,
        "person_id": person_id,
        "polygon": polygon,
        "created_at": datetime.datetime.now()
    }
    result = geofences_col.insert_one(gf_doc)
    print(f"[GEOFENCE] Created geofence {result.inserted_id} for person {person_id}")
    return jsonify({"_id": str(result.inserted_id), "message": "Geofence created"}), 201

@app.route('/geofences/<gf_id>', methods=['DELETE'])
def delete_geofence(gf_id):
    try:
        geofences_col.delete_one({"_id": ObjectId(gf_id)})
        return jsonify({"status": "deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/geofences/check', methods=['POST'])
def manual_geofence_check():
    # Trigger logic is handled continuously by background thread
    return jsonify({"status": "ok"}), 200

@app.route('/alerts', methods=['GET'])
def get_alerts():
    """Return all triggered alerts, most-recent first."""
    alerts = list(alerts_col.find({"type": "geofence_trigger"}).sort("created_at", -1))
    for a in alerts:
        a['_id'] = str(a['_id'])
        if 'created_at' in a:
            a['created_at'] = a['created_at'].isoformat()
        if 'triggered_at' in a:
            a['triggered_at'] = a['triggered_at'].isoformat()
    return jsonify(alerts), 200


@app.route('/alerts/<alert_id>/seen', methods=['PATCH'])
def mark_alert_seen(alert_id):
    """Mark a specific alert as seen."""
    try:
        alerts_col.update_one({"_id": ObjectId(alert_id)}, {"$set": {"seen": True}})
        return jsonify({"status": "updated"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/alerts/<alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """Delete an alert rule."""
    try:
        alerts_col.delete_one({"_id": ObjectId(alert_id)})
        return jsonify({"status": "deleted"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/alerts/mark-all-seen', methods=['PATCH'])
def mark_all_alerts_seen():
    """Mark all alerts as seen."""
    alerts_col.update_many({"seen": False}, {"$set": {"seen": True}})
    return jsonify({"status": "all marked seen"}), 200


def is_point_in_polygon(lat, lon, polygon):
    x, y = lon, lat
    inside = False
    n = len(polygon)
    
    def get_coords(pt):
        if isinstance(pt, dict):
            return pt.get('lon', pt.get('lng', 0)), pt.get('lat', 0)
        elif isinstance(pt, list) and len(pt) >= 2:
            return pt[1], pt[0]
        return 0, 0
        
    if n < 3: return False
    p1x, p1y = get_coords(polygon[0])
    for i in range(n + 1):
        p2x, p2y = get_coords(polygon[i % n])
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# ==============================================================================
# BACKGROUND GEOFENCE CHECKER
# ==============================================================================
def geofence_checker():
    print("[GEOFENCE] Geofence checker started")
    while processing_active:
        try:
            geofences = list(geofences_col.find({}))
            # print("geofences are", geofences)
            if not geofences:
                # print("none: ", geofences)
                time.sleep(15)
                continue

            cutoff = datetime.datetime.now() - datetime.timedelta(minutes=5)
            recent_photos = list(photos_col.find({
                "status": "done",
                # "processed_at": {"$gte": cutoff}
            }))
            # print("here")

            for photo in recent_photos:
                # print("[GEOFENCE] Geofence checker is checking a photo")

                loc = photo.get('location_data', {})
                coords = loc.get('coords', {})
                p_lat = coords.get('latitude') or coords.get('lat') or loc.get('lat')
                p_lon = coords.get('longitude') or coords.get('lon') or loc.get('lon')

                if p_lat is None or p_lon is None:
                    continue

                persons = photo.get('persons_present', [])
                for gf in geofences:
                    if gf.get('person_id') not in persons:
                        continue
                    
                    poly = gf.get('polygon', [])
                    if not poly:
                        continue

                    if is_point_in_polygon(float(p_lat), float(p_lon), poly):
                        existing = alerts_col.find_one({
                            "geofence_id": str(gf['_id']),
                            "photo_id": str(photo['_id'])
                        })
                        if not existing:
                            person_label = "Unknown"
                            if gf.get('person_id'):
                                u = users_col.find_one({"user_id": gf['person_id']})
                                if u: person_label = u.get('name') or u.get('email')
                                else:
                                    s = strangers_col.find_one({"face_id": gf['person_id']})
                                    if s: person_label = f"Unknown ({s['face_id'][-4:]})"

                            notification = {
                                "geofence_id": str(gf['_id']),
                                "geofence_name": gf.get('name'),
                                "person_id": gf['person_id'],
                                "person_label": person_label,
                                "photo_id": str(photo['_id']),
                                "image_url": photo.get('image_url'),
                                "lat": float(p_lat),
                                "lon": float(p_lon),
                                "seen": False,
                                "triggered_at": datetime.datetime.now(),
                                "created_at": datetime.datetime.now(),
                                "type": "geofence_trigger"
                            }
                            alerts_col.insert_one(notification)
                            print(f"[GEOFENCE] 🚨 TRIGGERED: {person_label} entered {gf.get('name')}!")

        except Exception as e:
            print(f"[GEOFENCE] Error in geofence checker: {e}")
        time.sleep(15)


geofence_thread = threading.Thread(target=geofence_checker, daemon=True)
geofence_thread.start()
print("[APP] Geofence checker thread started")


def run_server():
    # This function is what hupper will execute to start the worker
    print("[SERVER] Started Waitress with hot reloading...")
    serve(app, host='0.0.0.0', port=5069, threads=64, connection_limit=200, channel_timeout=30)

if __name__ == '__main__':
    # Point to the function: 'filename.function_name'
    # Use 'app.run_server' because your file is named app.py
    reloader = hupper.start_reloader('app.run_server') 
    
    # Start the server the first time
    run_server()