import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
import numpy as np
from deepface import DeepFace
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

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- FACE RECOGNITION FUNCTIONS ---

def get_face_embeddings(image_path):
    """
    Extract face locations and embeddings from image using DeepFace FaceNet512
    Returns: (locations, encodings) where locations are bounding boxes and encodings are embeddings
    """
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            enforce_detection=True,
            detector_backend="retinaface", # Uses state-of-the-art RetinaFace detector
            align=True                     # Applies affine transformation to level eyes
        )
        
        locations = []
        encodings = []
        
        for face_data in results:
            # Extract facial area (bounding box)
            area = face_data['facial_area']
            x = area['x']
            y = area['y']
            w = area['w']
            h = area['h']
            # Convert to (top, right, bottom, left) format
            box = (y, x + w, y + h, x)
            locations.append(box)
            
            # Extract and convert embedding to numpy array
            embedding = np.array(face_data['embedding'])
            encodings.append(embedding)
        
        return locations, encodings
    except Exception as e:
        print(f"[FACE] Error extracting face embeddings: {e}")
        return [], []


def l2_normalize(vector):
    """Normalize vector to unit length (required for FaceNet512 L2 distance)"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def faces_match(emb1, emb2):
    """
    Compare two face embeddings using 4 metrics with OR logic.
    Returns True if ANY metric indicates a match.
    
    Metrics (FaceNet512 thresholds):
    1. Cosine similarity >= 0.70
    2. Cosine distance <= 0.30
    3. L2 normalized Euclidean distance <= 1.04
    4. Raw Euclidean distance <= 23.56
    """
    try:
        # Convert to numpy arrays if needed
        emb1 = np.array(emb1)
        emb2 = np.array(emb2)
        
        # Strict threshold: 0.25 (High Security), Default: 0.30 (Standard)
        COSINE_THRESHOLD = 0.30 
        
        distance = cosine(emb1, emb2)
        is_match = distance <= COSINE_THRESHOLD
        
        print(f"[FACE] Comparison: cos_dist={distance:.4f} | Threshold: {COSINE_THRESHOLD} -> {'MATCH' if is_match else 'NO MATCH'}")
        
        return is_match
        
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
        if stored_embedding and faces_match(encoding, stored_embedding):
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
    tlsCAFile=certifi.where(),       # Fixes Windows SSL handshake issues
    maxIdleTimeMS=45000,             # Forces PyMongo to recycle connections before Atlas kills them
    connectTimeoutMS=20000,          
    serverSelectionTimeoutMS=20000,
    retryWrites=True                 # Tells Mongo to automatically retry a failed write once
)
db = client.new
users_col = db.users
strangers_col = db.strangers
shared_queue_col = db.shared_queue
photos_col = db.photos  # New collection for all photos with processing status

# --- BACKGROUND WORKER ---
processing_active = True

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
                        # bbox = {"y": top, "x": right, "h": bottom-top, "w": right-left}
                        bbox = {
                            "y": top, 
                            "x": right, 
                            "h": bottom-top, 
                            "w": right-left,
                            "img_w": img_w,
                            "img_h": img_h
                        }
                        print("bbox",bbox)

                        person_id = None
                        person_type = None
                        person_email = None
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
                                person_name = f"Unknown_{person_id[-4:]}"  # Assign a readable placeholder name
                                person_email = None
                                strangers_col.insert_one({
                                    "face_id": person_id,
                                    "name": person_name,
                                    "embedding": encoding.tolist(),
                                    "source_photo_id": str(photo_id),
                                    "image_url": photo_doc.get('image_url'),
                                    "local_path": local_path,
                                    "bbox": bbox,
                                    "created_at": datetime.datetime.now()
                                })
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
                    
                    # Update photo with processing results and persons_present
                    photos_col.update_one(
                        {"_id": photo_id},
                        {"$set": {
                            "status": "done",
                            "faces_found": len(detected_people),
                            "ai_info": detected_people,
                            "persons_present": persons_present_ids,
                            "processed_at": datetime.datetime.now()
                        }}
                    )
                    
                    print(f"[WORKER] Finished processing photo {photo_id}. Faces found: {len(detected_people)}, Persons: {persons_present_ids}")
                    
                except Exception as e:
                    print(f"[WORKER] Error processing photo: {e}")
                    import traceback
                    traceback.print_exc()
                    photos_col.update_one({"_id": photo_doc['_id']}, {"$set": {"status": "error", "error": str(e)}})
            
            # Sleep before next check
            time.sleep(5)
        
        except Exception as e:
            print(f"[WORKER] Background processor error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

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
            image_url = f"http://{my_ip}:5000/uploads/{filename}"
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
    image_url = f"http://{my_ip}:5000/uploads/{filename}"

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
    image_url = f"http://{my_ip}:5000/uploads/{filename}"
    
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
    image_url = f"http://{my_ip}:5000/uploads/{filename}"
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

def run_server():
    # This function is what hupper will execute to start the worker
    print("[SERVER] Started Waitress with hot reloading...")
    serve(app, host='0.0.0.0', port=5000, threads=8)

if __name__ == '__main__':
    # Point to the function: 'filename.function_name'
    # Use 'app.run_server' because your file is named app.py
    reloader = hupper.start_reloader('app.run_server') 
    
    # Start the server the first time
    run_server()