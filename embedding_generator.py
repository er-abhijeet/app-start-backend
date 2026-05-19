import os
import time
import json
from pymongo import MongoClient
from bson.objectid import ObjectId
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument
from sentence_transformers import SentenceTransformer
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/") # Update if using Atlas
KEYS_FILE = "keys.txt"

# Exact prompt used in your app.py to guarantee schema consistency
GEMINI_PROMPT = """
Analyze this image for a photo gallery search engine. 
Provide a brief, objective description of the scene, location, and activities.
Provide an array of 10 to 15 highly relevant search tags/keywords (e.g., "beach", "sunset", "wedding", "crowd").

Return ONLY a JSON object with this exact schema:
{
    "description": "string",
    "tags": ["string", "string"]
}
"""

# ==========================================
# KEY MANAGER
# ==========================================
class GeminiKeyManager:
    def __init__(self, keys_file):
        if not os.path.exists(keys_file):
            raise FileNotFoundError(f"Key file '{keys_file}' not found. Please create it with one API key per line.")
        
        with open(keys_file, 'r') as f:
            # Read lines, strip whitespace, remove empty lines
            self.keys = [line.strip() for line in f if line.strip()]
            
        if not self.keys:
            raise ValueError(f"No keys found in '{keys_file}'.")
            
        print(f"[KEY MANAGER] Loaded {len(self.keys)} API keys.")
        self.current_idx = 0
        self.invalid_keys = set()
        self.configure_current_key()

    def configure_current_key(self):
        current_key = self.keys[self.current_idx]
        genai.configure(api_key=current_key)
        print(f"[KEY MANAGER] Switched to Key ending in ...{current_key[-4:]}")

    def rotate_key(self, reason="Rate Limit"):
        print(f"[KEY MANAGER] Rotating key due to: {reason}")
        start_idx = self.current_idx
        
        while True:
            self.current_idx = (self.current_idx + 1) % len(self.keys)
            
            # If we cycled completely around, all keys are exhausted/invalid
            if self.current_idx == start_idx:
                valid_count = len(self.keys) - len(self.invalid_keys)
                if valid_count == 0:
                    raise SystemExit("[FATAL] All provided API keys are invalid or revoked. Cannot proceed.")
                
                print("[KEY MANAGER] All valid keys are currently rate-limited. Sleeping for 60 seconds...")
                time.sleep(60)
                # After sleeping, we assume rate limits have reset.
                self.configure_current_key()
                return

            if self.current_idx not in self.invalid_keys:
                self.configure_current_key()
                return

    def mark_current_key_invalid(self, reason="Invalid/Revoked"):
        print(f"[KEY MANAGER] WARNING: Key ending in ...{self.keys[self.current_idx][-4:]} marked INVALID ({reason}). Dropping from rotation.")
        self.invalid_keys.add(self.current_idx)
        self.rotate_key(reason="Invalid Key Override")


# ==========================================
# GEMINI EXTRACTION WITH RETRIES
# ==========================================
def extract_metadata_robust(image_path, key_manager):
    """
    Attempts to extract metadata using Gemini, handling rate limits and rotating keys as needed.
    """
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        generation_config={"response_mime_type": "application/json"}
    )
    
    while True:
        try:
            with Image.open(image_path) as img:
                response = model.generate_content([GEMINI_PROMPT, img])
                metadata = json.loads(response.text)
                return metadata
                
        except ResourceExhausted as e:
            # 429 Quota Exceeded or Rate Limited
            key_manager.rotate_key(reason="ResourceExhausted (429 Rate Limit/Quota)")
            
        except (PermissionDenied, InvalidArgument) as e:
            # 403 or 400 Invalid API Key
            key_manager.mark_current_key_invalid(reason=str(e))
            
        except json.JSONDecodeError:
            print("[GEMINI] Error: Model did not return valid JSON. Retrying...")
            time.sleep(2)
            
        except Exception as e:
            print(f"[GEMINI] Unexpected Error: {e}. Retrying in 5s...")
            time.sleep(5)


# ==========================================
# MAIN BACKFILL SCRIPT
# ==========================================
def main():
    print("[INIT] Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client.new
    photos_col = db.photos

    print("[INIT] Initializing Key Manager...")
    key_manager = GeminiKeyManager(KEYS_FILE)

    print("[INIT] Loading local text embedding model (all-MiniLM-L6-v2)...")
    text_embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Query for photos that are 'done' but missing either tags or embeddings
    # Using $size to check for empty arrays as per app.py initialization
    query = {
        "status": "done",
        "$or": [
            {"text_embedding": {"$exists": False}},
            {"text_embedding": []},
            {"tags": {"$exists": False}},
            {"tags": []}
        ]
    }
    
    photos_to_process = list(photos_col.find(query))
    total = len(photos_to_process)
    print(f"\n[SCAN] Found {total} photos requiring metadata/embedding backfill.\n")

    success_count = 0

    for index, photo in enumerate(photos_to_process, 1):
        photo_id = photo['_id']
        local_path = photo.get('local_path')
        
        print(f"--- Processing {index}/{total}: Photo {photo_id} ---")
        
        if not local_path or not os.path.exists(local_path):
            print(f"[SKIP] Local image file missing at {local_path}.")
            continue

        update_payload = {}
        
        # 1. Determine what is missing
        has_tags = bool(photo.get('tags'))
        
        description = photo.get('description', '')
        tags = photo.get('tags', [])

        # 2. Call Gemini if Tags/Description are missing
        if not has_tags:
            print("[ACTION] Missing tags. Calling Gemini AI...")
            metadata = extract_metadata_robust(local_path, key_manager)
            
            description = metadata.get("description", "")
            tags = metadata.get("tags", [])
            
            update_payload["description"] = description
            update_payload["tags"] = tags
            print(f" -> Extracted {len(tags)} tags.")
        else:
            print("[ACTION] Tags exist. Skipping Gemini AI.")

        # 3. Generate Embeddings (Always done if we reach here, because if tags existed, 
        # the query ensures text_embedding was missing to get into this loop)
        print("[ACTION] Generating local text embeddings...")
        text_for_embedding = f"{description} " + " ".join(tags)
        
        if text_for_embedding.strip():
            embedding = text_embedder.encode(text_for_embedding).tolist()
            update_payload["text_embedding"] = embedding
            print(" -> Vector embedding generated successfully.")
        else:
            print("[ERROR] Cannot generate embedding: Text string is empty.")

        # 4. Update Database
        if update_payload:
            photos_col.update_one(
                {"_id": ObjectId(photo_id)},
                {"$set": update_payload}
            )
            success_count += 1
            print(f"[SUCCESS] Document {photo_id} updated.\n")

    print(f"=== BACKFILL COMPLETE ===")
    print(f"Successfully processed and updated {success_count} out of {total} identified photos.")

if __name__ == "__main__":
    main()