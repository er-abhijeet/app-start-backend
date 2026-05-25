import os
import time
import json
import random
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
load_dotenv()


# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_FOLDER = "./test_images/adii"  # <-- Change this to your folder of images
OWNER_EMAIL = "testing@example.com"  # <-- Change to the user you are simulating
BACKEND_URL = "https://gallery.snorlax.codes"
MONGO_URI = os.getenv("MONGO_URI", "your_mongodb_connection_string_here")


# Pre-defined coordinates for Naya Raipur
LOCATIONS = [
    # {"name": "IIIT Naya Raipur", "lat": 21.1285, "lon": 81.7662},
    # {"name": "Sector 28", "lat": 21.1449, "lon": 81.7820},
    # {"name": "Sector 29", "lat": 21.1415, "lon": 81.7915},
    # {"name": "North Block", "lat": 21.1610, "lon": 81.7876},
    # {"name": "Cricket Stadium", "lat": 21.1643, "lon": 81.7836},
    # {"name": "Telibandha Chowk", "lat": 21.238691, "lon": 81.671619},
    # {"name": "Budha Talab", "lat": 21.233882, "lon": 81.633333},
    {"name": "Ghadi Chowk", "lat": 21.245407, "lon": 81.641741},
]

def generate_expo_location(loc):
    """Formats the location exactly like React Native's expo-location"""
    return {
        "coords": {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "altitude": 300,
            "accuracy": 5.0,
            "heading": 0,
            "speed": 0
        },
        "address": {
            "city": "Naya Raipur",
            "name": loc["name"]
        }
    }

def main():
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Folder '{IMAGE_FOLDER}' not found.")
        return

    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("No images found in the directory.")
        return

    print(f"Found {len(image_files)} images. Starting upload simulation...")

    # Start date: April 14, 2026 with current time
    now = datetime.now()
    current_simulated_date = datetime(2026, 4, 14, now.hour, now.minute, now.second) 
    uploaded_records = []

    # ==========================================
    # PHASE 1: HIT THE UPLOAD ENDPOINT
    # ==========================================
    for img_file in image_files:
        filepath = os.path.join(IMAGE_FOLDER, img_file)
        
        # Pick a random location
        loc = random.choice(LOCATIONS)
        expo_location_data = generate_expo_location(loc)
        
        # Keep current time (no random hours added)

        try:
            with open(filepath, 'rb') as f:
                print(f"Uploading {img_file} at {loc['name']}...")
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    data={
                        "owner_email": OWNER_EMAIL,
                        "location_data": json.dumps(expo_location_data)
                    },
                    files={"file": (img_file, f, "image/jpeg")}
                )

            if response.status_code == 200:
                res_data = response.json()
                photo_id = res_data.get("photo_id")
                print(f" -> Success! Photo ID: {photo_id}")
                
                # Keep track of the ID and the date we want it to have
                uploaded_records.append({
                    "photo_id": photo_id,
                    "simulated_date": current_simulated_date
                })
            else:
                print(f" -> Failed: {response.text}")

        except Exception as e:
            print(f" -> Error uploading {img_file}: {e}")

        # Move to the next day for the next photo to build a multi-day history
        current_simulated_date += timedelta(days=1)

    # ==========================================
    # PHASE 2: WAIT FOR PROCESSING & TIME TRAVEL
    # ==========================================
    print("\nAll images uploaded to the pending queue.")
    print("Connecting to database to monitor AI processing queue...")
    
    try:
        client = MongoClient(MONGO_URI)
        db = client.new
        photos_col = db.photos

        # Get a list of the ObjectIds we uploaded
        uploaded_ids = [ObjectId(record["photo_id"]) for record in uploaded_records]

        # Active Polling Loop
        while True:
            # Count how many of our uploaded photos are still not finished
            unfinished_count = photos_col.count_documents({
                "_id": {"$in": uploaded_ids},
                "status": {"$in": ["pending", "processing"]}
            })
            
            if unfinished_count == 0:
                print("[QUEUE] All uploaded images have finished AI processing!")
                break
                
            print(f"[QUEUE] Waiting for AI processing... {unfinished_count} images remaining in queue.")
            time.sleep(5)  # Poll every 5 seconds

        # Run the Time Travel timestamp modifications safely
        print("Applying simulated historical timestamps...")
        updated_count = 0
        for record in uploaded_records:
            sim_date = record["simulated_date"]
            sim_processed_date = sim_date + timedelta(seconds=5)

            result = photos_col.update_one(
                {"_id": ObjectId(record["photo_id"])},
                {"$set": {
                    "created_at": sim_date,
                    "processed_at": sim_processed_date
                }}
            )
            if result.modified_count > 0:
                updated_count += 1

        print(f"Done! Successfully simulated history for {updated_count} photos safely.")

    except Exception as e:
        print(f"Database operation failed: {e}")

if __name__ == "__main__":
    main()