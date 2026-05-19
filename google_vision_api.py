import json
import os
from google.cloud import vision
from typing import List, Dict

def get_image_keywords(image_path: str, max_results: int = 10) -> Dict:
    """
    Detect labels/keywords from an image using Google Cloud Vision API.
    
    Args:
        image_path (str): Path to the local image file
        max_results (int): Number of keywords to return (default 10)
    
    Returns:
        Dict: JSON-like structure with keywords and confidence scores
    """
    # Initialize the client
    client = vision.ImageAnnotatorClient()

    # Read the image file
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Perform label detection
    response = client.label_detection(image=image, max_results=max_results)
    
    # Check for errors
    if response.error.message:
        raise Exception(f"Vision API Error: {response.error.message}")

    labels = response.label_annotations

    # Prepare clean JSON output
    keywords = []
    for label in labels:
        keywords.append({
            "keyword": label.description,
            "score": round(float(label.score), 4)  # Confidence score 0-1
        })

    result = {
        "image": os.path.basename(image_path),
        "keywords": keywords,
        "total_keywords": len(keywords)
    }

    return result


# Example usage
if __name__ == "__main__":
    image_path = "abb.png"   # ← Change this
    
    try:
        result = get_image_keywords(image_path, max_results=10)
        
        # Print nicely formatted JSON
        print(json.dumps(result, indent=2))
        
        # Optionally save to file
        with open("image_keywords.json", "w") as f:
            json.dump(result, f, indent=2)
            
    except Exception as e:
        print(f"Error: {e}")