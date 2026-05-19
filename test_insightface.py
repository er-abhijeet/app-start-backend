import cv2
import numpy as np
from insightface.app import FaceAnalysis

# 1. Initialize the App
app = FaceAnalysis(
    name="buffalo_l", 
    providers=['CPUExecutionProvider']
)

# 2. Prepare the execution context
app.prepare(ctx_id=0, det_size=(640, 640))

# 3. Load the image
image_path = 'ab2.png'
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

# 4. Run the inference pipeline
faces = app.get(img)

print(f"Detected {len(faces)} face(s).")

# 5. Extract Embeddings and Draw Bounding Boxes
for i, face in enumerate(faces):
    # The bounding box coordinates [x1, y1, x2, y2]
    bbox = face.bbox.astype(int)
    
    # The L2-normalized 512-dimensional embedding
    embedding = face.normed_embedding
    
    print(f"\nFace {i+1}:")
    print(f"Bounding Box: {bbox}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding vector (first 5 values): {embedding[:5]}")
    
    # --- New Visualization Code ---
    # Draw the bounding box (color format is BGR: (0, 255, 0) = Green)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Add a label above the bounding box
    label = f"Face {i+1}"
    cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 6. Display the annotated image
# cv2.imshow creates a window. It requires a GUI environment.
cv2.imshow("Detected Faces", img)

# Wait indefinitely until the user presses a key
cv2.waitKey(0)

# Clean up and close the window
cv2.destroyAllWindows()

# Optional: To save the annotated image to disk, use:
cv2.imwrite('annotated_output.jpg', img)