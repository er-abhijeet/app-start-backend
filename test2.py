import numpy as np
from scipy.spatial.distance import cosine, euclidean

def l2_normalize(vector):
    """Normalizes a vector to length 1. Crucial for face recognition metrics."""
    return vector / np.linalg.norm(vector)

def compare_face_embeddings(emb1_path, emb2_path):
    # 1. Load the saved .npy embeddings
    emb1 = np.load(emb1_path)
    emb2 = np.load(emb2_path)
    
    # L2 Normalize the embeddings (Standard practice before Euclidean comparison)
    emb1_l2 = l2_normalize(emb1)
    emb2_l2 = l2_normalize(emb2)

    # 2. Calculate Metrics
    # Scipy's 'cosine' computes Distance. Similarity is (1 - distance).
    cos_distance = cosine(emb1, emb2)
    cos_similarity = 1 - cos_distance 
    
    euclidean_raw = euclidean(emb1, emb2)
    euclidean_l2_norm = euclidean(emb1_l2, emb2_l2)

    # 3. FaceNet512 Specific Thresholds (Bandwidths)
    # These are the standard boundary lines where a "Bad" score becomes a "Good" match
    thresholds = {
        "cosine_similarity": 0.70, # GOOD: > 0.70
        "cosine_distance": 0.30,   # GOOD: < 0.30
        "euclidean_l2": 1.04,      # GOOD: < 1.04
        "euclidean_raw": 23.56     # GOOD: < 23.56
    }

    print("--- FACE VERIFICATION RESULTS ---")
    
    # --- Cosine Similarity ---
    # Bandwidth: 1.0 is a perfect match, 0.0 is completely different.
    is_match = cos_similarity >= thresholds['cosine_similarity']
    status = "GOOD (MATCH)" if is_match else "BAD (NO MATCH)"
    print(f"1. Cosine Similarity:  {cos_similarity:.4f} | Threshold: >= {thresholds['cosine_similarity']} -> {status}")

    # --- Cosine Distance ---
    # Bandwidth: 0.0 is a perfect match, 1.0+ is completely different.
    is_match = cos_distance <= thresholds['cosine_distance']
    status = "GOOD (MATCH)" if is_match else "BAD (NO MATCH)"
    print(f"2. Cosine Distance:    {cos_distance:.4f} | Threshold: <= {thresholds['cosine_distance']} -> {status}")

    # --- L2 Normalized Euclidean Distance ---
    # Bandwidth: 0.0 is perfect, > 1.04 is bad. 
    # Highly recommended for FaceNet models.
    is_match = euclidean_l2_norm <= thresholds['euclidean_l2']
    status = "GOOD (MATCH)" if is_match else "BAD (NO MATCH)"
    print(f"3. Euclidean (L2):     {euclidean_l2_norm:.4f} | Threshold: <= {thresholds['euclidean_l2']} -> {status}")

    # --- Raw Euclidean Distance ---
    # Bandwidth: 0.0 is perfect. For FaceNet512, raw distance scales differently.
    is_match = euclidean_raw <= thresholds['euclidean_raw']
    status = "GOOD (MATCH)" if is_match else "BAD (NO MATCH)"
    print(f"4. Euclidean (Raw):    {euclidean_raw:.4f} | Threshold: <= {thresholds['euclidean_raw']} -> {status}")

# --- Example Usage ---
if __name__ == "__main__":
    # Assuming you have two generated embeddings
    compare_face_embeddings("user_123b_embedding.npy", "user_123c_embedding.npy")
    pass