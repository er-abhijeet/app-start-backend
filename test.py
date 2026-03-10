import numpy as np
from deepface import DeepFace

def generate_and_save_embedding(image_path, save_filename):
    """
    Extracts a 512-dimensional embedding using FaceNet512 and saves it as a NumPy file.
    """
    try:
        # Generate the representation
        # enforce_detection=True ensures the function throws an error if no face is found,
        # preventing garbage data from being saved.
        result = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet512",
            enforce_detection=True 
        )

        # DeepFace returns a list of dictionaries (one for each face detected in the image).
        # We extract the embedding vector for the first face detected.
        embedding_vector = result[0]['embedding']

        # Convert the Python list to a NumPy array
        embedding_np = np.array(embedding_vector)

        # Save the array to disk
        # np.save automatically appends the .npy extension if it's not present
        np.save(save_filename, embedding_np)
        
        print(f"Success! 512D embedding saved to {save_filename}.npy")
        return embedding_np

    except ValueError as e:
        print(f"Detection Error: Could not find a face in the image. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace with your actual image path
    input_image = "ad1.png" 
    output_file = "user_123d_embedding"
    
    generate_and_save_embedding(input_image, output_file)
    
    # To load the embedding later for comparison:
    # loaded_embedding = np.load("user_123_embedding.npy")