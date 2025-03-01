import faiss
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------- FUNCTIONS ----------------------

def load_faiss_embedding(index_path="index.faiss", vector_id=0):
    """
    Load an embedding from a FAISS index for a given vector ID.
    """
    index = faiss.read_index(index_path)
    return np.array(index.reconstruct(vector_id))  # Retrieve stored embedding


def get_google_embedding(text, api_key="AIzaSyA40IBKQ9c-x5TUIFS8CM9BXZDimuStj7Y"):
    """
    Generate an embedding for the given text using Google Generative AI Embeddings.
    """
    genai.configure(api_key=api_key)
    response = genai.embed_content(model="models/embedding-001", content=text)
    return np.array(response['embedding'])


def compute_cosine_similarity(vector1, vector2):
    """
    Compute cosine similarity between two vectors.
    """
    # Reshape vectors to 2D arrays for the cosine_similarity function
    v1 = vector1.reshape(1, -1)
    v2 = vector2.reshape(1, -1)
    return cosine_similarity(v1, v2)[0][0]


def recall_at_k(relevant_docs, retrieved_docs, k):
    """
    Compute Recall@K.

    Parameters:
    - relevant_docs: Set of ground truth relevant document IDs.
    - retrieved_docs: List of retrieved document IDs in ranked order.
    - k: Number of top-ranked items to consider.

    Returns:
    - Recall@K score (float between 0 and 1).
    """
    retrieved_top_k = retrieved_docs[:k]  # Take the top K retrieved items
    relevant_in_k = len(set(relevant_docs) & set(retrieved_top_k))  # Count how many relevant docs are in top K
    return relevant_in_k / len(relevant_docs) if relevant_docs else 0.0


# ---------------------- MAIN EXECUTION ----------------------

def main():
    # Example for Cosine Similarity
    query_text = "I feel like killing myself"

    # Generate an embedding for the query using Google Generative AI Embeddings
    query_embedding = get_google_embedding(query_text, api_key="AIzaSyA40IBKQ9c-x5TUIFS8CM9BXZDimuStj7Y")

    # Load an embedding for a stored legal document from the FAISS index
    stored_embedding = load_faiss_embedding("index.faiss", vector_id=0)

    # Compute cosine similarity between the query and the stored document
    similarity_score = compute_cosine_similarity(query_embedding, stored_embedding)
    print(f"Cosine Similarity between query and stored document 0: {similarity_score:.4f}")

    # Example for Recall@K
    # Assume these are our ground truth relevant document IDs
    relevant_docs = {"doc1", "doc3", "doc7"}
    # And these are the retrieved documents in ranked order from a search system
    retrieved_docs = ["doc1", "doc4", "doc3", "doc8", "doc2", "doc7"]

    k = 3  # We will consider the top 3 retrieved results
    recall_score = recall_at_k(relevant_docs, retrieved_docs, k)
    print(f"Recall@{k}: {recall_score:.2f}")


if __name__ == "__main__":
    main()
