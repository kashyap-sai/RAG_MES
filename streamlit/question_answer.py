from sklearn.metrics.pairwise import cosine_similarity
import torch

# Function to get the most relevant answer based on embeddings
def get_answer(query_embedding, documents):
    best_answer = None
    max_similarity = -1

    for doc_name, doc_embedding in documents:
        # Compute cosine similarity between the query embedding and document embeddings
        similarity = cosine_similarity(query_embedding.detach().numpy(), doc_embedding.detach().numpy())
        
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = doc_name  # Return the document name or content as the best answer

    return best_answer
