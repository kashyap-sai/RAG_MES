import streamlit as st
import os
import torch
from transformers import BertTokenizer, BertModel
from io import BytesIO
import numpy as np
import time

# Function to extract embeddings using a transformer model
def extract_embeddings(text):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Get the mean of token embeddings
    return embeddings

# Page configuration
st.set_page_config(layout="wide", page_title="Text Embedding Search", page_icon="📚")

# Initialize session state variables
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'embedding_results' not in st.session_state:
    st.session_state.embedding_results = None
if 'documents' not in st.session_state:
    st.session_state.documents = []

# Sidebar for file upload
st.sidebar.header("Upload Text Files")
uploaded_files = st.sidebar.file_uploader("Choose a text file", type="txt", accept_multiple_files=True)

# Function to handle uploaded files
def handle_uploaded_files(uploaded_files):
    embeddings = []
    documents = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_type = uploaded_file.type

        # Process text file
        if file_type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")
            documents.append(text)  # Store the raw text for later reference
            st.text_area(f"Text Content: {file_name}", text, height=200)
            embedding = extract_embeddings(text)
            embeddings.append(embedding)
    
    return embeddings, documents

# Handle the uploaded files and display embeddings if uploaded_files is not None
if uploaded_files:
    embeddings, documents = handle_uploaded_files(uploaded_files)
    st.session_state.embedding_results = embeddings
    st.session_state.documents = documents

# Display embeddings results
if st.session_state.embedding_results is not None and len(st.session_state.embedding_results) > 0:
    st.write("Extracted Embeddings:")
    for result in st.session_state.embedding_results:
        st.write(result.shape)  # Display the shape of embeddings

# Question input
st.subheader("Ask a Question")
question = st.text_input("Enter your question:")

# Function to split text into smaller chunks (like sentences or paragraphs)
def split_text_into_chunks(text, chunk_size=500):
    # Split the document into chunks of size 'chunk_size'
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# Function to process question and find similarity with the document embeddings
def find_similar_answer(question, embeddings, documents):
    if question:
        # Convert the question into embeddings
        question_embedding = extract_embeddings(question)
        
        # Initialize variables for finding the best chunk
        best_similarity = -1
        best_chunk = ""
        
        # For each document, split into chunks and calculate similarity for each chunk
        for doc_text in documents:
            chunks = split_text_into_chunks(doc_text)
            for chunk in chunks:
                # Extract embeddings for the chunk
                chunk_embedding = extract_embeddings(chunk)

                # Calculate cosine similarity (using numpy for easier handling)
                question_embedding_flat = question_embedding.detach().numpy().flatten()
                chunk_embedding_flat = chunk_embedding.detach().numpy().flatten()
                
                similarity = np.dot(question_embedding_flat, chunk_embedding_flat) / (np.linalg.norm(question_embedding_flat) * np.linalg.norm(chunk_embedding_flat))

                # Update best chunk if the current chunk is more similar
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_chunk = chunk
        
        return best_chunk, best_similarity
    
    return None, None

# Check and respond to the question
if question and st.session_state.embedding_results:
    similar_answer, similarity_score = find_similar_answer(question, st.session_state.embedding_results, st.session_state.documents)
    if similar_answer is not None:
        st.write(f"The most relevant document snippet has a similarity score of: {similarity_score:.4f}")
        st.write(f"Answer: {similar_answer}")  # Show the relevant document snippet as the answer
    else:
        st.write("No relevant answer found. Try rephrasing your question.")

# Footer or additional information (optional)
st.markdown("""
    ## About
    This app allows you to upload text files, extract embeddings using a transformer model, and ask questions.
    The app calculates similarity between the question and document chunks to provide the most relevant answer.
""")
