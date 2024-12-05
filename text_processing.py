from transformers import BertTokenizer, BertModel
import torch

# Function to process text and return embeddings
def process_text(text_path):
    # Open and read the text file
    with open(text_path, 'r') as file:
        text = file.read()
    return text

# Function to extract embeddings using a transformer model
def extract_embeddings(text):
    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    
    # Tokenize the input text and get the embeddings
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    
    # Extract embeddings (mean of token embeddings)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings
