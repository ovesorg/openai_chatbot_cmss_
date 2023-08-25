from transformers import BertTokenizer, BertModel
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def generate_embedding(text):
    if not isinstance(text, (str, list)):
        raise ValueError(f"Unexpected type for text: {type(text)}")

    # Tokenize the text without truncating
    tokens = tokenizer.tokenize(text)

    # If the text is too long, split into chunks and obtain embeddings for each chunk
    MAX_LEN = 512 - 2  # 2 tokens are reserved for [CLS] and [SEP]
    chunks = [tokens[i:i + MAX_LEN] for i in range(0, len(tokens), MAX_LEN)]

    total_embedding = np.zeros(768)  # Initialize with zeros
    chunk_count = 0

    with torch.no_grad():
        for chunk in chunks:
            # Convert chunk tokens to input format
            inputs = tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            
            # Get embeddings
            outputs = model(**inputs)
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Ensure the result is 1D
            if chunk_embedding.ndim == 2:  # If still 2D, take the first row
                chunk_embedding = chunk_embedding[0]
            
            # Accumulate the embeddings
            total_embedding += chunk_embedding
            chunk_count += 1
    
    # Average the embeddings
    avg_embedding = total_embedding / chunk_count

    # Check for NaN values and replace them if they exist
    if np.isnan(avg_embedding).any():
        avg_embedding[np.isnan(avg_embedding)] = 0

    # Ensure the embedding is the correct shape
    if avg_embedding.shape != (768,):
        raise ValueError(f"Unexpected embedding shape: {avg_embedding.shape}")

    return avg_embedding
