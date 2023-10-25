from transformers import BertTokenizer, BertModel
import numpy as np
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

def generate_embedding(text):
    if not isinstance(text, (str, list)):
        raise ValueError(f"Unexpected type for text: {type(text)}")

    tokens = tokenizer.tokenize(text)
    MAX_LEN = 512 - 2  # 2 tokens are reserved for [CLS] and [SEP]
    chunks = [tokens[i:i + MAX_LEN] for i in range(0, len(tokens), MAX_LEN)]

    total_embedding = np.zeros(1536)  # Initialize with zeros
    chunk_count = 0

    with torch.no_grad():
        for chunk in chunks:
            inputs = tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
            outputs = model(**inputs)
            chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            if chunk_embedding.ndim == 2:  # If still 2D, take the first row
                chunk_embedding = chunk_embedding[0]
            
            # Concatenate the embedding with itself to achieve size 1536
            chunk_embedding = np.concatenate([chunk_embedding, chunk_embedding])
            
            total_embedding += chunk_embedding
            chunk_count += 1
    
    avg_embedding = total_embedding / chunk_count

    if np.isnan(avg_embedding).any():
        avg_embedding[np.isnan(avg_embedding)] = 0

    if avg_embedding.shape != (1536,):
        raise ValueError(f"Unexpected embedding shape: {avg_embedding.shape}")

    return avg_embedding

