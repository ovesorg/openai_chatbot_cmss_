from elasticsearch import Elasticsearch, NotFoundError
import numpy as np
import pinecone

# Define Elasticsearch credentials
ELASTIC_SEARCH_URL = 'https://elastic-prod.omnivoltaic.com/'
ELASTIC_SEARCH_USERNAME = 'elastic'
ELASTIC_SEARCH_PASSWORD = '2BY2fzVYEgChBQp5'
EXPECTED_DIMENSION = 768

# Initialize Elasticsearch client with the provided credentials
es = Elasticsearch(
    hosts=[ELASTIC_SEARCH_URL],
    http_auth=(ELASTIC_SEARCH_USERNAME, ELASTIC_SEARCH_PASSWORD)
)

# Initialize Pinecone
pinecone.init(api_key="b52d829d-5b87-4b29-aa8d-02dbf49ce32c", environment="gcp-starter")
index_name = "omnivoltaic-company-data"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=768, metric="cosine", shards=1)

index = pinecone.Index(index_name=index_name)

def add_to_elasticsearch(doc_id, doc):
    es.index(index="omnivoltaic-company-data", id=doc_id, body=doc)

def query_elasticsearch(query_str):
    try:
        response = es.search(index="omnivoltaic-company-data", body={
            "query": {
                "match": {
                    "content": query_str
                }
            }
        })
        return response["hits"]["hits"]
    except NotFoundError:
        return []
    
    
def add_to_pinecone(doc_id, vector, metadata={}):
    # Ensure vector is in the correct format. If it's a string, convert it back to a list.
    def convert_to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert_to_list(item) for item in obj]
        return obj
    
    # Convert ndarray vectors to lists
    vector = convert_to_list(vector)

    print(f"Original vector dimension: {len(vector)}")  # Log the original vector's length

    # Ensure the vector matches the expected dimension
    vector = truncate_or_pad_vector(vector, EXPECTED_DIMENSION)

    print(f"Modified vector dimension: {len(vector)}")  # Log the modified vector's length

    # Now upsert the vector into Pinecone
    index.upsert(vectors=[(doc_id, vector, metadata)])


# def add_to_pinecone(doc_id, vector, metadata={}):
#     # Ensure vector is in the correct format. If it's a string, convert it back to a list.
#     if isinstance(vector, str):
#         try:
#             import json
#             vector = json.loads(vector)
#         except:
#             raise ValueError("Unable to parse the vector string back to a list.")

#     # If the vector is a numpy array, convert it to a list.
#     if isinstance(vector, np.ndarray):
#         vector = vector.tolist()

#     # Ensure the vector is actually a list (or similar iterable).
#     if not isinstance(vector, (list, tuple)):
#         raise ValueError(f"Vector is not a list or tuple but is: {type(vector)}")

#     print(f"Original vector dimension: {len(vector)}")  # Log the original vector's length

#     # Ensure the vector matches the expected dimension
#     vector = truncate_or_pad_vector(vector, EXPECTED_DIMENSION)

#     print(f"Modified vector dimension: {len(vector)}")  # Log the modified vector's length

#     # Now upsert the vector into Pinecone
#     index.upsert(vectors=[(doc_id, vector, metadata)])


def truncate_or_pad_vector(vector, target_dimension):
    """Truncate or pad the given vector to the target dimension."""
    if len(vector) > target_dimension:
        return vector[:target_dimension]
    elif len(vector) < target_dimension:
        padding = [0] * (target_dimension - len(vector))
        return vector + padding
    else:
        return vector

# def truncate_or_pad_vector(vector, target_dimension):
#     """Truncate or pad the given vector to the target dimension."""
#     if len(vector) > target_dimension:
#         return vector[:target_dimension]
#     elif len(vector) < target_dimension:
#         padding = [0] * (target_dimension - len(vector))
#         return vector + padding
#     else:
#         return vector


def query_pinecone(vector):
    # Ensure the query vector matches the expected dimension
    vector = truncate_or_pad_vector(vector, EXPECTED_DIMENSION)
    return index.query(queries=[vector], top_k=5)
