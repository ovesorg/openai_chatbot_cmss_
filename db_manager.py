from elasticsearch import Elasticsearch, NotFoundError
from config import ELASTIC_SEARCH_USERNAME, ELASTIC_SEARCH_PASSWORD, ELASTIC_SEARCH_URL
import numpy as np
import pinecone

# Define Elasticsearch credentials
EXPECTED_DIMENSION = 1536

# Initialize Elasticsearch client with the provided credentials
es = Elasticsearch(
    hosts=[ELASTIC_SEARCH_URL],
    http_auth=(ELASTIC_SEARCH_USERNAME, ELASTIC_SEARCH_PASSWORD)
)

# Initialize Pinecone
pinecone.init(api_key="b52d829d-5b87-4b29-aa8d-02dbf49ce32c", environment="gcp-starter")
index_name = "omnivoltaic-company-data"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=1536, metric="cosine", shards=1)

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
    

def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [convert_to_list(item) for item in obj]
    return obj
    
    
def add_to_pinecone(doc_id, vector, metadata={}):
    
    # Convert ndarray vectors to lists
    vector = convert_to_list(vector)

    print(f"Original vector dimension: {len(vector)}")  # Log the original vector's length

    # Ensure the vector matches the expected dimension
    vector = truncate_or_pad_vector(vector, EXPECTED_DIMENSION)

    print(f"Modified vector dimension: {len(vector)}")  # Log the modified vector's length

    # Now upsert the vector into Pinecone
    index.upsert(vectors=[(doc_id, vector, metadata)])
    
def truncate_or_pad_vector(vector, target_dimension):
    """Truncate or pad the given vector to the target dimension."""
    if len(vector) > target_dimension:
        return vector[:target_dimension]
    elif len(vector) < target_dimension:
        padding = [0] * (target_dimension - len(vector))
        return vector + padding
    else:
        return vector


def query_pinecone(vector):
    # Convert ndarray to list for serialization
    print(f"vector query_pinecone: {vector}")
    vector_list = convert_to_list(vector)
    print(f"vector list query_pinecone: {vector_list}")
    
    
    # Ensure the query vector matches the expected dimension
    vector_list = truncate_or_pad_vector(vector_list, EXPECTED_DIMENSION)
    print(f"vector list truncate: {vector_list}")
    return index.query(vector=[vector_list], top_k=5)


