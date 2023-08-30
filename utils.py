from config import NOTION_TOKEN, DATABASE_ID
from langchain.document_loaders import NotionDBLoader
from models import generate_embedding
from db_manager import add_to_elasticsearch, add_to_pinecone

def load_and_process_notion_data():
    loader = NotionDBLoader(NOTION_TOKEN, DATABASE_ID, request_timeout_sec=50)
    docs = loader.load()
    sources = []

    for f in docs:
        for u in f:        
            for l in u:
                sources.append(l)

    for source in sources:
    # for i in range(1):
        if isinstance(source, str): 

            print(source)
            vector = generate_embedding(source)
            doc_id = str(hash(source))  # Create a unique id for the document.
            
            # Add to Elasticsearch
            add_to_elasticsearch(doc_id, {"content": source})

            # Add to Pinecone
            add_to_pinecone(doc_id, vector)

    return "Data processed and stored in Pinecone and Elasticsearch."


# def load_and_process_notion_data():
#     loader = NotionDBLoader(NOTION_TOKEN, DATABASE_ID, request_timeout_sec=50)
#     docs = loader.load()
#     sources = []
#     for f in docs:
#         for u in f:        
#             for l in u:
#                 sources.append(l)
#     o = str(sources)
#     r = o.translate(str.maketrans('', '', '{}'))
#     v = r.replace('\t', '').replace('\n', '')
#     with open(os.path.join(OUTPUT_DIR, 'notiondata.txt'), 'w') as file1:
#         file1.write(v)
#     return os.path.join(OUTPUT_DIR, 'notiondata.txt')

# def load_document(file_path):
#     loader = TextLoader(file_path)
#     return loader.load()
