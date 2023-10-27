import pinecone
index_name = 'chatbot'
pinecone.init(
    api_key=input('enter pinecone key:'),
    environment=input('enter pinecone env:')
)

if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
index = pinecone.Index(index_name)
print(index.describe_index_stats())
#pinecone.delete_index(index_name)
