from fastapi import FastAPI, WebSocket
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings.openai import OpenAIEmbeddings
from models import generate_embedding
from db_manager import add_to_elasticsearch, query_elasticsearch, add_to_pinecone, query_pinecone
from config import OPENAI_API_KEY, ELASTIC_SEARCH_USERNAME, ELASTIC_SEARCH_PASSWORD, ELASTIC_SEARCH_URL
from langchain.retrievers import EnsembleRetriever
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate

app = FastAPI()

global_context = ""  # This is where we'll store the globally accessible context
user_query = ""

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
elastic_vector_search = ElasticsearchStore(
    index_name="chatbot",
    embedding=embeddings,
    es_url=ELASTIC_SEARCH_URL,
    es_user=ELASTIC_SEARCH_USERNAME,
    es_password=ELASTIC_SEARCH_PASSWORD,
    strategy=ElasticsearchStore.SparseVectorRetrievalStrategy()
)



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = process_query(data)
        await websocket.send_text(response)


def combine_data():
    # Load data from Pinecone and Elasticsearch
    pinecone_results = query_pinecone(generate_embedding(""))
    es_results = query_elasticsearch("")
    # Combine results
    if len(es_results) > 0:
        combined_data = pinecone_results + \
            [res["_source"]["content"] for res in es_results]
        return combined_data
    else:
        combined_data = pinecone_results
        return combined_data


def process_query(user_query: str):
    # Generate a response using LangChain and ChatGPT/OpenAI
    user_query = user_query
    response = qa.run({"context": global_context, "history": "",
                      "question": user_query, "query": user_query})
    return response


@app.on_event("startup")
async def startup_event():
    # load_and_process_notion_data()

    # Make these variables global so they can be accessed by the endpoint function
    global prompt, qa, user_query
    
    pinecone_retriever = Pinecone.from_existing_index(
        "chatbot", embeddings)
    elasticsearch_retriever = elastic_vector_search.as_retriever()

    # Create a weighted average retriever that combines the results from both retrievers.
    # retriever = EnsembleRetriever(retrievers=[elasticsearch_retriever, pinecone_retriever.as_retriever(
    # )], weights=[1, 0])
    # db = Chroma.from_documents([global_context], embeddings)
    llm = OpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY)
    retriever = pinecone_retriever.as_retriever()
template = """
You are here to assist clients who want information about our products. Combine chat history for the user together with his question and give a response that is considerate of his previous conversation and present question.
Address each client based on their username when responding and dont be monotonous.
Use the following context (delimited by <ctx></ctx>) and the chat history    (delimited by <hs></hs>) to answer the question. 


You are a domain-specific assistant for Oves. Please note that your responses should be based exclusively on our data, and you should not rely on external sources.

<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}

Answer:
"""
embeddings = OpenAIEmbeddings(model_name="gpt-4.0-turbo", openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
)
index_name = "chatbot"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
llm = OpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY)
retriever = docsearch.as_retriever()
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    verbose=0,
    chain_type_kwargs={
        "verbose": False,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)


@app.post("/query/")
async def get_response(query: str):
    if not query:
        return {"error": "Query not provided"}

    response = qa.run({"query": query})
    return {"response": response}


@app.post("/update_data/")
async def update_data_endpoint(content: str):
    vector = generate_embedding(content)
    # Create a unique id for the document. Modify as needed.
    doc_id = str(hash(content))
    add_to_elasticsearch(doc_id, {"content": content})
    add_to_pinecone(doc_id, vector)
    return {"status": "success", "message": "Data updated successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8111)
