from fastapi import FastAPI, WebSocket, HTTPException
from models import generate_embedding
from db_manager import add_to_elasticsearch, query_elasticsearch, add_to_pinecone, query_pinecone
from config import OPENAI_API_KEY
from utils import load_and_process_notion_data
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import NotionDBLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.retrievers import TFIDFRetriever
from docx import Document

import os

app = FastAPI()

global_context = ""  # This is where we'll store the globally accessible context

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = process_query(data)
        await websocket.send_text(response)

def combine_data():
    # Load data from Pinecone and Elasticsearch
    pinecone_results = query_pinecone(generate_embedding(""))  # We need some query to get results. Decide what's appropriate.
    es_results = query_elasticsearch("")
    # Combine results
    combined_data = pinecone_results + [res["_source"]["content"] for res in es_results]
    return combined_data

def process_query(user_query: str):
    # Use the function to get combined_data
    combined_data = combine_data()
    
    # Generate a response using LangChain and ChatGPT/OpenAI
    response = qa.run({"context": global_context, "history": "", "question": user_query})
    return response['response']

@app.on_event("startup")
async def startup_event():
    load_and_process_notion_data()
    combined_data = combine_data()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    global global_context
    global_context = " ".join(text_splitter.split_documents(combined_data))

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma.from_documents([global_context], embeddings)
    llm = OpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY)
    retriever = db.as_retriever()   
    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question and answer starting with Ovsmart then full colon and the response: 
    incase of maximum token limit tell the user to be specific and dont through in an error message.
    Be creative enough and keep conversational history to have humanly conversation.  Give short and correct answers based on the the content given.
    ------
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

    global prompt, qa  # Make these variables global so they can be accessed by the endpoint function

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
    doc_id = str(hash(content))  # Create a unique id for the document. Modify as needed.
    add_to_elasticsearch(doc_id, {"content": content})
    add_to_pinecone(doc_id, vector)
    return {"status": "success", "message": "Data updated successfully."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8111)
