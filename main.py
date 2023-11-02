from langchain import OpenAI
from fastapi import FastAPI, WebSocket
# from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from fastapi import FastAPI,Request, WebSocket, Form,Depends, HTTPException
from fastapi.security import HTTPBasicCredentials
from passlib.context import CryptContext
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
import base64
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
import os
import uvicorn
from langchain.vectorstores import Pinecone
import pinecone
import secrets
import json


app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Access the variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = "eu-west4-gcp"

index_name = "chatbot"
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        response = process_query(data)
        await websocket.send_text(response)


def process_query(user_query: str):
    user_query = user_query
    response = qa.run({"context": global_context, "history": "",
                      "question": user_query, "query": user_query})
    return response
    
@app.on_event("startup")
async def startup_event():
    pinecone_retriever = Pinecone.from_existing_index(index_name, embeddings)

    llm = OpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY, model='gpt-4')
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8111)
