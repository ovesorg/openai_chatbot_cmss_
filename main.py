from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from fastapi import FastAPI, WebSocket
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
import asyncio
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
PINECONE_API_KEY = "199b3561-863a-41a7-adfb-db5f55e505ac"
PINECONE_ENVIRONMENT = "eu-west4-gcp"

@app.on_event("startup")
async def startup_event():
    # load_and_process_notion_data()

    # Make these variables global so they can be accessed by the endpoint function
    global prompt, qa, user_query
    
    pinecone_retriever = Pinecone.from_existing_index(
        "chatbot", embeddings)

    llm = OpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY, model='gpt-4')
    retriever = pinecone_retriever.as_retriever()
template = """
As a representative of our organization, please provide a professional and informative response based on the available information. Ensure your response is concise and reflects our commitment to quality and accuracy.
Use top three conversations only when maintaining conservation memory
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
llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)
retriever = docsearch.as_retriever()
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
    max_tokens=300
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
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_text()
            response = qa.run(data,max_tokens=50)
            await websocket.send_text(response)
        except Exception as e:
            # Send a response to the customer to be more specific
            error_message = "We couldn't process your request. Please provide more specific information. If it persists refresh your page"
            await websocket.send_text(error_message)

            # Implement reconnection logic with a delay
            await asyncio.sleep(2)  # Delay for reconnection
            continue
@app.post("/query/")
async def get_response(query: str):
    if not query:
        return {"error": "Query not provided"}

    response = qa.run({"query": query})
    return {"response": response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8111)
