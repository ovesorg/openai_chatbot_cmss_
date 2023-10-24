from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request, Form,WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from langchain.vectorstores import Pinecone
import getpass
import pinecone
import requests
from typing import Annotated
import uvicorn
import secrets
from typing import Annotated
import json
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import FastAPI
from pydantic import BaseModel

class Request(BaseModel):
    query: str

    
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
security = HTTPBasic()
def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"osokoto"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"osokoto"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question. 
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
embeddings = OpenAIEmbeddings(openai_api_key=input("openai:"))
pinecone.init(
    api_key=input("pinecone:"), 
    environment=input("env"), 
)

index_name = "omnivoltaic-company-data"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
llm = OpenAI(temperature=0.8, openai_api_key=input("openai:"))
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

@app.get("/")  
async def root():  
    return {"message": "Hello World"}

@app.post("/user/")  
async def user(username: Annotated[str, Depends(get_current_username)],request: Request):
    data = request.json()
    data2 = json.loads(data)

    return str(qa.run(data2["query"]))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)


