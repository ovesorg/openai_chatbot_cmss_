from langchain import OpenAI
from fastapi import FastAPI, WebSocket
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
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
PINECONE_API_KEY = "199b3561-863a-41a7-adfb-db5f55e505ac"
PINECONE_ENVIRONMENT = "eu-west4-gcp"

@app.on_event("startup")
async def startup_event():
    # load_and_process_notion_data()

    # Make these variables global so they can be accessed by the endpoint function
    global prompt, qa, user_query
    
    pinecone_retriever = Pinecone.from_existing_index(
        "chatbot", embeddings)


    # Create a weighted average retriever that combines the results from both retrievers.
    # retriever = EnsembleRetriever(retrievers=[elasticsearch_retriever, pinecone_retriever.as_retriever(
    # )], weights=[1, 0])
    # db = Chroma.from_documents([global_context], embeddings)
    # llm = OpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY)
    llm = OpenAI(temperature=0.8, openai_api_key=OPENAI_API_KEY, model='gpt-4')
    retriever = pinecone_retriever.as_retriever()
template = """
 You are business assitant, you will use context deliminated by </ctx> and history deliminated by  </hs> to answer customer questions. Follow the format of  example deliminated  by  </example> when making your response. The example contains  question and the response that was provided by you when we were training you .Our context is arranged with columns in a table. column one for product titles, and column two for product description. Get customer question, use logic to understand his intent, scan through the content, and compile only short, direct and truthful answers. Dont formulate answers that are not true. After scanning the content and you dont get any answer kindly tell the user we dont have the information or the product yet. Also when customer replies with 1 it shows that you have given right response. If he replies with zero it means the response was wrong. So you must use all customer feedback to improve your learning
<ctx>
{context}
 </ctx>
------
<hs>
{history}
</hs>
------

<example>
question : I need a smart solar tv
response : We have a 40" Smart TV Pack S3, 32" Smart TV Pack S2, and 24" Smart TV Pack S1.All of these products are PAYG available, come with a warranty.

question : I am looking for tricycles
response : Yes, we have the ovEgo™ CET-3 electric cargo tricycle. It has excellent acceleration, a rated speed of 40 km/h, high load carrying capability, steep climbing capability, and a center-mount DC brushless motor.

question : give diffrerences between l190 and m600
response : - The Solar Light System M600X has a radio, while the Solar Light System L190 does not.
           - The M600X also has a higher power output than the L190
</example>
{question}

Answer:


 Repeat the above template for other questions with appropriate modifications.
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
    max_tokens=2000
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    verbose=0,
    chain_type_kwargs={
        "verbose": False,
        "prompt": prompt,
        "memory": ConversationSummaryBufferMemory(
            memory_key="history",
            input_key="question",
            llm=llm,max_token_limit=200),
    }
)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()        
        try:
          response = qa.run(data)
          await websocket.send_text(response)
        except Exception as e:
          # Handle the exception (e.g., log it)
          print(f"Error: {str(e)}")
          # Continue the loop to keep the connection alive
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
