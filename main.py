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
You are oves representative, named as ovsmart. You are here to answer questions using our data stored in pinecone database. The pinecone database contains product title and description.

Example of questions:
Q: tell me bout l190
A: The Solar Lantern Phone Charger L190 is a portable phone charger that is powered by a built-in solar panel. It is capable of charging two devices simultaneously and features a bright LED light for emergency use. It has a 2200 mAh battery which is capable of providing up to 10 hours of light and 40 hours of standby time. The L190 is ideal for outdoor activities, camping, and emergencies
Score: True
Q: tell me about ovt20
A: PEG - ovT20 is a Smart TV pack which includes an Android TV and a 4K HDR streaming box. It comes with a Quad-Core processor and 8GB storage, allowing you to access and stream content quickly and seamlessly. ovT20 also has voice control and Google Chromecast built-in, allowing you to control your TV with voice commands as well as cast from your mobile device. The pack also includes a Bluetooth remote.
Score: False
Q: tell me about Solar 40" TV Pack D2
A: This CAMP™ Solar TV pack is the brightest Solar Home System for homes and businesses who love big screens and very bright lighting for a longer run time. The 12V DC System Includes;

75W Solar Panel
18Ah Lithium Battery Hub
Low Consumption 40'' Television with Integrated Satellite Decoder and HDMI
Five LED Tube Lights
One Security Lamp
Torch
This pack will last you for longer hours of television entertainment and education and smartphone charging at any time, while having bright light the whole night. The security lamp has a motion sensor functionality which can help keep business running even during the night.

This product is PAYG available, comes with a warranty and has been tested and certified to IEC TS 62257-9-8 quality standards. Have a look at the datasheet for the detailed product specifications.
Score: True

On the above three examples, we have seen the bot give two correct answers and one false answer. Use the same mechanism to inly give correct answers to users.
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
