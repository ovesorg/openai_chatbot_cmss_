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
Q1: tell me bout l190
A: The LUMN™ Solar Lantern is an excellent entry-level solar product designed to meet the lighting needs of various applications.

2.4W Solar Panel
A Lantern Equipped With a Built-in Lithium Battery Pack for Energy Storage.
The solar lantern offers three brightness levels, making it versatile and ideal for use as a desk lamp, camping/tent lamp, emergency light, phone charger, and more.

With a full charge, the LUMN™ Solar Lantern can operate for up to 6-36 hours, depending on the selected brightness level. This feature ensures long-lasting performance and reliable illumination, even during extended periods of use. The lantern is designed to be energy-efficient, making it an eco-friendly and cost-effective lighting solution.

The LUMN™ Solar Lantern is a high-quality product that meets the IEC TS 62257-9-8 quality standard. It has been tested and certified to ensure optimal performance and durability, making it a reliable lighting solution for various applications. The product comes with a warranty, providing customers with peace of mind and assurance of its quality.Score: True
Q2: tell me about Farm Lighting Pack
A: This CAMP™ Solar Home Lighting Pack is the most popular and convenient when setting up large business spaces that require continuous exposure to bright light. The 12V DC System Includes; - 75W Solar Panel - 18Ah Lithium Battery Hub - Eight LED Tube Lights - Torch

This pack is very convenient in bringing out the feel of a modern business such as a chicken farm among others, as the tube lights have a unique modern design that shine brightest in the dark, and a remote-control torch that can be used to switch the LED tube lights on and off, or change the brightness level modes.

This product is PAYG available, comes with a warranty and has been tested and certified to IEC TS 62257-9-8 quality standards.

Have a look at the datasheet for the detailed product specifications
Score: True
Q3: tell me about Solar 40" TV Pack D2
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
Q:mention to me any electric motorcycle
A:The electric motorcycle ovEgo E-3 Plus is benchmarked against Boxer 150cc ICE motorcycle and Upgraded version, Consisting of Body, Motor, Drives, Brakes, and Wheels - Excellent Acceleration - Rated 80 km/h with Burst Speed Up To 100 km/h - High Load Carrying Capability - Steep Climbing Capability - Center-Mount Brushless DC Motor with High-Efficiency SVPWM Motor Controllers - Proprietary LFP Battery Module
Score:True
Follow the above examples in answering questions.
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
