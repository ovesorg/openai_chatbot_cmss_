from langchain import OpenAI
import asyncio
from fastapi import FastAPI, WebSocket
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from fastapi import FastAPI,Request, WebSocket, Form,Depends, HTTPException, WebSocketDisconnect
from fastapi.security import HTTPBasicCredentials

from passlib.context import CryptContext
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
from feedback import save_feedback_to_sheets
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
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configure CORS (enable all origins, allow credentials, allow all methods, allow all headers)
# Load environment variables from .env file
load_dotenv()
OUTPUT_DIR= os.getcwd()
f = open(os.path.join(OUTPUT_DIR, 'feedback.txt'), 'a')
f.write('\nTest User feedback.')
f.close()
# Access the variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
#OPENAI_API_KEY = "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"
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
    llm = OpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo-instruct')
    retriever = pinecone_retriever.as_retriever()
#You are business assistant to help people with our product information maintain business tone when answering, you will use context deliminated by </ctx> and history deliminated by </hs> to answer customer questions. Follow the format of example deliminated by </example> when making your response. The example contains question and the response that was provided by you when we were training you .Our context is arranged with columns in a table. column one for product titles, and column two for product description. Get customer question, use logic to understand his intent, scan through the content, and compile only short, direct and truthful answers. Dont formulate answers that are not true. After scanning the content and you dont get any answer kindly tell the user we dont have the information or the product yet. Also when customer replies with 1 it shows that you have given right response. Avoid hallucination always. Only give the product description by keping the title and description from corresponsing rows
#The following is a business conversation between a human and an AI. The AI is professional and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. The AI follows the examples provided to formulate its answers by scanning through the context and avoids hallucination by not providing wrong answers or answers that doesnt belong to certain product

template = """
The following is a business conversation between a human and an AI. The AI is professional and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. The AI follows the examples provided to formulate its answers by scanning through the context and avoids hallucination by not providing wrong answers or answers that doesnt belong to certain product
<ctx>
 {context}
 </ctx>
------
<hs>
{history}
</hs>
------

<example>
Humanu : Hello
AI: Hello, how can I helpyou today?
Human : I need a smart solar tv.
AI : We have a 40" Smart TV Pack S3, 32" Smart TV Pack S2, and 24" Smart TV Pack S1.All of these products are PAYG available, come with a warranty.

Human : I am looking for tricycles
AI : Yes, we have the ovEgo™ CET-3 electric cargo tricycle. It has excellent acceleration, a rated speed of 40 km/h, high load carrying capability, steep climbing capability, and a center-mount DC brushless motor.

Human : i need to know properties of m400
AI :The M400 solar light system is a versatile and powerful system designed for various lighting needs. It features a 400 lumen LED light, a 10W solar panel, and a 7.4V 2600mAh lithium battery. It also has a USB port for charging devices and is compact, lightweight, and waterproof.
Human : I need a smart solar tv.
AI : We have a 40" Smart TV Pack S3, 32" Smart TV Pack S2, and 24" Smart TV Pack S1.All of these products are PAYG available, come with a warranty.

Human : i need to know properties of m600
AI : The Solar Light System M600 is a versatile and powerful system designed for various lighting needs. It features a 600 lumen LED light, a 10W solar panel, and a 7.4V 2600mAh lithium battery. It also has a USB port for charging devices and is compact, lightweight, and waterproof. The M600X model has the same features as the M600, but with a higher power output of 600 lumen. The M600X also has a radio, while the M600 does not. The M630X model has a higher power output of 630 lumen and also includes a radio.
Human : give me models odf oasis
AI : The PEG - Oasis™ series offers a range of outdoor energy storage products designed for various applications. The Oasis 5x3, 0.67x0.7, and 2.38x2.4 models all feature LFP chemistry for safety, a smart MPPT for solar efficiency, and a crystalline silicon solar panel. These products are designed to provide professional off-grid power and have a high full cycle life of 80+%/2500 cycles. The Oasis series is compact, lightweight, and waterproof, making it suitable for outdoor use.

Human : kindly let me know the energy storage facilities you have
AI : We offer a range of energy storage products, including the Town™ energy storage systems and the PEG - Oasis™ solar power generators. These products provide flexible, reliable, and space-saving solutions for a variety of energy storage applications. They also offer cost savings, ease of maintenance, and increased flexibility, making them an attractive option for off-grid areas with limited access to maintenance resources. Additionally, our products have higher energy density, improved reliability, and can be designed to provide different types of energy storage, such as fast-acting, high-power storage for frequency regulation and slow-acting storage for longer discharge times and greater storage capacity.

Human :tell me the specifications of AA  batteries
AI : The AA rechargeable batteries have a high capacity rating of 2800mAh, providing a longer working time for devices. They are built with NiMH environmental protection materials, allowing for up to 1600 recharge cycles. The batteries have a voltage of 1.5V and come with a 5V/2A charger. It is recommended to fully charge the batteries before first use, as they are shipped at 30% charge for safety purposes. These batteries are not suitable for use with flash, blink series, Apple mouse, or keyboard.
Human :  Calculate the maximum number of "PEG - Oasis™ 2.0x2.0" units that can be powered by a 10 kW solar panel system, if each unit requires 1.5 kW.
AI : Calculation: Number of Units = Total Power / Power per Unit = 10 kW / 1.5 kW = 6.67 ≈ 6 units (assuming only whole units can be utilized).

Human : defferentiate between all motorcycles you have
AI : - ovEgo™ E-3 Plus: Benchmark performance against 150cc ICE motorcycles, high load carrying and steep climbing capability, center-mount DC brushless motor, and proprietary LFP battery module.
    - ovEgo™ F-3: New fast and fashionable design, excellent acceleration, high load carrying and steep climbing capability, center-mount DC brushless motor, and proprietary LFP battery module.
    - ovEgo™ T-5: Cross motorcycle design, excellent acceleration, high load carrying and steep climbing capability, center-mount DC brushless motor, and proprietary LFP battery module.
    - ovEgo™ S-3: City and community use, excellent acceleration, high load carrying and steep climbing capability, center-mount DC brushless motor, and proprietary LFP battery module.

Human : which products can i use for production and business purpose?
AI : To recommend products from your provided data for production and business purposes, it's crucial to consider the nature of your business, its location, energy requirements, and specific operational needs. Based on the products in the data, here are some suggestions:

            Solar Sewing Machine Packs (e.g., 42Ah Solar Fan & Sewing Machine Pack, 18Ah Solar Sewing Machine Pack):

            Suitable For: Tailoring businesses, especially in areas with limited access to electricity. They offer a sustainable and cost-effective solution for garment production.
            Advantages: These packs allow for continuous operation during power outages and reduce electricity costs. The inclusion of a fan also improves working conditions in hot climates.
            Solar Smart TV Packs (e.g., 32" Smart TV Pack S2, 40" Smart TV Pack S3):
            
            Suitable For: Businesses in hospitality, such as hotels, cafes, and waiting areas where entertainment for customers is beneficial. They're also ideal for remote educational settings where a TV can be used as a learning tool.
            Advantages: These TVs provide entertainment without reliance on the electrical grid, reducing operational costs and enhancing customer experience in off-grid locations.
            Solar Power Generators (e.g., PEG - Oasis™ 0.67x0.7):
            
            Suitable For: A wide range of businesses that require backup power solutions, such as small offices, shops, and outdoor event organizers.
            Advantages: They offer a reliable power source for essential equipment during power outages, and their portable nature makes them useful for outdoor events or remote operations.
            Electric Motorcycles (e.g., ovEgo™ models):
            
            Suitable For: Delivery services, courier companies, or businesses requiring short-range transportation.
            Advantages: These vehicles are cost-effective for frequent, short trips, especially in urban areas. They reduce fuel costs and contribute to a lower carbon footprint.
            Intelligent Battery Swapping Cabinet:
            
            Suitable For: Businesses that use a fleet of electric vehicles or devices.
            Advantages: This system facilitates quick battery swaps, reducing downtime for vehicles or equipment that rely on battery power.
            Each of these products offers unique benefits and can be instrumental in different types of businesses, particularly those focusing on sustainability and energy efficiency. The best choice depends on your specific business needs, operational environment, and sustainability goals.
question : tell me about solar 32" pack B4
response : The CAMP solar 32" TV Pack B4 is an excellent Solar Home System designed for homes and businesses that prefer large screens and well-lit rooms. The solar 32" TV Pack B4 system comes in a 75W Solar Panel, 12Ah lithium battery Hub for energy storage and a low consumption 32” TV with integrated satellite decoder and HDMI. Other accessories include four efficient LED tube lights, a multi-functional torch and a motion sensor security light.
</example>
{question}

Answer:
"""

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
)
index_name = "chatbot"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
llm = OpenAI(temperature=0.3, openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo-instruct')
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
        "memory": ConversationBufferWindowMemory(
            k=0,
            memory_key="history",
            llm=llm,
            input_key="question"),
    }
)
'''qa = RetrievalQA.from_chain_type(
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

# A dictionary to keep track of connected clients
clients = {}

@app.websocket("/ws/{email}")
async def websocket_endpoint(websocket: WebSocket, email: str):
    if email in clients:
        await websocket.close(code=4000, reason="email already in use")
        return
    try:
        await websocket.accept()
        print(f"Connected: {email}", flush=True)

        # Store the WebSocket object in the clients dictionary
        clients[email] = websocket
        while True:
            data = await websocket.receive_text()
            try:
                json_data = json.loads(data)
                if "input" in json_data:
                    # This is a user query
                    print("This is user query")
                    try:
                        if "email" in json_data:
                            email = json_data["email"]
                            if email in clients:
                                response = qa.run(json_data["input"])
                                await clients[email].send_text(f"{response} response send to {email}")
                            else:
                                print(f"Email {email} not found in clients dictionary")
                        else:
                            print("Email not provided in the JSON data")
                    except Exception as e:
                        # Handle the exception (e.g., log it)
                        print(f"Error: {str(e)}")
                        # Continue the loop to keep the connection alive
                        continue

                elif "user_query" in json_data:
                    save_feedback_to_sheets(data)
                    print("This is feedback message", flush=True)
                else:
                    # Unexpected data format
                    print("Unknown data format")
            except json.JSONDecodeError:
                # Handle JSON decoding error
                print("Error decoding JSON")
                continue
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected with code {e.code}: {e.reason}")

        # Remove the WebSocket object from the clients dictionary
        if email in clients:
            del clients[email]'''
'''@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        print(data)
        try:
            json_data = json.loads(data)
            if "input" in json_data:
                # This is a user query
                print("This is user query")
                try:
                    response = qa.run(json_data["input"])
                    await websocket.send_text(response)
                except Exception as e:
                    # Handle the exception (e.g., log it)
                    print(f"Error: {str(e)}")
                    # Continue the loop to keep the connection alive
                    continue
            elif "user_query" in json_data:
                save_feedback_to_sheets(data)
                print("This is feedback message", flush=True)
            else:
                # Unexpected data format
                print("Unknown data format")
        except json.JSONDecodeError:
            # Handle JSON decoding error
            print("Error decoding JSON")
            continue'''

@app.websocket("/ws/{email}")
async def websocket_endpoint(websocket: WebSocket, email: str):
    try:
        await websocket.accept()
        print(f"Connected: {email}",flush=True)

        dialogue_history = [
                {'type': 'bot', 'text': 'Hello to you'},
                {'type': 'user', 'text': 'Hello'},
                {'type': 'bot', 'text': 'yes we have'},
                {'type': 'user', 'text': 'I want tv'},
                {'type': 'bot', 'text': 'Yes we have l190'},
                {'type': 'user', 'text': 'I need l190'}
            ]
        dialogue_history_string = json.dumps(dialogue_history)
        await websocket.send_text(dialogue_history_string)

        while True:
            data = await websocket.receive_text()
            try:
                json_data = json.loads(data)
                if "input" in json_data:
                    # This is a user query
                    print("This is user query")
                    try:
                        response = qa.run(json_data["input"])
                        await websocket.send_text(response)
                    except Exception as e:
                        # Handle the exception (e.g., log it)
                        print(f"Error: {str(e)}")
                        # Continue the loop to keep the connection alive
                        continue
                elif "user_query" in json_data:
                    save_feedback_to_sheets(data)
                    print("This is feedback message", flush=True)
                else:
                    # Unexpected data format
                    print("Unknown data format")
            except json.JSONDecodeError:
                # Handle JSON decoding error
                print("Error decoding JSON")
                continue
    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected with code {e.code}: {e.reason}")
        # Perform any necessary cleanup or logging here
           
@app.post("/query/")
async def get_response(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query not provided")
    if isinstance(query, dict) or (isinstance(query, str) and query.startswith('{') and query.endswith('}')):
        print("This is feedback message", flush=True)

        # Assuming save_feedback_to_sheets is a function in feedback_module that handles feedback saving
        try:
            save_feedback_to_sheets(query)
            print("Feedback saved to Google Sheets", flush=True)
            return {"message": "Feedback saved successfully"}
        except Exception as e:
            print(f"Error saving feedback: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    else:
        print("user query received")
        print(query)
        try:
            response = qa.run({"query": query})
            print(response, flush=True)
            return {"response": response}
        except Exception as e:
            # Handle the exception (e.g., log it)
            print(f"Error: {str(e)}")
            # Customize the response message for your specific use case
            raise HTTPException(status_code=400, detail="Make your question more specific")


@app.post("/submit-form/")
async def submit_form(user_query: str, bot_response: str,user_expected_response:str,user_rating:int):
    # Process the form data, you can save it to a database or perform any other actions
    return {"user_query": user_query, "bot_response": bot_response,"user_expected_response":user_expected_response,"user_rating":user_rating}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8111)
