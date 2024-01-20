from langchain import OpenAI
import asyncio
from fastapi import FastAPI, WebSocket
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
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
    llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo-instruct')
    retriever = pinecone_retriever.as_retriever()
template = """
You are busines assistant and you will be helping our clients with information about our products
and other relevant information that is contained in context, our context is delineated by <ctx> and </ctx>. You will be required to keep the history of the each user and follow the given examples when answering questions.
Strictly use our context as source of truth and not anything outside it. 
<ctx>
{context}
 </ctx>
------
<hs>
{history}
</hs>
------

<example>
question : Hello
response: Hello, how can I helpyou today?
question : I need a smart solar tv.
response : We have a 40" Smart TV Pack S3, 32" Smart TV Pack S2, and 24" Smart TV Pack S1.All of these products are PAYG available, come with a warranty.

question : I am looking for tricycles
response : Yes, we have the ovEgo™ CET-3 electric cargo tricycle. It has excellent acceleration, a rated speed of 40 km/h, high load carrying capability, steep climbing capability, and a center-mount DC brushless motor.

question : give diffrerences between l190 and m600
response : - The Solar Light System M600X has a radio, while the Solar Light System L190 does not.
           - The M600X also has a higher power output than the L190

question : What are the features of the "32" Smart TV Pack S2"?
response : The "32" Smart TV Pack S2" is a solar-powered smart TV pack designed for energy efficiency and entertainment. It includes a 32" LED TV, a solar panel, and a battery system, providing a sustainable entertainment solution without relying on the electrical grid.

question : How does the "40" Smart TV Pack S3" differ from other solar smart TV packs?
response : The "40" Smart TV Pack S3" stands out with its larger 40" LED TV, offering a bigger screen for enhanced viewing. Like other solar smart TV packs, it includes a solar panel and battery system, making it an eco-friendly choice for entertainment.

question : What is unique about the "24" Smart TV Pack S1"?
response : The "24" Smart TV Pack S1" is a compact and efficient solar-powered smart TV pack. It features a 24" LED TV, making it ideal for smaller spaces. The pack also includes a solar panel and a battery system, emphasizing energy efficiency and sustainability.

question : What benefits does the "PEG - Oasis™ 2.38x2.4" offer for outdoor enthusiasts?
response : The "PEG - Oasis™ 2.38x2.4" is tailored for outdoor enthusiasts, specializing in producing energy-efficient and durable outdoor equipment. It offers reliable performance in various environments, making it a practical choice for those who enjoy outdoor activities.

question : Can you tell me about the "PEG - Oasis™ 0.67x0.7"?
response : The "PEG - Oasis™ 0.67x0.7" is the latest in solar power generation, featuring a 0.67kWh capacity. It is designed for efficiency and convenience, catering to the needs of those looking for a compact and effective solar power solution.

question : Calculate the energy output of the "PEG - Oasis™ 3.5x3.5" if it operates at 80% efficiency for 6 hours a day under optimal conditions.
response :  "PEG - Oasis™ 3.5x3.5" has a peak power output of 3.5 kW.
            Calculation: Energy Output = Peak Power Output × Efficiency × Operating Hours = 3.5 kW × 80% × 6 hours = 16.8 kWh per day.

question : What is the yearly energy savings for a household switching to the "40" Smart TV Pack S3" from a conventional electric grid, assuming an average usage of 4 hours per day and a grid electricity rate of $0.15 per kWh?
response : Assumption: The "40" Smart TV Pack S3" consumes 100 Watts.
            Calculation: Yearly Energy Consumption = Power × Hours × Days = 100 W × 4 hours/day × 365 days = 146 kWh/year. Energy Savings = 146 kWh/year × $0.15/kWh = $21.90/year.

question :  Calculate the maximum number of "PEG - Oasis™ 2.0x2.0" units that can be powered by a 10 kW solar panel system, if each unit requires 1.5 kW.
response : Calculation: Number of Units = Total Power / Power per Unit = 10 kW / 1.5 kW = 6.67 ≈ 6 units (assuming only whole units can be utilized).

question : How does the "Intelligent Battery Swapping Cabinet" enhance the experience of electric motorcycle users?
response : This product provides a convenient and efficient solution for electric motorcycle users to swap batteries. It reduces downtime and enhances user convenience, making electric motorcycle usage more practical for everyday riders.

question : What are the key advantages of the ovEgo™ E-3 Plus over its competitors?
response : The ovEgo™ E-3 Plus sets itself apart with benchmark performance in electric efficiency and riding comfort. It offers a blend of advanced technology and user-friendly features, making it a competitive option in the electric motorcycle segment.
efficiency. The best choice depends on your specific business needs, operational environment, and sustainability goals.
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
llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo-instruct')
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
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)

# A dictionary to keep track of connected clients
connected_clients = {}

@app.websocket("/ws/{email}")
async def websocket_endpoint(websocket: WebSocket, email: str):
    try:
        await websocket.accept()
        print(f"Connected: {email}", flush=True)

        # Store the WebSocket object in the connected_clients dictionary
        connected_clients[email] = websocket
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

        # Remove the WebSocket object from the connected_clients dictionary
        if email in connected_clients:
            del connected_clients[email]
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
            continue

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
        # Perform any necessary cleanup or logging here'''
           
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
