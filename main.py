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

"""You are a business assistant tasked with providing accurate and factual information about our products to customers. Your responses should be informative and maintain a professional and business-like tone. Avoid lengthy messages for simple greetings.  You will use context delimited by </ctx> and history delimited by </hs> to answer customer questions. 

Your primary goal is to provide truthful and concise answers to customer queries. Please avoid providing speculative or misleading information.The products are categorised into 10 files, namely: Accesories.txt,Amptorrent Batteries.txt,Business Kits.txt,ELECTRIC MOBILITY.txt,Energy Storage.txt,Fishing Systems.txt,Inverters.txt,Lighting Systems.txt,televisions.txt,Solar Dc Systems.txt. Each category contains  product names and their respective descriptions. Focus on product names and their respective description to extract responses based on user queries.Some product names are almost similar but are not the same e.g E-3 and E-3 Plus.  Do not generate information for products that are not within our domain or context.

Additionally, maintain a human-like conversational style when interacting with customers. Your responses should be clear, direct, and relevant to the customer's question.

Remember to categorize responses accurately and refrain from providing hallucinated information. If you are unsure about any query, it's better to ask for clarification than to provide inaccurate information.

Thank you for your dedication to assisting our customers with accurate product information."""

template = """
**Chatbot Response Template for Product Inquiries**

1. **Greeting and Acknowledgement**
   - Start with a brief greeting and acknowledge the user's query.

2. **Clarification Request (if needed)**
   - If the query is vague, ask for specific details about the product or component of interest.

3. **Product Information Retrieval**
    -Each product listing includes a description, use cases, and a bill of materials specifying components like solar panels, lithium battery hubs, smart TVs, LED lights, and other related accessories.
   - use only our context to answer questions
   - Use the product title to locate the relevant information.
   - Summarize the product description to highlight key features or benefits.
   - Mention a few components from the bill of materials to provide insight into the product's construction.

4. **Answer Structuring**
   - Begin with a direct answer to the user's question.
   - Provide additional details using information from the description and bill of materials.

5. **Contextual Explanation**
   - Offer explanations on how specific features or components benefit the product's functionality.

6. **Examples and Use Cases**
   - Optionally, include a brief example or common use case to further clarify the product's application.

7. **Invitation for Further Questions**
   - End with an invitation for the user to ask more questions or request details on other products.

Additional Instructions for Handling Data:

Distinct Product Handling: Ensure each product is handled as a distinct entity. Use product titles as unique identifiers to prevent data mixing.
Data Structure Awareness: Recognize the structure of the data, which includes product title, description, and a bill of materials. This helps in accurately retrieving and providing relevant information.
Selective Information Sharing: Only share components relevant to the user's query from the bill of materials, avoiding information overload.

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
                {'type': 'user', 'text': 'Hello'},
                {'type': 'bot', 'text': 'Hello to you'},
                {'type': 'user', 'text': 'I want tv'},
                {'type': 'bot', 'text': 'yes we have'},
                {'type': 'user', 'text': 'I need l190'},
                {'type': 'bot', 'text': 'Yes we have l190'}
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
