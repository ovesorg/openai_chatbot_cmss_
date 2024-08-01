from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Cookie, Request, Response
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
from langchain import PromptTemplate
import os
import json
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
from langchain import PromptTemplate
import os
import json
import uvicorn
import pinecone
import logging
from langchain.chat_models import ChatOpenAI  # Use chat models

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global qa
    
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_existing_index("chatbot", embeddings)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo')  # Use ChatOpenAI
    retriever = docsearch.as_retriever()
    prompt_template = """
    As our online assistant, your task is to be short, precise, and truthful to the customer by effectively using the content provided in JSON format from our website to address customer inquiries. Always summarize your responses to make sense and give direct answers to users. Here's a structured approach:

    Greetings: Give short and precise answers when customers greet. Don't add any information that the customer has not asked.

    Content Understanding: Familiarize yourself with the structure and details of the website content provided in JSON format to ensure accurate information retrieval.

    Answering Inquiries: Utilize the website content to respond to user questions, ensuring responses are based solely on the provided information.

    Handling Unavailable Information: If a question arises that isn't covered by the website content, inform the customer politely that we currently don't have the information available.

    Ensure Precision and Pertinence: Many of our products bear names that are closely related, yet they exhibit significant distinctions. It is imperative that you exercise diligence to avoid conflating information between products. Your task is to meticulously gather and distill relevant data, ensuring that the insights provided to customers are unambiguous and succinct, thereby preventing any confusion stemming from product mix-ups.

    Accuracy and Relevance: Carefully compile and summarize relevant information without mixing product details to provide customers with clear, concise answers.

    from the website structure, use product, collection, product description, description properties.

    We have also added articles from our websites that will give more insights about general information that might be asked by users,

    Generate responses based on products and articles that we have on our websites only.
    
    <ctx>
    {context}
    </ctx>
    -----
    <hs>
    {history}
    </hs>
    ------
    {question}

    Answer:"""
    
    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template, max_tokens=2000)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=0,
        chain_type_kwargs={
            "verbose": False,
            "prompt": prompt,
            "memory": ConversationBufferWindowMemory(k=0, memory_key="history", llm=llm, input_key="question"),
        }
    )

@app.websocket("/ws/{email}")
async def websocket_endpoint(websocket: WebSocket, email: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            try:
                json_data = json.loads(data)
                if "input" in json_data:
                    try:
                        response = qa.run(json_data["input"])
                        await websocket.send_text(response)
                    except Exception as e:
                        # Handle the exception (e.g., log it)
                        print(f"Error: {str(e)}")
                        # Continue the loop to keep the connection alive
                        continue
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
    try:
        response = qa.run(query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)
