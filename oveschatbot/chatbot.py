from langchain import OpenAI
import sqlite3  
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
from decouple import config
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Access the variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT')
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
templates = Jinja2Templates(directory="templates")
# SQLite3 database setup
conn = sqlite3.connect("your_database.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS my_tables (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        password TEXT NOT NULL,
        Telephone TEXT NOT NULL,
        email TEXT NOT NULL,
        country_of_origin TEXT NOT NULL
    );
''')
conn.commit()

# Function to hash a password
def get_password_hash(password: str):
    return pwd_context.hash(password)

# Function to verify a password
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

# Function to create a new user
def create_user(username: str, password: str):
    hashed_password = get_password_hash(password)
    cursor.execute('INSERT INTO my_tables (username, password) VALUES (?, ?)', (username, hashed_password))
    conn.commit()

# Function to authenticate a user
def authenticate_user(credentials: HTTPBasicCredentials):
    cursor.execute('SELECT username, password FROM my_tables WHERE username = ?', (credentials.username,))
    user = cursor.fetchone()
    if user is None or not verify_password(credentials.password, user[1]):
        return None
    return credentials.username

html = '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WebSocket Client</title>
</head>
<body>
  <div id="loginSection">
    <h2>Login</h2>
    <input type="text" id="username" placeholder="Username"><br>
    <input type="password" id="password" placeholder="Password"><br>
    <button onclick="login()">Login</button>
  </div>

  <div id="chatSection" style="display: none;">
    <h2>Chat</h2>
    <textarea id="messages" rows="10" cols="100" disabled></textarea><br>
    <input type="text" id="messageInput" placeholder="Type a message">
    <button onclick="sendMessage()">Send</button>
  </div>

  <script>
    let ws;
    let isLoggedIn = false;

    function login() {
      const username = document.getElementById('username').value;
      const password = document.getElementById('password').value;
      const credentials = btoa(`${username}:${password}`);

      ws = new WebSocket(`ws://http://127.0.0.1:8000//ws?credentials=${credentials}`);

      ws.onopen = (event) => {
        console.log('WebSocket connection opened:', event);
        document.getElementById('loginSection').style.display = 'none';
        document.getElementById('chatSection').style.display = 'block';
        isLoggedIn = true;
      };

      ws.onmessage = (event) => {
        const messagesTextarea = document.getElementById('messages');
        messagesTextarea.value += `Received: ${event.data}\n`;
      };
    }

    function sendMessage() {
      if (!isLoggedIn) {
        alert('Please log in first.');
        return;
      }

      const messageInput = document.getElementById('messageInput');
      const message = messageInput.value;
      ws.send(message);
      messageInput.value = '';
    }
  </script>
</body>
</html>

'''

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
async def get():
    return HTMLResponse(html)


# Initialize an empty dictionary to store user conversations
user_conversations = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, credentials: str):
    try:
        # Decode base64-encoded credentials
        decoded_credentials = base64.b64decode(credentials).decode("utf-8")
        # Split credentials into username and password
        username, password = decoded_credentials.split(":")

        # Check the credentials against the SQLite database
        cursor.execute('SELECT username, password FROM my_tables WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()

        if user:
            await websocket.accept()

            # Initialize or get the conversation for the user
            user_data = user_conversations.get(username, [])
            
            while True:
                data = await websocket.receive_text()

                # Append the user's message to their conversation history
                user_data.append(f"{username}: {data}")

                # Update the user's conversation history in the dictionary
                user_conversations[username] = user_data

                # Concatenate the user's chat history for context
                chat_history = "\n".join(user_data)
                
                # Prepare the query with context for embeddings
                query_with_context = {
                    "context": chat_history,
                    "question": data,
                }

                # Convert the query to a string
                query_string = json.dumps(query_with_context)

                # Get the response from embeddings using the query with context
                response = qa.run(query_string)

                # Send the chatbot's response back to the user
                await websocket.send_text(response)
        else:
            raise HTTPException(status_code=403, detail="Invalid credentials")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid credentials format")
    
class User(BaseModel):
    username: str
    password: str
    Telephone: int
    email: str
    country_of_origin: str
@app.post("/insert/")
async def insert_data(user: User):
    try:
        # Insert data into the database
        cursor.execute("INSERT INTO my_tables (username, password,Telephone,email,country_of_origin) VALUES (?, ?,?,?,?)", (user.username, user.password,user.Telephone,user.email,user.country_of_origin))
        conn.commit()
        return {"message": "Data inserted successfully"}
    except Exception as e:
        return {"error": f"Failed to insert data: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1:8000/", port=8000)
