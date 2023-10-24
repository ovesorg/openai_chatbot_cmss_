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
import os
import uvicorn
from langchain.vectorstores import Pinecone
import getpass
import pinecone
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
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
embeddings = OpenAIEmbeddings(openai_api_key=input("Enter openai api key:"))
pinecone.init(
    api_key=input("enter pinecone api key:"), 
    environment=input("Enter pinecone env:"), 
)

index_name = "omnivoltaic-company-data"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
llm = OpenAI(temperature=0.8, openai_api_key=input("Enter openai api key:"))
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

app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
   return templates.TemplateResponse("socket.html", {"request": request})
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
   await websocket.accept()
   while True:
      data = await websocket.receive_text()
      response = qa.run(data)
      await websocket.send_text(f"{response}")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

