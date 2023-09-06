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

os.environ["PINECONE_API_KEY"] = getpass.getpass("Pinecone API Key:")
os.environ["PINECONE_ENV"] = getpass.getpass("Pinecone Environment:")
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
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
embeddings = OpenAIEmbeddings()
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"), 
    environment=os.getenv("PINECONE_ENV"), 
)

index_name = "omnivoltaic-company-data"
docsearch = Pinecone.from_existing_index(index_name, embeddings)
llm = OpenAI(temperature=0.8, openai_api_key=getpass.getpass("OpenAI API Key:"))
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

@app.get("/oves")
def form_post(request: Request):
    response = "enter query"
    return templates.TemplateResponse('okay.html', context={'request': request, 'response': response})

@app.post("/oves")
def form_post(request: Request, num: str = Form(...)):
    response = qa.run(num)
    return templates.TemplateResponse('okay.html', context={'request': request, 'response': response})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


