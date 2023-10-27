
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import re
import requests
import json
import pandas as pd
import uvicorn
from langchain.vectorstores import Pinecone
import getpass
import pinecone
from decouple import config
from dotenv import load_dotenv

k= os.getcwd()
ke = (f"{k}\\")
loader = DirectoryLoader(ke, "product_data.csv")
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = text_splitter.split_documents(doc)
embeddings = OpenAIEmbeddings(openai_api_key=input("enter openai :"))
pinecone.init(
    api_key=input("enter pinecone api:"), 
    environment=input("enter environment:"), 
)

index_name = "chatbot"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=1536,
       shards=1,
        pods=2
)
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
