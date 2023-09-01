from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.retrievers import TFIDFRetriever
import os
import re
import requests
import json
import pandas as pd
import uvicorn
from getpass import getpass
OUTPUT_DIR= os.getcwd()
f = open(os.path.join(OUTPUT_DIR, 'notiondata.txt'), 'w')
f.write('This is the new file.')
f.close()
NOTION_TOKEN = getpass()
DATABASE_ID = getpass()
from langchain.document_loaders import NotionDBLoader
loader = NotionDBLoader(NOTION_TOKEN, DATABASE_ID,request_timeout_sec=50)
docs = loader.load()
sources = []
for f in docs:
    for u in f:        
        for l in u:
            sources.append(l)
o = str(sources)
r = o.translate(str.maketrans('', '','{}'))
mod_string = r.replace('\t', '').replace('\n', '')
list_of_char = ['“', '”','’']
pattern = '[' + ''.join(list_of_char) + ']'
v= re.sub(pattern, '', mod_string)
file1 = open('notiondata.txt', 'w')
file1.write(v)
file1.close()

url = "https://oves-2022.myshopify.com/api/2023-04/graphql.json"
page =1
def get_json(url, page):
    try:
        url = "https://oves-2022.myshopify.com/api/2023-04/graphql.json"

        payloads = "{\"query\":\"query MyQuery {\\r\\n  articles(first: 100) {\\r\\n    nodes {\\r\\n      content\\r\\n    }\\r\\n  }\\r\\n}\",\"variables\":{}}"
        headers = {
          'Content-Type': 'application/json',
          'X-Shopify-Storefront-Access-Token': 'c7d58bb21938849add72ce28c71303f3',
          'X-Shopify-Api-Version': '2023-04'
        }

        responses = requests.request("POST", url, headers=headers, data=payloads)
        products_json = responses.text[20:-2]
        products_dicts = json.loads(products_json)
        df = pd.DataFrame.from_dict(products_dicts['nodes'])
        OUTPUT_DIR= os.getcwd()
        f = open(os.path.join(OUTPUT_DIR, 'articles.txt'), 'w',encoding='utf-8')
        df.to_csv('articles.txt', header=None, index=None, sep=' ', mode='a')

        payload = json.dumps({
          "query": "query Product($first: Int) {\n  products(first: $first) {\n    nodes {\n      id\n      title\n      handle\n      descriptionHtml\n      publishedAt\n      createdAt\n      updatedAt\n      vendor\n      tags\n      createdAt\n      \n    }\n  }\n}",
          "variables": {
            "first": 100
          }
        })
        headers = {
          'Content-Type': 'application/json',
          'X-Shopify-Storefront-Access-Token': 'c7d58bb21938849add72ce28c71303f3',
          'X-Shopify-Api-Version': '2023-04'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        products_json = response.text[20:-2]
        products_dict = json.loads(products_json)
        df = pd.DataFrame.from_dict(products_dict['nodes'])
        OUTPUT_DIR= os.getcwd()
        f = open(os.path.join(OUTPUT_DIR, 'products.txt'), 'w',encoding='utf-8')
        df.to_csv('products.txt', header=None, index=None, sep=' ', mode='a')
    except requests.exceptions.HTTPError as error_http:
        print("HTTP Error:", error_http)

    except requests.exceptions.ConnectionError as error_connection:
        print("Connection Error:", error_connection)

    except requests.exceptions.Timeout as error_timeout:
        print("Timeout Error:", error_timeout)

    except requests.exceptions.RequestException as error:
        print("Error: ", error)

get_json(url, page)
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
k= os.getcwd()
ke = (f"{k}\\")
loader = DirectoryLoader(ke, glob="**/*.txt")
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = text_splitter.split_documents(doc)
embeddings = OpenAIEmbeddings(openai_api_key="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")
db = Chroma.from_documents(docs, embeddings)
llm = OpenAI(temperature=0.8, openai_api_key="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")
retriever = db.as_retriever()
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
# Setup RetrievalQA chain
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
    
while True:
    print(qa.run({"query": input('\n'"customer:")}))
