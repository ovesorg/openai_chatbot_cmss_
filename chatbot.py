from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import NotionDBLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.chains import ConversationChain
from langchain.retrievers import TFIDFRetriever
from getpass import getpass
#from docx import Document
import os
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
v = r.replace('\t', '').replace('\n', '')
print(v)
file1 = open('notiondata.txt', 'w')
file1.write(v)
file1.close()
loader = TextLoader('notiondata.txt')
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
docs = text_splitter.split_documents(doc)
embeddings = OpenAIEmbeddings(openai_api_key="enter_open_ai_key")
db = Chroma.from_documents(docs, embeddings)
llm = OpenAI(temperature=0.8, openai_api_key="enter_open_ai_key")
retriever = db.as_retriever()    
template = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question and answer starting with Ovsmart then full colon and the response: 
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
while True:
    print(qa.run({"query": input('\n'"customer:")}))
