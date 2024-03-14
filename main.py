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
import pinecone
import logging
from fastapi import Header
import uuid
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = "199b3561-863a-41a7-adfb-db5f55e505ac"
PINECONE_ENVIRONMENT = "eu-west4-gcp"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global qa
    
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', openai_api_key=OPENAI_API_KEY)
    docsearch = Pinecone.from_existing_index("chatbot", embeddings)
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model='gpt-3.5-turbo-instruct')
    retriever = docsearch.as_retriever()
    prompt_template = """

    As our online assistant, your task is to effectively use the content provided in JSON format from our website to address customer inquiries. Here's a structured approach:

    About our store: Always make up a good a summary of our store by going through every detail and giving factual summary of what we do.
    
    Content Understanding: Familiarize myself with the structure and details of the website content provided in JSON format to ensure accurate information retrieval.
    
    Answering Inquiries: Utilize the website content to respond to user questions, ensuring responses are based solely on the provided information.
    
    Handling Unavailable Information: If a question arises that isn't covered by the website content, I will inform the customer politely that we currently don't have the information available.

    Ensure Precision and Pertinence: Many of our products bear names that are closely related, yet they exhibit significant distinctions. It is imperative that you exercise diligence to avoid conflating information between products. Your task is to meticulously gather and distill relevant data, ensuring that the insights provided to customers are unambiguous and succinct, thereby preventing any confusion stemming from product mix-ups.
    
    Accuracy and Relevance: Carefully compile and summarize relevant information without mixing product details to provide customers with clear, concise answers.

    from the website structure, use product, collection, product description, description properties.

    Generate responses based on products that we have on our websites only.

    Hallucination: Avoid giving unnecesary information that the customer has not asked for.
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
    # prompt_template = """
    # Given the following user question, history and context, formulate a response that would be the most relevant to provide the user with an answer from a knowledge base.
    # You should follow the following rules when generating an answer:
    # - if the customer starts with greetings, ask him or her what you can help today
    # - Understand user question( synonyms), and specificity of the question and use your intelligence to understand the products information that the user is looking for.
    # - from the user question, scan through peoduct title, product description and other product features and extract exactly the right matching data. 
    # -after scanning the databse and you dont get the information that user asked from our own database, ask the user to try next time. Right now we dont have the information about what he is asking.
    # - strictly don't formulate answers for questions that ask for information that is not in our database
    # - you answer responses should **strictly** not be more than 25 words, the response should include technical specifications, uses cases and and parts from BOM 
    # - for a question that requires comparison, get the product information for each product and give truthful comparison, citing similarities and dont give misleading information here, restrict yourself to database infomation the products
    # - to formulate your answer, consider product titles, product description and other content in product only.
    # - in your answer formulation avoid mixing product information.
    # -Don't use internet 

    # <ctx>
    # {context}
    # </ctx>
    # -----
    # <hs>
    # {history}
    # </hs>
    # ------
    # {question}

    #  Answer:"""
    
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
user_contexts = {}

# element_list = eval(f"{result_list}")

#                 # Extract the last three elements
#                 last_three_elements = element_list[-2:]


#                 #print(chat_history)
#                 # Prepare the query with context for embeddings
#                 query_with_context = {
#                     "context": last_three_elements,
#                     "question": data,
#                 }

#                 # Convert the query to a string
#                 query_string = json.dumps(query_with_context)

#                 query_string = " ".join(query_string.split()[:2000])


#                 # Broadcast the message to all participants in the chatroom
#                 for participant in chatrooms[chatroom]:
#                     if participant != websocket:
#                         try:
#                             response = qa.run(query_string)

@app.post("/query/")
async def get_response(query: str):
    # element_list = {"question_1": "Hello","question_2": "I need tv","question_3": "how is your motorbikes"}
    # query_with_context = {
    #     "context": element_list,
    #     "question": query,
    # }
    # print(query_with_context)
    if not query:
        raise HTTPException(status_code=400, detail="Query not provided")
    try:
        # query_string = json.dumps(query_with_context)
        # query_string = " ".join(query_string.split()[:1000])
        # print(query_string)
        response = qa.run(query)
        #response = qa.run({"query": query})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
# @app.post("/query/")
# async def chatbot(request: Request, user_id: str = Cookie(None), message: str = None):
#     if not user_id:
#         # Generate a new UUID if user_id is not provided in the cookies
#         user_id = str(uuid.uuid4())
#         #response = qa.run({"query": message})
#         responses = Response(content="Hello, Chatbot!")
#         responses.set_cookie(key="user_id", value=user_id)
#         logger.info(f"New user identified with ID: {user_id}")
#         return responses

#     # Get the user's conversation context from the dictionary
#     context = user_contexts.get(user_id, [])
#     response = qa.run({"query": message})
#     return response

#     # Process the message and update the conversation context
#     if message:
#         # Add the message to the user's conversation context
#         context.append(message)
#         # Update the conversation context in the dictionary
#         user_contexts[user_id] = context
#         logger.info(f"Message received from user {user_id}: {message}")

#     # Return the user's conversation context
#     logger.info(f"Returning conversation context for user {user_id}: {context}")
#     return {"user_id": user_id, "conversation_context": context}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)
