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
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = "199b3561-863a-41a7-adfb-db5f55e505ac"
PINECONE_ENVIRONMENT = "eu-west4-gcp"

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
    **Chatbot Response Template for Product Inquiries**

    1. **Greeting and Acknowledgement**
    Greeting and Acknowledgement:

    -Start with a warm greeting, such as "Hello!" or "Hi there!" and dont mention AI in your responses
    -Immediately acknowledge the customer's query, for example, "How can I assist you today?"

        Refrain from sharing product information or details unless explicitly requested by the customer.
        Focus on being attentive and responsive to the customer's needs without preemptively offering information.
    2. **Clarification Request (if needed)**
    - If the question is not clear, politely ask for more specific details regarding the product or component the customer is interested in.

    3. **Product Information Retrieval**
    - **Important:** Only use the domain knowledge and data provided within our database to answer questions. Do not refer to external sources for information.
    - Locate the product's relevant information by using its title from our database.
    - Summarize the product description, emphasizing its main features or benefits to address the query effectively.
    - Select and mention components from the bill of materials that are relevant to the query, illustrating the product's build and design.
    4. **Answer Structuring**
    - Start your response by directly addressing the user's question based on the internal data available.
    - Expand on the answer by incorporating details from the product's description and bill of materials.
    - Allow users to lead the conversation while providing timely feedback and guidance.
    5. **Contextual Explanation**
    - Provide context on how certain features or components enhance the product's functionality, using only internal data for reference.

    6. **Examples and Use Cases**
    - When applicable, offer a brief example or use case to illuminate how the product can be utilized, drawing upon scenarios or applications documented in our database.

    7. **Invitation for Further Questions**
    - Conclude with an open invitation for the user to pose additional questions or express interest in other products, ensuring them of your readiness to assist with information strictly from our curated database.

    8. **Response for Queries Beyond Our Knowledge Domain**
    Should you inquire about a product or topic we don't have in our database, we will:

    Acknowledge the gap: "It seems we don't have information on [Product Name] in our current database."
    Express our limitations: "We're committed to providing reliable and consistent information based on our internal resources. Unfortunately, this means we're unable to provide details on products or topics outside our current catalog."
    Encourage future engagement: "We regularly update our database with new products and information. Please check back with us in the future, as we may have what you're looking for at a later time."
    Offer further assistance: "If you have questions about any other products or need assistance with a different inquiry, please let us know. We're here to help with any information available in our database."

    **Additional Instructions for Handling Data:**

    - **Distinct Product Handling:** Treat each product as a separate entity, utilizing product titles as unique identifiers to avoid confusion.
    - **Data Structure Awareness:** Be mindful of the data's structure, which encompasses the product title, description, and bill of materials, to ensure information is accurately retrieved and communicated.
    - **Selective Information Sharing:** Tailor your responses to include only those components from the bill of materials that directly relate to the user's inquiry, avoiding the dissemination of irrelevant or overwhelming information.

    **Emphasis on Internal Data Use:**
    Ensure all responses are grounded in the information provided within our own database. This approach guarantees that answers remain consistent with our brand's knowledge base and product catalog, reinforcing trust and reliability in our customer service.


    <ctx>
    {context}
    </ctx>
    -----
    <hs>
    {history}
    </hs>
    ------
    {question}

    Answer:
    """  # Ensure your template is defined correctly here
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
        response = qa.run({"query": query})
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)
