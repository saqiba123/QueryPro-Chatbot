import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import List, Tuple
load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0,openai_api_key=os.getenv("OPENAI_API_KEY"))
# Convert chat history list to LangChain message format
def convert_history_to_messages(history: List[Tuple[str, str]]) -> List:
    messages = []
    for q, a in history:
        messages.append(HumanMessage(content=q))
        messages.append(AIMessage(content=a))
    return messages

def create_vectorstore(documents, save_path):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local(save_path)

def load_vectorstore(path):
    embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)

def call_plain_llm(query: str, history: list):
    messages = convert_history_to_messages(history)
    messages.append(HumanMessage(content=query))
    response = llm.invoke(messages)
    history.append((query, response.content))
    return response.content

def create_rag_chain(vectorstore, history: list):
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True)
    memory.chat_memory.messages = convert_history_to_messages(history)
    chain = ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever,memory=memory,return_source_documents=False,verbose=False)
    return chain