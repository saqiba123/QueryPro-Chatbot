from utils.rag_chain import load_vectorstore, create_rag_chain

def get_chat_response(query: str, session_id: str, history: list):
    vectorstore = load_vectorstore(f"faiss_index/{session_id}")
    rag_chain = create_rag_chain(vectorstore, history)
    result = rag_chain.invoke({"question": query})
    history.append((query, result["answer"]))  # store as tuple
    return result["answer"], history
