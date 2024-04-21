import streamlit as st
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field
from typing import List
import os

# Parameters for RAG
class RAGParams(BaseModel):
    include_summarization: bool = Field(default=False)
    top_k: int = Field(default=2)
    chunk_size: int = Field(default=1024)
    embed_model: str = Field(OpenAIEmbedding())
    llm: str = Field(default="gpt-4-1106-preview")

def load_data(directory: str) -> List[Document]:
    from llama_index import SimpleDirectoryReader
    reader = SimpleDirectoryReader(input_dir=directory)
    return reader.load_data()

def construct_agent(system_prompt: str, rag_params: RAGParams, docs: List[Document]) -> CondensePlusContextChatEngine:
    llm = OpenAI(model=rag_params.llm)
    service_context = ServiceContext.from_defaults(chunk_size=rag_params.chunk_size, llm=llm, embed_model=rag_params.embed_model)
    vector_index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    agent = CondensePlusContextChatEngine.from_defaults(vector_index.as_retriever(similarity_top_k=rag_params.top_k), system_prompt=system_prompt)
    return agent

# Streamlit UI setup
def main():
    st.title('Wealth Management Chatbot')
    openai_api_key = st.secrets["openai_api_key"]
    os.environ["OPENAI_API_KEY"] = openai_api_key
    system_prompt = "You are a wealth management chatbot that can answer questions based on the provided documents."

    # Initialize parameters and load data only once
    if 'agent' not in st.session_state:
        rag_params = RAGParams()
        docs = load_data(directory="/docs")
        st.session_state.agent = construct_agent(system_prompt, rag_params, docs)

    # Chat interface
    user_input = st.text_input("You:", help='Type your query and press enter.')
    if st.button('Submit') and user_input:
        # Append user input to chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append(f"You: {user_input}")

        # Perform chat
        response = st.session_state.agent.chat(user_input)
        st.session_state.chat_history.append(f"Bot: {response.response}")

        # Display chat history
        for message in st.session_state.chat_history:
            st.text(message)

        # Optionally, display the top K results
        top_k_results = response.source_nodes[:rag_params.top_k]
        st.write("Top results for prompt:")
        for i, result in enumerate(top_k_results, start=1):
            st.write(f"{i}. {result.node.text[:1000]} (Score: {result.score})")

if __name__ == "__main__":
    main()
