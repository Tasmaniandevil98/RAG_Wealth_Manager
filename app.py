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

def main():
    st.title('Wealth Management Chatbot')
    system_prompt = "You are a wealth management chatbot that can answer questions based on the provided documents."
    rag_params = RAGParams()
    docs = load_data(directory="/docs")  # Make sure this path is correct and accessible
    agent = construct_agent(system_prompt, rag_params, docs)

    user_input = st.text_input("You:", help='Type your query and press enter.')
    if st.button('Submit'):
        if user_input:
            # Perform chat
            response = agent.chat(user_input)
            if hasattr(response, 'response'):
                st.write(f"Bot: {response.response}")
                # Check if 'source_nodes' exists
                if hasattr(response, 'source_nodes') and len(response.source_nodes) >= rag_params.top_k:
                    top_k_results = response.source_nodes[:rag_params.top_k]
                    st.write("Top results for prompt:")
                    for i, result in enumerate(top_k_results, start=1):
                        st.write(f"{i}. {result.node.text[:1000]} (Score: {result.score})")
                else:
                    st.write("No sufficient source nodes available.")
            else:
                st.write("No response from the chat agent.")


if __name__ == "__main__":
    main()
