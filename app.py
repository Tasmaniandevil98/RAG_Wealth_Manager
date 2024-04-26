import streamlit as st
from llama_index.llms import OpenAI as LlamaOpenAI
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field
from typing import List
import os
from tenacity import retry, wait_fixed, stop_after_attempt, after_log, RetryError
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    llm = LlamaOpenAI(model=rag_params.llm)
    service_context = ServiceContext.from_defaults(chunk_size=rag_params.chunk_size, llm=llm, embed_model=rag_params.embed_model)
    vector_index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    return CondensePlusContextChatEngine.from_defaults(vector_index.as_retriever(similarity_top_k=rag_params.top_k), system_prompt=system_prompt)

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5), after=after_log(logger, logging.INFO))
def make_api_request(agent, user_input):
    try:
        response = agent.chat(user_input)
        logger.info(f"Received response: {response.response}")
        return response
    except Exception as e:
        logger.error(f"Failed to get response: {e}")
        raise

def main():
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    st.title('AI powered Wealth Manager')
    if 'docs' not in st.session_state:
        st.session_state.docs = load_data(directory="docs/")
    if 'agent' not in st.session_state:
        st.session_state.agent = construct_agent("You are a wealth management chatbot.", RAGParams(), st.session_state.docs)
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_context' not in st.session_state:
        st.session_state.current_context = ""

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    prompt = st.chat_input("Hello! Please ask me any wealth management questions here...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = make_api_request(st.session_state.agent, prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.response})

        # Update context in the expander
        top_k_results = [
            f"{i + 1}. {result.node.text[:1000]} (Score: {result.score})"
            for i, result in enumerate(response.source_nodes[:2])
        ]
        st.session_state.current_context = "\n".join(top_k_results)

    # Expander with context details
    with st.expander("See response details", expanded=True):
        st.write(st.session_state.current_context)
    

if __name__ == "__main__":
    main()
