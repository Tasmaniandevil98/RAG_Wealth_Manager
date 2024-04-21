import streamlit as st
from llama_index.llms import OpenAI
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
    llm = OpenAI(model=rag_params.llm)
    service_context = ServiceContext.from_defaults(chunk_size=rag_params.chunk_size, llm=llm, embed_model=rag_params.embed_model)
    vector_index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    agent = CondensePlusContextChatEngine.from_defaults(vector_index.as_retriever(similarity_top_k=rag_params.top_k), system_prompt=system_prompt)
    return agent

@retry(wait=wait_fixed(2), stop=stop_after_attempt(5), after=after_log(logger, logging.INFO))
def make_api_request(agent, user_input):
    """Attempt to make an API request with retry mechanism."""
    return agent.chat(user_input)

# Streamlit UI setup
def main():
    st.title('Wealth Management Chatbot')
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    system_prompt = "You are a wealth management chatbot that can answer questions based on the provided documents."
    rag_params = RAGParams()

    if 'docs' not in st.session_state:
        st.session_state.docs = load_data(directory="docs/")  # Adjust directory as necessary

    if 'agent' not in st.session_state:
        st.session_state.agent = construct_agent(system_prompt, rag_params, st.session_state.docs)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    user_input = st.text_input("You:", key="user_input")

    if st.button('Submit', key='submit_button'):
        if user_input:
            user_prompt_display = f"You: {user_input}\n"
            response = make_api_request(st.session_state.agent, user_input)
            bot_response_display = f"Bot: {response.response}\n"
            top_k_results_display = "Top results:\n"
            top_k_results = response.source_nodes[:rag_params.top_k]
            for i, result in enumerate(top_k_results, start=1):
                top_k_results_display += f"{i}. {result.node.text[:1000]} (Score: {result.score})\n"

            # Append the full conversation block (user prompt, bot response, and top-k results)
            full_conversation_block = user_prompt_display + bot_response_display + top_k_results_display
            st.session_state.conversation_history.append(full_conversation_block)
            st.text_area("Conversation:", value="\n".join(st.session_state.conversation_history), height=300, disabled=True)

            # Clear the input field
            st.rerun()

if __name__ == "__main__":
    main()
