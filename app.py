import streamlit as st
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field
from typing import List, Optional
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
    docs = load_data(directory="./data/")  # Adjust directory as necessary

    if 'agent' not in st.session_state:
        st.session_state.agent = construct_agent(system_prompt, rag_params, docs)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Display previous conversation
    for index, exchange in enumerate(st.session_state.conversation_history):
        st.text_area("Conversation:", value=exchange, height=100, disabled=True, key=f"conversation_{index}")

    user_input = st.text_input("You:", help='Type your query and press enter.', key="user_input")

    if st.button('Submit'):
        if user_input:
            # Append user prompt to conversation history
            user_prompt_display = f"You: {user_input}"
            st.session_state.conversation_history.append(user_prompt_display)

            try:
                # Generate response using retry mechanism
                response = make_api_request(st.session_state.agent, user_input)
                bot_response_display = f"Bot: {response.response}"
                st.session_state.conversation_history.append(bot_response_display)
            except RetryError as e:
                st.error("Failed to connect to the API after several attempts. Please try again later.")

            # Clear input by updating the state used to manage the input value
            st.session_state['reset_input'] = True

            # Display updated conversation
            st.experimental_rerun()

    # Check if we need to reset the input
    if st.session_state.get('reset_input', False):
        st.session_state['user_input'] = ""  # Clear input
        del st.session_state['reset_input']  # Remove the flag to prevent further resets

if __name__ == "__main__":
    main()
