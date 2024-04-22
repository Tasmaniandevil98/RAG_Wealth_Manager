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
        
# Function to set the background
def set_background(image_path):
    css_style = f"""
    <style>
    .stApp {{
        background-image: url({image_path});
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(css_style, unsafe_allow_html=True)
    
def main():
    set_background("chatbot Background.jpg")
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]
    st.title('Wealth Management Chatbot')
    system_prompt = "You are a wealth management chatbot that can answer questions based on the provided documents."
    rag_params = RAGParams()

    if 'docs' not in st.session_state:
        st.session_state.docs = load_data(directory="docs/")
    if 'agent' not in st.session_state:
        st.session_state.agent = construct_agent(system_prompt, rag_params, st.session_state.docs)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'context_history' not in st.session_state:
        st.session_state.context_history = []

    # Display previous conversations without numbering the context
    for index, (conversation, context) in enumerate(zip(st.session_session.conversation_history, st.session_state.context_history)):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.text_area("Conversation:", value=conversation, height=150, disabled=True, key=f"conv_{index}")
        with col2:
            st.text_area("Context:", value=context, height=150, disabled=True, key=f"context_{index}")

    # Unique key for input widget using count of inputs
    user_input_key = f"user_input_{len(st.session_state.conversation_history)}"
    user_input = st.text_input("User Input:", key=user_input_key)

    if st.button('Submit', key='submit_button'):
        if user_input:
            response = make_api_request(st.session_state.agent, user_input)
            bot_response = f"You: {user_input}\nBot: {response.response}\n"
            st.session_state.conversation_history.append(bot_response)

            top_k_results = [
                f"{i + 1}. {result.node.text[:1000]} (Score: {result.score})"
                for i, result in enumerate(response.source_nodes[:rag_params.top_k])
            ]
            top_k_results_text = "\n".join(top_k_results)
            st.session_state.context_history.append(top_k_results_text)

            # Refresh UI
            st.experimental_rerun()

if __name__ == "__main__":
    main()
