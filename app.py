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

# Custom CSS to include custom avatars and style the chat messages
st.markdown(
    """
    <style>
    .chat-message {
        position: relative;
        padding-left: 60px; /* Make space for avatar image */
        min-height: 50px; /* Ensure the message box is large enough to display the avatar */
    }
    .chat-avatar {
        position: absolute;
        top: 0;
        left: 0;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-size: cover;
        background-position: center;
    }
    .user-avatar {
        background-image: url('https://raw.githubusercontent.com/Tasmaniandevil98/RAG_Wealth_Manager/85277fcf16ba85e667326d8a9d23b6e3688e0743/you-are-here.png'); /* Direct link to an image file */
    }
    .bot-avatar {
        background-image: url('https://raw.githubusercontent.com/Tasmaniandevil98/RAG_Wealth_Manager/85277fcf16ba85e667326d8a9d23b6e3688e0743/asset.png'); /* Direct link to an image file */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Example function to display chat messages
def display_chat_message(role, text):
    avatar_class = "user-avatar" if role == "user" else "bot-avatar"
    
    st.markdown(
        f"""
        <div class="chat-message">
            <div class="{avatar_class} chat-avatar"></div>
            <p>{text}</p>
        </div>
        """,
        unsafe_allow_html=True 
        )

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
            f"{i + 1}. {result.node.text[:1000]}"
            for i, result in enumerate(response.source_nodes[:2])
        ]
        st.session_state.current_context = "\n".join(top_k_results)
        # Use rerun here to update the interface after processing the input
        st.rerun()

    # Expander with context details
    with st.expander("See the details about the source of information", expanded=False):
        st.write(st.session_state.current_context)
    

if __name__ == "__main__":
    main()
