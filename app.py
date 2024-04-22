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


def main():
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

    if 'input_count' not in st.session_state:
        st.session_state.input_count = 0

    # Display all conversation history before input fields
    for index, conversation in reversed(list(enumerate(st.session_state.conversation_history))):
        st.text_area(f"Conversation {index + 1}:", value=conversation, height=300, disabled=True, key=f"conv_{index}")

    # Generate a unique key for the input widget using the count of inputs
    user_input_key = f"user_input_{st.session_state.input_count}"
    user_input = st.text_input("You:", key=user_input_key)

    if st.button('Submit', key='submit_button'):
        if user_input:
            response = make_api_request(st.session_state.agent, user_input)
            conversation_text = f"You: {user_input}\nBot: {response.response}\n"
            top_k_results_text = "\n".join(
                f"{i + 1}. {result.node.text[:1000]} (Score: {result.score})"
                for i, result in enumerate(response.source_nodes[:rag_params.top_k])
            )
            full_conversation_text = conversation_text + top_k_results_text
            st.session_state.conversation_history.append(full_conversation_text)

            # Increment the input count to generate a new key for the next input
            st.session_state.input_count += 1
            # Clear the input field and rerun the app to refresh the UI
            st.experimental_rerun()

if __name__ == "__main__":
    main()
