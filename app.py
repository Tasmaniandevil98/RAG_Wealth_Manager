import streamlit as st
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.chat_engine import CondensePlusContextChatEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from pydantic import BaseModel, Field
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
    return agent.chat(user_input)

def main():
    st.title('AI powered Wealth Manager')
    os.environ["OPENAI_API_KEY"] = st.secrets["openai_api_key"]

    rag_params = RAGParams()
    if 'docs' not in st.session_state:
        st.session_state.docs = load_data(directory="docs/")
    if 'agent' not in st.session_state:
        st.session_state.agent = construct_agent("You are a wealth management chatbot that can answer questions based on the provided documents.", rag_params, st.session_state.docs)

    # Chat history container
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Input box and button
    user_input = st.text_input("You:", placeholder="Type your wealth management questions here...", key="user_input")
    if st.button('Submit'):
        if user_input:
            # Process the input
            response = make_api_request(st.session_state.agent, user_input)
            st.session_state.messages.append(f"You: {user_input}\nBot: {response.response}")

            # Display top k results in an expander
            top_k_results = "\n".join([f"{i + 1}. {result.node.text[:1000]} (Score: {result.score})"
                                       for i, result in enumerate(response.source_nodes[:rag_params.top_k])])
            st.session_state.messages.append(f"Context:\n{top_k_results}")

    # Display messages
    for message in reversed(st.session_state.messages):
        st.text_area("", value=message, height=200, disabled=True)

if __name__ == "__main__":
    main()
