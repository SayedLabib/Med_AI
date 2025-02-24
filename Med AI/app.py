import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Custom Styling
st.set_page_config(page_title="MediBot - AI Clinical Assistant", page_icon="🏥", layout="wide")

# Inject custom CSS for enhanced UI

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #87CEEB, #FFFFFF);
        margin: 0;
        padding: 0;
        font-family: Arial, sans-serif;
        height: 100vh;
        width: 100vw;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("""
    <style>
        body {
            background: #a3ff05;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }

        .image-container {
            width: 300px;
            height: 300px;
            overflow: hidden;
            margin-left: 40%;
        }
        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 50%
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 10px;
            margin: 10px 0;
            max-width: 70%;
        }
        .stChatMessage.user {
            background-color: #d0e8ff;
            margin-left: auto;
        }
        .stChatMessage.assistant {
            background-color: #f0f0f0;
            margin-right: auto;
        }
        .stTextInput input {
            width: 100% !important;
            height: 50px !important;
            border-radius: 8px !important;
            border: 2px solid #ccc !important;
            padding: 10px !important;
            font-size: 16px !important;
        }
        .clear-button {
            margin-top: 20px;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 8px;
            background-color: #ff4b4b;
            color: white;
            border: none;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .clear-button:hover {
            background-color: #ff1c1c;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #333;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def set_custom_prompt():
    return PromptTemplate(template="""
        Use the provided context to answer the user's question concisely. 
        If unsure, simply state so—avoid making up information.
        
        Context: {context}
        Question: {question}
    """, input_variables=["context", "question"])

def load_llm():
    return HuggingFaceEndpoint(repo_id=HUGGINGFACE_REPO_ID, temperature=0.5, model_kwargs={"token": HF_TOKEN, "max_length": 512})

def main():
    
    st.markdown('<div class="image-container"><img src="https://t3.ftcdn.net/jpg/05/28/97/80/360_F_528978063_hoXUak6fcFDiboHUVIbjxdAEK8sFMFZX.jpg" alt="Clinical AI Bot"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="title">MediBot 🏥 - AI Clinical Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your intelligent assistant for clinical insights</div>', unsafe_allow_html=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            role_class = "user" if message["role"] == "user" else "assistant"
            st.markdown(f'<div class="stChatMessage {role_class}">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Clear chat history and reset input field
    if st.button("Clear Chat History", key="clear-history", help="Click to clear chat history", use_container_width=True):
        if not st.session_state.messages:
            st.warning("Nothing to remove! Chat history is already empty.")
        else:
            st.session_state.messages = []
            st.session_state["input"] = ""  # Reset input field
            st.success("Chat history cleared successfully!")
            st.rerun()
    
    # Input field with session state management
    prompt = st.text_input("Ask a medical question:", value=st.session_state.get("input", ""), key="input", help="Enter your medical question here")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f'<div class="stChatMessage user">{prompt}</div>', unsafe_allow_html=True)
        
        try:
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )
            response = qa_chain.invoke({'query': prompt})
            result = response.get("result", "No response available.")
            st.session_state.messages.append({"role": "assistant", "content": result})
            st.markdown(f'<div class="stChatMessage assistant">{result}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
