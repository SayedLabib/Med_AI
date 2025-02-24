import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 

from dotenv import load_dotenv
from pathlib import Path

# Load .env file
dotenv_path = Path('.env')  # Ensure the .env file is in the same directory as your script
load_dotenv(dotenv_path=dotenv_path)

# Set up LLM (Mistral with Hugging Face)
HF_TOKEN = os.environ.get('HF_TOKEN')  # Retrieve S_TOKEN from .env file

hug_repo_id = 'mistralai/Mistral-7B-Instruct-v0.3'

def load_llm(hug_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=hug_repo_id,
        temperature=0.5,
        model_kwargs={'token': HF_TOKEN, 'max_length': '512'}
    )
    return llm

# Custom prompt template
custom_prompts = """
Use the information provided to answer the question. If you don't know the answer, just say
you don't know the answer. Do not make anything up.

Context: {context}
Question: {question}

Directly start the answer:
"""

def set_custom_prompt(custom_prompts):
    prompt = PromptTemplate(template=custom_prompts, input_variables=['context', 'question'])
    return prompt

# Load FAISS database
DB_FAISS_PATH = 'vectorstore/db_faiss'
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create Q&A chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(hug_repo_id),
    chain_type='stuff',
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(custom_prompts)}
)

# Invoke the Q&A chain
user_query = input('Write Query here: ')
response = qa_chain.invoke({'query': user_query})

print("Result: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])