# RAG Q&A Chatbot with Chat History
import streamlit as st       
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
import os

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Setup a Streamlit App
st.title("Conversational RAG with PDF uploads and Chat History")
st.write("Upload Pdf and chat with their content")


# Input the GROQ API Key
api_key = st.text_input("Enter your GROQ API Key", type="password")

# Check if GROQ API Key is provided
if api_key:
    llm = ChatGroq(api_key= api_key, model="gemma2-9b-it")
    
    # Chat Interface
    session_id = st.text_input("Session ID", value="default_session")
    
    # Statefully manage chat histories
    if "store" not in st.session_state:
        st.session_state.store = {}
        
    uploaded_files = st.file_uploader("Choose a Pdf file", type="pdf", accept_multiple_files=True)
    
    # Process my uploaded files
    if uploaded_files:
        document = []
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
                
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            document.extend(docs)
    # Split and create embeddings for the document
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(document)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()
        
        
    
        contextualized_q_system_prompt = (
            """
            Given a chat history and latest user question
            which might reference context in chat history,
            formlate a standalone question which can be understood
            without chat history. Do NOT answer the question,
            just reformulate it if needed and otherwise return it as is.
            """
            )
        contextualized_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualized_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
    
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualized_q_prompt)
        
        
        # Answer Question Prompt
        system_prompt = (
        """
            You are an assistant for question-answering tasks.
            Use the following piece of retrieved context to answer the question.
            If you don't know the answer, say that you don't know. Use three sentences 
            maximum and keep the answer concise.
            \n\n
            {context}
        """
        )
        
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        
        def get_session_history(session_id:str)-> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        user_input = st.text_input("Your Questions")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable":{"session_id":session_id}
                }, # constructs a session_id in store
            )
            
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
        
else:
    st.write("Please provide the GROQ API Key")