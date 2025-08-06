import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from preprocessing import extract_and_clean_pdf, split_into_chunks
from retriver import create_embeddings, create_multiquery_retriever
from RAG import build_chain

load_dotenv()

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="wide")

# LangChain LLM
model_groq = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="Gemma2-9b-It",
    temperature=0.5
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metadata" not in st.session_state:
    st.session_state.metadata = {}
if "multiquery_retriever" not in st.session_state:
    st.session_state.multiquery_retriever = None

# ---------------- CSS ---------------- #
st.markdown(
    """
    <style>
    /* Fix title at top */
    .chat-header {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        padding: 10px;
        border-bottom: 1px solid #ddd;
        background-color: white;
        position: sticky;
        top: 0;
        z-index: 10;
    }

    /* Scrollable chat area */
    .chat-container {
        height: calc(100vh - 200px); /* leave space for header + input */
        overflow-y: auto;
        padding: 10px;
    }

    /* Fix input at bottom */
    .stChatInput {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 50%;
        z-index: 1000;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- SIDEBAR UPLOAD ---------------- #
with st.sidebar:
    st.header("📂 Upload PDF")
    uploaded_file = st.file_uploader("Choose your PDF", type="pdf")
    if st.button("Submit") and uploaded_file:
        clean_text, metadata = extract_and_clean_pdf(uploaded_file)
        chunks = split_into_chunks(clean_text)
        vector_store = create_embeddings(chunks)
        multiquery_retriever = create_multiquery_retriever(vector_store, model_groq)

        st.session_state.metadata = metadata
        st.session_state.multiquery_retriever = multiquery_retriever
        st.success("PDF processed successfully!")

# ---------------- LAYOUT ---------------- #
col_left, col_center, col_right = st.columns([1, 2, 1])

# ---------------- CENTER: CHAT UI ---------------- #
with col_center:
    st.markdown("<div class='chat-header'>💬 Chat with your PDF</div>", unsafe_allow_html=True)

    if st.session_state.metadata:
        # Capture input BEFORE rendering chat history
        user_input = st.chat_input("Ask something about your PDF...")

        if user_input:
            # Append user message immediately
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Generate bot response immediately
            bot_response = build_chain(user_input, st.session_state.multiquery_retriever)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Now render chat messages (including the new one)
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.chat_message("user").markdown(msg["content"])
            else:
                st.chat_message("assistant").markdown(msg["content"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Quit & Save Chat
        if st.button("Quit & Save Chat"):
            chat_text = ""
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Assistant"
                chat_text += f"{role}: {msg['content']}\n\n"

            file_name = "chat_history.txt"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(chat_text)

            with open(file_name, "rb") as f:
                st.download_button(
                    label="📥 Download Chat History",
                    data=f,
                    file_name=file_name,
                    mime="text/plain"
                )

            st.success("✅ Chat history saved! Click the download button above.")
    else:
        st.info("Please upload and submit a PDF to start chatting.")


# ---------------- RIGHT SIDEBAR ---------------- #
with col_right:
    if st.session_state.metadata:
        st.header("📑 PDF Info")
        for key, value in st.session_state.metadata.items():
            st.write(f"**{key}:** {value}")
