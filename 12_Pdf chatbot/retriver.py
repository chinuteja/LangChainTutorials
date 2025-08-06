import os
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader

from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics.pairwise import cosine_similarity
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import re
from langchain.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# 1. Initialize LLM (Groq)


# 2. Create embeddings and vector store
def create_embeddings(chunks):
    """
    Creates a FAISS vector store from given document chunks.
    """
    embeddings = CohereEmbeddings(model="embed-english-v3.0")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# 3. Create retriever with MMR + multi-query
def create_multiquery_retriever(vector_store,model_groq):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "lambda_mult": 0.5}
    )
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=model_groq
    )
    return multiquery_retriever