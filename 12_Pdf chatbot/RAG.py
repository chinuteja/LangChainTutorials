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
import preprocessing
from preprocessing import extract_and_clean_pdf, split_into_chunks
from retriver import create_embeddings, create_multiquery_retriever

load_dotenv()

llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="Gemma2-9b-It",
        temperature=0.5 ## this is creative parameter
    )

model_groq = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="Gemma2-9b-It",
    temperature=0
)

def augumentation(question,multiquery_retriever):


    prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say the question is out of context.
      Always cite the sources by page number in square brackets.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
    retrieved_docs    = multiquery_retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    # final_prompt = prompt.invoke({"context": context_text, "question": question})

    return prompt


def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

def build_chain(question,multiquery_retriever):
    
    parallel_chain = RunnableParallel({
    'context': multiquery_retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
    prompt = augumentation(question,multiquery_retriever)
    parser = StrOutputParser()
    # print("promt:", type(prompt))
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain.invoke(question)



## to test thje code
# clean_text, metadata = extract_and_clean_pdf("PDF Documents/attention_all_you_need.pdf")
# chunks = split_into_chunks(clean_text)
# vector_store = create_embeddings(chunks)
# multiquery_retriever = create_multiquery_retriever(vector_store, model_groq)
# print(build_chain("What is the main idea of the paper?", multiquery_retriever))