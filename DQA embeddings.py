# Install Libraries
# pip install unstructured --user

# importing libraries
import os
import openai
import pinecone
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# OpenAI credentials
openai_api_key = "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"

# Pinecone credentials
api_key = "ef9a9434-5233-4b29-a794-355b106be8d7"
environment = "us-west4-gcp-free"
index_name = "techplaybook-dev"

# Source Files
directory_path = '.\dataset'

def load_docs(directory_path):
    loader = DirectoryLoader(directory_path)
    documents = loader.load()
    return documents

documents = load_docs(directory_path)
print("Total number of documents :",len(documents))

# Document Splitting for Efficient Processing with LangChain:
def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs
docs = split_docs(documents)
print("Length of documents: ", len(docs))

# Create the embeddings object
embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002") # type: ignore

# Example
query_result = embeddings.embed_query("large language model")
len(query_result)
print("query_result: ", query_result)

#  Create the vector store object
pinecone.init(
    api_key="ef9a9434-5233-4b29-a794-355b106be8d7",
    environment="us-west4-gcp-free"
)

index_name = "techplaybook-dev"
print("index_name: ", index_name)

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
