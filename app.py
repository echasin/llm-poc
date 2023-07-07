import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

#loading environment variables
load_dotenv()
OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')

#loading data
directory = 'Data'
def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents
documents = load_docs(directory)
#print(len(documents))


def split_docs(documents, chunk_size=1500, chunk_overlap=75):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs
docs = split_docs(documents)
# print(len(docs))


embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002")
# text-embedding-ada-002 is getting better values than ada

# creating pinecone index 
pinecone.init(
    api_key= PINECONE_API_KEY,
    environment=PINECONE_ENV
)
index_name = "llmchatbot"
index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

#gives out 4 similar documents by doing semantic search of vector database 
def get_similiar_docs(query, k=4, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(OpenAI(model ="text-davinci-003", temperature = 0), index.as_retriever(), memory = memory)


#chainlit 
import chainlit as cl
from chainlit import langchain_factory
from chainlit import AskUserMessage, Message, on_chat_start
from chainlit import on_message
from chainlit import user_session

@langchain_factory(use_async=True)
def model():
   qa = ConversationalRetrievalChain.from_llm(OpenAI(model ="text-davinci-003", temperature = 0), index.as_retriever(), memory = memory)
   return qa

@on_chat_start
async def main():
    await Message( content= 'Hello! How can I help you?').send()
