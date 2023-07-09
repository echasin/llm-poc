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

# Loading environment variables
load_dotenv()
OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
INDEX_NAME= os.getenv('INDEX_NAME')

# Clearing the Screen
os.system('clear')

# Create the embeddings object
# text-embedding-ada-002 is getting better values than ada
embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002") # type: ignore

# Connect to  pinecone index 
pinecone.init(
    api_key= PINECONE_API_KEY, # type: ignore
    environment=PINECONE_ENV # type: ignore
)
index_name = INDEX_NAME
print("index_name: ", index_name)

index = Pinecone.from_existing_index(index_name, embeddings) # type: ignore

# Gives out 4 similar documents by doing semantic search of vector database 
def get_similiar_docs(query, k=4, score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = index.similarity_search(query, k=k)
  return similar_docs

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(OpenAI(model ="text-davinci-003", temperature = 0), index.as_retriever(), memory = memory) # type: ignore


# chainlit 
import chainlit as cl
from chainlit import langchain_factory
from chainlit import AskUserMessage, Message, on_chat_start
from chainlit import on_message
from chainlit import user_session

@langchain_factory(use_async=True)
def model():
   qa = ConversationalRetrievalChain.from_llm(OpenAI(model ="text-davinci-003", temperature = 0), index.as_retriever(), memory = memory) # type: ignore
   return qa

@on_chat_start
async def main():
    await Message( content= 'Hello! How can I help you?').send()
