# importing libraries
import os
from dotenv import load_dotenv
import openai
import pinecone
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

# Loading environment variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
INDEX_NAME= os.getenv('INDEX_NAME')

# Clearing the Screen
os.system('clear')

# # Create the embeddings object
# # text-embedding-ada-002 is getting better values than ada
# embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002") # type: ignore

# Connect to  pinecone index 
pinecone.init(
    api_key= PINECONE_API_KEY, # type: ignore
    environment=PINECONE_ENV # type: ignore
)
index_name = INDEX_NAME
index = Pinecone(index_name=index_name) # type: ignore
print("index_name: ", index_name)

# Query
query = "Apples?"
# query = "Who is a database expert and what are they known for?"
# query = "Who is Ralph Kibmball and what are they known for?"
# query = "Who is Ralph Kibmball and what are they known for?  Do not compare to other people."  
# query = "What is a data warehouse? Answer in bullets.  Provide explaination for each bullet."  
# query = "What is a data warehouse? Answer in bullets.  Provide explaination for each bullet. Write answers like you are talking to very technical engineer"
# query = "What is a Bills house? Answer in bullets. If you don't have any context and are unsure of the answer, reply that you don't know about this topic and are always learning."
# query = "how many books did Bill Innmon write a book on Data Lakehous? If yes what was the name of the book? Whats was the publish date?"

# create the query vector
question = openai.Embedding.create(
    input=[query],
    model="text-embedding-ada-002"
)


