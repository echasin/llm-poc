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
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
INDEX_NAME= os.getenv('INDEX_NAME')

# Clearing the Screen
os.system('clear')

# Connect to existing Pinecone index 
pinecone.init(
    api_key= PINECONE_API_KEY, # type: ignore
    environment=PINECONE_ENV # type: ignore
)
index_name = INDEX_NAME
index = pinecone.Index(index_name) # type: ignore
print("index_name: ", index_name)
print(index.describe_index_stats())

# Query
# query = " Eric Chasin "
# query = "Who is a database expert and what are they known for?"
query = "Who is Bill Inmon and what are they known for?"
# query = "Who is Ralph Kibmball and what are they known for?  Do not compare to other people."  
# query = "What is a data warehouse? Answer in bullets.  Provide explaination for each bullet."  
# query = "What is a data warehouse? Answer in bullets.  Provide explaination for each bullet. Write answers like you are talking to very technical engineer"
# query = "What is a Bills house? Answer in bullets. If you don't have any context and are unsure of the answer, reply that you don't know about this topic and are always learning."
# query = "how many books did Bill Innmon write a book on Data Lakehous? If yes what was the name of the book? Whats was the publish date?"

model_name = 'text-embedding-ada-002'

embeddings = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
) # type: ignore

docsearch = Pinecone.from_existing_index(index_name, embeddings) # type: ignore

similar_docs = docsearch.similarity_search_with_relevance_scores(query, k=100) 
# print("similar_docs: ", similar_docs)

# Assuming similar_docs contains the list of similar documents
threshold = 0.75
filtered_docs = [doc for doc in similar_docs if doc[1] > threshold]

for doc in filtered_docs:
     page_content = doc[0].page_content[:50] 
     metadata = doc[0].metadata # type: ignore
     score = doc[1] # type: ignore
     print(f"Page Content: {page_content}, Metadata: {metadata}, Score: {score}")
  
