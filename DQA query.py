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

# Create the vector store object
pinecone.init(
    api_key="ef9a9434-5233-4b29-a794-355b106be8d7",
    environment="us-west4-gcp-free"
)

index_name = "techplaybook-dev"
print("index_name: ", index_name)


# Cnnect to the Pinecone Index
index = pinecone.Index(index_name)
# Create the embeddings object
# Create the embeddings object
embeddings = OpenAIEmbeddings(model ="text-embedding-ada-002") # type: ignore

#Finding Similar Documents using Pinecone Index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

model_name = "text-davinci-003"
# model_name = "gpt-3.5-turbo"
llm = OpenAI(model=model_name) # type: ignore
#llm = openai.ChatCompletion.create(model=model_name)
chain = load_qa_chain(llm, chain_type="stuff")

# query = "Who is database expert and what are they known for?"
# query = "Who is Ralph Kibmball and what are they known for?"
# query = "Who is Ralph Kibmball and what are they known for?  Do not compare to other people."  
# query = "What is a data warehouse? Answer in bullets.  Provide explaination for each bullet."  
# query = "What is a data warehouse? Answer in bullets.  Provide explaination for each bullet. Write answers like you are talking to very technical engineer"
#query = "What is a data Lake? Answer in bullets. If you don't have any context and are unsure of the answer, reply that you don't know about this topic and are always learning."
query = "how many books did Bill Innmon write a book on Data Lakehous? If yes what was the name of the book? Whats was the publish date?"

def get_answer(query):
    similar_docs = docsearch.similarity_search(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

def get_similiar_docs(query, k=2, score=True):
    if score:
        similar_docs = docsearch.similarity_search_with_score(query, k=k)
    else:
        similar_docs = docsearch.similarity_search_with_score(query, k=k)
    return similar_docs


docs = docsearch.similarity_search(query)

# print(docs[0].page_content)

# found_docs = docsearch.max_marginal_relevance_search(query, k=2, fetch_k=10)
# for i, doc in enumerate(found_docs):
#     print(f"{i + 1}.", doc.page_content, "\n")

answer = get_answer(query)
print("query: ", query)
print(answer)





