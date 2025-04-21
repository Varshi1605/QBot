from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from langchain.chains import RetrievalQA
import shutil
import json
import sys
import os
from openai import OpenAI

# Load data
with open("cleaned_conf_data.json", "r") as f1, open("cleaned_jira_data.json", "r") as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)
final_data=data1+data2

# Text splitter for large content
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=75
)

# Process data into chunks
documents = []
metadata = []
for entry in final_data:
    chunks = text_splitter.split_text(entry['content'])
    documents.extend(chunks)
    metadata.extend([{"id": entry['id'], "title": entry['title']}] * len(chunks))

print("length of metadata is ",len(metadata))

# Embedding model
model_kwargs = {'device':'cpu'}
encode_kwargs = {'normalize_embeddings': False}
model_name="sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name,model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

if os.path.exists("vectorizing/chroma_path"):
    shutil.rmtree("vectorizing/chroma_path")

# Initialize and populate ChromaDB
vectorstore = Chroma.from_texts(documents, embeddings, metadatas=metadata, persist_directory="vectorizing/chroma_path")

#Initializing MobileBERT and tokenizer
model_name="google/mobilebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name) - Looks like RetrievalQA requires a runnable model and not raw model
#model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Wrap in HuggingFacePipeline
qa_pipeline = pipeline("question-answering", model=model_name, tokenizer=tokenizer)
hf_pipeline = HuggingFacePipeline(pipeline=qa_pipeline)

# RAG Pipeline with LangChain
template_ = "Answer the following question based on the provided context: {context}\nQuestion: {question}" 
prompt_= PromptTemplate(template=template_,input_variables=["context", "question"])



"""

OpenAi based Retrival QA - But works only if you pay :(

"""
api_key=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def retrieve_and_answer(query):
     results = vectorstore.similarity_search(query, k=5)
     context = "\n".join([doc.page_content for doc in results])
     return context

def get_openai_response(question,context):
    prompt= f"Context : {context} \n\nQuestion: {question} \n Answer: "
    response= client.chat.completions.create(model="gpt-4o-search-preview",messages=[{"role":"system",
                                                                        "content":"Answer based on the context provided"},
                                                                        {"role":"user", "content":prompt}])
    return response.choices[0].messsage.content

query ="How to onboard instruments"
context = retrieve_and_answer(query)
answer= get_openai_response(query,context) # This method works but its not free :(

print(answer)

breakpoint()


# RAG pipeline with LangChain
qa_chain = RetrievalQA.from_chain_type(
    llm=hf_pipeline, 
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type="stuff",  # Combines multiple chunks for longer answers
    chain_type_kwargs = {"prompt": prompt_}
)

def retrieve_and_answer(query):
     results = vectorstore.similarity_search(query, k=5)
     context = "\n".join([doc.page_content for doc in results])
     return qa_chain.invoke({"context":context, "question":query})

# Example Usage
query = "How do I onboard new instruments?"
#response = retrieve_and_answer(query)
#response=qa_chain.invoke({"query":query})
