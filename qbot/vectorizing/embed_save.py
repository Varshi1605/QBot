# embed_and_save.py

import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_path"
COLLECTION_NAME = "knowledge_base"

# Initialize embedding function
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    device="cpu"
)

# Load data
with open("cleaned_conf_data.json", "r") as f1, open("cleaned_jira_data.json", "r") as f2:
    #final_data = json.load(f1) + json.load(f2)
    final_data = json.load(f1)

# Chunking function
def chunk_content(content, max_length=500):
    sentences = content.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Process documents
documents = []
metadatas = []
ids = []

for entry in final_data:
    chunks = chunk_content(entry['content'])
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "id": entry['id'],
            "title": entry['title'],
            "chunk_num": i,
            "source": "conf" if "conf" in entry.get('source', '') else "jira"
        })
        ids.append(f"{entry['id']}_{i}")

# Save to ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func
)

if collection.count() == 0:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Saved {len(documents)} chunks to ChromaDB at '{CHROMA_PATH}'")
else:
    print("ChromaDB already has data. Skipping add.")
