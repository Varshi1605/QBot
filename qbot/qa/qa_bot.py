# qa_retrieve.py

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import chromadb
from chromadb.utils import embedding_functions

# Configuration
QA_MODEL_NAME = "bert-large-uncased-whole-word-masking-finetuned-squad"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_path"
COLLECTION_NAME = "knowledge_base"

# Initialize embedding function and ChromaDB client
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL,
    device="cpu"
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_func
)

# Initialize QA pipeline
tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
qa_pipeline = pipeline(
    "question-answering",
    model=model,
    tokenizer=tokenizer,
    device="cpu"
)

#summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device="cpu")
summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    tokenizer="google/flan-t5-large",
    device="cpu"
)

# QA function
def retrieve_and_answer(question: str, top_k: int = 3):
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )

    print("\nRetrieved Contexts:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {meta['title']} (ID: {meta['id']})")
        print(f"Content: {doc[:200]}...")

    combined_context = "\n\n---\n\n".join([
        f"Source: {meta['title']}\nContent: {doc}"
        for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    ])

    try:
        answer = qa_pipeline(
            question=question,
            context=combined_context,
            max_answer_len=100,
            handle_impossible_answer=False
        )
    except Exception as e:
        print(f"QA Error: {e}")
        return {
            "answer": "Sorry, I couldn't process that question.",
            "confidence": 0,
            "sources": []
        }

    return {
        "answer": answer['answer'],
        "confidence": answer['score'],
        "sources": [
            {
                "content": doc,
                "source": meta['title'],
                "source_id": meta['id']
            }
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
    }

# Example usage
if __name__ == "__main__":
    question = "What is rtos?"
    print(f"\nQuestion: {question}")
    response = retrieve_and_answer(question)

    print("\nFinal Answer:")
    print(f"Answer: {response['answer']}")
    print(f"Confidence: {response['confidence']:.4f}")
    print("Sources:")
    for src in response['sources']:
        print(f"- {src['source']} (ID: {src['source_id']})")

    def summarize_question_answer_style(question: str, response: dict) -> str:
        if not response["sources"]:
            return "No relevant information found to summarize."

        # Prepare the context
        context = "\n\n".join([src['content'] for src in response['sources']])

        # Prompt format
        prompt = f"Answer the question based on the context.\n\nQuestion: {question}\n\nContext: {context}"

        # Generate summary/answer
        output = summarizer(prompt, max_length=200, do_sample=False)[0]['generated_text']
        return output

    answer = summarize_question_answer_style(question, response)

    print("\nFinal Answer (Summary):")
    print(answer)
    # Summarize retrieved chunks from response
    # if response["sources"]:
    #     retrieved_text = "\n\n".join([src['content'] for src in response['sources']])
        
    #     summary = summarizer(
    #         retrieved_text,
    #         max_length=250,
    #         min_length=50,
    #         do_sample=False
    #     )[0]['summary_text']

    #     print("\nSummarised Answer:")
    #     print(summary)
    # else:
    #     print("\nNo sources found to summarize.")
