import faiss
from transformers import pipeline

class RAGModel:
    def __init__(self, embedding_dim=384):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.documents = []
        self.qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

    def build_index(self, embeddings, documents):
        """Build FAISS index with embeddings."""
        self.index.add(embeddings)
        self.documents = documents

    def retrieve(self, query_embedding, top_k=3):
        """Retrieve top-k relevant documents."""
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]

    def answer_query(self, query, embedding_model):
        """Generate answer for a given query."""
        query_embedding = embedding_model.encode([query])
        relevant_docs = self.retrieve(query_embedding)
        context = " ".join(relevant_docs)
        return self.qa_pipeline(f"question: {query} context: {context}")[0]['generated_text']
