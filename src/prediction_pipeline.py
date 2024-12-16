from src.data_preprocessing import Preprocessor
from src.rag_model import RAGModel

class QAPipeline:
    def __init__(self, text_file, model_name='all-MiniLM-L6-v2'):
        self.preprocessor = Preprocessor(model_name)
        self.rag_model = RAGModel()

        # Load and process data
        self.documents = self.preprocessor.load_text(text_file)
        embeddings = self.preprocessor.vectorize_text(self.documents)
        self.rag_model.build_index(embeddings, self.documents)

    def predict(self, query):
        """Answer a question using the RAG pipeline."""
        return self.rag_model.answer_query(query, self.preprocessor.model)
