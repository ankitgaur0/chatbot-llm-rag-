import os
from sentence_transformers import SentenceTransformer

class Preprocessor:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def load_text(self, file_path):
        """Load text data from a file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()

    def vectorize_text(self, documents):
        """Transform text documents into embeddings."""
        return self.model.encode(documents, show_progress_bar=True)
