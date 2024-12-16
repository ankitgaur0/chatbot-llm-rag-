import os
import streamlit as st
from src.data_preprocessing import Preprocessor
from src.rag_model import RAGModel
from src.prediction_pipeline import QAPipeline


project_dir=os.path.dirname(os.path.abspath(__file__))
TEXT_PATH=os.path.join(project_dir,"data","dialogs.txt")
def main():
    st.title("Question Answering Chatbot with RAG")

    query = st.text_input("Enter your question:")

    if query:
        preprocessor = Preprocessor()
        documents = preprocessor.load_text(TEXT_PATH)
        embeddings = preprocessor.vectorize_text(documents)

        rag_model = RAGModel()
        rag_model.build_index(embeddings, documents)

        qa_pipeline = QAPipeline(rag_model)
        answer = qa_pipeline.predict(query)
        
        st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
