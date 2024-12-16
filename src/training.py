import os
from src.data_preprocessing import Preprocessor
from src.rag_model import RAGModel
from src.prediction_pipeline import QAPipeline


project_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_FILE = os.path.join(project_dir,"data","dialogs.txt")
print(project_dir)
print("="*40)
print(TEXT_FILE)

def run_preprocessing():
    # Initialize preprocessor and load text data
    print("Running preprocessing...")
    preprocessor = Preprocessor()
    documents = preprocessor.load_text(TEXT_FILE)
    embeddings = preprocessor.vectorize_text(documents)
    return embeddings, documents

def run_rag_model(embeddings, documents):
    # Initialize RAG model and build index
    print("Building RAG model...")
    rag_model = RAGModel()
    rag_model.build_index(embeddings, documents)
    return rag_model

def run_prediction(rag_model):
    # Initialize the prediction pipeline and make predictions
    print("Running prediction pipeline...")
    qa_pipeline = QAPipeline(text_file=TEXT_FILE)
    query = "What is the capital of France?"  # Sample query
    answer = qa_pipeline.predict(query)
    print(f"Answer: {answer}")
    return answer

def run_streamlit_app():
    # Run the Streamlit app (this is a separate command, so just call streamlit run here)
    import os
    os.system("streamlit run app.py")

if __name__ == "__main__":
    embeddings, documents = run_preprocessing()         # Step 1: Preprocessing
    rag_model = run_rag_model(embeddings, documents)    # Step 2: Build RAG Model
    run_prediction(rag_model)                           # Step 3: Run Prediction Pipeline
    run_streamlit_app()                                 # Step 4: Run Streamlit app
