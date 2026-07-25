from langchain_chroma import Chroma
from embedding import embedding_model

embeddings = embedding_model

vector_store1 = Chroma(
    persist_directory="D:/Niraj/rag/vector_db",
    collection_name="sql",
    embedding_function=embeddings
)

vector_store2 = Chroma(
    persist_directory="D:/Niraj/rag/vector_db",
    collection_name="machine_learning",
    embedding_function=embeddings
)