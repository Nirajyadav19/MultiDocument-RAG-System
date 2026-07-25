from langchain_chroma import Chroma
from embedding import embedding_model, text_to_chunk

pdf_path1="C:/Users/Downloads/rag (2)/rag/data/SQL-Manual.pdf"

pdf_path2="C:/Users/Downloads/rag (2)/rag/data/Niraj_Yadav_Resume.pdf"

chunks1 = text_to_chunk(pdf_path1)

chunks2 = text_to_chunk(pdf_path2)

Chroma.from_documents(
    documents=chunks1,
    embedding=embedding_model,
    collection_name="sql",
    persist_directory="D:/Niraj/Agentic ai/rag/vector_db"
)

Chroma.from_documents(
    documents=chunks2,
    collection_name="niraj",
    embedding=embedding_model,
    persist_directory="D:/Niraj/Agentic ai/rag/vector_db"
)

print("Vector DB created successfully!")