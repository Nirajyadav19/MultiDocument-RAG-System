from pdf_loader import extract_text_from_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

huggingface_token=os.getenv("HF_TOKEN")

embedding_model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={
        "token": huggingface_token
    }
    )

def text_to_chunk(pdf_path):
    """ This function for chunk creation """

    documents = extract_text_from_pdf(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)
    print(chunks)
    return chunks


def query_embedding(query):
    """ this function for query vector genration."""

    query_vector=embedding_model.embed_query(query)

    return query_vector

