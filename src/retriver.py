from src.vector_store import vector_store1,vector_store2
from sentence_transformers import CrossEncoder
import asyncio

reranker = CrossEncoder(
    "BAAI/bge-reranker-large"
)

async def retriver1(query:str):
    
    retriver=vector_store1.as_retriever(
         search_type="similarity",
        search_kwargs={"k": 7}
    )

    results=await retriver.invoke(query)

    pairs = [
        [query, doc.page_content]
        for doc in results
    ]

    scores =await reranker.predict(pairs)

    ranked_docs = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    final_docs = [
        doc
        for doc, score in ranked_docs[:3]
    ]

    page_contents = [
        doc.page_content 
        for doc in final_docs
    ]
    return page_contents

async def retriver2(query:str):
    
    retriver=vector_store2.as_retriever(
         search_type="similarity",
        search_kwargs={"k": 7}
    )

    results=await retriver.invoke(query)

    pairs = [
        [query, doc.page_content]
        for doc in results
    ]

    scores =await reranker.predict(pairs)

    ranked_docs = sorted(
        zip(results, scores),
        key=lambda x: x[1],
        reverse=True
    )

    final_docs = [
        doc
        for doc, score in ranked_docs[:3]
    ]

    page_contents = [
        doc.page_content 
        for doc in final_docs
    ]
    return page_contents