from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from src.retriver import retriver1,retriver2
import time
from dotenv import load_dotenv
import os
import yaml
import asyncio

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

class query_model(BaseModel):
    question: str

class bulkquestion_model(BaseModel):
    bulkrequest:list[query_model]

app=FastAPI()

with open("./config/prompt.yaml","r") as file:
    data=yaml.safe_load(file)

prompt=data["system_prompt"]

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=api_key
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt
)

classification_prompt=data["classification_prompt"]

classification_template = PromptTemplate(
    input_variables=["question"],
    template=classification_prompt
)

@app.post("/query")
async def retrieve_ans(query: bulkquestion_model):

    async def process_question(item):

        question = item.question

        # 1. Classification
        classification_prompt_text = classification_template.format(
            question=question
        )

        result = await llm.ainvoke(
            classification_prompt_text
        )

        category = result.content.strip().lower()

        # 2. Select retriever
        retrievers = {
            "sql": retriver1,
            "general": retriver2,
        }

        retriever = retrievers.get(category)

        if retriever is None:
            return {
                "question": question,
                "error": f"Unknown category: {category}"
            }

        # 3. Retrieve context
        context = retriever(question)

        # 4. Generate prompt
        formatted_prompt = prompt_template.format(
            context=context,
            question=question
        )

        # 5. Generate answer
        response = await llm.ainvoke(
            formatted_prompt
        )

        return {
            "question": question,
            "answer": response.content
        }
    
    results = await asyncio.gather(
        *[
            process_question(q)
            for q in query.bulkrequest
        ]
    )

    return {
        "results": results
    }