from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from src.retriver import retriver1,retriver2
import time
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")

class query_model(BaseModel):
    question: str

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
async def retrive_ans(query: query_model):
    time1=time.time()
    classification_prompt_text=classification_template.format(
    question=query.question
)
    result =await llm.ainvoke(classification_prompt_text)
    category = result.content.strip().lower()
    print(category)

    retrievers = {
    "sql": retriver1,
    "general": retriver2,
    }

    retriever = retrievers.get(category)
    print(retriever)
    if retriever is None:
        return {
            "error": f"Unknown category returned by classifier: {category}"
        }

    context = retriever(query.question)
    print(context)
    """ if category == "sql":
        context = retriver1(query.question)
        print(context)
        formatted_prompt = prompt_template.format(
            context=context,
            question=query.question
        ) """
    """ else:
        context = retriver2(query.question)
        print(context) """
    formatted_prompt = prompt_template.format(
        context=context,
        question=query.question
    )
    response = await llm.ainvoke(formatted_prompt)
    return response.content
    