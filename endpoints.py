import logging
from fastapi import FastAPI
from pydantic import BaseModel
from rag_handler import RAGHandler
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.DEBUG)
app = FastAPI()

class AskQuestion(BaseModel):
    question: str

CONNECTION_STRING = "postgresql://postgres@localhost:5432/imdb" # replace with your connection string
OPENAI_API_KEY = ""

# Instantiate the RAGHandler
rag_handler = RAGHandler(
    connection_string=CONNECTION_STRING,
    openai_api_key=OPENAI_API_KEY,
    table_name='imdb',   # replace with your table you want to query
    schema='public'
)

@app.post("/ask-question")
def ask_question(question: AskQuestion):
    """
    Endpoint to handle user questions. Uses an LLM to decide
    whether to perform database queries or respond directly.
    """
    rag_handler.add_user_message(question.question)

    result = rag_handler.run_conversation()

    return {
        "response": result["response"],
        "executed_queries": result["executed_queries"],
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


if __name__ == '__main__':
    uvicorn.run("endpoints:app", host='localhost', port=4000, reload=True)