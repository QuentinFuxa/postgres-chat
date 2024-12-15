import logging
from fastapi import FastAPI
from pydantic import BaseModel
from postgres_chat import PostgresChat
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import os
logging.basicConfig(level=logging.INFO)
app = FastAPI()

class AskQuestion(BaseModel):
    question: str

CONNECTION_STRING = "postgresql://postgres@localhost:5432/imdb" # replace with your connection string

with open("api_key.txt", "r") as f:
    os.environ["OPENAI_API_KEY"] = f.read().strip()


# Instantiate the RAGHandler
rag_handler = PostgresChat(
    connection_string=CONNECTION_STRING,
    table_name='imdb',   # replace with your table you want to query
    schema='public',
    system_prompt_path='dd.txt'
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


@app.get("/reinitialize")
def reinitialize():
    """
    Reinitializes the RAGHandler object.
    """
    rag_handler.reinitialize_messages()
    return {"message": "RAGHandler reinitialized."}

@app.get("/show-system-prompt", response_class=PlainTextResponse)
def show_system_prompt():
    """
    Shows the system prompt to the user.
    """
    return rag_handler.system_prompt

if __name__ == '__main__':
    uvicorn.run("endpoints:app", host='localhost', port=4000, reload=True)