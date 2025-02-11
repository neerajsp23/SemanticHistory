from fastapi import FastAPI, BackgroundTasks
from .rag_history import RAGHistory

app = FastAPI()

@app.post("/embed_async")
async def embed_async(text: str, background_tasks: BackgroundTasks):
    rag_history = RAGHistory()
    background_tasks.add_task(rag_history.embed_history())
    return {"message": "Embedding started in background"}

@app.post("/query")
async def query(text: str):
    rag_history = RAGHistory()
    response = rag_history.query_history_assistant(text)
    return {"response": response}
