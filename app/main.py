import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from app.rag.graph import rag_graph

load_dotenv()
app = FastAPI()
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
    # temperature=0,
)


class ChatRequest(BaseModel):
    q: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str


class AskRequest(BaseModel):
    q: str = Field(min_length=1, max_length=2000)


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []
    usage: dict = {}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    msg = await llm.ainvoke(req.q)
    return ChatResponse(answer=msg.text)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    out = await rag_graph.ainvoke({"question": req.q, "strategy": "struct"})
    docs = out.get("docs") or []
    return AskResponse(
        answer=out.get("answer", ""),
        sources=[f"{d['source']}#{d['chunk_index']}" for d in docs],
        usage=dict(out.get("usage") or {}),
    )
