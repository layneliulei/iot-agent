import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
app = FastAPI()
llm = ChatOpenAI(
    model="deepseek-chat",
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

class ChatRequest(BaseModel):
    q: str = Field(min_length=1,max_length=2000)

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
async def health():
    return {"status":"ok"}

@app.post("/chat",response_model=ChatResponse)
async def chat(req: ChatRequest):
    msg = await llm.ainvoke(req.q)
    return ChatResponse(answer=msg.content)