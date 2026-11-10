# iot-agent

AI Agent 应用练习项目：FastAPI 服务 + 工具调用 + RAG 检索。

## 跑起来
python -m pytest -q          # 3 passed
uvicorn app.main:app --reload

## 容器
docker build -t iot-agent .
docker run -p 8000:8000 --env-file .env iot-agent
curl.exe http://127.0.0.1:8000/health

## Agent 状态图
python -m app.graph          # 计算器工具 + 设备状态工具（mock）
# 图结构：START → llm_call →(条件边)→ tool_node → llm_call → END

## 下一步
LangGraph 状态图 / 工具工程化 / RAG + 评测