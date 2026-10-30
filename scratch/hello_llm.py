import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()



# 1.同步方式调用
# llm = ChatOpenAI(
#     model="deepseek-chat",
#     base_url=os.environ["OPENAI_BASE_URL"],
#     api_key=os.environ["OPENAI_API_KEY"],
# )
# print(llm.invoke("用一句话解释什么是AI Agent").content)

# 2.异步方式调用

async def main():
    llm = ChatOpenAI(
        model="deepseek-chat",
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
    )

    resp = await llm.ainvoke("用一句话解释什么是AI Agent")
    print(resp.content)

if __name__ == "__main__":
    asyncio.run(main())