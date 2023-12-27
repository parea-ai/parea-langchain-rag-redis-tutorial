from dotenv import load_dotenv
from fastapi import FastAPI

from rag.chain import handler

load_dotenv()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/rag-redis")
async def chain() -> str:
    response = chain.invoke(
        "What was Nike's revenue in 2023?", config={"callbacks": [handler]}
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", reload=True)
