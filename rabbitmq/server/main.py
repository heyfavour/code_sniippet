from fastapi import FastAPI
import asyncio
from fastapi import APIRouter

from rabbitmq import AsyncioRabbitMQ

app = FastAPI()
router = APIRouter

ep = None


@app.on_event("startup")
async def startup() -> None:
    loop = asyncio.get_running_loop()
    task = loop.create_task(app.pika_client.consume(loop))
    await task


@app.router("/pika")
async def test_pika(msg) -> None:
    ep.publish_message(msg)
    return {"code": 200}


app.include_router(router, prefix="")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app='main:app', host="0.0.0.0", port=8080)
