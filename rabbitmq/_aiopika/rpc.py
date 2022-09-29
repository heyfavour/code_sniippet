import json
import os
import asyncio

from aio_pika import DeliveryMode, ExchangeType, Message, connect_robust
from aio_pika.abc import AbstractIncomingMessage
from multiprocessing import Process

import uuid

HOST = "49.235.242.224"
PORT = 50009


async def client(i):
    data = {"id": i, "status": "成功"}

    connection = await connect_robust(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    channel = await connection.channel()  # publisher_confirms 交换机确认 默认True
    rpc_ex = await channel.declare_exchange("rpc-ex", ExchangeType.DIRECT)
    rpc_queue = await channel.declare_queue(exclusive=True)
    await rpc_queue.bind(rpc_ex,routing_key="rpc")

    def callback(message: AbstractIncomingMessage) -> None:
        if message.correlation_id !=correlation_id:
            print(f"[C{i}]Bad message {message.body}")
            return
        print(f"[C{i}] Received message is: {json.loads(message.body)}")

    await rpc_queue.consume(callback)
    correlation_id = str(uuid.uuid4())

    await rpc_ex.publish(
        Message(
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            correlation_id=correlation_id,
            reply_to=rpc_queue.name,
        ),
        routing_key="rpc",
    )
    await channel.close()


async def server():
    connection = await connect_robust(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    rpc_ex = await channel.declare_exchange("rpc-ex", ExchangeType.DIRECT)
    queue = await channel.declare_queue("rpc_queue")
    await queue.bind(rpc_ex,routing_key="rpc")

    print("[S] Awaiting RPC requests")

    async with queue.iterator() as qiterator:
        message: AbstractIncomingMessage
        async for message in qiterator:
            async with message.process(requeue=False):
                assert message.reply_to is not None
                data = json.loads(message.body)
                data["id"] = -1*data["id"]
                await rpc_ex.publish(
                    Message(
                        json.dumps(data, ensure_ascii=False).encode("utf-8"),
                        delivery_mode=DeliveryMode.PERSISTENT,
                        correlation_id=message.correlation_id,
                    ),
                    routing_key=message.reply_to,
                )
def run_client():
    for i in range(10):asyncio.run(client(i))

def run_server():
    asyncio.run(server())

def run():
    s = Process(target=run_server)
    c = Process(target=run_client)
    s.start()
    c.start()

if __name__ == '__main__':
    run()

