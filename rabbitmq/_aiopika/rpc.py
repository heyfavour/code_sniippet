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
    loop = asyncio.get_running_loop()
    connection = await connect_robust(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    channel = await connection.channel()  # publisher_confirms 交换机确认 默认True
    rpc_ex = await channel.declare_exchange("rpc-ex", ExchangeType.DIRECT)
    callback_queue = await channel.declare_queue(exclusive=True)

    await callback_queue.bind(rpc_ex, routing_key=callback_queue.name)
    future = loop.create_future()

    def callback(message: AbstractIncomingMessage) -> None:
        if message.correlation_id != correlation_id: raise Exception(f"[C{i}]Bad message {message.body}")
        print(f"[C{i}] Received message is: {json.loads(message.body)}")
        future.set_result(message.body)

    await callback_queue.consume(callback)
    correlation_id = str(uuid.uuid1())
    await rpc_ex.publish(
        Message(
            json.dumps(data, ensure_ascii=False).encode("utf-8"),
            correlation_id=correlation_id,
            reply_to=callback_queue.name,
        ),
        routing_key="rpc",
    )
    await future
    await channel.close()
    print(f"[C{i}] END RPC REQUEST")


async def server():
    connection = await connect_robust(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    rpc_ex = await channel.declare_exchange("rpc-ex", ExchangeType.DIRECT)
    queue = await channel.declare_queue("rpc_queue")
    await queue.bind(rpc_ex, routing_key="rpc")

    print("[S] Awaiting RPC requests")

    async with queue.iterator() as qiterator:
        message: AbstractIncomingMessage
        async for message in qiterator:
            async with message.process(requeue=False):
                assert message.reply_to is not None
                data = json.loads(message.body)
                print(f"[S] revice data {message.reply_to} {data}")
                await asyncio.sleep(0.1)
                data["id"] = -1 * data["id"]
                await rpc_ex.publish(
                    Message(
                        json.dumps(data, ensure_ascii=False).encode("utf-8"),
                        delivery_mode=DeliveryMode.PERSISTENT,
                        correlation_id=message.correlation_id,
                    ),
                    routing_key=message.reply_to,
                )


async def gather_client():
    tasks = [client(i) for i in range(10)]
    await asyncio.gather(*tasks)


def run_client():
    asyncio.run(gather_client())


def run_server():
    asyncio.run(server())


def run():
    s = Process(target=run_server)
    c = Process(target=run_client)
    s.start()
    c.start()


if __name__ == '__main__':
    run()
