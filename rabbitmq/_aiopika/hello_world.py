#!/usr/bin/env python
# pip install pika
import json
import asyncio
import aio_pika
from aio_pika.abc import AbstractIncomingMessage
from aio_pika import DeliveryMode

"""
    producer->channel->exchange->queue->channel->consumer
    49.235.242.224
"""
HOST = "49.235.242.224"
PORT = 50009


async def init():
    connection = await aio_pika.connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    # connection  = await aio_pika.connect(url="amqp://admin:admin@49.235.242.224:50009/")#->Connection
    # connection  = await aio_pika.connect_robust(host=HOST, port=PORT, login="admin",password="admin")#->RobustConnection -> reconnect
    async with connection:  # ->await self.close()
        channel = await connection.channel()
        queue = await channel.declare_queue("hello-world", durable=True)


async def producer():
    connection = await aio_pika.connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection

    async with connection:
        channel = await connection.channel()
        data = {"id": 1, "status": "成功"}
        await channel.default_exchange.publish(
            aio_pika.Message(
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key="hello-world",
        )
        print("[x] Sent 'Hello World!'")


async def consumer():  # auto_ack
    connection = await aio_pika.connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        queue = await channel.declare_queue("hello-world", durable=True)

        async def callback(message: AbstractIncomingMessage) -> None:
            try:
                print(f"[x] Received message is: {json.loads(message.body)}")
            except:
                print(f"[x] Received message is: {message.body}")


        await queue.consume(callback, no_ack=True)

        print("[*] Waiting for messages. To exit press CTRL+C")
        await asyncio.Future()


async def run():
    await init()
    await producer()
    await consumer()


if __name__ == '__main__':
    asyncio.run(run())
