import datetime
import json
import os
import asyncio
from aiormq.abc import DeliveredMessage
from aio_pika import DeliveryMode, ExchangeType, Message, connect
from multiprocessing import Process

HOST = "49.235.242.224"
PORT = 50009


async def init():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        priority_ex = await channel.declare_exchange("priority-ex", ExchangeType.DIRECT)

        arguments = {'x-max-priority': 5}  # 0-255越大越优先 不建议太大会因为排序浪费性能
        queue = await channel.declare_queue(name="priority-queue", arguments=arguments)
        await queue.bind(exchange=priority_ex, routing_key="")


async def producer(i):
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel(publisher_confirms=False)  # True aiormq.exceptions.ChannelNotFoundEntity
        ex = await channel.get_exchange("priority-ex",ensure=True)  # passive=True aiormq.exceptions.ChannelNotFoundEntity
        data = {"id": i, "status": "成功", "priority": i % 5}
        ack = await ex.publish(
            Message(
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=data['priority'],
            ),
            routing_key="",
            mandatory=True,
        )


async def consumer():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        queue = await channel.get_queue("priority-queue")

        async def callback(message) -> None:
            async with message.process():
                await asyncio.sleep(0.2)
                print(f"[info] Received message is: {json.loads(message.body)}")

        await queue.consume(callback)

        print("[info] Waiting for logs. To exit press CTRL+C")
        await asyncio.Future()


def run_producer():
    for i in range(100): asyncio.run(producer(i))


def run_consumer():
    asyncio.run(consumer())


def run():
    asyncio.run(init())
    c = Process(target=run_consumer)
    p = Process(target=run_producer)
    c.start()
    p.start()


if __name__ == '__main__':
    run()
