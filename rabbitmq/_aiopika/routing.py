import datetime
import json
import os
import asyncio
import time

from aio_pika import DeliveryMode, ExchangeType, Message, connect
from multiprocessing import Process

HOST = "49.235.242.224"
PORT = 50009


async def producer(i):
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()  # publisher_confirms 交换机确认 默认True
        direct_ex = await channel.declare_exchange("direct-ex", ExchangeType.DIRECT)
        level_dict = {0: "info", 1: "warning", 2: "error"}
        data = {"id": i, "status": "成功", "level": level_dict[i % 3]}
        try:
            await direct_ex.publish(
                Message(
                    json.dumps(data, ensure_ascii=False).encode("utf-8"),
                    delivery_mode=DeliveryMode.PERSISTENT,
                ),
                routing_key=data["level"]
            )
        except Exception as e:
            print(f'Message could not be confirmed{str(e)}')


async def consumer_info():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        direct_ex = await channel.declare_exchange("direct-ex", ExchangeType.DIRECT)
        queue = await channel.declare_queue("info")
        await queue.bind(direct_ex, routing_key="info")
        await queue.bind(direct_ex, routing_key="warning")
        await queue.bind(direct_ex, routing_key="error")

        async def callback(message) -> None:
            async with message.process():
                await asyncio.sleep(0.2)
                print(f"[info] Received message is: {json.loads(message.body)}")

        await queue.consume(callback)

        print("[info] Waiting for logs. To exit press CTRL+C")
        await asyncio.Future()


async def consumer_error():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        direct_ex = await channel.declare_exchange("direct-ex", ExchangeType.DIRECT)
        queue = await channel.declare_queue("error")
        await queue.bind(direct_ex, routing_key="error")

        async def callback(message) -> None:
            async with message.process():
                await asyncio.sleep(0.2)
                print(f"[error] Received message is: {json.loads(message.body)}")

        await queue.consume(callback)

        print("[error] Waiting for logs. To exit press CTRL+C")
        await asyncio.Future()


def run_producer():
    for i in range(10): asyncio.run(producer(i))


def run_consumer_1():
    asyncio.run(consumer_info())


def run_consumer_2():
    asyncio.run(consumer_error())


def run():
    p = Process(target=run_producer)
    c1 = Process(target=run_consumer_1)
    c2 = Process(target=run_consumer_2)

    c1.start()
    c2.start()
    time.sleep(1)
    p.start()


if __name__ == '__main__':
    run()
