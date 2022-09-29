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
        channel = await connection.channel()#publisher_confirms 交换机确认 默认True
        fanout_ex = await channel.declare_exchange("fanout-ex", ExchangeType.FANOUT)
        data = {"id": i, "status": "成功"}
        try:
            await fanout_ex.publish(
                Message(
                    json.dumps(data, ensure_ascii=False).encode("utf-8"),
                    delivery_mode=DeliveryMode.PERSISTENT,
                ),
                routing_key=""
            )
        except Exception as e:
            print(f'Message could not be confirmed{str(e)}')


async def consumer1():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        fanout_ex = await channel.declare_exchange("fanout-ex", ExchangeType.FANOUT)
        queue = await channel.declare_queue("fanout-1")
        await queue.bind(fanout_ex)
        async def callback(message) -> None:
            async with message.process():
                await asyncio.sleep(0.2)
                print(f"[C1] Received message is: {json.loads(message.body)}")

        await queue.consume(callback)

        print("[C1] Waiting for logs. To exit press CTRL+C")
        await asyncio.Future()


async def consumer2():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        fanout_ex = await channel.declare_exchange("fanout-ex", ExchangeType.FANOUT)
        queue = await channel.declare_queue("fanout-2")
        await queue.bind(fanout_ex)
        async def callback(message) -> None:
            async with message.process():
                await asyncio.sleep(0.2)
                print(f"[C2] Received message is: {json.loads(message.body)}")

        await queue.consume(callback)

        print("[C2] Waiting for logs. To exit press CTRL+C")
        await asyncio.Future()


def run_producer():
    for i in range(10):asyncio.run(producer(i))


def run_consumer_1():
    asyncio.run(consumer1())


def run_consumer_2():
    asyncio.run(consumer2())


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
