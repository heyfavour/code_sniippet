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
        lazy_ex = await channel.declare_exchange("lazy-ex", ExchangeType.DIRECT)

        arguments = {"x-queue-mode":"lazy"}
        queue = await channel.declare_queue(name="lazy-queue",arguments=arguments)
        await queue.bind(exchange=lazy_ex,routing_key="")


async def producer():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel(publisher_confirms=False)  # True aiormq.exceptions.ChannelNotFoundEntity
        ex = await channel.get_exchange("lazy-ex",ensure=True)  # passive=True aiormq.exceptions.ChannelNotFoundEntity
        data = {"id": 1, "status": "成功"}
        for i in range(10000):
            ack = await ex.publish(
                Message(
                    json.dumps(data, ensure_ascii=False).encode("utf-8"),
                    delivery_mode=DeliveryMode.PERSISTENT,
                ),
                routing_key="",
                mandatory=False,
            )



def run():
    asyncio.run(init())
    asyncio.run(producer())


if __name__ == '__main__':
    run()
