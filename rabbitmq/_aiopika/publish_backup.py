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
        confirm_ex = await channel.declare_exchange("confirm-ex", ExchangeType.DIRECT)
        confirm_queue = await channel.declare_queue(name="confirm-queue")
        await confirm_queue.bind(exchange=confirm_ex, routing_key="")


async def init_backup():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        backup_ex = await channel.declare_exchange("backup-ex", ExchangeType.FANOUT)
        backup_queue = await channel.declare_queue(name="backup-queue")
        await backup_queue.bind(exchange=backup_ex, routing_key="")
        # 备份交换机
        arguments = {'alternate-exchange': 'backup-ex'}
        confirm_ex = await channel.declare_exchange("confirm-ex", ExchangeType.DIRECT, arguments=arguments)
        confirm_queue = await channel.declare_queue(name="confirm-queue")
        await confirm_queue.bind(exchange=confirm_ex, routing_key="")


async def producer_ex():  # 错误的ex
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel(publisher_confirms=False)  # True aiormq.exceptions.ChannelNotFoundEntity
        ex = await channel.get_exchange("fake-ex",ensure=False)  # passive=True aiormq.exceptions.ChannelNotFoundEntity
        data = {"id": 1, "status": "成功"}
        ack = await ex.publish(
            Message(
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key="",
        )
        if isinstance(ack, DeliveredMessage): print("[EX] message not send")

async def producer_queue():#错误的queue
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel(publisher_confirms=True)  # True aiormq.exceptions.ChannelNotFoundEntity
        ex = await channel.get_exchange("confirm-ex", ensure=True)  # passive=True aiormq.exceptions.ChannelNotFoundEntity
        data = {"id": 1, "status": "成功"}
        ack = await ex.publish(
            Message(
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key="fake-key",
        )
        if isinstance(ack, DeliveredMessage): print("[QUEUE] message not send")
#
#
async def producer_backup():#错误的ex->backup
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel(publisher_confirms=True)  # True aiormq.exceptions.ChannelNotFoundEntity
        ex = await channel.get_exchange("confirm-ex", ensure=True)  # passive=True aiormq.exceptions.ChannelNotFoundEntity
        data = {"id": 1, "status": "成功"}
        ack = await ex.publish(
            Message(
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key="fake-key",
            mandatory=True,
        )
        #此处不会报错 因为已经被备份交换机处理
        if isinstance(ack, DeliveredMessage): print("[BACKUP] message not send")


def no_backup():
    asyncio.run(init())
    #asyncio.run(producer_ex())  # 不存在的ex
    asyncio.run(producer_queue())  # 不存在的queue


def backup():
    asyncio.run(init_backup())
    asyncio.run(producer_backup())  # 不存在的ex


def run():
    no_backup()
    backup()


if __name__ == '__main__':
    run()
