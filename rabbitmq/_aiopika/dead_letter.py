"""
死信队列:超时 超长 拒绝
1:消息被消费端拒绝，使用 channel.basicNack 或 channel.basicReject ，并且此时requeue 属性被设置为false
2: 消息在队列的存活时间超过设置的TTL时间。
3:消息队列的消息数量已经超过最大队列长度，无法再继续新增消息到MQ中
4:一个队列中的消息的TTL对其他队列中同一条消息的TTL没有影响
"""
import json
from aiormq.abc import DeliveredMessage
import asyncio
import time

from aio_pika import DeliveryMode, ExchangeType, Message, connect
from multiprocessing import Process

HOST = "49.235.242.224"
PORT = 50009


async def init():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        normal_ex = await channel.declare_exchange("normal-ex", ExchangeType.DIRECT)
        arguments = {}
        arguments["x-dead-letter-exchange"] = "dead-ex"  # 死信交换机
        arguments["x-dead-letter-routing-key"] = ""  # 死信routing-key
        arguments["x-max-length"] = 100  # 死信routing-key
        normal_queue = await channel.declare_queue(name="normal-queue", arguments=arguments)
        await normal_queue.bind(normal_ex, routing_key="normal-queue")

        dead_ex = await channel.declare_exchange("dead-ex", ExchangeType.DIRECT)
        dead_queue = await channel.declare_queue(name="dead-queue")
        await dead_queue.bind(exchange=dead_ex, routing_key="")


async def producer():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel(publisher_confirms=True)
        normal_ex = await channel.declare_exchange("normal-ex", ExchangeType.DIRECT)
        data = {"id": 1, "status": "成功"}
        for i in range(110):
            data["index"] = i
            ack = await normal_ex.publish(
                Message(
                    json.dumps(data, ensure_ascii=False).encode("utf-8"),
                    delivery_mode=DeliveryMode.PERSISTENT,
                    expiration=1,#seconds
                ),
                routing_key="normal-queue",
                mandatory=True
            )
            if isinstance(ack, DeliveredMessage): print("message not send")


async def consumer_normal():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        arguments = {}
        arguments["x-dead-letter-exchange"] = "dead-ex"  # 死信交换机
        arguments["x-dead-letter-routing-key"] = ""  # 死信routing-key
        arguments["x-max-length"] = 100  # 死信routing-key
        normal_queue = await channel.declare_queue(name="normal-queue", arguments=arguments)

        async def callback(message) -> None:
            async with message.process():
                await asyncio.sleep(0.2)
                print(f"[NORMAL] Received message is: {json.loads(message.body)}")

        await normal_queue.consume(callback)

        print("[info] Waiting for logs. To exit press CTRL+C")
        await asyncio.Future()


async def consumer_dead():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        dead_queue = await channel.declare_queue(name="dead-queue")

        async def callback(message) -> None:
            async with message.process():
                await asyncio.sleep(0.2)
                #print(f"[DEAD] Received message is: {json.loads(message.body)}")

        await dead_queue.consume(callback)

        print("[info] Waiting for logs. To exit press CTRL+C")
        await asyncio.Future()


def run_producer():
    asyncio.run(producer())


def run_consumer_normal():
    asyncio.run(consumer_normal())


def run_consumer_dead():
    asyncio.run(consumer_dead())


def run():
    asyncio.run(init())
    time.sleep(1)
    p = Process(target=run_producer)
    cn = Process(target=run_consumer_normal)
    cd = Process(target=run_consumer_dead)
    p.start()
    cn.start()
    cd.start()



if __name__ == '__main__':
    run()
