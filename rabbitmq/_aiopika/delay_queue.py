"""
1.TTL+死信队列
过于麻烦
2.pluguin
rabbitmq-plugins enable rabbitmq_delayed_message_exchange

wget https://github.com/rabbitmq/rabbitmq-delayed-message-exchange/releases/download/3.10.2/rabbitmq_delayed_message_exchange-3.10.2.ez
docker cp rabbitmq_delayed_message_exchange-3.10.2.ez 98e0d1d698de:/plugins
docker exec -it 98e0d1d698de /bin/bash
>rabbitmq-plugins enable rabbitmq_delayed_message_exchange
docker restart 98e0d1d698de
"""
import json
import time,datetime
from aiormq.abc import DeliveredMessage
import asyncio


from aio_pika import DeliveryMode, ExchangeType, Message, connect
from multiprocessing import Process

HOST = "49.235.242.224"
PORT = 50009


async def init():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        normal_ex = await channel.declare_exchange("delay-ex", ExchangeType.X_DELAYED_MESSAGE,arguments={"x-delayed-type": "direct"})
        delay_queue = await channel.declare_queue(name="delay-queue")
        await delay_queue.bind(exchange=normal_ex, routing_key="")


async def producer():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        normal_ex = await channel.get_exchange("delay-ex")
        data = {}
        for i in range(10):
            data["delay-time"] = i*1000 #seconds = delay-time/1000
            data["index"] = i
            data["time"] = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
            ack = await normal_ex.publish(
                Message(
                    json.dumps(data, ensure_ascii=False).encode("utf-8"),
                    delivery_mode=DeliveryMode.PERSISTENT,
                    headers={'x-delay': data["delay-time"]},
                ),
                routing_key="",
                # mandatory=True #无法使用 会返回错误的 ack
            )
            # if isinstance(ack, DeliveredMessage): print("message not send")


async def consumer():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        delay_queue = await channel.get_queue("delay-queue")

        async def callback(message) -> None:
            async with message.process():
                msg_time = datetime.datetime.now()
                msg = json.loads(message.body)
                print(f"[DELAY] Received message is: {msg} {msg_time-datetime.datetime.strptime(msg['time'],'%Y%m%d %H%M%S')}")
        await delay_queue.consume(callback)

        print("[info] Waiting for logs. To exit press CTRL+C")
        await asyncio.Future()

def run():
    asyncio.run(init())
    asyncio.run(producer())
    asyncio.run(consumer())
if __name__ == '__main__':
    run()
