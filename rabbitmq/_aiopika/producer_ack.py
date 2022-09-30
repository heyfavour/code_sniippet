import datetime
import json
import os
import asyncio
from aiormq.abc import DeliveredMessage
from aio_pika import DeliveryMode, ExchangeType, Message, connect
from multiprocessing import Process

HOST = "49.235.242.224"
PORT = 50009

"""
mandatory
True:如果exchange根据自身类型和消息routeKey无法找到一个符合条件的queue，那么会调用basic.return方法将消息返回给生产者
DeliveredMessage(delivery=Basic.Return header=Content-Header,body=Content-Body)

immediate
True:如果exchange在将消息路由到queue(s)时发现对于的queue上么有消费者，那么这条消息不会放入队列中
"""

async def ack_async():
    connection = await connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()  # publisher_confirms 交换机确认 默认True
        direct_ex = await channel.declare_exchange("direct-ex", ExchangeType.DIRECT)
        # ack_queue = await channel.declare_queue("ack_async",durable=True)
        # await ack_queue.bind(direct_ex,routing_key="ack_async")
        data = {"id": 0, "status": "成功"}
        try:
            start = datetime.datetime.now()
            for i in range(10):
                ack = await direct_ex.publish(
                    Message(
                        json.dumps(data, ensure_ascii=False).encode("utf-8"),
                        delivery_mode=DeliveryMode.PERSISTENT,
                    ),
                    routing_key="ack_async",
                    mandatory=True,
                )
                if isinstance(ack,DeliveredMessage):print("message not send")
            end = datetime.datetime.now()

            print(f"{end-start}")
        except Exception as e:
            print(f'Message could not be confirmed{str(e)}')



if __name__ == '__main__':
    asyncio.run(ack_async())
