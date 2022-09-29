import json
import aio_pika
import asyncio

from multiprocessing import Process

HOST = "49.235.242.224"
PORT = 50009


async def init():
    pass


async def producer(i):
    connection = await aio_pika.connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection
    async with connection:
        channel = await connection.channel()
        data = {"id": i, "status": "成功"}
        await channel.default_exchange.publish(
            aio_pika.Message(
                json.dumps(data, ensure_ascii=False).encode("utf-8"),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            ),
            routing_key="work-queues",
        )
        #print(f"[x] Sent {data}")


async def consumer1():
    connection = await aio_pika.connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.declare_queue("work-queues", durable=True)

        async def callback(message) -> None:
            # async with message.process():
            #     print(f"[C1] Received message is: {json.loads(message.body)}")
            # if True:await message.reject(requeue=False)
            print(f"[C1] Received message is: {json.loads(message.body)}")
            await message.ack()

        await queue.consume(callback,no_ack=False)

        print("[C1] Waiting for messages. To exit press CTRL+C")
        await asyncio.Future()


async def consumer2():
    connection = await aio_pika.connect(host=HOST, port=PORT, login="admin", password="admin")  # ->Connection

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        queue = await channel.declare_queue("work-queues", durable=True)

        async def callback(message) -> None:
            # try:
            #     async with message.process(requeue=True):
            #         print(f"[C2] Received message is: {json.loads(message.body)}")
            #         raise Exception("[C2] Exception")
            # except Exception as exc:
            #     print(str(exc))
            async with message.process(ignore_processed=True):# Now (with ignore_processed=True) you may reject (or ack) message manually too
                print(f"[C2] Received message is: {json.loads(message.body)}")
                if True:  # some reasonable condition here
                    await message.reject(requeue=True)


        await queue.consume(callback,no_ack=False)

        print("[C2] Waiting for messages. To exit press CTRL+C")
        await asyncio.Future()

def run_producer():
    for i in range(4):
        asyncio.run(producer(i))

def run_consumer_1():
    asyncio.run(consumer1())


def run_consumer_2():
    asyncio.run(consumer2())


def run():
    p = Process(target=run_producer)
    c1 = Process(target=run_consumer_1)
    c2 = Process(target=run_consumer_2)
    p.start()
    c1.start()
    c2.start()



if __name__ == '__main__':
    run()
