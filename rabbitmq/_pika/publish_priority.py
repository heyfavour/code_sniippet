import datetime
import time

import pika
import json, os
from multiprocessing import Process

HOST = "49.235.242.224"
PORT = 50009


def init():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.exchange_declare(exchange="priority-ex", exchange_type="direct")

    arguments = {'x-max-priority': 5}  # 0-255越大越优先 不建议太大会因为排序浪费性能
    channel.queue_declare(queue="priority-queue",arguments=arguments)
    channel.queue_bind(queue="priority-queue", exchange="priority-ex",routing_key="")
    channel.close()


def producer():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.confirm_delivery()  # 仅仅确认交换机收到
    data = {"id": 1, "status": "成功"}
    for i in range(20):
        data["priority"] = i % 5
        data["id"] = i
        channel.basic_publish(
            exchange='priority-ex',
            routing_key="",
            body=json.dumps(data, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent,priority=data["priority"] ),
            mandatory=True,
        )
    connection.close()
    print('[PRODUCER] SEND ALL.')


def consumer():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()

    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}] Received :{json.loads(body)}')
        time.sleep(0.5)
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)  # 手动应答 multiple = True 批量应答 可能会造成消息丢失

    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发
    channel.basic_consume(queue="priority-queue", on_message_callback=callback, auto_ack=False)
    channel.start_consuming()


def run():
    init()
    p = Process(target=producer)
    c = Process(target=consumer)
    p.start()
    c.start()


if __name__ == '__main__':
    run()
