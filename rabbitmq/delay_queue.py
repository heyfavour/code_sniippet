"""
1.TTL+死信队列
过于麻烦
2.pluguin
rabbitmq-plugins enable rabbitmq_delayed_message_exchange

wget https://github.com/rabbitmq/rabbitmq-delayed-message-exchange/releases/download/3.10.2/rabbitmq_delayed_message_exchange-3.10.2.ez
docker cp rabbitmq_delayed_message_exchange-3.10.2.ez 98e0d1d698de:/plugins
docker exec -it 98e0d1d698de /bin/bash
>rabbitmq-plugins enable rabbitmq_delayed_message_exchang
docker restart 98e0d1d698de
"""
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

    channel.exchange_declare(exchange="delay-ex", exchange_type="x-delayed-message", arguments={"x-delayed-type": "direct"})
    channel.queue_declare(queue="delay-queue")
    channel.queue_bind(queue="delay-queue", exchange="delay-ex", routing_key="")
    channel.close()


def producer():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.confirm_delivery()
    data = {"id": 1, "status": "成功"}
    try:
        for i in range(10):
            data["delay-time"] = i*1000
            data["index"] = i
            data["time"] = datetime.datetime.now().strftime("%Y%m%d %H%M%S")
            channel.basic_publish(
                exchange='delay-ex',
                routing_key="",
                body=json.dumps(data, ensure_ascii=False),
                properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent,headers={'x-delay': data["delay-time"]}),
            )
    except Exception as e:
        print(f'ACK ERROR : {str(e)}')
    # 程序退出前，确保刷新网络缓冲以及消息发送给rabbitmq，需要关闭本次连接
    connection.close()
    print('[PRODUCER] SEND ALL.')


def consumer():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}:Delay:{datetime.datetime.now()}] Received :{json.loads(body)}')
        print("sleep",datetime.datetime.now() - datetime.datetime.strptime(json.loads(body)["time"],"%Y%m%d %H%M%S"))
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)

    channel.basic_consume(queue="delay-queue", on_message_callback=callback, auto_ack=False)
    print('[NORMAL] Waiting for message.')
    channel.start_consuming()


if __name__ == '__main__':
    init()
    produce = Process(target=producer)
    con = Process(target=consumer)
    produce.start()
    time.sleep(20)
    con.start()
