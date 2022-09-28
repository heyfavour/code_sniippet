import datetime
import json
import os
import time

import pika

from multiprocessing import Process

"""
    producer->channel->exchange->Routingkey(binding)->queue->channel->consumer
                               ->Routingkey(binding)->queue->channel->consumer

    exchange  type "" direct topic headers fanout-广播

    binding:queue->exchange

    49.235.242.224
"""
HOST = "49.235.242.224"
PORT = 50009
QUEUE_NAME = "BANK_CALLBACK"


def producer():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.exchange_declare(exchange='topic—ex', exchange_type='topic')
    channel.confirm_delivery()
    data = {"id": 1, "status": "成功", "time": datetime.datetime.now().strftime("%Y%m%d %H%M%S")}
    direct_dict = {0: "bank.info", 1: "bank.warnning", 2: "bank.error"}
    try:
        for i in range(3):
            routing_key = direct_dict[i % 3]
            data["level"] = routing_key
            channel.basic_publish(
                exchange='topic—ex',
                routing_key=routing_key,
                body=json.dumps(data, ensure_ascii=False),
                properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent),
            )
    except Exception as e:
        print(f'ACK ERROR : {str(e)}')
    # 程序退出前，确保刷新网络缓冲以及消息发送给rabbitmq，需要关闭本次连接
    connection.close()


def consumer_bank():  # auto_ack=False
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    # 2. 创建一个channel
    channel = connection.channel()
    # 指定交换机
    channel.exchange_declare(exchange='topic—ex', exchange_type='topic')
    queue = channel.queue_declare(queue="BANK", durable=True)
    queue_name = queue.method.queue
    channel.queue_bind(queue=queue_name, exchange='topic—ex', routing_key="bank.*")

    # 4. 定义消息处理程序
    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}:BANK] Received :{json.loads(body)}')
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)  # 手动应答 multiple = True 批量应答 可能会造成消息丢失

    # 5. 接收来自指定queue的消息
    # 在prefetch_count表示接收的消息数量，当我接收的消息没有处理完（用basic_ack标记消息已处理完毕）之前不会再接收新的消息了
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发
    # queue 接收指定queue的消息 on_message_callback 接收到消息后的处理程序 auto_ack 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息 arguments
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)
    print('[*] Waiting for message.')
    channel.start_consuming()


def consumer_error():  # auto_ack=False
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    queue = channel.queue_declare(queue="ERROR", durable=True)
    queue_name = queue.method.queue
    channel.queue_bind(queue=queue_name,exchange='topic—ex', routing_key="*.error")
    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}:ERROR] Received :{json.loads(body)}')
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)  # 手动应答 multiple = True 批量应答 可能会造成消息丢失
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)
    print('[*] Waiting for message.')
    channel.start_consuming()


if __name__ == '__main__':
    producer()
    info = Process(target=consumer_bank)
    error = Process(target=consumer_error)
    info.start()
    error.start()
