"""
 'alternate-exchange': 'backup-ex'
 备份交换机 交换机收到路由不到队列的消息就会发送到备用交换机绑定的队列中
"""
import datetime
import time

import pika
import json, os

HOST = "49.235.242.224"
PORT = 50009


def init():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    # 确认交换机
    channel.exchange_declare(exchange="confirm-ex", exchange_type="direct")
    channel.queue_declare(queue="confirm-queue")
    channel.queue_bind(queue="confirm-queue", exchange="confirm-ex", routing_key="")
    channel.close()

def init_backup():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.exchange_declare(exchange="backup-ex", exchange_type="fanout")
    channel.queue_declare(queue="backup-queue")
    channel.queue_bind(queue="backup-queue", exchange="backup-ex")
    # 确认交换机
    arguments = { 'alternate-exchange': 'backup-ex'}
    channel.exchange_declare(exchange="confirm-ex", exchange_type="direct",arguments=arguments)
    channel.queue_declare(queue="confirm-queue")
    channel.queue_bind(queue="confirm-queue", exchange="confirm-ex", routing_key="")
    channel.close()


def producer_ex():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.confirm_delivery()  # 仅仅确认交换机收到
    data = {"id": 1, "status": "成功"}
    channel.basic_publish(
        exchange='fake-ex',
        routing_key="",
        body=json.dumps(data, ensure_ascii=False),
        properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent),
    )
    # 程序退出前，确保刷新网络缓冲以及消息发送给rabbitmq，需要关闭本次连接
    # pika.exceptions.ChannelClosedByBroker
    connection.close()
    print('[PRODUCER] SEND ALL.')


def producer_queue():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.confirm_delivery()  # 仅仅确认交换机收到
    data = {"id": 1, "status": "成功"}
    channel.basic_publish(
        exchange='confirm-ex',
        routing_key="fake-ex",
        body=json.dumps(data, ensure_ascii=False),
        properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent),
        mandatory=True,  # 确认队列收到
    )
    # 程序退出前，确保刷新网络缓冲以及消息发送给rabbitmq，需要关闭本次连接
    # pika.exceptions.UnroutableError
    connection.close()
    print('[PRODUCER] SEND ALL.')


def producer_backup():
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.confirm_delivery()  # 仅仅确认交换机收到
    data = {"id": 1, "status": "成功"}
    try:
        channel.basic_publish(
            exchange='confirm-ex',
            routing_key="fake-ex",
            body=json.dumps(data, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent),
            mandatory=True,  # 确认队列收到
        )
        # 程序退出前，确保刷新网络缓冲以及消息发送给rabbitmq，需要关闭本次连接
    except (pika.exceptions.ChannelClosedByBroker, pika.exceptions.UnroutableError) as e:
        print(f"[BACKUP] END ERROR{str(e)}")
    connection.close()
    print('[PRODUCER] SEND ALL.')


def producer():
    #init()
    # producer_ex()
    # producer_queue()
    init_backup()
    producer_backup()


if __name__ == '__main__':
    producer()
