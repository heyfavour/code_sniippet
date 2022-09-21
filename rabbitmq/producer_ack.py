#!/usr/bin/env python
# pip install pika
import json
import os
import time

import pika

from multiprocessing import Process

"""
    producer->channel->exchange->queue->channel->consumer
    49.235.242.224
"""
HOST = "49.235.242.224"
PORT = 50009
QUEUE_NAME = "BANK_CALLBACK"


def ack_block():
    # 1.创建凭证，使用rabbitmq用户密码登录
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)#heartbeat=600,blocked_connection_timeout=300
    connection = pika.BlockingConnection(con_para)
    # 2.创建channel

    channel = connection.channel()
    # 3. 创建队列，queue_declare可以使用任意次数，
    # 如果指定的queue不存在，则会创建一个queue，如果已经存在，
    # 则不会做其他动作，官方推荐，每次使用时都可以加上这句
    # queue_declare:passive 队列是否存在  durable 持久化 exclusive 其他信道无法访问 auto_delete 数据消费完成后是否删除队列  arguments 延时 死信
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    # 注意在rabbitmq中，消息想要发送给队列，必须经过交换(exchange)，
    # 初学可以使用空字符串交换(exchange='')，
    # 它允许我们精确的指定发送给哪个队列(routing_key=''),
    # 参数body值发送的数据
    data = {"id": 1, "status": "成功"}
    channel.confirm_delivery()  # 发布确认
    # channel.basic_publish(exchange='', routing_key=QUEUE_NAME, body=json.dumps(data, ensure_ascii=False))
    # Send a message
    try:
        for i in range(100):
            ack = channel.basic_publish(
                exchange='',
                routing_key=QUEUE_NAME,
                body=json.dumps(data, ensure_ascii=False),
                properties=pika.BasicProperties(content_type='text/plain', delivery_mode=pika.DeliveryMode.Persistent)
            )
            print(f'ack is {ack}')
    except Exception as e:
        print('ERROR ACK')
    # 程序退出前，确保刷新网络缓冲以及消息发送给rabbitmq，需要关闭本次连接
    connection.close()


def ack_async():
    pass


if __name__ == '__main__':
    ack_block()
    ack_async()
