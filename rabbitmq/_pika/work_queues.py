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


def producer():
    # 1.创建凭证，使用rabbitmq用户密码登录
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
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
        channel.basic_publish(
            exchange='',
            routing_key=QUEUE_NAME,
            body=json.dumps(data, ensure_ascii=False),
            properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent),
            # Persistent = 2 声明消息在队列中持久化 Transient = 1 消息非持久化
        )
        print('Message publish was confirmed')
    except Exception as e:
        print('Message could not be confirmed')
    # 程序退出前，确保刷新网络缓冲以及消息发送给rabbitmq，需要关闭本次连接
    connection.close()


def consumer():  # auto_ack=False
    # 1. 创建一个到RabbitMQ server的连接，如果连接的不是本机，
    # 则在pika.ConnectionParameters中传入具体的ip和port即可
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    # 2. 创建一个channel
    channel = connection.channel()

    # 3. 创建队列，queue_declare可以使用任意次数，
    # 如果指定的queue不存在，则会创建一个queue，如果已经存在， 则不会做其他动作，官方推荐，每次使用时都可以加上这句
    # queue_declare:passive 队列是否存在  durable 持久化 exclusive 其他信道无法访问 auto_delete 数据消费完成后是否删除队列
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    # 4. 定义消息处理程序
    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}] Received :{json.loads(body)}')
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)  # 手动应答 multiple = True 批量应答 可能会造成消息丢失

    # 5. 接收来自指定queue的消息
    # 在prefetch_count表示接收的消息数量，当我接收的消息没有处理完（用basic_ack标记消息已处理完毕）之前不会再接收新的消息了
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发
    # queue 接收指定queue的消息 on_message_callback 接收到消息后的处理程序 auto_ack 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息 arguments
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=False)
    print('[*] Waiting for message.')
    # 6. 开始循环等待，一直处于等待接收消息的状态
    channel.start_consuming()


def consumer1():  # auto_ack=False
    # 1. 创建一个到RabbitMQ server的连接，如果连接的不是本机，
    # 则在pika.ConnectionParameters中传入具体的ip和port即可
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    # 2. 创建一个channel
    channel = connection.channel()

    # 3. 创建队列，queue_declare可以使用任意次数，
    # 如果指定的queue不存在，则会创建一个queue，如果已经存在， 则不会做其他动作，官方推荐，每次使用时都可以加上这句
    # queue_declare:passive 队列是否存在  durable 持久化 exclusive 其他信道无法访问 auto_delete 数据消费完成后是否删除队列
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    # 4. 定义消息处理程序
    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}] Received :{json.loads(body)}')
        time.sleep(1)
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)  # 手动应答 multiple = True 批量应答 可能会造成消息丢失

    # 5. 接收来自指定queue的消息
    # 在prefetch_count表示接收的消息数量，当我接收的消息没有处理完（用basic_ack标记消息已处理完毕）之前不会再接收新的消息了
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发
    # queue 接收指定queue的消息 on_message_callback 接收到消息后的处理程序 auto_ack 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息 arguments
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=False)
    print('[*] Waiting for message.')
    # 6. 开始循环等待，一直处于等待接收消息的状态
    channel.start_consuming()


def consumer2():  # auto_ack=False
    # 1. 创建一个到RabbitMQ server的连接，如果连接的不是本机，
    # 则在pika.ConnectionParameters中传入具体的ip和port即可
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    # 2. 创建一个channel
    channel = connection.channel()

    # 3. 创建队列，queue_declare可以使用任意次数，
    # 如果指定的queue不存在，则会创建一个queue，如果已经存在， 则不会做其他动作，官方推荐，每次使用时都可以加上这句
    # queue_declare:passive 队列是否存在  durable 持久化 exclusive 其他信道无法访问 auto_delete 数据消费完成后是否删除队列
    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    # 4. 定义消息处理程序
    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}] Received :{json.loads(body)}')
        time.sleep(2)
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)  # 手动应答 multiple = True 批量应答 可能会造成消息丢失

    # 5. 接收来自指定queue的消息
    # 在prefetch_count表示接收的消息数量，当我接收的消息没有处理完（用basic_ack标记消息已处理完毕）之前不会再接收新的消息了
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发
    # queue 接收指定queue的消息 on_message_callback 接收到消息后的处理程序 auto_ack 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息 arguments
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback, auto_ack=False)
    print('[*] Waiting for message.')
    # 6. 开始循环等待，一直处于等待接收消息的状态
    channel.start_consuming()


if __name__ == '__main__':
    for i in range(100): producer()
    procs = []
    # for _ in range(5):
    #     proc = Process(target=consumer)
    #     procs.append(proc)
    #     proc.start()
    # for proc in procs:
    #     proc.join()
    proc1 = Process(target=consumer1)
    proc2 = Process(target=consumer2)
    proc1.start()
    proc2.start()
