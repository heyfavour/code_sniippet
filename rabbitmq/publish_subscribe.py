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
    # 1.创建凭证，使用rabbitmq用户密码登录
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    # 创建一个指定名称的交换机，并指定类型为fanout，用于将接收到的消息广播到所有queue中
    channel.exchange_declare(exchange='fanout—ex', exchange_type='fanout')
    channel.confirm_delivery()  # 发布确认
    # 将消息发送给指定的交换机，在fanout类型中，routing_key=''表示不用发送到指定queue中，而是将发送到绑定到此交换机的所有queue
    data = {"id": 1, "status": "成功", "time": datetime.datetime.now().strftime("%Y%m%d %H%M%S")}
    try:
        for i in range(5):
            channel.basic_publish(
                exchange='fanout—ex',
                routing_key="",
                body=json.dumps(data, ensure_ascii=False),
                properties=pika.BasicProperties(delivery_mode=pika.DeliveryMode.Persistent),
                # Persistent = 2 声明消息在队列中持久化 Transient = 1 消息非持久化
            )
    except Exception as e:
        print(f'Message could not be confirmed{str(e)}')
    # 程序退出前，确保刷新网络缓冲以及消息发送给rabbitmq，需要关闭本次连接
    connection.close()


def consumer():  # auto_ack=False
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    channel = connection.channel()
    channel.exchange_declare(exchange='fanout—ex', exchange_type='fanout')

    # 使用RabbitMQ给自己生成一个专有的queue
    #queue = channel.queue_declare(queue='', exclusive=True)
    queue = channel.queue_declare(queue="EMAIL",durable=True)
    queue_name = queue.method.queue
    # 将queue绑定到指定交换机
    channel.queue_bind(exchange='fanout—ex', queue=queue_name)

    # 4. 定义消息处理程序
    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}] Received :{json.loads(body)}')
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)  # 手动应答 multiple = True 批量应答 可能会造成消息丢失

    # 5. 接收来自指定queue的消息
    # 在prefetch_count表示接收的消息数量，当我接收的消息没有处理完（用basic_ack标记消息已处理完毕）之前不会再接收新的消息了
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发
    # queue 接收指定queue的消息 on_message_callback 接收到消息后的处理程序 auto_ack 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息 arguments
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)
    print('[*] Waiting for message.')
    # 6. 开始循环等待，一直处于等待接收消息的状态
    channel.start_consuming()


def consumer1():  # auto_ack=False
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    # 2. 创建一个channel
    channel = connection.channel()
    # 指定交换机
    channel.exchange_declare(exchange='fanout—ex', exchange_type='fanout')

    # 使用RabbitMQ给自己生成一个专有的queue
    #queue = channel.queue_declare(queue='', exclusive=True)
    queue = channel.queue_declare(queue="EMAIL",durable=True)
    queue_name = queue.method.queue
    # 将queue绑定到指定交换机
    channel.queue_bind(exchange='fanout—ex', queue=queue_name)

    # 4. 定义消息处理程序
    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}] Received :{json.loads(body)}')
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)  # 手动应答 multiple = True 批量应答 可能会造成消息丢失

    # 5. 接收来自指定queue的消息
    # 在prefetch_count表示接收的消息数量，当我接收的消息没有处理完（用basic_ack标记消息已处理完毕）之前不会再接收新的消息了
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发
    # queue 接收指定queue的消息 on_message_callback 接收到消息后的处理程序 auto_ack 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息 arguments
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)
    print('[*] Waiting for message.')
    # 6. 开始循环等待，一直处于等待接收消息的状态
    channel.start_consuming()


def consumer2():  # auto_ack=False
    credentials = pika.PlainCredentials("product", "product")
    con_para = pika.ConnectionParameters(host=HOST, port=PORT, virtual_host='/product', credentials=credentials)
    connection = pika.BlockingConnection(con_para)
    # 2. 创建一个channel
    channel = connection.channel()
    # 指定交换机
    channel.exchange_declare(exchange='fanout—ex', exchange_type='fanout')

    # 使用RabbitMQ给自己生成一个专有的queue
    queue = channel.queue_declare(queue="WECHAT",durable=True)
    queue_name = queue.method.queue
    # 将queue绑定到指定交换机
    channel.queue_bind(exchange='fanout—ex', queue=queue_name)

    # 4. 定义消息处理程序
    def callback(ch, method, properties, body):
        print(f'[{os.getpid()}] Received :{json.loads(body)}')
        ch.basic_ack(delivery_tag=method.delivery_tag, multiple=False)  # 手动应答 multiple = True 批量应答 可能会造成消息丢失

    # 5. 接收来自指定queue的消息
    # 在prefetch_count表示接收的消息数量，当我接收的消息没有处理完（用basic_ack标记消息已处理完毕）之前不会再接收新的消息了
    channel.basic_qos(prefetch_count=1)  # 避免性能不同但是公平分发
    # queue 接收指定queue的消息 on_message_callback 接收到消息后的处理程序 auto_ack 指定为True，表示消息接收到后自动给消息发送方回复确认，已收到消息 arguments
    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=False)
    print('[*] Waiting for message.')
    # 6. 开始循环等待，一直处于等待接收消息的状态
    channel.start_consuming()


if __name__ == '__main__':
    """
    发布订阅  持久化是queue的能力 exchange只负责转发给queue 如果exchange转发的时候没有queue订阅，则丢弃报文
             路由键在 fanout 类型的交换机中不起作用
             因此可以1.consumer先订阅 producer再生产
                        channel.queue_declare(queue='', exclusive=True)
                        consumer_p1 = Process(target=consumer1)
                        consumer_p2 = Process(target=consumer2)
                        consumer_p1.start()
                        consumer_p2.start()
                        producer()
                        queue = channel.queue_declare(queue="name",durable=True)
                    2.先持久化queue  producer再生产 consumer再消费
                        producer()
                        consumer_p1 = Process(target=consumer1)
                        consumer_p2 = Process(target=consumer2)
                        consumer_p1.start()
                        consumer_p2.start()
    """
    producer()
    consumer_p1 = Process(target=consumer1)
    consumer_p2 = Process(target=consumer2)
    consumer_p1.start()
    consumer_p2.start()

